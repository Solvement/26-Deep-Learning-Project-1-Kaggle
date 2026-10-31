"""Independent fixed SAM160 + SE-WRN22-4/r16 scratch experiment.

Foret et al., ICLR 2021: https://arxiv.org/abs/2010.01412
Two forwards/backwards reuse one already augmented/Mixup minibatch. The SAM
helper freezes second-pass BN statistics and restores perturbations safely.
Only native CUDA BF16 or CPU FP32 is supported; no GradScaler/FP16 path exists.
Use a budgeted supervisor. No confirmation or hidden-image loader is called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import environment, seed_all, write_json
import numpy as np
import torch
from data import load_training_arrays
from model import count_parameters
from research.se_model import ARCHITECTURE, PARAMETER_COUNT, build_model
from research.train_sam import prepare_batch, loss_closure, resolve_sam_precision
from research.sam import sam_step
from research.train_shadow import capture_rng, evaluate, restore_rng, validate_partition
from train import (DEFAULT_CONFIG, MEAN, STD, code_provenance, save_checkpoint,
                   schedule, sha256, tensor_images, training_source)

SCHEMA = "research_se_sam_training_v1"
ENGINEERING_SCHEMA = "research_se_sam_engineering_v1"
SAM_RECIPE_SHA256 = "0418de7f24086160d06651af13f77016ffbd3c03f26245731570c27b02fb762a"
SOURCE_KEYS = ("research/train_se_sam.py", "research/se_model.py", "research/train_sam.py", "research/sam.py",
               "research/train_shadow.py", "research/augmentations.py", "research/configs/wrn22_sam.json")


def fixed_config(seed=2026):
    if type(seed) is not int or seed not in (2026, 2027):
        raise ValueError("SE permits only primary2026 and a later independent2027")
    return {**DEFAULT_CONFIG, "sam_rho": .05, "architecture": ARCHITECTURE, "se_reduction": 16, "seed": seed}


def validate_config(config):
    if not isinstance(config, dict) or config != fixed_config(config.get("seed")) or set(config) != set(fixed_config()):
        raise ValueError("SE requires the exact fixed SAM160 recipe; only architecture and SE reduction are added")
    if any(type(config[k]) is not type(v) for k, v in fixed_config(config["seed"]).items()):
        raise ValueError("SE configuration types must match the fixed recipe exactly")
    if sha256(ROOT/"research/configs/wrn22_sam.json") != SAM_RECIPE_SHA256:
        raise ValueError("The fixed SAM reference recipe changed")


def model_configuration(config):
    validate_config(config)
    return {"architecture": ARCHITECTURE, **{k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")},
            "reduction": config["se_reduction"]}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def input_config(config, raw=None):
    validate_config(config)
    raw = raw if raw is not None else (json.dumps(config, indent=2)+"\n").encode()
    if not isinstance(raw, bytes) or {**DEFAULT_CONFIG, **json.loads(raw)} != config:
        raise ValueError("Exact raw input configuration differs from the effective fixed SE recipe")
    return raw


def research_provenance(precision, config, input_config_sha256=None):
    validate_config(config)
    if precision not in ("bf16", "fp32"):
        raise ValueError("SE SAM requires native BF16 or CPU FP32, with no scaler")
    raw_sha = input_config_sha256 or hashlib.sha256(input_config(config)).hexdigest()
    if not isinstance(raw_sha, str) or len(raw_sha) != 64 or any(c not in "0123456789abcdef" for c in raw_sha):
        raise ValueError("SE raw configuration fingerprint is required")
    return {"schema": SCHEMA, "method": "se_sam", "architecture": ARCHITECTURE, "reduction": 16,
            "rho": config["sam_rho"], "adaptive": False, "precision": precision, "grad_scaler_enabled": False,
            "source_sha256": {key:sha256(ROOT/key) for key in SOURCE_KEYS},
            "effective_config_sha256": config_digest(config), "input_config_sha256": raw_sha,
            "training_seed": config["seed"], "split_seed": 2026,
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def validate_resume(last, config, signature, provenance, steps_per_epoch):
    validate_config(config)
    if last.get("schema") != SCHEMA or last.get("config") != config or last.get("data_signature") != signature:
        raise ValueError("SE resume configuration, schema or official split differs")
    if last.get("model_config") != model_configuration(config) or last.get("parameter_count") != PARAMETER_COUNT:
        raise ValueError("SE resume architecture or parameter count differs")
    previous = last.get("provenance", {})
    if (previous.get("source_sha256") != provenance["source_sha256"] or
            previous.get("research_training") != provenance["research_training"]):
        raise ValueError("SE resume source/config/precision/seed provenance differs")
    epoch = last.get("epoch")
    if (type(epoch) is not int or not 1 <= epoch < config["epochs"] or steps_per_epoch != 313 or
            [row.get("epoch") for row in last.get("history", [])] != list(range(1, epoch+1)) or
            type(last.get("updates")) is not int or last["updates"] != epoch*steps_per_epoch or
            last.get("skipped_updates") != 0 or last.get("sam_forward_passes") != 2*last["updates"]):
        raise ValueError("SE resume needs every completed two-pass update and an exact partial history")
    for row in last["history"]:
        if row.get("successful_updates") != 313 or row.get("skipped_updates") != 0 or row.get("sam_forward_passes") != 626:
            raise ValueError("SE resume history has a missing update/pass")
    if any(k not in last for k in ("model", "ema", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")) or "grad_scaler" in last:
        raise ValueError("SE resume needs model/EMA/SGD/RNG and forbids a scaler")
    if (not isinstance(last["torch_rng"], torch.Tensor) or last["torch_rng"].dtype != torch.uint8 or
            not isinstance(last["cuda_rng"], list) or
            (provenance["research_training"]["precision"] == "bf16" and not last["cuda_rng"])):
        raise ValueError("SE resume RNG state is incomplete")
    groups = last["optimizer"].get("param_groups", [])
    if (len(groups) != 1 or groups[0].get("momentum") != .9 or groups[0].get("weight_decay") != .0005 or
            groups[0].get("nesterov") is not True or not groups[0].get("params")):
        raise ValueError("SE resume must retain the same SGD optimizer state")
    validate_saved_state(last, config)


def validate_saved_state(last, config):
    """Validate actual SE tensor names/shapes and all SGD momentum buffers."""
    reference = build_model(model_configuration(config))
    expected = reference.state_dict()
    for kind in ("model", "ema"):
        state = last.get(kind, {})
        if set(state) != set(expected):
            raise ValueError("SE saved model/EMA state keys do not match the architecture")
        for key, tensor in expected.items():
            actual = state[key]
            if (not isinstance(actual, torch.Tensor) or actual.shape != tensor.shape or actual.dtype != tensor.dtype or
                    not torch.isfinite(actual).all()):
                raise ValueError("SE saved model/EMA tensor shape, dtype or finite values differ")
    parameters = list(reference.parameters())
    optimizer = last.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    expected_ids = list(range(len(parameters)))
    if len(groups) != 1 or groups[0].get("params") != expected_ids or set(optimizer.get("state", {})) != set(expected_ids):
        raise ValueError("SE saved SGD parameter topology or momentum state is incomplete")
    for index, parameter in enumerate(parameters):
        state = optimizer["state"][index]
        tensor = state.get("momentum_buffer")
        if (set(state) != {"momentum_buffer"} or not isinstance(tensor, torch.Tensor) or
                tensor.shape != parameter.shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError("SE saved SGD momentum buffer shape, dtype or finite values differ")
    del reference


def gate_parameters(model):
    gates = {name:parameter for name, parameter in model.named_parameters() if ".se." in name}
    if len(gates) != 36 or len({name.split(".se.")[0] for name in gates}) != 9:
        raise ValueError("Expected all four parameters from each of nine SE gates")
    return gates


def gate_update_evidence(model, before, gradients):
    parameters = gate_parameters(model)
    if set(before) != set(parameters) or set(gradients) != set(parameters):
        raise ValueError("Every SE gate parameter needs a pre-update tensor and actual gradient")
    report = {}
    for gate in sorted({name.split(".se.")[0] for name in parameters}):
        names = [name for name in parameters if name.startswith(gate+".se.")]
        grads, changes = [gradients[name].detach().float() for name in names], [parameters[name].detach()-before[name] for name in names]
        if any(gradients[name].shape != parameters[name].shape or before[name].shape != parameters[name].shape for name in names):
            raise ValueError("SE gate gradient/state dimensions differ")
        grad_norm = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item())
        change_norm = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in changes])).item())
        if not math.isfinite(grad_norm) or grad_norm <= 0 or not math.isfinite(change_norm) or change_norm <= 0:
            raise FloatingPointError("Each SE gate needs finite positive gradient and actual parameter change")
        report[gate] = {"gradient_l2":grad_norm, "update_l2":change_norm}
    return report


def train(config, output, data_dir, device, precision="auto", stop_after_epoch=None, resume=False, input_config_bytes=None):
    validate_config(config)
    precision, dtype = resolve_sam_precision(precision, device)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must lie within configured epochs")
    raw = input_config(config, input_config_bytes)
    raw_sha = hashlib.sha256(raw).hexdigest()
    output, data_dir = Path(output), Path(data_dir)
    artifacts = ("last.pt", "best.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json", "input_config.json")
    if not resume and any((output/name).exists() for name in artifacts):
        raise FileExistsError("Existing SAM artifacts require explicit resume or a new directory")
    if resume and not (output/"last.pt").is_file():
        raise FileNotFoundError("SAM resume requires last.pt")
    seed_all(config["seed"])
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    manifest = json.loads((data_dir/"training_manifest.json").read_text(encoding="utf-8"))
    validate_partition(train_ids, val_ids, manifest, config)
    source = training_source(data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    provenance = code_provenance()
    marker = research_provenance(precision, config, raw_sha)
    provenance["research_training"] = marker
    train_x, val_x = tensor_images(images[train_ids], device), tensor_images(images[val_ids], device)
    train_y, val_y = torch.as_tensor(labels[train_ids], device=device), torch.as_tensor(labels[val_ids], device=device)
    del images, labels
    model_config = model_configuration(config)
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    parameters = count_parameters(model)
    if parameters != PARAMETER_COUNT or count_parameters(model, trainable_only=True) != PARAMETER_COUNT:
        raise ValueError("SAM model exceeds the 5,000,000 parameter limit")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    steps = math.ceil(len(train_ids)/config["batch_size"])
    history, start_epoch, updates, previous_seconds = [], 1, 0, 0.0
    best_accuracy, best_loss = -1.0, float("inf")
    if resume:
        last = torch.load(output/"last.pt", map_location=device, weights_only=False)
        if (output/"input_config.json").read_bytes() != raw:
            raise ValueError("Raw SE resume input configuration changed")
        validate_resume(last, config, signature, provenance, steps)
        model.load_state_dict(last["model"])
        ema.load_state_dict(last["ema"])
        optimizer.load_state_dict(last["optimizer"])
        history, start_epoch, updates = last["history"], last["epoch"]+1, last["updates"]
        best_accuracy, best_loss, previous_seconds = last["best_accuracy"], last["best_loss"], last["elapsed_seconds"]
        restore_rng(last, device)
        del last
    end_epoch = min(config["epochs"], stop_after_epoch or config["epochs"])
    if end_epoch < start_epoch:
        raise ValueError("stop-after-epoch precedes the resumed epoch")
    output.mkdir(parents=True, exist_ok=True)
    (output/"input_config.json").write_bytes(raw)
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {**environment(), "schema": SCHEMA, "se_sam_training_precision": precision,
                                         "grad_scaler_enabled": False, "research_training": marker})
    write_json(output/"provenance.json", {"schema": SCHEMA, "code": provenance, "official_data": source, "data_signature": signature,
               "train_examples": len(train_ids), "validation_examples": len(val_ids), "parameter_count": parameters,
               "normalization": {"mean": MEAN, "std": STD}})
    print(json.dumps({"event": "se_sam_training_started", "precision": precision, "parameters": parameters,
                      "training_examples": len(train_ids), "validation_examples": len(val_ids), "config": config}), flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, end_epoch+1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_y), device=device)
        loss_sum = sam_loss_sum = perturbation_sum = grad_norm_sum = 0.0
        correct = torch.zeros((), device=device)
        seen, epoch_updates = 0, 0
        for batch, start in enumerate(range(0, len(order), config["batch_size"])):
            lr = schedule((epoch-1)*steps+batch, config["epochs"]*steps, steps, config)
            for group in optimizer.param_groups: group["lr"] = lr
            ids = order[start:start+config["batch_size"]]
            y = train_y[ids]
            # Both SAM passes reuse these exact tensors, permutation and lambda.
            prepared = prepare_batch(train_x[ids], y, config)
            _, _, permutation, lam = prepared
            first = {}
            closure = loss_closure(model, prepared, config, device, dtype, first)
            report = sam_step(model, optimizer, closure, rho=config["sam_rho"], precision=precision)
            if report["updated"] is not True or report["forward_backward_passes"] != 2:
                raise RuntimeError("SAM helper did not perform exactly one two-pass SGD update")
            updates += 1
            epoch_updates += 1
            with torch.no_grad():
                decay = min(config["ema_decay"], (1+updates)/(10+updates))
                torch._foreach_lerp_(ema_params, model_params, 1-decay)
                predicted = first["logits"].argmax(1)
                correct += predicted.eq(y).sum()*lam
                if permutation is not None: correct += predicted.eq(y[permutation]).sum()*(1-lam)
            loss_sum += report["first_loss"]*len(y)
            sam_loss_sum += report["second_loss"]*len(y)
            perturbation_sum += report["perturbation_norm"]*len(y)
            grad_norm_sum += report["grad_norm"]*len(y)
            seen += len(y)
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers()): average.copy_(online)
        regular, average = evaluate(model, val_x, val_y, device, dtype), evaluate(ema, val_x, val_y, device, dtype)
        choice = "ema" if (average["accuracy"], -average["loss"]) >= (regular["accuracy"], -regular["loss"]) else "model"
        metric = average if choice == "ema" else regular
        elapsed = previous_seconds+time.perf_counter()-started
        record = {"epoch": epoch, "train_loss": loss_sum/seen, "sam_loss": sam_loss_sum/seen,
            "mean_perturbation_norm": perturbation_sum/seen, "mean_first_grad_norm": grad_norm_sum/seen,
            "train_accuracy": correct.item()/seen, "validation_loss": regular["loss"], "validation_accuracy": regular["accuracy"],
            "ema_validation_loss": average["loss"], "ema_validation_accuracy": average["accuracy"], "selected_variant": choice,
            "lr": lr, "epoch_seconds": time.perf_counter()-epoch_started, "elapsed_seconds": elapsed,
            "successful_updates": epoch_updates, "skipped_updates": 0, "sam_forward_passes": 2*epoch_updates}
        history.append(record)
        if (metric["accuracy"], -metric["loss"]) > (best_accuracy, -best_loss):
            best_accuracy, best_loss = metric["accuracy"], metric["loss"]
            selected = ema if choice == "ema" else model
            save_checkpoint(output/"best.pt", {"schema": SCHEMA, "model": selected.state_dict(), "model_config": model_config, "config": config,
                "epoch": epoch, "variant": choice, "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                "parameter_count": parameters, "data_signature": signature, "provenance": provenance})
        write_json(output/"history.json", history)
        write_json(output/"metrics.json", {"schema": SCHEMA, "status": "complete" if epoch == config["epochs"] else "running",
            "epochs_completed": epoch, "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
            "parameter_count": parameters, "elapsed_seconds": elapsed, "training_examples": len(train_ids),
            "validation_examples": len(val_ids), "config": config, "research_training": marker,
            "successful_updates": updates, "skipped_updates": 0, "sam_forward_passes": 2*updates})
        save_checkpoint(output/"last.pt", {"schema": SCHEMA, "model_config": model_config, "parameter_count": parameters,
            "sam_forward_passes": 2*updates, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "updates": updates, "skipped_updates": 0, "config": config, "data_signature": signature,
            "provenance": provenance, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss,
            "elapsed_seconds": elapsed, **capture_rng(device)})
        print(json.dumps(record), flush=True)
    print(json.dumps({"event": "se_sam_training_complete" if end_epoch == config["epochs"] else "se_sam_training_paused",
                      "epochs_completed": end_epoch, "best_validation_accuracy": best_accuracy,
                      "successful_updates": updates, "skipped_updates": 0}), flush=True)


def benchmark(config, output, device, precision, steps=20, input_config_bytes=None):
    """Synthetic two-pass SAM engineering check; never loads official examples."""
    validate_config(config)
    precision, dtype = resolve_sam_precision(precision, device)
    if type(steps) is not int or steps != 20:
        raise ValueError("The predeclared SE preflight requires 3 warmup and 20 timed steps")
    raw = input_config(config, input_config_bytes)
    path = Path(output)/"engineering_benchmark.json"
    if path.exists():
        raise FileExistsError("SE engineering evidence is immutable")
    seed_all(config["seed"])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_configuration(config)).to(device=device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=.9, weight_decay=.0005, nesterov=True)
    images = torch.randint(0, 256, (128, 32, 32, 3), dtype=torch.uint8, device=device)
    labels = torch.arange(128, device=device) % 10
    gates, gradients, handles = gate_parameters(model), {}, []
    for name, parameter in gates.items():
        def keep_gradient(gradient, name=name):
            gradients[name] = gradient.detach()
        handles.append(parameter.register_hook(keep_gradient))
    durations, reports, gate_reports = [], [], []
    try:
        for index in range(23):
            gradients.clear()
            before = {name:parameter.detach().clone() for name, parameter in gates.items()}
            counters = [(module, module.num_batches_tracked.detach().clone()) for module in model.modules()
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            if device.type == "cuda": torch.cuda.synchronize(device)
            began = time.perf_counter()
            prepared = prepare_batch(images, labels, config)
            report = sam_step(model, optimizer, loss_closure(model, prepared, config, device, dtype, {}),
                              rho=.05, precision=precision)
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-began
            if report["updated"] is not True or report["forward_backward_passes"] != 2 or not 0 <= report["perturbation_norm"] <= .050001:
                raise RuntimeError("SE preflight did not prove the fixed two-pass SAM update")
            if any(not torch.equal(module.num_batches_tracked, old+1) for module, old in counters):
                raise RuntimeError("SE preflight second SAM pass changed BN counters")
            gate_reports.append(gate_update_evidence(model, before, gradients))
            if index >= 3:
                durations.append(elapsed)
                reports.append(report)
    finally:
        for handle in handles: handle.remove()
    median = float(np.median(durations))
    result = {"schema": ENGINEERING_SCHEMA, "status": "complete", "synthetic_only": True,
        "precision": precision, "parameters": count_parameters(model), "model_config": model_configuration(config),
        "config": config, "input_config_sha256": hashlib.sha256(raw).hexdigest(),
        "batch_size": 128, "warmup_steps": 3, "timed_steps": 20, "successful_updates": 23,
        "sam_forward_passes": 46, "skipped_updates": 0, "gate_changed_steps": 23,
        "gate_gradient_verified_steps": 23, "gate_count": 9, "gate_evidence": gate_reports,
        "second_pass_bn_verified": True, "median_step_seconds": median,
        "p90_step_seconds": float(np.quantile(durations, .9)), "images_per_second": 128/median,
        "estimated_40k_training_seconds": median*313*160,
        "required_training_and_benchmark_seconds": 1.5*median*313*160+300,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "mean_first_loss": float(np.mean([row["first_loss"] for row in reports])),
        "mean_sam_loss": float(np.mean([row["second_loss"] for row in reports])),
        "mean_perturbation_norm": float(np.mean([row["perturbation_norm"] for row in reports])),
        "research_training": research_provenance(precision, config, hashlib.sha256(raw).hexdigest()),
        "environment": environment(), "training_data_loaded": False, "confirmation_evaluated": False,
        "hidden_images_loaded": False, "external_data_loaded": False,
        "interpretation": "Synthetic engineering only. Fixed 1.5 pipeline margin and 300s finalization reserve; no measured accuracy."}
    write_json(path, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG, **json.loads(args.config.read_text(encoding="utf-8"))}
    if args.benchmark_steps:
        if args.resume or args.stop_after_epoch is not None:
            parser.error("Synthetic benchmark cannot be combined with training resume/stop")
        benchmark(config, args.output_dir, torch.device(args.device), args.precision, args.benchmark_steps, input_config_bytes=args.config.read_bytes())
    else:
        train(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision, args.stop_after_epoch, args.resume, input_config_bytes=args.config.read_bytes())


if __name__ == "__main__":
    main()
