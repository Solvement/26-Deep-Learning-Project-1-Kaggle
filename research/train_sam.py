"""Separate scratch SAM experiment: frozen official 40k train / 5k development.

Foret et al., ICLR 2021: https://arxiv.org/abs/2010.01412
Two forwards/backwards reuse one already augmented/Mixup minibatch. The SAM
helper freezes second-pass BN statistics and restores perturbations safely.
Only native CUDA BF16 or CPU FP32 is supported; no GradScaler/FP16 path exists.
Use a budgeted supervisor. No confirmation or hidden-image loader is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import environment, seed_all, write_json
import numpy as np
import torch
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from research.sam import sam_step
from research.train_shadow import capture_rng, evaluate, resolve_precision, restore_rng, validate_partition
from research.train_shadow import validate_config as validate_base_config
from train import (DEFAULT_CONFIG, MEAN, STD, augment, code_provenance, save_checkpoint,
                   schedule, sha256, tensor_images, training_source)

SCHEMA = "research_sam_training_v1"
SOURCE_KEYS = ("research/train_sam.py", "research/sam.py", "research/train_shadow.py", "research/augmentations.py")


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG) | {"sam_rho"}:
        raise ValueError("SAM requires exactly the base training keys plus sam_rho")
    validate_base_config({key: value for key, value in config.items() if key != "sam_rho"})
    rho = config["sam_rho"]
    if isinstance(rho, bool) or not isinstance(rho, (float, int)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("sam_rho must be positive and finite")
    if config["num_classes"] != 10:
        raise ValueError("SAM requires the official ten classes")


def resolve_sam_precision(requested, device):
    if requested == "fp16":
        raise ValueError("SAM FP16 is unsupported; use native BF16 or CPU FP32")
    precision, dtype = resolve_precision(requested, device)
    if (device.type == "cuda" and precision != "bf16") or (device.type == "cpu" and precision != "fp32"):
        raise ValueError("SAM requires native CUDA BF16 or CPU FP32; no FP16/scaler path")
    return precision, dtype


def research_provenance(precision, config):
    return {"schema": SCHEMA, "method": "sam", "rho": config["sam_rho"], "adaptive": False,
            "source_sha256": {key: sha256(ROOT/key) for key in SOURCE_KEYS},
            "precision": precision, "grad_scaler_enabled": False,
            "training_seed": config["seed"], "split_seed": config["split_seed"],
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def prepare_batch(images, labels, config):
    """Draw augmentation/Mixup once outside the two-pass SAM closure."""
    x = augment(images, config["cutout"])
    mixed = config["mixup_alpha"] > 0 and random.random() < config["mixup_probability"]
    lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mixed else 1.0
    permutation = torch.randperm(len(labels), device=labels.device) if mixed else None
    if mixed: x = x*lam+x[permutation]*(1-lam)
    return x, labels, permutation, lam


def loss_closure(model, batch, config, device, dtype, first):
    x, y, permutation, lam = batch
    def closure():
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"])*lam
            if permutation is not None:
                loss += F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"])*(1-lam)
        if "logits" not in first: first["logits"] = logits.detach()
        return loss
    return closure


def validate_resume(last, config, signature, provenance, steps_per_epoch):
    if last.get("config") != config or last.get("data_signature") != signature:
        raise ValueError("SAM resume configuration or official data/split mismatch")
    previous = last.get("provenance", {})
    if (previous.get("source_sha256") != provenance["source_sha256"] or
            previous.get("research_training") != provenance["research_training"]):
        raise ValueError("SAM resume requires identical core/helper/trainer sources, precision and seeds")
    epoch = last.get("epoch")
    if (type(epoch) is not int or not 1 <= epoch < config["epochs"] or
            [row.get("epoch") for row in last.get("history", [])] != list(range(1, epoch+1)) or
            last.get("updates") != epoch*steps_per_epoch or last.get("skipped_updates") != 0):
        raise ValueError("SAM resume requires a partial complete history with every successful update")


def train(config, output, data_dir, device, precision="auto", stop_after_epoch=None, resume=False):
    validate_config(config)
    precision, dtype = resolve_sam_precision(precision, device)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must lie within configured epochs")
    output, data_dir = Path(output), Path(data_dir)
    artifacts = ("last.pt", "best.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json")
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
    marker = research_provenance(precision, config)
    provenance["research_training"] = marker
    train_x, val_x = tensor_images(images[train_ids], device), tensor_images(images[val_ids], device)
    train_y, val_y = torch.as_tensor(labels[train_ids], device=device), torch.as_tensor(labels[val_ids], device=device)
    del images, labels
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    parameters = count_parameters(model)
    if parameters > 5000000 or count_parameters(model, trainable_only=True) > 5000000:
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
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {**environment(), "sam_training_precision": precision,
                                         "grad_scaler_enabled": False, "research_training": marker})
    write_json(output/"provenance.json", {"code": provenance, "official_data": source, "data_signature": signature,
               "train_examples": len(train_ids), "validation_examples": len(val_ids), "parameter_count": parameters,
               "normalization": {"mean": MEAN, "std": STD}})
    print(json.dumps({"event": "sam_training_started", "precision": precision, "parameters": parameters,
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
            save_checkpoint(output/"best.pt", {"model": selected.state_dict(), "model_config": model_config, "config": config,
                "epoch": epoch, "variant": choice, "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                "parameter_count": parameters, "data_signature": signature, "provenance": provenance})
        write_json(output/"history.json", history)
        write_json(output/"metrics.json", {"status": "complete" if epoch == config["epochs"] else "running",
            "epochs_completed": epoch, "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
            "parameter_count": parameters, "elapsed_seconds": elapsed, "training_examples": len(train_ids),
            "validation_examples": len(val_ids), "config": config, "research_training": marker,
            "successful_updates": updates, "skipped_updates": 0, "sam_forward_passes": 2*updates})
        save_checkpoint(output/"last.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "updates": updates, "skipped_updates": 0, "config": config, "data_signature": signature,
            "provenance": provenance, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss,
            "elapsed_seconds": elapsed, **capture_rng(device)})
        print(json.dumps(record), flush=True)
    print(json.dumps({"event": "sam_training_complete" if end_epoch == config["epochs"] else "sam_training_paused",
                      "epochs_completed": end_epoch, "best_validation_accuracy": best_accuracy,
                      "successful_updates": updates, "skipped_updates": 0}), flush=True)


def benchmark(config, output, device, precision, steps):
    """Engineering throughput only: synthetic batches, no official loader."""
    validate_config(config)
    precision, dtype = resolve_sam_precision(precision, device)
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("benchmark-steps must be an integer from 1 through 1000")
    path = Path(output)/"engineering_benchmark.json"
    if path.exists():
        raise FileExistsError("Existing engineering benchmark is immutable")
    seed_all(config["seed"])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    images = torch.randint(0, 256, (config["batch_size"], 32, 32, 3), dtype=torch.uint8, device=device)
    labels = torch.arange(config["batch_size"], device=device) % 10
    durations, reports, changed = [], [], 0
    probe = next(model.parameters())
    for index in range(3+steps):
        if device.type == "cuda": torch.cuda.synchronize(device)
        before = probe.detach().clone()
        began = time.perf_counter()
        prepared = prepare_batch(images, labels, config)
        report = sam_step(model, optimizer, loss_closure(model, prepared, config, device, dtype, {}),
                          rho=config["sam_rho"], precision=precision)
        if device.type == "cuda": torch.cuda.synchronize(device)
        duration = time.perf_counter()-began
        if report["updated"] is not True or report["forward_backward_passes"] != 2:
            raise RuntimeError("Engineering SAM update failed")
        changed += int(not torch.equal(probe.detach(), before))
        if index >= 3:
            durations.append(duration)
            reports.append(report)
    if changed != steps+3:
        raise RuntimeError("Engineering SAM steps did not each change real model parameters")
    median = float(np.median(durations))
    result = {"schema": "research_sam_engineering_v1", "status": "complete", "synthetic_only": True,
        "precision": precision, "parameters": count_parameters(model), "model_config": model_config,
        "batch_size": config["batch_size"], "warmup_steps": 3, "timed_steps": steps,
        "successful_updates": steps+3, "parameter_changed_steps": changed, "sam_forward_passes": 2*(steps+3),
        "median_step_seconds": median, "p90_step_seconds": float(np.quantile(durations, .9)),
        "images_per_second": config["batch_size"]/median,
        "estimated_40k_epoch_seconds": median*math.ceil(40000/config["batch_size"]),
        "estimated_40k_training_seconds": median*math.ceil(40000/config["batch_size"])*config["epochs"],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "mean_first_loss": float(np.mean([r["first_loss"] for r in reports])),
        "mean_sam_loss": float(np.mean([r["second_loss"] for r in reports])),
        "mean_perturbation_norm": float(np.mean([r["perturbation_norm"] for r in reports])),
        "research_training": research_provenance(precision, config), "environment": environment(),
        "training_data_loaded": False, "confirmation_evaluated": False, "hidden_images_loaded": False,
        "interpretation": "Synthetic engineering throughput; training estimates exclude validation, EMA, checkpoint I/O and full pipeline overhead; no accuracy evidence."}
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
        benchmark(config, args.output_dir, torch.device(args.device), args.precision, args.benchmark_steps)
    else:
        train(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision, args.stop_after_epoch, args.resume)


if __name__ == "__main__":
    main()
