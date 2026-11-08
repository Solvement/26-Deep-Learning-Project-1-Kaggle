"""Typed fixed SWA tail of an audited own scratch WRN34-3 prefix.

No tail development selection, original confirmation, hidden inputs or uploads.
The frozen 160-epoch cosine prefix is never recomputed as a 200-epoch schedule.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import environment, seed_all, write_json
import numpy as np
import torch
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from research.swa import average_parameters_, recompute_training_bn_
from research.train_shadow import capture_rng, evaluate, restore_rng, validate_partition
from train import DEFAULT_CONFIG, augment, code_provenance, save_checkpoint, schedule, sha256, tensor_images, training_source

SCHEMA = "research_swa_training_v1"
SOURCE_KEYS = ("research/train_swa.py", "research/swa.py", "research/train_shadow.py", "research/augmentations.py")
SWA_KEYS = {"swa_prefix_epochs", "swa_tail_epochs", "swa_lr", "swa_bn_seed"}
BASE_CONFIG = {**DEFAULT_CONFIG, "depth": 34, "widen_factor": 3}
PREFIX_EPOCHS, TAIL_EPOCHS, TRAIN_EXAMPLES, VALIDATION_EXAMPLES = 160, 40, 40000, 5000
PARAMETERS = 4166298
CACHED_PARENT_SHA = "0686e9e7698512fb87233586c00e956e05876fdb869633019c1b394a4029bb8b"
CACHED_RECEIPT_SHA = "2e8da4c929b703f31f7daa380c2b68d2794692b89087809ac137947a70bbd349"
STABILITY_SHA = "258c4f4714f1382ace8864141a817a14e629d15b01628ef0c94497247cacd049"


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def config_hash(config): return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config):
    expected = {**BASE_CONFIG, "epochs": PREFIX_EPOCHS+TAIL_EPOCHS, "swa_prefix_epochs": PREFIX_EPOCHS,
        "swa_tail_epochs": TAIL_EPOCHS, "swa_lr": .01, "swa_bn_seed": 2026}
    require(set(config) == set(expected) and type(config.get("seed")) is int and config["seed"] in (2026, 2027) and
        all(config[k] == v and type(config[k]) is type(v) for k,v in expected.items() if k != "seed"),
        "SWA accepts only the predeclared WRN34-3 160+40 recipe and seeds 2026/2027")


def prefix_config(config):
    validate_config(config)
    return {key: PREFIX_EPOCHS if key == "epochs" else value for key,value in config.items() if key not in SWA_KEYS}


def resolve_precision(requested, device):
    require(requested in ("bf16", "fp32"), "SWA FP16/automatic precision is unsupported")
    if device.type == "cpu":
        require(requested == "fp32", "CPU SWA requires FP32")
        return "fp32", torch.float32
    require(device.type == "cuda" and requested == "bf16", "CUDA SWA requires native BF16")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is unavailable")
    return "bf16", torch.bfloat16


def research_provenance(precision, config):
    return {"schema": SCHEMA, "method": "swa", "precision": precision,
        "source_sha256": {name: sha256(ROOT/name) for name in SOURCE_KEYS}, "training_seed": config["seed"]}


def snapshots(paths): return {str(Path(path).resolve()): sha256(path) for path in set(map(Path, paths))}


def unchanged(record): require(snapshots(record) == record, "SWA immutable parent/source evidence changed")


def validate_origin(path, config, core, precision):
    folder = path.parent
    origin_path = folder/"prefix_origin.json"
    if origin_path.is_file():
        origin = read(origin_path)
        raw_path = folder/"input_config.json"
        require(origin.get("schema") == "research_swa_prefix_origin_v1" and origin.get("status") == "complete" and
            origin.get("init") == "scratch_seeded" and origin.get("parent_weights_loaded") is False and origin.get("resumed") is False and
            origin.get("seed") == config["seed"] and origin.get("source_sha256") == core and
            origin.get("checkpoint_sha256") == sha256(path) and origin.get("config_sha256") == sha256(folder/"config.json") and
            Path(origin.get("rawinputconfig_path", "")).resolve() == raw_path.resolve() and
            origin.get("rawinputconfig_sha256") == sha256(raw_path) and {**DEFAULT_CONFIG, **read(raw_path)} == config,
            "Fresh prefix requires a source-bound scratch origin and exact original inputs")
        command = origin.get("command", [])
        require(len(command) == 10 and Path(command[0]).resolve() == Path(sys.executable).resolve() and command[1:] == ["train.py", "--config", str(raw_path.resolve()), "--output-dir",
            str(folder.resolve()), "--data-dir", str((ROOT/"data").resolve()), "--device", "cuda" if precision == "bf16" else "cpu"],
            "Prefix command must run only frozen train.py from scratch without resume/parent weights")
        first, last = [datetime.fromisoformat(origin[k]) for k in ("started_at", "finished_at")]
        require(first.tzinfo is not None and last.tzinfo is not None and first < last <= datetime.now(timezone.utc), "Prefix origin timestamps are invalid")
        return origin, [origin_path, raw_path]
    receipt_path = folder/"supervisor.json"
    require(path.resolve() == (ROOT/"results/research/wrn34_3_s2026/last.pt").resolve() and config["seed"] == 2026 and
        sha256(path) == CACHED_PARENT_SHA and sha256(receipt_path) == CACHED_RECEIPT_SHA,
        "Only the exact previously audited own primary prefix may use a cached origin")
    receipt = read(receipt_path)
    require(receipt.get("status") == "complete" and receipt.get("resumed") is False and
        receipt.get("confirmation_evaluated") is False and receipt.get("submitted") is False,
        "Cached primary prefix was not a completed scratch-only run")
    return {"schema": "audited_cached_scratch_prefix_v1", "init": "scratch_seeded", "resumed": False,
        "parent_weights_loaded": False, "started_at": receipt["started_at"], "finished_at": receipt["finished_at"]}, [receipt_path]


def validate_parent(parent_last_path, config, data_signature=None):
    """Deep CPU audit; return tensors plus separately source-bound parent metadata."""
    original = prefix_config(config)
    path = Path(parent_last_path).resolve()
    require(path.name == "last.pt" and "stability_seed2027" not in path.parts and sha256(path) != STABILITY_SHA,
        "SWA prefix must be the final own last.pt, never best or original stability")
    folder = path.parent
    paths = [path, *(folder/name for name in ("config.json", "metrics.json", "history.json", "provenance.json", "environment.json"))]
    before = snapshots(paths)
    last = torch.load(path, map_location="cpu", weights_only=False)
    saved, metrics, history, provenance, env = [read(p) for p in paths[1:]]
    core = code_provenance()["source_sha256"]
    steps = math.ceil(TRAIN_EXAMPLES/original["batch_size"])
    require(saved == original == last.get("config") == metrics.get("config") and last.get("epoch") == PREFIX_EPOCHS and
        metrics.get("status") == "complete" and metrics.get("epochs_completed") == PREFIX_EPOCHS and
        metrics.get("training_examples") == provenance.get("train_examples") == TRAIN_EXAMPLES and
        metrics.get("validation_examples") == provenance.get("validation_examples") == VALIDATION_EXAMPLES and
        last.get("updates") == PREFIX_EPOCHS*steps and last.get("history") == history and
        [row.get("epoch") for row in history] == list(range(1, PREFIX_EPOCHS+1)), "Prefix configuration/history/actual counts are not the fixed completed 160 epochs")
    signature = last.get("data_signature")
    require(signature == provenance.get("data_signature") and (data_signature is None or signature == data_signature) and
        last.get("provenance", {}).get("source_sha256") == core and provenance.get("code") == last["provenance"] and
        "research_training" not in last["provenance"] and provenance.get("official_data", {}).get("source") == "official_kaggle_competition_training",
        "Prefix uses different sources, a research method or unofficial data")
    require(all(type(row.get(key)) in (float, int) and math.isfinite(row[key]) for row in history
        for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "ema_validation_loss", "ema_validation_accuracy", "lr")) and
        all(math.isclose(row["lr"], schedule(i*steps-1, PREFIX_EPOCHS*steps, steps, original), abs_tol=1e-12)
        for i,row in enumerate(history, 1)), "Prefix finite history or original 160-epoch cosine schedule differs")
    model_config = {key: original[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config)
    require(count_parameters(model) == count_parameters(model, True) == metrics.get("parameter_count") == provenance.get("parameter_count") == PARAMETERS,
        "Prefix actual model parameter count differs")
    for key in ("model", "ema"):
        model.load_state_dict(last[key], strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()), "Prefix contains nonfinite model state")
    groups = last.get("optimizer", {}).get("param_groups", [])
    parameters = list(model.parameters())
    require(len(groups) == 1 and groups[0].get("params") == list(range(len(parameters))) and
        groups[0].get("lr") == history[-1]["lr"] and groups[0].get("momentum") == original["momentum"] and
        groups[0].get("weight_decay") == original["weight_decay"] and groups[0].get("nesterov") is True and
        groups[0].get("dampening") == 0 and groups[0].get("maximize") is False,
        "Prefix SGD momentum/configuration is missing or mismatched")
    state = last["optimizer"].get("state", {})
    require(set(state) == set(range(len(parameters))) and all(set(state[i]) == {"momentum_buffer"} and
        state[i]["momentum_buffer"].shape == p.shape and state[i]["momentum_buffer"].dtype == torch.float32 and
        torch.isfinite(state[i]["momentum_buffer"]).all().item() for i,p in enumerate(parameters)), "Prefix SGD momentum tensors differ")
    try:
        random.Random().setstate(last["python_rng"])
        np.random.RandomState().set_state(last["numpy_rng"])
        with torch.random.fork_rng(devices=[]): torch.set_rng_state(last["torch_rng"])
    except (KeyError, ValueError, TypeError, RuntimeError) as error: raise ValueError("Prefix RNG state is missing or invalid") from error
    precision = "bf16" if env.get("device") == "cuda" and env.get("amp_dtype") == "torch.bfloat16" else "fp32" if env.get("device") == "cpu" else None
    require(precision is not None and isinstance(last.get("cuda_rng"), list) and
        ((precision == "fp32" and not last["cuda_rng"]) or (precision == "bf16" and bool(last["cuda_rng"]) and
        all(isinstance(r, torch.Tensor) and r.dtype == torch.uint8 and r.ndim == 1 and r.numel() > 0 for r in last["cuda_rng"]))),
        "Prefix native training precision/CUDA RNG provenance differs")
    origin, origin_paths = validate_origin(path, original, core, precision)
    paths += origin_paths+[ROOT/name for name in core]
    unchanged(before)
    metadata = {"checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "epoch": PREFIX_EPOCHS,
        "config": original, "config_sha256": sha256(folder/"config.json"), "precision": precision,
        "successful_updates": last["updates"], "training_examples": TRAIN_EXAMPLES, "validation_examples": VALIDATION_EXAMPLES,
        "data_signature": signature, "source_sha256": core, "origin": origin, "evidence_sha256": snapshots(paths)}
    return {"last": last, "metadata": metadata, "paths": paths, "prefix_history": history}


def update(model, optimizer, images, labels, config, device, dtype):
    x = augment(images, config["cutout"])
    mixed = random.random() < config["mixup_probability"]
    lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mixed else 1.
    permutation = torch.randperm(len(labels), device=device) if mixed else None
    if mixed: x = x*lam+x[permutation]*(1-lam)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        logits = model(x)
        loss = F.cross_entropy(logits, labels, label_smoothing=config["label_smoothing"])*lam
        if mixed: loss += F.cross_entropy(logits, labels[permutation], label_smoothing=config["label_smoothing"])*(1-lam)
    if not torch.isfinite(loss).item(): raise FloatingPointError("Nonfinite SWA tail loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack(torch._foreach_norm(gradients, math.inf)).isfinite().all().item():
        raise FloatingPointError("Nonfinite SWA tail gradient")
    optimizer.step()
    predicted = logits.detach().argmax(1)
    correct = predicted.eq(labels).sum()*lam
    if mixed: correct += predicted.eq(labels[permutation]).sum()*(1-lam)
    return loss.detach().item(), correct.detach().item()


def validate_resume(last, config, signature, provenance, parent):
    tail = last.get("history", {}).get("tail", [])
    epoch, steps = last.get("epoch"), math.ceil(TRAIN_EXAMPLES/config["batch_size"])
    require(last.get("schema") == "research_swa_last_v1" and last.get("config") == config and
        last.get("data_signature") == signature and last.get("provenance") == provenance and last.get("parent_training") == parent and
        type(epoch) is int and PREFIX_EPOCHS < epoch < PREFIX_EPOCHS+TAIL_EPOCHS and
        [r.get("epoch") for r in tail] == list(range(PREFIX_EPOCHS+1, epoch+1)) and
        last.get("swa_count") == len(tail) and last.get("tail_updates") == len(tail)*steps and
        last.get("updates") == (PREFIX_EPOCHS+len(tail))*steps and
        all(r.get("swa_count") == i and r.get("lr") == config["swa_lr"] and r.get("successful_updates") == steps and
        r.get("skipped_updates") == 0 and all(r.get(k) is None for k in ("validation_loss", "validation_accuracy", "ema_validation_loss", "ema_validation_accuracy"))
        for i,r in enumerate(tail, 1)), "SWA resume requires identical source/parent/config and contiguous unselected tail state")


def train(config, parent_last, output, data_dir, device, precision="bf16", stop_after_epoch=None, resume=False):
    validate_config(config)
    precision, dtype = resolve_precision(precision, device)
    output, data_dir = Path(output), Path(data_dir)
    require(stop_after_epoch is None or type(stop_after_epoch) is int and PREFIX_EPOCHS < stop_after_epoch <= config["epochs"], "SWA stop epoch must lie inside the fixed tail")
    outputs = ("last.pt", "swa.pt", "raw_tail.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json")
    if not resume and any((output/name).exists() for name in outputs): raise FileExistsError("Use a new immutable SWA directory")
    parent = validate_parent(parent_last, config)
    require(parent["metadata"]["precision"] == precision, "SWA tail precision must match its own prefix")
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    manifest = read(data_dir/"training_manifest.json")
    validate_partition(train_ids, val_ids, manifest, config)
    require(len(train_ids) == TRAIN_EXAMPLES and len(val_ids) == VALIDATION_EXAMPLES, "SWA requires the fixed official 40k/5k split")
    source = training_source(data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(signature == parent["metadata"]["data_signature"], "SWA actual official data differs from parent prefix")
    marker = research_provenance(precision, config)
    provenance = {**code_provenance(), "research_training": marker}
    evidence = {**parent["metadata"]["evidence_sha256"], **snapshots([ROOT/name for name in SOURCE_KEYS]),
        **snapshots([data_dir/"training_manifest.json"])}
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    average = copy.deepcopy(model).eval().requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["swa_lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    model.load_state_dict(parent["last"]["model"])
    ema.load_state_dict(parent["last"]["ema"])
    optimizer.load_state_dict(parent["last"]["optimizer"])
    history = {"prefix": parent["prefix_history"], "tail": []}
    start, count, elapsed_before = PREFIX_EPOCHS+1, 0, 0.
    trajectory = parent["last"]
    if resume:
        trajectory = torch.load(output/"last.pt", map_location=device, weights_only=False)
        validate_resume(trajectory, config, signature, provenance, parent["metadata"])
        require(trajectory["history"]["prefix"] == parent["prefix_history"], "SWA resumed prefix history differs")
        model.load_state_dict(trajectory["model"]); average.load_state_dict(trajectory["swa_model"])
        ema.load_state_dict(trajectory["ema"])
        optimizer.load_state_dict(trajectory["optimizer"])
        history, start, count, elapsed_before = trajectory["history"], trajectory["epoch"]+1, trajectory["swa_count"], trajectory["elapsed_seconds"]
    train_x, val_x = tensor_images(images[train_ids], device), tensor_images(images[val_ids], device)
    train_y, val_y = torch.as_tensor(labels[train_ids], device=device), torch.as_tensor(labels[val_ids], device=device)
    del images, labels
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    write_json(output/"provenance.json", {"code": provenance, "official_data": source, "data_signature": signature,
        "train_examples": TRAIN_EXAMPLES, "validation_examples": VALIDATION_EXAMPLES, "parameter_count": PARAMETERS, "parent_training": parent["metadata"]})
    write_json(output/"environment.json", {**environment(), "swa_training_precision": precision, "grad_scaler_enabled": False, "research_training": marker})
    restore_rng(trajectory, device)  # After allocations/initializers; preserve the exact parent/raw trajectory.
    del parent["last"], trajectory
    steps = math.ceil(TRAIN_EXAMPLES/config["batch_size"])
    began = time.perf_counter()
    end = stop_after_epoch or config["epochs"]
    require(end >= start, "SWA stop epoch precedes the resumed tail")
    metrics = None
    for epoch in range(start, end+1):
        epoch_started = time.perf_counter(); model.train()
        for group in optimizer.param_groups: group["lr"] = config["swa_lr"]
        order = torch.randperm(TRAIN_EXAMPLES, device=device)
        loss_sum = correct = 0.
        for batch,begin in enumerate(range(0, TRAIN_EXAMPLES, config["batch_size"])):
            ids = order[begin:begin+config["batch_size"]]
            loss, accuracy = update(model, optimizer, train_x[ids], train_y[ids], config, device, dtype)
            updates = (epoch-1)*steps+batch+1
            with torch.no_grad():
                decay = min(config["ema_decay"], (1+updates)/(10+updates))
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
            loss_sum += loss*len(ids); correct += accuracy
        with torch.no_grad():
            for target,source_buffer in zip(ema.buffers(), model.buffers()): target.copy_(source_buffer)
        count = average_parameters_(average, model, count)
        elapsed = elapsed_before+time.perf_counter()-began
        row = {"epoch": epoch, "train_loss": loss_sum/TRAIN_EXAMPLES, "train_accuracy": correct/TRAIN_EXAMPLES,
            "lr": config["swa_lr"], "successful_updates": steps, "skipped_updates": 0, "swa_count": count,
            "validation_loss": None, "validation_accuracy": None, "ema_validation_loss": None, "ema_validation_accuracy": None,
            "epoch_seconds": time.perf_counter()-epoch_started, "elapsed_seconds": elapsed}
        history["tail"].append(row)
        unchanged(evidence)
        metrics = {"status": "running", "selection": "fixed_swa_final", "no_best_selection": True,
            "epochs_completed": epoch, "prefix_epochs": PREFIX_EPOCHS, "tail_epochs": count, "swa_count": count,
            "prefix_updates": PREFIX_EPOCHS*steps, "tail_updates": count*steps, "successful_updates": (PREFIX_EPOCHS+count)*steps,
            "skipped_updates": 0, "training_examples": TRAIN_EXAMPLES, "validation_examples": VALIDATION_EXAMPLES,
            "config": config, "parameter_count": PARAMETERS, "research_training": marker, "parent_training": parent["metadata"]}
        metrics.update(ema_tracked=True, ema_selected=False, confirmation_evaluated=False, hidden_images_loaded=False, external_data_loaded=False)
        write_json(output/"history.json", history); write_json(output/"metrics.json", metrics)
        save_checkpoint(output/"last.pt", {"schema": "research_swa_last_v1", "model": model.state_dict(), "swa_model": average.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "updates": (PREFIX_EPOCHS+count)*steps, "tail_updates": count*steps,
            "swa_count": count, "config": config, "data_signature": signature, "provenance": provenance,
            "parent_training": parent["metadata"], "history": history, "elapsed_seconds": elapsed, **capture_rng(device)})
        print(json.dumps(row), flush=True)
    if end != config["epochs"]: return metrics
    require(count == TAIL_EPOCHS, "SWA final sample count is not the predeclared40")
    final_metrics, bn = {}, {}
    for variant, selected in (("swa", average), ("raw_tail_control", model)):
        bn[variant] = recompute_training_bn_(selected, train_x, batch_size=config["batch_size"], seed=config["swa_bn_seed"], precision=precision, device=device)
        selected.eval()
        final_metrics[variant] = {**evaluate(selected, val_x, val_y, device, dtype), "examples": VALIDATION_EXAMPLES}
        save_checkpoint(output/("swa.pt" if variant == "swa" else "raw_tail.pt"), {"schema": "research_swa_checkpoint_v1",
            "model": selected.state_dict(), "model_config": model_config, "config": config, "epoch": config["epochs"], "variant": variant,
            "parameter_count": PARAMETERS, "data_signature": signature, "provenance": provenance, "parent_training": parent["metadata"],
            "swa_count": count, "prefix_updates": PREFIX_EPOCHS*steps, "tail_updates": count*steps,
            "successful_updates": (PREFIX_EPOCHS+count)*steps, "skipped_updates": 0, "bn_recomputation": bn[variant],
            "final_validation": final_metrics[variant], "selection": "fixed_swa_final" if variant == "swa" else "fixed_raw_tail_control", "no_best_selection": True})
    unchanged(evidence)
    require(bn["swa"]["augmented_input_sha256"] == bn["raw_tail_control"]["augmented_input_sha256"],
        "SWA/control BN recomputation did not consume identical fixed-seed training views")
    metrics.update(status="complete", final_validation=final_metrics, bn_recomputation=bn)
    write_json(output/"metrics.json", metrics)
    return metrics


def benchmark(config, output, device, precision, steps):
    validate_config(config); precision, dtype = resolve_precision(precision, device)
    require(type(steps) is int and 1 <= steps <= 1000, "Positive bounded synthetic benchmark steps required")
    path = Path(output)/"engineering_benchmark.json"
    if path.exists(): raise FileExistsError("Engineering evidence is immutable")
    marker = research_provenance(precision, config)
    evidence = snapshots([ROOT/name for name in (*SOURCE_KEYS, *code_provenance()["source_sha256"])])
    seed_all(config["seed"])
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last).train()
    require(count_parameters(model) == PARAMETERS <= 5000000, "Unexpected SWA parameter count")
    average = copy.deepcopy(model).eval().requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["swa_lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    images = torch.randint(0, 256, (config["batch_size"],32,32,3), device=device, dtype=torch.uint8)
    labels = torch.arange(config["batch_size"], device=device)%10
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    durations, changed, count = [], 0, 0
    for index in range(steps+3):
        if device.type == "cuda": torch.cuda.synchronize(device)
        probe = next(model.parameters()); before = probe.detach().clone(); began = time.perf_counter()
        update(model, optimizer, images, labels, config, device, dtype)
        with torch.no_grad(): torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-config["ema_decay"])
        count = average_parameters_(average, model, count)
        if device.type == "cuda": torch.cuda.synchronize(device)
        duration = time.perf_counter()-began
        changed += int(not torch.equal(before, probe.detach()))
        if index >= 3: durations.append(duration)
    require(changed == steps+3, "Every engineering update must change real parameters")
    unchanged(evidence)
    median = float(np.median(durations))
    result = {"schema": "research_swa_engineering_v1", "status": "complete", "synthetic_only": True,
        "precision": precision, "model_config": model_config, "parameter_count": PARAMETERS, "steps": steps,
        "successful_updates": steps, "timed_steps": steps, "warmup_steps": 3, "total_successful_updates": steps+3,
        "parameter_changed_steps": changed, "averaging_calls": count, "median_step_seconds": median,
        "ema_tracked": True, "ema_selected": False,
        "p90_step_seconds": float(np.quantile(durations,.9)), "estimated_40k_epoch_seconds": median*math.ceil(TRAIN_EXAMPLES/config["batch_size"]),
        "estimated_tail_training_seconds": median*math.ceil(TRAIN_EXAMPLES/config["batch_size"])*TAIL_EPOCHS,
        "native_bf16_verified": precision == "bf16", "finite_losses_verified": True, "finite_parameters_verified": True,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "training_data_loaded": False, "hidden_images_loaded": False, "confirmation_evaluated": False,
        "config_sha256": config_hash(config), "source_sha256": marker["source_sha256"], "research_training": marker,
        "estimates_exclude": "validation, BN recomputation and I/O; averaging each update conservatively exceeds epoch-only averaging"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-last", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); config = {**DEFAULT_CONFIG, **read(args.config)}
    if args.benchmark_steps:
        if args.parent_last or args.resume or args.stop_after_epoch: parser.error("Synthetic mode never accepts a real parent/resume")
        result = benchmark(config, args.output_dir, torch.device(args.device), args.precision, args.benchmark_steps)
    else:
        if args.parent_last is None: parser.error("Formal SWA requires its audited own --parent-last")
        result = train(config, args.parent_last, args.output_dir, args.data_dir, torch.device(args.device), args.precision, args.stop_after_epoch, args.resume)
    print(json.dumps(result))


if __name__ == "__main__": main()
