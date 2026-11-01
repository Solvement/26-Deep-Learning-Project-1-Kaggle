"""Exploratory 40k/5k scratch training with explicit native AMP precision.

This independent entrypoint reuses the frozen primary model/data/augmentation
functions without modifying them. CUDA FP16 uses torch.amp.GradScaler; skipped
overflow steps never advance EMA. No confirmation or hidden-image loader is used.
AMP reference: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
Run only through a budgeted local/cloud supervisor.
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
from train import (DEFAULT_CONFIG, MEAN, STD, augment, code_provenance, normalize,
                   save_checkpoint, schedule, sha256, tensor_images, training_source)
from research.augmentations import cutmix, validate_cutmix

SCHEMA = "research_shadow_training_v1"
SOURCE_KEY = "research/train_shadow.py"
SOURCE_KEYS = (SOURCE_KEY, "research/augmentations.py")
CUTMIX_KEYS = {"cutmix_alpha", "cutmix_probability"}


def validate_config(config):
    if config.get("train_all") is not False:
        raise ValueError("Shadow training permits only held-out 40k/5k; full-data training is forbidden")
    extra_keys = set(config)-set(DEFAULT_CONFIG)
    if not set(DEFAULT_CONFIG).issubset(config) or extra_keys not in (set(), CUTMIX_KEYS):
        raise ValueError("Shadow configuration contains missing or unknown training keys")
    if config["split_seed"] != 2026:
        raise ValueError("Shadow training requires the frozen official split seed 2026")
    for key in ("seed", "epochs", "batch_size", "warmup_epochs"):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"Configuration {key} must be an integer")
    if (config["epochs"] < 1 or config["batch_size"] < 1 or
            not 0 <= config["warmup_epochs"] < config["epochs"]):
        raise ValueError("Positive epochs/batch size and 0 <= warmup < epochs are required")
    if (not 0 <= config["mixup_probability"] <= 1 or config["mixup_alpha"] < 0 or
            not 0 <= config["ema_decay"] < 1 or not 0 <= config["label_smoothing"] < 1):
        raise ValueError("Invalid mixing, EMA or label-smoothing setting")
    alpha, probability = config.get("cutmix_alpha", 0.0), config.get("cutmix_probability", 0.0)
    validate_cutmix(alpha, probability)
    if alpha > 0 and probability > 0 and (config["mixup_alpha"] > 0 or config["cutout"] != 0):
        raise ValueError("Enabled CutMix must replace both MixUp and Cutout")


def resolve_precision(requested, device):
    """Fail before loading real data if the requested hardware precision is invalid."""
    if requested not in ("auto", "fp16", "bf16", "fp32"):
        raise ValueError("Unknown precision")
    if device.type == "cpu":
        if requested not in ("auto", "fp32"):
            raise ValueError("CPU shadow checks require fp32; native FP16/BF16 require CUDA")
        return "fp32", torch.float32
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    supported = torch.cuda.is_bf16_supported(including_emulation=False)
    selected = ("bf16" if supported else "fp16") if requested == "auto" else requested
    if selected == "bf16" and not supported:
        raise RuntimeError("Native BF16 is unsupported on this GPU; use --precision fp16 for T4")
    return selected, {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[selected]


def research_provenance(precision, config):
    return {"schema": SCHEMA, "source_sha256": {key: sha256(ROOT/key) for key in SOURCE_KEYS},
            "precision": precision, "grad_scaler_enabled": precision == "fp16",
            "training_seed": config["seed"], "split_seed": config["split_seed"],
            "confirmation_evaluated": False, "hidden_images_loaded": False,
            "external_data_loaded": False}


def validate_partition(train_ids, val_ids, manifest, config):
    if (len(train_ids) != 40000 or len(val_ids) != 5000 or
            len(np.unique(train_ids)) != 40000 or len(np.unique(val_ids)) != 5000 or
            np.intersect1d(train_ids, val_ids).size):
        raise ValueError("Shadow training requires disjoint official 40k training / 5k development")
    if manifest["split"]["seed"] != config["split_seed"]:
        raise ValueError("Configured split seed differs from audited official split")


def optimizer_step(loss, optimizer, scaler):
    """Return whether an SGD update occurred; GradScaler suppresses overflow steps."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    scale_before = scaler.get_scale()
    if not math.isfinite(scale_before) or scale_before <= 0:
        raise FloatingPointError("GradScaler scale must remain positive and finite")
    scaler.scale(loss).backward()
    if not scaler.is_enabled():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite unscaled training gradient")
    scaler.step(optimizer)
    scaler.update()
    # Default GradScaler backoff is <1, growth >=1; scale reduction means skip.
    scale_after = scaler.get_scale()
    if not math.isfinite(scale_after) or scale_after <= 0:
        raise FloatingPointError("GradScaler scale must remain positive and finite")
    return scale_after >= scale_before


def capture_rng(device):
    return {"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def restore_rng(state, device):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def validate_resume(last, config, signature, provenance):
    if last["config"] != config or last["data_signature"] != signature:
        raise ValueError("Resume configuration or official data/split mismatch")
    if (last["provenance"]["source_sha256"] != provenance["source_sha256"] or
            last["provenance"].get("research_training") != provenance["research_training"]):
        raise ValueError("Resume requires identical frozen core, shadow source, precision and seeds")
    if "grad_scaler" not in last:
        raise ValueError("Shadow resume requires its saved GradScaler state")


@torch.no_grad()
def evaluate(model, images, labels, device, dtype, batch_size=256):
    model.eval()
    correct, loss = 0, 0.0
    for start in range(0, len(labels), batch_size):
        x = normalize(images[start:start+batch_size])
        target = labels[start:start+batch_size]
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite development logits")
        correct += logits.argmax(1).eq(target).sum().item()
        loss += F.cross_entropy(logits.float(), target, reduction="sum").item()
    return {"loss": loss/len(labels), "accuracy": correct/len(labels)}


def train(config, output, data_dir, device, precision="auto", stop_after_epoch=None, resume=False):
    validate_config(config)
    precision, dtype = resolve_precision(precision, device)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must lie within configured training epochs")
    output, data_dir = Path(output), Path(data_dir)
    artifacts = ("last.pt", "best.pt", "config.json", "metrics.json", "history.json", "provenance.json")
    if not resume and any((output/name).exists() for name in artifacts):
        raise FileExistsError("Existing experiment artifacts: use --resume or a new output directory")
    if resume and not (output/"last.pt").is_file():
        raise FileNotFoundError("Shadow resume requires last.pt")
    seed_all(config["seed"])
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    source = training_source(data_dir)
    manifest = json.loads((data_dir/"training_manifest.json").read_text(encoding="utf-8"))
    validate_partition(train_ids, val_ids, manifest, config)
    signature = {"archive_sha256": source["archive_sha256"],
                 "split_sha256": manifest["split"]["indices_sha256"]}
    provenance = code_provenance()
    provenance["research_training"] = research_provenance(precision, config)
    train_x, val_x = tensor_images(images[train_ids], device), tensor_images(images[val_ids], device)
    train_y = torch.as_tensor(labels[train_ids], device=device)
    val_y = torch.as_tensor(labels[val_ids], device=device)
    del images, labels
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    parameters = count_parameters(model)
    if parameters > 5000000:
        raise ValueError("Model exceeds the 5,000,000 parameter limit")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    steps_per_epoch = math.ceil(len(train_ids)/config["batch_size"])
    total_steps = config["epochs"]*steps_per_epoch
    history, start_epoch, updates, skipped = [], 1, 0, 0
    best_accuracy, best_loss, previous_seconds = -1.0, float("inf"), 0.0
    if resume:
        last = torch.load(output/"last.pt", map_location=device, weights_only=False)
        validate_resume(last, config, signature, provenance)
        model.load_state_dict(last["model"])
        ema.load_state_dict(last["ema"])
        optimizer.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["grad_scaler"])
        history, start_epoch, updates = last["history"], last["epoch"]+1, last["updates"]
        skipped = last["skipped_updates"]
        best_accuracy, best_loss = last["best_accuracy"], last["best_loss"]
        previous_seconds = last["elapsed_seconds"]
        restore_rng(last, device)
        if start_epoch > config["epochs"]:
            raise ValueError("Completed training cannot be resumed")
        del last
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {**environment(), "shadow_training_precision": precision,
                                         "grad_scaler_enabled": scaler.is_enabled()})
    write_json(output/"provenance.json", {"code": provenance, "official_data": source,
               "data_signature": signature, "train_examples": len(train_ids),
               "validation_examples": len(val_ids), "parameter_count": parameters,
               "normalization": {"mean": MEAN, "std": STD}})
    print(json.dumps({"event": "shadow_training_started", "precision": precision,
                      "parameters": parameters, "training_examples": len(train_ids),
                      "validation_examples": len(val_ids), "config": config}), flush=True)
    started = time.perf_counter()
    end_epoch = min(config["epochs"], stop_after_epoch or config["epochs"])
    if end_epoch < start_epoch:
        raise ValueError("stop-after-epoch precedes the resumed epoch")
    for epoch in range(start_epoch, end_epoch+1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_y), device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen, epoch_updates, epoch_skipped, cutmix_batches = 0, 0, 0, 0
        for batch, start in enumerate(range(0, len(order), config["batch_size"])):
            step = (epoch-1)*steps_per_epoch+batch
            lr = schedule(step, total_steps, steps_per_epoch, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = order[start:start+config["batch_size"]]
            y = train_y[ids]
            x = augment(train_x[ids], config["cutout"])
            if config.get("cutmix_alpha", 0.0) > 0 and config.get("cutmix_probability", 0.0) > 0:
                x, permutation, lam, _ = cutmix(x, config["cutmix_alpha"], config["cutmix_probability"])
                mix = permutation is not None
                cutmix_batches += int(mix)
            else:
                mix = config["mixup_alpha"] > 0 and random.random() < config["mixup_probability"]
                lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mix else 1.0
                permutation = torch.randperm(len(y), device=device) if mix else None
                if mix:
                    x = x*lam+x[permutation]*(1-lam)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"])*lam
                if mix:
                    loss += F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"])*(1-lam)
            updated = optimizer_step(loss, optimizer, scaler)
            if updated:
                updates += 1
                epoch_updates += 1
                with torch.no_grad():
                    decay = min(config["ema_decay"], (1+updates)/(10+updates))
                    torch._foreach_lerp_(ema_params, model_params, 1-decay)
            else:
                skipped += 1
                epoch_skipped += 1
            with torch.no_grad():
                predicted = logits.detach().argmax(1)
                correct += predicted.eq(y).sum()*lam
                if mix:
                    correct += predicted.eq(y[permutation]).sum()*(1-lam)
                loss_sum += loss.detach()*len(y)
                seen += len(y)
        if epoch_updates == 0:
            raise FloatingPointError("No successful SGD updates in the epoch")
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers()):
                average.copy_(online)
        regular = evaluate(model, val_x, val_y, device, dtype)
        average = evaluate(ema, val_x, val_y, device, dtype)
        choice = "ema" if (average["accuracy"], -average["loss"]) >= (regular["accuracy"], -regular["loss"]) else "model"
        metric = average if choice == "ema" else regular
        elapsed = previous_seconds+time.perf_counter()-started
        record = {"epoch": epoch, "train_loss": loss_sum.item()/seen, "train_accuracy": correct.item()/seen,
                  "validation_loss": regular["loss"], "validation_accuracy": regular["accuracy"],
                  "ema_validation_loss": average["loss"], "ema_validation_accuracy": average["accuracy"],
                  "selected_variant": choice, "lr": lr, "epoch_seconds": time.perf_counter()-epoch_started,
                  "elapsed_seconds": elapsed, "successful_updates": epoch_updates,
                  "skipped_updates": epoch_skipped, "grad_scaler_scale": scaler.get_scale(),
                  "cutmix_batches": cutmix_batches}
        history.append(record)
        if (metric["accuracy"], -metric["loss"]) > (best_accuracy, -best_loss):
            best_accuracy, best_loss = metric["accuracy"], metric["loss"]
            selected = ema if choice == "ema" else model
            save_checkpoint(output/"best.pt", {"model": selected.state_dict(), "model_config": model_config,
                "config": config, "epoch": epoch, "variant": choice, "validation_accuracy": best_accuracy,
                "validation_loss": best_loss, "parameter_count": parameters, "data_signature": signature,
                "provenance": provenance})
        write_json(output/"history.json", history)
        write_json(output/"metrics.json", {"status": "complete" if epoch == config["epochs"] else "running",
            "epochs_completed": epoch, "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
            "parameter_count": parameters, "elapsed_seconds": elapsed, "training_examples": len(train_ids),
            "validation_examples": len(val_ids), "config": config, "research_training": provenance["research_training"],
            "successful_updates": updates, "skipped_updates": skipped})
        save_checkpoint(output/"last.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "grad_scaler": scaler.state_dict(), "epoch": epoch,
            "updates": updates, "skipped_updates": skipped, "config": config, "data_signature": signature,
            "provenance": provenance, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss,
            "elapsed_seconds": elapsed, **capture_rng(device)})
        print(json.dumps(record), flush=True)
    print(json.dumps({"event": "shadow_training_complete" if end_epoch == config["epochs"] else "shadow_training_paused",
                      "epochs_completed": end_epoch, "best_validation_accuracy": best_accuracy,
                      "successful_updates": updates, "skipped_updates": skipped}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG, **json.loads(args.config.read_text(encoding="utf-8"))}
    train(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision,
          args.stop_after_epoch, args.resume)


if __name__ == "__main__":
    main()
