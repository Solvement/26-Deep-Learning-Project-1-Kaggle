"""Independent official 40k/5k ResizeMix training, with native FP16 GradScaler.

Qin et al., https://arxiv.org/abs/2012.11101: sample a SIDE ratio uniformly,
resize the full donor image and mix labels using the actual pasted pixel area.
Only frozen core/shadow utilities are reused; no module is monkeypatched.
Run through a budgeted supervisor; no confirmation or hidden-image loader.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import environment, seed_all, write_json
import torch
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from research.resizemix import resizemix, validate_resizemix
from research.train_shadow import (capture_rng, evaluate, optimizer_step, resolve_precision,
                                   restore_rng, validate_partition)
from research.train_shadow import validate_config as validate_base_config
from train import (DEFAULT_CONFIG, MEAN, STD, augment, code_provenance, save_checkpoint,
                   schedule, sha256, tensor_images, training_source)

SCHEMA = "research_resizemix_training_v1"
KEYS = {"resizemix_probability", "resizemix_tau_min", "resizemix_tau_max"}
SOURCE_KEYS = ("research/train_resizemix.py", "research/resizemix.py", "research/train_shadow.py", "research/augmentations.py")


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG) | KEYS:
        raise ValueError("ResizeMix requires exactly the base training keys plus its probability/tau bounds")
    validate_base_config({key: value for key, value in config.items() if key not in KEYS})
    validate_resizemix(config["resizemix_tau_min"], config["resizemix_tau_max"], config["resizemix_probability"])
    if config["cutout"] != 0 or config["mixup_alpha"] != 0 or config["mixup_probability"] != 0:
        raise ValueError("ResizeMix must replace both Mixup and Cutout")
    if config["num_classes"] != 10: raise ValueError("ResizeMix requires the official ten classes")


def research_provenance(precision, config):
    return {"schema": SCHEMA, "method": "resizemix",
            "source_sha256": {key: sha256(ROOT/key) for key in SOURCE_KEYS},
            "precision": precision, "grad_scaler_enabled": precision == "fp16",
            "tau_min": config["resizemix_tau_min"], "tau_max": config["resizemix_tau_max"],
            "probability": config["resizemix_probability"], "tau_sampling": "uniform_side_ratio",
            "interpolation": "bilinear", "bbox_sampling": "per_image_uniform_center",
            "boundary_policy": "shift_full_patch_inside_image",
            "training_seed": config["seed"], "split_seed": config["split_seed"],
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def validate_resume(last, config, signature, provenance, steps):
    if last.get("config") != config or last.get("data_signature") != signature:
        raise ValueError("ResizeMix resume configuration or official data/split mismatch")
    previous = last.get("provenance", {})
    if (previous.get("source_sha256") != provenance["source_sha256"] or
            previous.get("research_training") != provenance["research_training"]):
        raise ValueError("ResizeMix resume requires identical sources, precision and seeds")
    epoch, history = last.get("epoch"), last.get("history", [])
    if (type(epoch) is not int or not 1 <= epoch < config["epochs"] or
            [row.get("epoch") for row in history] != list(range(1, epoch+1)) or
            "grad_scaler" not in last):
        raise ValueError("ResizeMix resume requires a partial complete history and GradScaler state")
    successful = missed = 0
    for row in history:
        count, skipped = row.get("successful_updates"), row.get("skipped_updates")
        if type(count) is not int or type(skipped) is not int or count < 1 or skipped < 0 or count+skipped != steps:
            raise ValueError("ResizeMix resume update history is inconsistent")
        mixed, scale = row.get("resizemix_batches"), row.get("grad_scaler_scale")
        if (type(mixed) is not int or not 0 <= mixed <= steps or not isinstance(scale, (float, int)) or
                not math.isfinite(scale) or scale <= 0):
            raise ValueError("ResizeMix resume mixing/scaler history is inconsistent")
        successful += count
        missed += skipped
    if last.get("updates") != successful or last.get("skipped_updates") != missed:
        raise ValueError("ResizeMix resume update totals differ from history")


def prepare_batch(images, labels, config):
    images = augment(images, 0)
    mixed, permutation, retained, rectangle = resizemix(images, config["resizemix_tau_min"],
        config["resizemix_tau_max"], config["resizemix_probability"])
    return mixed, labels, permutation, retained, rectangle


def train(config, output, data_dir, device, precision="auto", stop_after_epoch=None, resume=False):
    validate_config(config)
    precision, dtype = resolve_precision(precision, device)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must lie within configured epochs")
    output, data_dir = Path(output), Path(data_dir)
    artifacts = ("last.pt", "best.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json")
    if not resume and any((output/name).exists() for name in artifacts):
        raise FileExistsError("Existing ResizeMix artifacts require explicit resume or a new directory")
    if resume and not (output/"last.pt").is_file(): raise FileNotFoundError("ResizeMix resume requires last.pt")
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
        raise ValueError("ResizeMix model exceeds the 5,000,000 parameter limit")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    steps = math.ceil(len(train_ids)/config["batch_size"])
    history, start_epoch, updates, skipped, previous_seconds = [], 1, 0, 0, 0.0
    best_accuracy, best_loss = -1., float("inf")
    if resume:
        last = torch.load(output/"last.pt", map_location=device, weights_only=False)
        validate_resume(last, config, signature, provenance, steps)
        model.load_state_dict(last["model"])
        ema.load_state_dict(last["ema"])
        optimizer.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["grad_scaler"])
        history, start_epoch, updates, skipped = last["history"], last["epoch"]+1, last["updates"], last["skipped_updates"]
        best_accuracy, best_loss, previous_seconds = last["best_accuracy"], last["best_loss"], last["elapsed_seconds"]
        restore_rng(last, device)
        del last
    end_epoch = min(config["epochs"], stop_after_epoch or config["epochs"])
    if end_epoch < start_epoch: raise ValueError("stop-after-epoch precedes the resumed epoch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {**environment(), "resizemix_training_precision": precision,
                                         "grad_scaler_enabled": scaler.is_enabled(), "research_training": marker})
    write_json(output/"provenance.json", {"code": provenance, "official_data": source, "data_signature": signature,
        "train_examples": len(train_ids), "validation_examples": len(val_ids), "parameter_count": parameters,
        "normalization": {"mean": MEAN, "std": STD}})
    print(json.dumps({"event": "resizemix_training_started", "precision": precision, "parameters": parameters,
                      "training_examples": len(train_ids), "validation_examples": len(val_ids), "config": config}), flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, end_epoch+1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_y), device=device)
        loss_sum, correct = torch.zeros((), device=device), torch.zeros((), device=device)
        seen, epoch_updates, epoch_skipped, mixed_batches = 0, 0, 0, 0
        for batch, start in enumerate(range(0, len(order), config["batch_size"])):
            lr = schedule((epoch-1)*steps+batch, config["epochs"]*steps, steps, config)
            for group in optimizer.param_groups: group["lr"] = lr
            ids = order[start:start+config["batch_size"]]
            x, y, permutation, lam, _ = prepare_batch(train_x[ids], train_y[ids], config)
            mixed = permutation is not None
            mixed_batches += int(mixed)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"])*lam
                if mixed: loss += F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"])*(1-lam)
            if optimizer_step(loss, optimizer, scaler):
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
                if mixed: correct += predicted.eq(y[permutation]).sum()*(1-lam)
                loss_sum += loss.detach()*len(y)
                seen += len(y)
        if epoch_updates == 0: raise FloatingPointError("No successful SGD updates in the epoch")
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers()): average.copy_(online)
        regular, average = evaluate(model, val_x, val_y, device, dtype), evaluate(ema, val_x, val_y, device, dtype)
        choice = "ema" if (average["accuracy"], -average["loss"]) >= (regular["accuracy"], -regular["loss"]) else "model"
        metric = average if choice == "ema" else regular
        elapsed = previous_seconds+time.perf_counter()-started
        record = {"epoch": epoch, "train_loss": loss_sum.item()/seen, "train_accuracy": correct.item()/seen,
            "validation_loss": regular["loss"], "validation_accuracy": regular["accuracy"],
            "ema_validation_loss": average["loss"], "ema_validation_accuracy": average["accuracy"], "selected_variant": choice,
            "lr": lr, "epoch_seconds": time.perf_counter()-epoch_started, "elapsed_seconds": elapsed,
            "successful_updates": epoch_updates, "skipped_updates": epoch_skipped,
            "grad_scaler_scale": scaler.get_scale(), "resizemix_batches": mixed_batches}
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
            "successful_updates": updates, "skipped_updates": skipped,
            "resizemix_batches": sum(row["resizemix_batches"] for row in history)})
        save_checkpoint(output/"last.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(), "epoch": epoch, "updates": updates, "skipped_updates": skipped,
            "config": config, "data_signature": signature, "provenance": provenance, "history": history,
            "best_accuracy": best_accuracy, "best_loss": best_loss, "elapsed_seconds": elapsed, **capture_rng(device)})
        print(json.dumps(record), flush=True)
    print(json.dumps({"event": "resizemix_training_complete" if end_epoch == config["epochs"] else "resizemix_training_paused",
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
    train(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision, args.stop_after_epoch, args.resume)


if __name__ == "__main__": main()
