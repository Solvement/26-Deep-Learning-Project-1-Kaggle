"""Exploratory scratch WRN with RandAugment, native AMP and fixed ten-view TTA.

This post-feedback experiment trains on ALL permitted 50k training images. It
never evaluates development/confirmation, and its fixed final EMA has no held-out
score. The main campaign's independent confirmation does not endorse this run.

RandAugment: Cubuk et al., https://arxiv.org/abs/1909.13719 . Implementation uses
torchvision v2's documented uint8 tensor transform, N=2, M=9, bilinear sampling.
"""
from __future__ import annotations

from runtime import ROOT, seed_all, environment, write_json
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, v2

from data import COMPETITION, load_training_arrays, load_kaggle_test
from model import build_model, count_parameters
from predict import validate_submission
from train import MEAN, STD, normalize, save_checkpoint, schedule, training_source


TTA_MODE = "reflectpad4_five_crops_plus_horizontal_flip_mean_logits"
TTA_OFFSETS = ((0, 0), (0, 8), (8, 0), (8, 8), (4, 4))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def amp_settings(device):
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU training")
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=not native_bf16)
    else:
        dtype = torch.float32
        scaler = torch.amp.GradScaler("cpu", enabled=False)
    return dtype, scaler


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype if device.type == "cuda" else torch.bfloat16,
                          enabled=device.type == "cuda")


def optimizer_update(loss, optimizer, scaler):
    """An overflow lowers the scale and skips SGD; do not count it as an EMA step."""
    previous_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale


class RobustAugment:
    def __init__(self, config):
        self.cutout = config["cutout"]
        self.policy = v2.RandAugment(
            num_ops=config["randaugment_n"], magnitude=config["randaugment_m"],
            interpolation=InterpolationMode.BILINEAR, fill=config["randaugment_fill"],
        )

    def __call__(self, images):
        n, device = len(images), images.device
        padded = F.pad(images.permute(0, 3, 1, 2), (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
        axis = torch.arange(32, device=device)[None]
        rows = axis + torch.randint(0, 9, (n, 1), device=device)
        cols = axis + torch.randint(0, 9, (n, 1), device=device)
        cols = torch.where(torch.rand(n, 1, device=device) < 0.5, cols.flip(1), cols)
        cropped = padded[torch.arange(n, device=device)[:, None, None], rows[:, :, None], cols[:, None, :]]
        augmented = self.policy(cropped.permute(0, 3, 1, 2))
        x = normalize(augmented.permute(0, 2, 3, 1))
        if self.cutout:
            cy = torch.randint(0, 32, (n, 1, 1), device=device)
            cx = torch.randint(0, 32, (n, 1, 1), device=device)
            axis = torch.arange(32, device=device)
            keep = ((axis[None, :, None] < cy - self.cutout // 2)
                    | (axis[None, :, None] >= cy + self.cutout // 2)
                    | (axis[None, None, :] < cx - self.cutout // 2)
                    | (axis[None, None, :] >= cx + self.cutout // 2))
            x = x * keep[:, None]
        return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def recalibrate_bn(model, images, device, dtype, deadline, batch_size=256):
    """One clean TRAINING-only cumulative-statistics pass, with no test adaptation."""
    model.eval()
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
            module.reset_running_stats()
            module.momentum = None
            module.train()
    try:
        for start in range(0, len(images), batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget exhausted during clean-training BN recalibration")
            with autocast(device, dtype):
                model(normalize(images[start : start + batch_size]))
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.eval()


@torch.inference_mode()
def ten_view_predictions(model, images, device, dtype, deadline, batch_size=128):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch_size):
        raw = images[start : start + batch_size].permute(0, 3, 1, 2)
        padded = F.pad(raw, (4, 4, 4, 4), mode="reflect")
        summed_logits = None
        for row, col in TTA_OFFSETS:
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget exhausted during fixed ten-view inference")
            view = normalize(padded[:, :, row : row + 32, col : col + 32].permute(0, 2, 3, 1))
            with autocast(device, dtype):
                logits = model(view).float() + model(view.flip(-1)).float()
            require(torch.isfinite(logits).all().item(), "Nonfinite ten-view logits")
            summed_logits = logits if summed_logits is None else summed_logits + logits
        predictions.extend((summed_logits / 10).argmax(1).cpu().tolist())
    return predictions


def source_provenance(output, config_path):
    names = ("model.py", "data.py", "runtime.py", "train.py", "train_robust.py")
    hashes = {name: sha256(ROOT / name) for name in names}
    hashes["configs/robust_full.json"] = sha256(config_path)
    companion = output / "launch_manifest.json"
    if companion.exists():
        manifest = json.loads(companion.read_text())
        commit = manifest["source_commit"]
        require(all(manifest["source_sha256"].get(name) == value for name, value in hashes.items()),
                "Frozen robust source bundle changed before training")
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"git_commit": commit, "source_sha256": hashes}


def gpu_preflight(config, device, dtype, scaler):
    require(device.type == "cuda" and torch.cuda.is_available(), "A real CUDA GPU is required")
    # Exercise uint8 RandAugment on the assigned GPU and the exact AMP update path.
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    pixels = torch.randint(0, 256, (4, 32, 32, 3), device=device, dtype=torch.uint8)
    augment = RobustAugment(config)
    successful = False
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, dtype):
            loss = F.cross_entropy(model(augment(pixels)), torch.arange(4, device=device))
        require(torch.isfinite(loss).item(), "CUDA augmentation/AMP preflight loss is nonfinite")
        if optimizer_update(loss, optimizer, scaler):
            successful = True
            break
    require(successful, "FP16 preflight never completed a finite optimizer step")
    torch.cuda.synchronize()
    return {"passed": True, "native_bf16": dtype == torch.bfloat16,
            "amp_dtype": str(dtype), "grad_scaler_enabled": scaler.is_enabled(),
            "gpu": torch.cuda.get_device_name(), "cuda_capability": list(torch.cuda.get_device_capability()),
            "gpu_count": torch.cuda.device_count(), "torch_cuda_version": torch.version.cuda}


def run(config, config_path, output, data_dir, device, max_seconds, resume=False):
    deadline = time.monotonic() + max_seconds
    dtype, scaler = amp_settings(device)
    preflight = gpu_preflight(config, device, dtype, scaler)
    print(json.dumps({"event": "robust_gpu_preflight", **preflight}), flush=True)
    # Reset every generator after preflight, so it cannot change scratch training.
    seed_all(config["seed"], threads=2)
    dtype, scaler = amp_settings(device)
    provenance = source_provenance(output, config_path)
    source = training_source(data_dir)
    manifest = json.loads((data_dir / "training_manifest.json").read_text())
    images, labels, _, _ = load_training_arrays(data_dir)
    require(len(images) == len(labels) == 50000, "Robust full-data training needs all 50k official examples")
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    pixels = torch.as_tensor(images, device=device)
    targets = torch.as_tensor(labels, device=device)
    del images, labels
    model_config = {name: config[name] for name in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    parameters = count_parameters(model)
    require(parameters == 4_298_970 and parameters <= 5_000_000, "Unexpected WRN-22-4 parameter count")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    augmentation = RobustAugment(config)
    history, start_epoch, successful_steps, skipped_steps, previous_seconds = [], 1, 0, 0, 0.0
    if resume:
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        require(saved["config"] == config and saved["data_signature"] == signature
                and saved["provenance"]["source_sha256"] == provenance["source_sha256"], "Resume source/config/data mismatch")
        require(saved["amp_dtype"] == str(dtype), "Resume must use the same AMP dtype")
        model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        history = saved["history"]; start_epoch = saved["epoch"] + 1
        successful_steps = saved["successful_steps"]; skipped_steps = saved["skipped_steps"]
        previous_seconds = saved["elapsed_seconds"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu()); torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    elif (output / "last.pt").exists():
        raise FileExistsError("Existing robust training state requires --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {**environment(), **preflight, "amp_dtype": str(dtype)})
    write_json(output / "provenance.json", {
        "code": provenance, "data_signature": signature, "official_data": source,
        "training_examples": 50000, "validation_examples": 0, "parameter_count": parameters,
        "exploratory_after_public_feedback": True, "main_confirmation_endorses_this_run": False,
        "normalization": {"mean": MEAN, "std": STD}, "bn_recalibration_source": "clean allowed training images only",
        "augmentation_order": "independent crop/flip; batch-policy RandAugment; normalization; Cutout; MixUp",
        "randaugment_policy_scope": "one sampled policy per minibatch; independent crop/flip per image",
        "randaugment_reference": "https://arxiv.org/abs/1909.13719", "tta_mode": TTA_MODE,
    })
    steps_per_epoch = math.ceil(len(targets) / config["batch_size"])
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    started = time.monotonic()

    def checkpoint(epoch):
        save_checkpoint(output / "last.pt", {
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "config": config, "model_config": model_config,
            "data_signature": signature, "provenance": provenance, "history": history,
            "successful_steps": successful_steps, "skipped_steps": skipped_steps,
            "elapsed_seconds": previous_seconds + time.monotonic() - started, "amp_dtype": str(dtype),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        })

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_started = time.monotonic(); model.train()
        order = torch.randperm(len(targets), device=device)
        loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device); seen = 0
        for batch, start in enumerate(range(0, len(order), config["batch_size"])):
            if time.monotonic() >= deadline - 90:
                raise TimeoutError("Robust training exceeded its fixed wall-time budget; periodic last.pt remains resumable")
            lr = schedule((epoch - 1) * steps_per_epoch + batch, config["epochs"] * steps_per_epoch, steps_per_epoch, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            idx = order[start : start + config["batch_size"]]; y = targets[idx]
            x = augmentation(pixels[idx])
            mix = random.random() < config["mixup_probability"]
            lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mix else 1.0
            permutation = torch.randperm(len(y), device=device) if mix else None
            if mix:
                x = x * lam + x[permutation] * (1 - lam)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"]) * lam
                if mix:
                    loss += F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"]) * (1 - lam)
            require(torch.isfinite(loss).item(), "Nonfinite robust loss")
            updated = optimizer_update(loss, optimizer, scaler)
            with torch.no_grad():
                if updated:
                    successful_steps += 1
                    torch._foreach_lerp_(ema_params, model_params, 1 - config["ema_decay"])
                else:
                    skipped_steps += 1
                predicted = logits.detach().argmax(1)
                correct += predicted.eq(y).sum() * lam
                if mix:
                    correct += predicted.eq(y[permutation]).sum() * (1 - lam)
                loss_sum += loss.detach() * len(y); seen += len(y)
            if epoch == start_epoch and batch == 24:
                seconds = (time.monotonic() - epoch_started) / 25
                print(json.dumps({"event": "robust_throughput", "amp_dtype": str(dtype),
                    "measured_steps": 25, "seconds_per_step": seconds, "estimated_epoch_seconds": seconds * steps_per_epoch,
                    "estimated_remaining_training_seconds": seconds * steps_per_epoch * (config["epochs"] - epoch + 1),
                    "remaining_budget_seconds": deadline - time.monotonic()}), flush=True)
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers()):
                average.copy_(online)
        record = {"epoch": epoch, "train_loss": loss_sum.item() / seen,
            "train_accuracy_mixup_weighted": correct.item() / seen, "validation_accuracy": None,
            "ema_validation_accuracy": None, "lr": lr, "successful_steps": successful_steps, "skipped_steps": skipped_steps,
            "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": previous_seconds + time.monotonic() - started}
        history.append(record); write_json(output / "history.json", history)
        write_json(output / "metrics.json", {"status": "training", "epochs_completed": epoch, "parameter_count": parameters,
            "training_examples": 50000, "validation_examples": 0, "best_validation_accuracy": None, "best_validation_loss": None,
            "elapsed_seconds": record["elapsed_seconds"], "config": config, "successful_steps": successful_steps,
            "skipped_steps": skipped_steps, "tta_mode": TTA_MODE})
        if epoch % config["checkpoint_interval"] == 0 or epoch == config["epochs"]:
            checkpoint(epoch)
        print(json.dumps(record), flush=True)
    require(successful_steps > 0, "No successful optimizer steps occurred")
    print("Recalibrating final EMA BN once on clean allowed training images only", flush=True)
    recalibrate_bn(ema, pixels, device, dtype, deadline)
    best = {"model": ema.state_dict(), "model_config": model_config, "config": config, "epoch": config["epochs"],
        "variant": "ema", "validation_accuracy": None, "validation_loss": None, "parameter_count": parameters,
        "data_signature": signature, "provenance": provenance, "bn_recalibrated_on_training_only": True,
        "tta_mode": TTA_MODE, "amp_dtype": str(dtype)}
    save_checkpoint(output / "best.pt", best)
    checkpoint_hash = sha256(output / "best.pt")
    # Only now read the official unlabeled test export, strictly for fixed inference.
    ids, test_images = load_kaggle_test(data_dir)
    test_pixels = torch.as_tensor(test_images, device=device)
    predictions = ten_view_predictions(ema, test_pixels, device, dtype, deadline)
    csv_path = output / "submission.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["id", "label"]); writer.writerows(zip(ids, predictions))
    submission = validate_submission(csv_path, data_dir)
    evaluation = {"epoch": config["epochs"], "variant": "ema", "model_config": model_config, "parameter_count": parameters,
        "trainable_parameter_count": parameters, "validation_examples": 0, "plain_validation_accuracy": None,
        "flip_validation_accuracy": None, "validation_accuracy": None, "validation_loss": None,
        "flip_tta": True, "tta_mode": TTA_MODE, "tta_offsets": [list(offset) for offset in TTA_OFFSETS],
        "tta_views": 10, "tta_aggregation": "mean_logits", "checkpoint_sha256": checkpoint_hash,
        "public_test_accuracy": None, "private_test_accuracy": None, "submitted_to_kaggle": False,
        "submission": submission, "data_signature": signature, "provenance": provenance,
        "exploratory_after_public_feedback": True, "main_confirmation_endorses_this_run": False}
    write_json(output / "evaluation.json", evaluation)
    test_manifest = json.loads((data_dir / "kaggle_test_manifest.json").read_text())
    metadata = {"run": "robust_full", "competition": COMPETITION, "hidden_test_labels_loaded": False,
        "original_evaluation_sha256": sha256(output / "evaluation.json"), "checkpoint_sha256": checkpoint_hash,
        "submission_sha256": submission["sha256"], "source_git_commit": provenance["git_commit"],
        "source_sha256": provenance["source_sha256"], "training_examples": 50000, "parameter_count": parameters,
        "seed": config["seed"], "model_config": model_config, "flip_tta": True, "tta_mode": TTA_MODE,
        "rows": submission["rows"], "columns": submission["columns"], "ids_match_official_order": True,
        "data_signature": signature, "test_input_signature": test_manifest["source_provenance"],
        "exploratory_after_public_feedback": True, "main_confirmation_endorses_this_run": False}
    write_json(output / "metadata.json", metadata)
    metrics = json.loads((output / "metrics.json").read_text())
    metrics.update(status="complete", elapsed_seconds=previous_seconds + time.monotonic() - started,
                   bn_recalibrated_on_training_only=True, submission_sha256=submission["sha256"])
    write_json(output / "metrics.json", metrics)
    print(json.dumps({"event": "robust_complete", "epochs": config["epochs"], "parameter_count": parameters,
                      "submission_sha256": submission["sha256"], "tta_mode": TTA_MODE}), flush=True)


def cpu_smoke(config):
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(3031)
    pixels = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    transformed = RobustAugment(config)(pixels)
    require(transformed.shape == (4, 3, 32, 32) and torch.isfinite(transformed).all().item(), "CPU augmentation failed")
    model = build_model(depth=10, widen_factor=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dtype, scaler = amp_settings(torch.device("cpu"))
    loss = F.cross_entropy(model(transformed), torch.arange(4))
    require(optimizer_update(loss, optimizer, scaler), "CPU optimizer update was skipped")
    require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()),
            "CPU optimizer gradients failed")
    bn = nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
    recalibrate_bn(bn, pixels, torch.device("cpu"), dtype, time.monotonic() + 20, batch_size=2)
    require(bn[0].num_batches_tracked.item() == 2 and not bn.training, "Training-only BN recalibration failed")
    predictions = ten_view_predictions(bn, pixels, torch.device("cpu"), dtype, time.monotonic() + 20, batch_size=2)
    require(len(predictions) == 4 and all(0 <= value < 10 for value in predictions), "Ten-view smoke failed")
    with tempfile.TemporaryDirectory(prefix="robust-csv-smoke-") as name:
        folder = Path(name)
        for filename in ("sample_submission.csv", "submission.csv"):
            with (folder / filename).open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["id", "label"])
                writer.writerows((str(i), str(predictions[i % 4])) for i in range(2000))
        require(validate_submission(folder / "submission.csv", folder)["rows"] == 2000, "CSV smoke failed")
    print("ROBUST_CPU_SMOKE_PASS: uint8 RandAugment/crop/flip/Cutout, optimizer, clean-training BN, ten-view TTA, CSV2000", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robust_full.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/robust_full")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--max-seconds", type=int, default=10500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    require(config.get("epochs") == 300 and config.get("train_all") is True and config.get("seed") == 3031
            and config.get("final_variant") == "ema" and config.get("tta_mode") == TTA_MODE,
            "Robust experiment must retain its predeclared full-data recipe")
    if args.cpu_smoke:
        cpu_smoke(config); return
    require(0 < args.max_seconds <= 10800, "Robust GPU budget must be at most three hours")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(config, args.config, args.output_dir, args.data_dir, torch.device(args.device), args.max_seconds, args.resume)


if __name__ == "__main__":
    main()
