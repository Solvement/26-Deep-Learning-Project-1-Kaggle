"""Train from local, authorized competition images; never download data or weights."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import CsvImages, research_split
from .model import CifarResNet
from .utils import (choose_device, provenance, restore_rng, rng_state, save_checkpoint,
                    seed_everything, seed_worker, write_json)


@dataclass
class Config:
    base_width: int = 32
    blocks_per_stage: int = 3
    seed: int = 42
    split_seed: int = 42
    epochs: int = 120
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1
    validation_fraction: float = 0.1
    confirmation_fraction: float = 0.1
    warmup_epochs: int = 5
    workers: int = 2
    amp: bool = True

    def __post_init__(self):
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.split_seed) is not int or not 0 <= self.split_seed < 2**32:
            raise ValueError("split_seed must be an integer in [0, 2**32)")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be between 0 and epochs - 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.momentum < 1 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid momentum or weight decay")
        if not 0 <= self.label_smoothing < 1 or not 0 < self.validation_fraction < 0.5:
            raise ValueError("Invalid label smoothing or validation fraction")
        if not 0 < self.confirmation_fraction < 0.5 or self.validation_fraction + self.confirmation_fraction >= 1:
            raise ValueError("Invalid confirmation fraction")
        if type(self.amp) is not bool:
            raise ValueError("amp must be a boolean")


def epoch_learning_rate(config, epoch):
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, batch_limit=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, examples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, targets) in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(scaler and scaler.is_enabled())):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss; stop this experiment and inspect the inputs/configuration")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            examples += len(targets)
    if examples == 0:
        raise RuntimeError("No examples were evaluated")
    return {"loss": loss_sum / examples, "accuracy": correct / examples, "examples": examples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", required=True, help="Authorized training CSV relative to data-root")
    parser.add_argument("--image-dir", default="train_images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int, help="Stop after this absolute epoch; retain the full LR schedule")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="Mark non-competition pipeline checks explicitly")
    args = parser.parse_args(argv)
    config = Config(**json.loads(args.config.read_text()))
    for limit in [args.stop_after_epoch, args.limit_train_batches, args.limit_val_batches]:
        if limit is not None and limit <= 0:
            parser.error("Epoch/batch limits must be positive")
    device = choose_device(args.device)
    torch.set_num_threads(2)
    seed_everything(config.seed)
    model = CifarResNet(config.base_width, config.blocks_per_stage).to(device)
    train_data = CsvImages(args.data_root / args.train_csv, args.data_root / args.image_dir, training=True)
    validation_data = CsvImages(args.data_root / args.train_csv, args.data_root / args.image_dir)
    split = research_split(train_data.labels, config.validation_fraction, config.confirmation_fraction, config.split_seed)
    train_indices, validation_indices = split["training"], split["validation"]
    split_sha = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    dataset_sha = train_data.fingerprint()
    smoke = args.smoke_test or args.limit_train_batches is not None or args.limit_val_batches is not None
    limits = {"train": args.limit_train_batches, "validation": args.limit_val_batches}
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=config.batch_size,
                              shuffle=True, generator=generator, num_workers=config.workers,
                              worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(Subset(validation_data, validation_indices), batch_size=config.batch_size,
                                   num_workers=config.workers, worker_init_fn=seed_worker)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    history, best_accuracy, best_epoch, best_model = [], -1.0, 0, None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {"config": asdict(config), "dataset_sha256": dataset_sha, "split_sha256": split_sha,
           "trainable_parameters": model.parameter_count, "smoke_test": smoke, "batch_limits": limits,
           "training_examples": len(train_indices), "validation_examples": len(validation_indices),
           "confirmation_examples": len(split["confirmation"])}
    current_provenance = provenance(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ["config", "dataset_sha256", "split_sha256", "smoke_test", "batch_limits"]:
            if checkpoint[key] != run[key]:
                raise ValueError(f"Cannot resume: {key} differs from checkpoint")
        if checkpoint["provenance"]["device"] != str(device):
            raise ValueError("Exact training resume requires the original device type; use inference to migrate devices")
        for key in ["source_sha256", "torch", "gpu"]:
            if checkpoint["provenance"][key] != current_provenance[key]:
                raise ValueError(f"Exact training resume requires matching {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_accuracy, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        best_model = checkpoint["best_model"]
        restore_rng(checkpoint["rng"], generator)
    elif any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory or use --resume")
    if args.resume and not history:
        raise ValueError("Checkpoint has no completed epochs")
    run["provenance"] = current_provenance
    run["resume_from"] = str(args.resume) if args.resume else None
    write_json(args.output_dir / "run.json", run)
    write_json(args.output_dir / "split.json", split)
    stop = min(config.epochs, args.stop_after_epoch or config.epochs)
    if len(history) >= stop:
        raise ValueError("No epochs remain for the requested stop point")
    for epoch in range(len(history), stop):
        start = time.monotonic()
        learning_rate = epoch_learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train = run_epoch(model, train_loader, device, criterion, optimizer, scaler, args.limit_train_batches)
        validation = run_epoch(model, validation_loader, device, nn.CrossEntropyLoss(), batch_limit=args.limit_val_batches)
        metrics = {"epoch": epoch + 1, "learning_rate": learning_rate,
                   "train": train, "validation": validation, "seconds": time.monotonic() - start}
        history.append(metrics)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch + 1
            best_model = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        common = {**run, "format_version": 1, "epoch": epoch + 1,
                  "best_accuracy": best_accuracy, "best_epoch": best_epoch}
        save_checkpoint(args.output_dir / "last.pt", {**common, "model": model.state_dict(),
                         "best_model": best_model, "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "history": history, "rng": rng_state(generator)})
        save_checkpoint(args.output_dir / "best.pt", {**common, "model": best_model, "epoch": best_epoch})
        (args.output_dir / "metrics.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
        print(json.dumps(metrics), flush=True)
    summary = {"completed_epochs": len(history), "planned_epochs": config.epochs,
               "best_epoch": best_epoch, "validation_accuracy": best_accuracy,
               "trainable_parameters": model.parameter_count, "smoke_test": smoke,
               "official_test_accuracy": None, "note": "Hidden test accuracy is only known after an authorized Kaggle submission."}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
