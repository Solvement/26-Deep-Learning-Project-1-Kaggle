"""One-time confirmation evaluation after a candidate has been selected on development data."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import CsvImages
from .model import CifarResNet
from .train import run_epoch
from .utils import choose_device, sha256_file, write_json


def wilson_interval(correct, total):
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True, help="Shared receipt directory for the whole research campaign")
    parser.add_argument("--train-csv", default="train_labels.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("smoke_test") and not args.allow_smoke_checkpoint:
        raise ValueError("A smoke checkpoint is not a final candidate")
    if not checkpoint.get("confirmation_examples"):
        raise ValueError("Checkpoint does not reserve an independent confirmation split")
    split = json.loads(args.split.read_text())
    split_sha = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    if split_sha != checkpoint["split_sha256"]:
        raise ValueError("Confirmation split does not match checkpoint")
    dataset = CsvImages(args.data_root / args.train_csv, args.data_root / "train_images")
    if dataset.fingerprint() != checkpoint["dataset_sha256"]:
        raise ValueError("Confirmation dataset differs from training provenance")
    groups = [set(split[name]) for name in ["training", "validation", "confirmation"]]
    if any(groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)]) or set.union(*groups) != set(range(len(dataset))):
        raise ValueError("Invalid or overlapping split partitions")
    device = choose_device(args.device)
    torch.set_num_threads(2)
    model = CifarResNet(checkpoint["config"]["base_width"], checkpoint["config"]["blocks_per_stage"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.campaign_dir / f"confirmation-used-{checkpoint['dataset_sha256'][:12]}-{split_sha[:12]}.json"
    record = {"checkpoint_sha256": sha256_file(args.checkpoint), "split_sha256": split_sha,
              "status": "started", "smoke_test": checkpoint["smoke_test"]}
    with receipt.open("x") as handle:
        json.dump(record, handle)
    try:
        metrics = run_epoch(model, DataLoader(Subset(dataset, split["confirmation"]), batch_size=128),
                            device, nn.CrossEntropyLoss())
        record.update({"status": "complete", "metrics": metrics,
                       "accuracy_wilson95": wilson_interval(round(metrics["accuracy"] * metrics["examples"]), metrics["examples"]),
                       "official_test_accuracy": None,
                       "interpretation": "One-time internal holdout check; not a Kaggle score or proof of superiority."})
    except Exception:
        record["status"] = "failed"
        write_json(receipt, record)
        raise
    write_json(receipt, record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
