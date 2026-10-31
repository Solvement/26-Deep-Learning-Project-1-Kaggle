"""Prepare only staff-provided Kaggle training inputs and a fixed split.

Supply train_images/ + train_labels.csv, or an audited official_train.npz,
official_source_manifest.json and train_labels.csv exported privately on Kaggle.
This command never downloads or reads external CIFAR training archives.
"""
from __future__ import annotations

import argparse
import json

try:
    from .data import DEFAULT_DATA_DIR, prepare_training_data
except ImportError:
    from data import DEFAULT_DATA_DIR, prepare_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = prepare_training_data(args.data_dir, seed=args.seed)
    print(json.dumps({
        "status": "TRAINING_DATA_READY",
        "data_dir": args.data_dir,
        "training_images": manifest["split"]["train_count"],
        "validation_images": manifest["split"]["validation_count"],
        "confirmation_images": manifest["split"]["confirmation_count"],
        "split_sha256": manifest["split"]["indices_sha256"],
        "source_type": manifest["source_type"],
        "external_training_data_loaded": manifest["external_training_data_loaded"],
        "official_test_labels_loaded": manifest["official_test_labels_loaded"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
