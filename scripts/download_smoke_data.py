"""Download a small, balanced subset of official competition data for plumbing checks.

This is not the full training dataset and cannot produce a valid full competition
submission. Authentication is supplied by the user's existing Kaggle environment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--test-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=1.5)
    args = parser.parse_args()
    if args.per_class < 2 or args.test_images < 1 or args.interval < 1.3:
        parser.error("Use at least 2 images per class, 1 test image, and a request interval >= 1.3 seconds")
    from kaggle import api
    slug = "cifar-10-res-net-architecture-challenge-dl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "_source"
    source.mkdir(exist_ok=True)
    def download(name, directory):
        target = directory / name
        if not target.is_file():
            api.competition_download_file(slug, name, path=str(directory), quiet=True)
            time.sleep(args.interval)
        if not target.is_file():
            raise FileNotFoundError(f"Expected downloaded file missing: {target}")
        return target
    for name in ["train_labels.csv", "test.csv", "sample_submission.csv"]:
        download(name, source)
    training = read(source / "train_labels.csv")
    rng = np.random.default_rng(args.seed)
    selected = []
    for label in range(10):
        group = [row for row in training if int(row["label"]) == label]
        if len(group) < args.per_class:
            raise ValueError("Insufficient training images for requested class-balanced sample")
        selected.extend(group[i] for i in rng.choice(len(group), size=args.per_class, replace=False))
    tests = read(source / "test.csv")[:args.test_images]
    ids = {row["id"] for row in tests}
    sample = [row for row in read(source / "sample_submission.csv") if row["id"] in ids]
    manifest = {"competition": slug, "seed": args.seed, "per_class": args.per_class,
                "training_rows": len(selected), "test_rows": len(tests), "smoke_test_only": True,
                "source_csv_sha256": hashlib.sha256((source / "train_labels.csv").read_bytes()).hexdigest(),
                "training_ids": [row["id"] for row in selected], "test_ids": [row["id"] for row in tests]}
    manifest_path = args.output_dir / "smoke_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing smoke dataset uses different sampling settings; choose another directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for folder, rows in [("train_images", selected), ("test_images", tests)]:
        for index, row in enumerate(rows):
            filename = row["filename"]
            if Path(filename).name != filename:
                raise ValueError("Expected a plain image filename")
            # The Kaggle CLI preserves the path inside the competition's filename.
            path = download(f"{folder}/{filename}", args.output_dir)
            with Image.open(path) as image:
                image.verify()
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"Verified {folder}: {index + 1}/{len(rows)}", flush=True)
    write(args.output_dir / "train_labels.csv", ["id", "filename", "label"], selected)
    write(args.output_dir / "test.csv", ["id", "filename"], tests)
    write(args.output_dir / "sample_submission.csv", ["id", "label"], sample)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
