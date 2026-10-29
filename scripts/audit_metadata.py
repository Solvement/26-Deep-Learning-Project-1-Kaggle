"""Validate the full competition CSV contracts against a completed file inventory."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if not inventory.get("complete"):
        raise ValueError("A complete file inventory is required")
    names = inventory["files"]
    if len(names) != len(set(names)):
        raise ValueError("Inventory contains duplicate file names")
    data, hashes = {}, {}
    columns = {"train_labels.csv": ["id", "filename", "label"],
               "test.csv": ["id", "filename"], "sample_submission.csv": ["id", "label"]}
    for name, expected in columns.items():
        path = args.data_root / name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected:
                raise ValueError(f"Unexpected column contract for {name}")
            rows = list(reader)
        if not rows or len(rows) != len({row["id"] for row in rows}):
            raise ValueError(f"Empty or duplicate IDs in {name}")
        data[name] = rows
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    training, tests, sample = [data[n] for n in columns]
    if len(training) != 50_000 or len(tests) != 2_000 or len(sample) != 2_000:
        raise ValueError("Full competition data must contain 50,000 training and 2,000 test/sample rows")
    if [r["id"] for r in tests] != [r["id"] for r in sample]:
        raise ValueError("Test IDs/order differ from the official sample")
    counts = Counter(int(row["label"]) for row in training)
    if counts != Counter({i: 5000 for i in range(10)}):
        raise ValueError("Unexpected training class distribution")
    expected_files = set(columns)
    for folder, rows in [("train_images", training), ("test_images", tests)]:
        filenames = [row["filename"] for row in rows]
        if len(filenames) != len(set(filenames)) or any(Path(n).name != n for n in filenames):
            raise ValueError("Duplicate or unsafe image filenames")
        expected_files.update(f"{folder}/{name}" for name in filenames)
    if set(names) != expected_files:
        raise ValueError("CSV references and full competition file inventory differ")
    result = {"competition": "cifar-10-res-net-architecture-challenge-dl", "inventory_complete": True,
              "files": len(names), "training_images": len(training), "test_images": len(tests),
              "class_counts": dict(sorted(counts.items())), "csv_columns": columns, "csv_sha256": hashes,
              "csv_inventory_consistent": True, "image_bytes_fully_downloaded_or_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
