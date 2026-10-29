"""Generate and validate a submission locally; never upload it to Kaggle."""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CsvImages, read_csv, unique_values
from .model import CifarResNet
from .utils import choose_device, sha256_file, write_json


def validate_submission(path, sample_path):
    for csv_path in [path, sample_path]:
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            if next(csv.reader(handle), None) != ["id", "label"]:
                raise ValueError("Submission and sample must have exactly the columns id,label in that order")
    rows = read_csv(path, {"id", "label"})
    sample = read_csv(sample_path, {"id", "label"})
    ids, expected = unique_values(rows, "id"), unique_values(sample, "id")
    if ids != expected:
        raise ValueError("Submission IDs and their order must exactly match sample_submission.csv")
    for row in rows:
        try:
            label = int(row["label"])
        except ValueError as exc:
            raise ValueError("Predicted labels must be integers from 0 to 9") from exc
        if not 0 <= label <= 9:
            raise ValueError("Predicted labels must be integers from 0 to 9")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", default="test.csv")
    parser.add_argument("--image-dir", default="test_images")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.output.name != "submission.csv":
        parser.error("The competition output file must be named submission.csv")
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        raise FileExistsError("Refusing to overwrite existing submission artifacts")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint["smoke_test"] and not args.allow_smoke_checkpoint:
        raise ValueError("Smoke-test checkpoints cannot be used for a competition submission")
    device = choose_device(args.device)
    torch.set_num_threads(2)
    model = CifarResNet(checkpoint["config"]["base_width"], checkpoint["config"]["blocks_per_stage"])
    if model.parameter_count != checkpoint["trainable_parameters"]:
        raise ValueError("Checkpoint parameter count does not match its architecture")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset = CsvImages(args.data_root / args.test_csv, args.data_root / args.image_dir, labeled=False)
    sample_path = args.data_root / args.sample
    sample = read_csv(sample_path, {"id", "label"})
    sample_ids = unique_values(sample, "id")
    test_ids = unique_values(dataset.rows, "id")
    if set(sample_ids) != set(test_ids):
        raise ValueError("Test CSV and sample submission IDs must match exactly")
    # Validate the reference header before expensive inference or writing output.
    validate_submission(sample_path, sample_path)
    predictions = {}
    with torch.inference_mode():
        for images, indices in DataLoader(dataset, batch_size=args.batch_size):
            labels = model(images.to(device)).argmax(1).cpu().tolist()
            for index, label in zip(indices.tolist(), labels):
                predictions[dataset.rows[index]["id"]] = label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows((identity, predictions[identity]) for identity in sample_ids)
    rows = validate_submission(temporary, sample_path)
    temporary.replace(args.output)
    metadata = {"rows": rows, "trainable_parameters": model.parameter_count,
                "checkpoint_sha256": sha256_file(args.checkpoint), "submission_sha256": sha256_file(args.output),
                "sample_sha256": sha256_file(sample_path), "test_index_sha256": sha256_file(args.data_root / args.test_csv),
                "validation_accuracy": checkpoint["best_accuracy"], "official_test_accuracy": None,
                "smoke_test": checkpoint["smoke_test"], "submitted_to_kaggle": False}
    write_json(args.output.with_suffix(".metadata.json"), metadata)
    print(metadata)
    return metadata


if __name__ == "__main__":
    main()
