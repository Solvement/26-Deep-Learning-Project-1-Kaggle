"""Generate fixed-policy single-model benchmark CSVs; never upload or retrain.

Use only completed 40k-training runs and the flip-TTA rule already selected in
each original evaluation.json. Outputs live in separate submission_<run>
directories; candidate evaluations, frozen selection and confirmation stay
unchanged. Hidden images are used only for inference, without test labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Process-local Windows DLL isolation must happen before torch is imported.
from runtime import ROOT, seed_all, write_json
import torch

from data import COMPETITION, load_kaggle_test, load_training_arrays
from model import build_model, count_parameters
from predict import validate_submission
from train import code_provenance, normalize, training_source


ALLOWED_RUNS = ("baseline", "architecture", "regularized", "stability_seed2027")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_run(results_dir, run, signature, current_source_hashes):
    folder = results_dir / run
    metrics = read_json(folder / "metrics.json")
    config = read_json(folder / "config.json")
    evaluation = read_json(folder / "evaluation.json")
    provenance = read_json(folder / "provenance.json")
    checkpoint_path = folder / "best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    check(metrics.get("status") == "complete" and metrics.get("epochs_completed") == config.get("epochs") == 160,
          "Run must have completed its fixed 160 epochs: " + run)
    check(metrics.get("training_examples") == 40000 and metrics.get("validation_examples") == 5000
          and config.get("train_all") is False, "Benchmark requires a 40k/5k candidate: " + run)
    check(metrics.get("config") == config, "Run configuration metadata mismatch: " + run)
    check(evaluation.get("validation_examples") == 5000 and isinstance(evaluation.get("flip_tta"), bool),
          "Missing original development-selected TTA policy: " + run)
    check(evaluation.get("checkpoint_sha256") == checkpoint_hash, "Stale original evaluation/checkpoint hash: " + run)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    check(checkpoint.get("config") == config and checkpoint.get("model_config") == evaluation.get("model_config"),
          "Checkpoint architecture/configuration mismatch: " + run)
    check(checkpoint.get("epoch") == evaluation.get("epoch") and checkpoint.get("variant") == evaluation.get("variant"),
          "Checkpoint epoch/variant differs from the frozen evaluation: " + run)
    for evidence in (checkpoint, evaluation, provenance):
        check(evidence.get("data_signature") == signature, "Official training source/split mismatch: " + run)
    checkpoint_code = checkpoint.get("provenance", {})
    check(checkpoint_code.get("source_sha256") == evaluation.get("provenance", {}).get("source_sha256")
          == provenance.get("code", {}).get("source_sha256") == current_source_hashes,
          "Frozen training source fingerprint mismatch: " + run)
    check(checkpoint_code.get("git_commit") == evaluation.get("provenance", {}).get("git_commit")
          == provenance.get("code", {}).get("git_commit"), "Source commit metadata mismatch: " + run)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    parameters = count_parameters(model)
    check(0 < parameters <= 5_000_000 and parameters == checkpoint.get("parameter_count")
          == evaluation.get("parameter_count") == metrics.get("parameter_count"), "Single-model parameter budget/count mismatch: " + run)
    metadata = {
        "run": run, "checkpoint_sha256": checkpoint_hash, "parameter_count": parameters,
        "model_config": checkpoint["model_config"], "seed": config["seed"], "training_examples": 40000,
        "flip_tta": evaluation["flip_tta"], "source_git_commit": checkpoint_code.get("git_commit"),
        "source_sha256": current_source_hashes, "data_signature": signature,
        "original_evaluation_sha256": sha256(folder / "evaluation.json"),
        "selected_epoch": checkpoint["epoch"], "variant": checkpoint["variant"],
        "competition": COMPETITION, "hidden_test_labels_loaded": False,
    }
    return model, metadata


@torch.inference_mode()
def infer(model, test_pixels, device, flip_tta, batch_size):
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    predictions = []
    for start in range(0, len(test_pixels), batch_size):
        batch = normalize(test_pixels[start : start + batch_size])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch)
            if flip_tta:
                logits = (logits + model(batch.flip(-1))) * 0.5
        check(torch.isfinite(logits).all().item(), "Nonfinite hidden-image inference logits")
        predictions.extend(logits.argmax(1).cpu().tolist())
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", choices=ALLOWED_RUNS, default=list(ALLOWED_RUNS[:3]))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    check(len(args.runs) == len(set(args.runs)), "Duplicate requested runs")
    check(1 <= args.batch_size <= 128, "Batch size must be in 1..128 to preserve GPU headroom")
    if args.device == "cuda":
        check(torch.cuda.is_available(), "CUDA requested but unavailable")
        check(torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported(),
              "Exact frozen inference requires native BF16 support; no precision fallback is applied")
    seed_all(2026, threads=2)
    device = torch.device(args.device)
    protected_paths = [args.results_dir / name / "evaluation.json" for name in args.runs]
    protected_paths += [args.results_dir / name for name in ("selection.json", "confirmation.json")]
    protected = {path: sha256(path) for path in protected_paths if path.is_file()}
    images, labels, train_ids, validation_ids = load_training_arrays(args.data_dir)
    check(len(images) == len(labels) == 50000 and len(train_ids) == 40000 and len(validation_ids) == 5000,
          "Official train/development cache has unexpected sizes")
    del images, labels, train_ids, validation_ids
    manifest = read_json(args.data_dir / "training_manifest.json")
    source = training_source(args.data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    current_source_hashes = code_provenance()["source_sha256"]
    # One validated read of all official hidden images; never load their labels.
    ids, test_images = load_kaggle_test(args.data_dir)
    test_manifest = read_json(args.data_dir / "kaggle_test_manifest.json")
    check(test_manifest.get("competition") == COMPETITION and test_manifest.get("count") == 2000
          and test_manifest.get("labels_loaded") is False and len(ids) == 2000,
          "Official test manifest is invalid")
    test_signature = {
        "test_csv_sha256": test_manifest["test_csv_sha256"],
        "sample_submission_sha256": test_manifest["sample_submission_sha256"],
        "ordered_test_png_sha256": test_manifest["source_provenance"]["ordered_test_png_sha256"],
        "archive_sha256": test_manifest["source_provenance"].get("archive_sha256"),
        "source_manifest_sha256": test_manifest["source_provenance"].get("source_manifest_sha256"),
    }
    test_pixels = torch.as_tensor(test_images, device=device)
    del test_images
    for run in args.runs:
        model, metadata = checked_run(args.results_dir, run, signature, current_source_hashes)
        metadata["test_input_signature"] = test_signature
        metadata["training_manifest_sha256"] = sha256(args.data_dir / "training_manifest.json")
        output = args.output_root / ("submission_" + run)
        csv_path, metadata_path = output / "submission.csv", output / "metadata.json"
        cached = read_json(metadata_path) if metadata_path.is_file() else {}
        cache_matches = csv_path.is_file() and all(cached.get(key) == value for key, value in metadata.items())
        validation = validate_submission(csv_path, args.data_dir) if cache_matches else None
        cache_matches = bool(cache_matches and cached.get("submission_sha256") == validation["sha256"]
                             and cached.get("rows") == validation["rows"] and cached.get("columns") == validation["columns"]
                             and cached.get("ids_match_official_order") is True)
        if cache_matches:
            event = {"run": run, "status": "cached", "rows": 2000, "submission_sha256": validation["sha256"]}
        else:
            predictions = infer(model, test_pixels, device, metadata["flip_tta"], args.batch_size)
            check(len(predictions) == len(ids), "Prediction count mismatch")
            output.mkdir(parents=True, exist_ok=True)
            temporary = csv_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "label"])
                writer.writerows(zip(ids, predictions))
            validation = validate_submission(temporary, args.data_dir)
            temporary.replace(csv_path)
            metadata.update(submission_sha256=validation["sha256"], rows=validation["rows"],
                            columns=validation["columns"], ids_match_official_order=validation["ids_match_official_order"])
            write_json(metadata_path, metadata)
            event = {"run": run, "status": "generated", "rows": 2000, "parameter_count": metadata["parameter_count"],
                     "flip_tta": metadata["flip_tta"], "submission_sha256": validation["sha256"]}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(event), flush=True)
    for path, original_hash in protected.items():
        check(sha256(path) == original_hash, "An original evaluation/selection/confirmation artifact changed")
    print("BENCHMARK_SUBMISSIONS_READY: no uploads performed; original evaluations and frozen selection preserved.", flush=True)


if __name__ == "__main__":
    main()
