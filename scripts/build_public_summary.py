"""Publishable scalar evidence from private artifacts; never copy private inputs.

The output contains allowlisted configurations, aggregate metrics and hashes.
It excludes images, weights, prediction rows, gold-label rows, credentials,
environment paths and arbitrary error messages. Missing measurements stay null
and appear in pending_fields. --require-complete also verifies checkpoint bytes.
--require-five-submissions requires all five distinct completed official entries.

Example: python scripts/build_public_summary.py --output docs/results.json --require-complete
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
CANDIDATES = ("baseline", "architecture", "regularized")
RUNS = (*CANDIDATES, "final_full")
SUBMISSION_RUNS = (*RUNS, "robust_full")
SAMPLE_SHA256 = "83aa7ddd58d46026c3b3f870f507c33b178ebfbdf24e0a69d9f7a9d6d8cfff59"
CORE_FILES = ("train.py", "model.py", "data.py", "runtime.py")
ROBUST_SOURCE_FILES = (*CORE_FILES, "train_robust.py", "configs/robust_full.json")
TTA_MODES = ("reflectpad4_five_crops_plus_horizontal_flip_mean_logits",
             "reflect-pad4-five-crops-and-flips-logit-average")
CONFIG_KEYS = (
    "depth", "widen_factor", "dropout", "num_classes", "epochs", "batch_size",
    "lr", "min_lr", "warmup_epochs", "momentum", "weight_decay",
    "label_smoothing", "cutout", "mixup_alpha", "mixup_probability", "ema_decay",
    "seed", "split_seed", "train_all", "final_variant", "fixed_flip_tta",
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    check(isinstance(value, dict), "Expected a JSON object: " + path.name)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    if value is not None:
        check(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256 evidence")
    return value


def number(value, *, accuracy=False):
    if value is None:
        return None
    check(not isinstance(value, bool), "Boolean supplied as a measurement")
    value = float(value)
    check(math.isfinite(value), "Nonfinite measurement")
    if accuracy:
        check(0 <= value <= 1, "Accuracy must be a fraction in [0,1]")
    return value


def configuration(raw):
    result = {}
    for key in CONFIG_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "final_variant":
            check(value in ("model", "ema"), "Unknown model variant")
        elif key in ("train_all", "fixed_flip_tta"):
            check(isinstance(value, bool), "Configuration boolean has wrong type")
        else:
            check(isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value), "Configuration must contain finite numeric values")
        result[key] = value
    return result


def code_provenance(raw):
    commit = raw.get("git_commit", raw.get("frozen_commit"))
    if commit is not None:
        check(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{7,40}", commit), "Invalid source commit")
    hashes = raw.get("source_sha256", raw.get("source_sha", {}))
    return {"git_commit": commit, "source_sha256": {name: digest(hashes[name]) for name in CORE_FILES if name in hashes}}


def aggregate_confusion(value, examples, accuracy):
    if value is None:
        return None
    check(isinstance(value, list) and len(value) == 10 and all(isinstance(row, list) and len(row) == 10 for row in value),
          "Confusion matrix must be 10x10")
    check(all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for row in value for n in row),
          "Confusion counts must be nonnegative integers")
    total = sum(map(sum, value))
    check(total == examples, "Confusion counts do not match evaluation size")
    if accuracy is not None:
        check(math.isclose(sum(value[i][i] for i in range(10)) / total, accuracy, abs_tol=1e-12),
              "Confusion counts do not match measured accuracy")
    return value


def validate_csv(path, sample_path):
    with sample_path.open(newline="", encoding="utf-8-sig") as stream:
        sample = csv.DictReader(stream)
        check(sample.fieldnames == ["id", "label"], "Official sample header mismatch")
        expected = [row["id"] for row in sample]
    check(len(expected) == len(set(expected)) == 2000, "Official sample must have 2000 unique IDs")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == ["id", "label"], "Submission header mismatch")
        count = 0
        for count, row in enumerate(reader, 1):
            check(count <= 2000 and set(row) == {"id", "label"}, "Malformed submission row")
            check(row["id"] == expected[count - 1], "Submission IDs/order mismatch")
            check(row["label"] in {str(i) for i in range(10)}, "Submission label outside 0..9")
    check(count == 2000, "Submission must contain exactly 2000 rows")
    return {"rows": count, "columns": ["id", "label"], "ids_match_official_order": True, "sha256": sha256(path)}


def robust_configuration(raw):
    result = configuration(raw)
    for key in ("randaugment_n", "randaugment_m", "checkpoint_interval"):
        if key in raw:
            check(isinstance(raw[key], int) and not isinstance(raw[key], bool) and raw[key] >= 0,
                  "Robust augmentation/checkpoint setting must be a nonnegative integer")
            result[key] = raw[key]
    if "randaugment_interpolation" in raw:
        check(raw["randaugment_interpolation"] == "bilinear", "Unknown robust interpolation setting")
        result["randaugment_interpolation"] = raw["randaugment_interpolation"]
    if "randaugment_fill" in raw:
        fill = raw["randaugment_fill"]
        check(isinstance(fill, list) and len(fill) == 3 and all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255 for value in fill),
            "Robust RGB fill must contain three uint8 channel values")
        result.update({"randaugment_fill_" + channel: value for channel, value in zip("rgb", fill)})
    for key in ("exploratory_after_public_feedback", "confirmation_endorsement"):
        if key in raw:
            check(isinstance(raw[key], bool), "Robust protocol flag must be boolean")
            result[key] = raw[key]
    if "tta_mode" in raw:
        check(raw["tta_mode"] in TTA_MODES, "Unknown robust test-time augmentation rule")
        result["tta_mode"] = raw["tta_mode"]
    return result


def robust_evidence(results_dir, split_sha, archive_sha, verify_checkpoints):
    """Separate post-feedback exploration from the unchanged primary study."""
    folder = results_dir / "robust_full"
    metrics = read_json(folder / "metrics.json")
    evaluation = read_json(folder / "evaluation.json")
    provenance = read_json(folder / "provenance.json")
    raw_config = read_json(folder / "config.json") or metrics.get("config", {})
    config = robust_configuration(raw_config)
    if raw_config:
        check(config.get("epochs") == 300 and config.get("train_all") is True
              and config.get("final_variant") == "ema", "Robust recipe must be the full-data 300-epoch EMA run")
        check(config.get("exploratory_after_public_feedback") is True and config.get("confirmation_endorsement") is False,
              "Robust experiment must remain post-public-feedback exploration without primary confirmation endorsement")
    if metrics:
        check(metrics.get("status") in ("training", "running", "complete", "failed"), "Unknown robust training status")
        check(raw_config == metrics.get("config"), "Robust configuration artifacts disagree")
        check(metrics.get("training_examples") == 50000 and metrics.get("validation_examples") == 0,
              "Robust run must train on all 50k images with no held-out development score")
        check(metrics.get("best_validation_accuracy") is None and metrics.get("best_validation_loss") is None,
              "Robust full-data training cannot claim a development score")
        check(isinstance(metrics.get("epochs_completed"), int) and 0 <= metrics["epochs_completed"] <= 300,
              "Invalid robust epoch count")
        if metrics["status"] == "complete":
            check(metrics["epochs_completed"] == 300, "Completed robust run has fewer than 300 epochs")
            check(bool(evaluation), "Completed robust run is missing its final inference evidence")
    params = metrics.get("parameter_count", evaluation.get("parameter_count"))
    if params is not None:
        check(isinstance(params, int) and not isinstance(params, bool) and 0 < params <= 5_000_000,
              "Robust parameter budget violated")
    if "bn_recalibrated_on_training_only" in metrics:
        check(isinstance(metrics["bn_recalibrated_on_training_only"], bool), "Robust BN calibration flag must be boolean")
    code_raw = provenance.get("code", evaluation.get("provenance", {}))
    code = code_provenance(code_raw)
    code["source_sha256"] = {name: digest(code_raw.get("source_sha256", {}).get(name))
                              for name in ROBUST_SOURCE_FILES if name in code_raw.get("source_sha256", {})}
    checkpoint = digest(evaluation.get("checkpoint_sha256"))
    verified = False
    if evaluation:
        check(metrics.get("status") == "complete", "Robust evaluation belongs to an unfinished run")
        check(evaluation.get("validation_examples") == 0 and all(evaluation.get(key) is None for key in (
            "plain_validation_accuracy", "flip_validation_accuracy", "validation_accuracy", "validation_loss")),
            "Robust full-data evaluation cannot claim overlapping development accuracy")
        check(evaluation.get("epoch") == 300 and evaluation.get("variant") == "ema"
              and evaluation.get("flip_tta") is True and evaluation.get("tta_mode") == config.get("tta_mode")
              and evaluation.get("tta_views") == 10 and evaluation.get("tta_aggregation") == "mean_logits",
              "Robust fixed epoch/variant/ten-view inference policy mismatch")
        check(evaluation.get("parameter_count") == params and evaluation.get("exploratory_after_public_feedback") is True
              and evaluation.get("main_confirmation_endorses_this_run") is False,
              "Robust evaluation lacks its separate exploratory context")
        for record in (evaluation, provenance):
            signature = record.get("data_signature", {})
            check(signature.get("split_sha256") == split_sha and signature.get("archive_sha256") == archive_sha,
                  "Robust official source/split fingerprint mismatch")
        check(code_provenance(evaluation.get("provenance", {})) == code_provenance(code_raw),
              "Robust checkpoint/code provenance mismatch")
        check(len(code["source_sha256"]) == len(ROBUST_SOURCE_FILES) and all(code["source_sha256"].values())
              and all(evaluation.get("provenance", {}).get("source_sha256", {}).get(name) == value
                      for name, value in code["source_sha256"].items()), "Robust extended source fingerprints disagree")
        if verify_checkpoints:
            check((folder / "best.pt").is_file() and sha256(folder / "best.pt") == checkpoint,
                  "Stale robust checkpoint/evaluation hash")
            verified = True
    return {
        "status": metrics.get("status", "pending"), "purpose": "post_public_feedback_exploratory_robustness",
        "main_confirmation_endorses_this_run": False, "development_accuracy": None,
        "config": config or None, "epochs_completed": metrics.get("epochs_completed"),
        "training_examples": metrics.get("training_examples"), "development_examples": metrics.get("validation_examples"),
        "parameter_count": params, "elapsed_seconds": number(metrics.get("elapsed_seconds")),
        "variant": evaluation.get("variant"), "flip_tta": evaluation.get("flip_tta"),
        "tta_mode": config.get("tta_mode"), "checkpoint_sha256": checkpoint,
        "checkpoint_bytes_verified": verified, "code": code,
        "bn_recalibrated_on_training_only": metrics.get("bn_recalibrated_on_training_only"),
    }


def official_submissions(results_dir, data_dir, *, require_five=False):
    path = results_dir / "kaggle_submissions.json"
    receipts = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else []
    check(isinstance(receipts, list), "Kaggle submission history must be a list")
    if receipts:
        check(sha256(data_dir / "sample_submission.csv") == SAMPLE_SHA256,
              "Official benchmark sample submission fingerprint mismatch")
    result, names, refs, csv_hashes = [], set(), set(), set()
    for receipt in receipts:
        check(isinstance(receipt, dict), "Official submission receipt must be an object")
        name = receipt.get("run")
        check(name in SUBMISSION_RUNS and name not in names, "Unknown or duplicate official submission run")
        check(receipt.get("competition") == COMPETITION and receipt.get("team") == "Saber",
              "Benchmark receipt belongs to another competition/team")
        check(receipt.get("status") == "complete", "Recorded official entry has not completed scoring")
        ref = receipt.get("submission_ref")
        check(not isinstance(ref, bool) and isinstance(ref, (int, str)) and re.fullmatch(r"[0-9]+", str(ref))
              and int(ref) > 0 and int(ref) not in refs, "Invalid or duplicate official submission reference")
        csv_sha = digest(receipt.get("submission_sha256"))
        check(csv_sha is not None and csv_sha not in csv_hashes, "Missing or duplicate official prediction hash")
        timestamp = receipt.get("submitted_at")
        check(isinstance(timestamp, str) and re.fullmatch(r"[0-9TtZz:.+ -]{10,50}", timestamp),
              "Invalid official submission timestamp")
        folder = results_dir / name
        metrics = read_json(folder / "metrics.json")
        config = read_json(folder / "config.json") or metrics.get("config", {})
        evaluation = read_json(folder / "evaluation.json")
        full = name in ("final_full", "robust_full")
        epochs = 300 if name == "robust_full" else 160
        examples = 50000 if full else 40000
        check(metrics.get("status") == "complete" and metrics.get("epochs_completed") == epochs
              and config.get("epochs") == epochs and config.get("train_all") is full,
              "Submitted run lacks its completed fixed training budget: " + name)
        check(metrics.get("config") == config and metrics.get("training_examples") == examples
              and metrics.get("validation_examples") == (0 if full else 5000), "Submitted run partition/config mismatch")
        params = metrics.get("parameter_count")
        check(isinstance(params, int) and not isinstance(params, bool) and 0 < params <= 5_000_000
              and receipt.get("parameter_count") == params and receipt.get("training_examples") == examples,
              "Submitted parameter/sample counts disagree with training")
        prediction_folder = results_dir / ("submission_" + name) if name in CANDIDATES else folder
        csv_path = prediction_folder / "submission.csv"
        validated = validate_csv(csv_path, data_dir / "sample_submission.csv")
        check(validated["sha256"] == csv_sha, "Official benchmark CSV no longer matches its receipt")
        metadata = read_json(prediction_folder / "metadata.json")
        if name == "final_full" and not metadata:
            check(evaluation.get("submission", {}).get("sha256") == csv_sha, "Primary final evaluation CSV hash mismatch")
            metadata = {
                "submission_sha256": csv_sha, "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
                "flip_tta": evaluation.get("flip_tta"), "source_git_commit": evaluation.get("provenance", {}).get("git_commit"),
                "source_sha256": evaluation.get("provenance", {}).get("source_sha256", {}),
                "parameter_count": evaluation.get("parameter_count"), "training_examples": examples,
            }
        else:
            check(metadata.get("run") == name and metadata.get("competition") == COMPETITION
                  and metadata.get("hidden_test_labels_loaded") is False, "Submitted prediction metadata context mismatch")
            check(metadata.get("original_evaluation_sha256") == sha256(folder / "evaluation.json"),
                  "Submitted run's original evaluation artifact changed")
        check(metadata.get("submission_sha256") == csv_sha and metadata.get("parameter_count") == params
              and metadata.get("training_examples") == examples, "Submitted metadata disagrees with CSV/training")
        checkpoint_sha = digest(evaluation.get("checkpoint_sha256"))
        check(checkpoint_sha is not None and metadata.get("checkpoint_sha256") == checkpoint_sha
              and sha256(folder / "best.pt") == checkpoint_sha, "Submitted checkpoint bytes/evaluation/metadata mismatch")
        check(isinstance(receipt.get("flip_tta"), bool) and receipt["flip_tta"] == metadata.get("flip_tta"),
              "Submitted inference averaging rule mismatch")
        source = code_provenance({"git_commit": metadata.get("source_git_commit"),
                                  "source_sha256": metadata.get("source_sha256", {})})
        check(source["git_commit"] is not None and receipt.get("source_git_commit") == source["git_commit"]
              and evaluation.get("provenance", {}).get("git_commit") == source["git_commit"],
              "Submitted source revision disagrees with checkpoint provenance")
        allowlist = ROBUST_SOURCE_FILES if name == "robust_full" else CORE_FILES
        source["source_sha256"] = {key: digest(metadata.get("source_sha256", {})[key]) for key in allowlist
                                     if key in metadata.get("source_sha256", {})}
        check(len(source["source_sha256"]) == len(allowlist) and all(source["source_sha256"].values()),
              "Submitted source fingerprints are incomplete")
        check(all(evaluation.get("provenance", {}).get("source_sha256", {}).get(key) == value
                  for key, value in source["source_sha256"].items()), "Submitted source hashes disagree with checkpoint provenance")
        public_score = number(receipt.get("public_score"), accuracy=True)
        check(public_score is not None, "Completed official receipt has no measured public score")
        public = {
            "run": name, "status": "complete", "competition": COMPETITION, "team": "Saber",
            "public_score": public_score, "private_score": number(receipt.get("private_score"), accuracy=True),
            "submission_ref": int(ref), "submitted_at": timestamp, "submission_sha256": csv_sha,
            "checkpoint_sha256": checkpoint_sha, "training_examples": examples, "parameter_count": params,
            "epochs_completed": epochs, "flip_tta": receipt["flip_tta"],
            "source_git_commit": source["git_commit"],
            "code": source, "config": robust_configuration(config) if name == "robust_full" else configuration(config),
            "primary_protocol_full_retrain": name == "final_full", "exploratory_after_public_feedback": name == "robust_full",
        }
        if name == "robust_full":
            public["main_confirmation_endorses_this_run"] = False
        mode = metadata.get("tta_mode", receipt.get("tta_mode"))
        if mode is not None:
            check(mode in TTA_MODES and ("tta_mode" not in receipt or receipt["tta_mode"] == mode),
                  "Submitted test-time augmentation metadata mismatch")
            public["tta_mode"] = mode
        names.add(name); refs.add(int(ref)); csv_hashes.add(csv_sha); result.append(public)
    result.sort(key=lambda item: SUBMISSION_RUNS.index(item["run"]))
    missing = [name for name in SUBMISSION_RUNS if name not in names]
    if require_five:
        check(not missing and len(result) == len(refs) == len(csv_hashes) == 5,
              "Five completed distinct official submissions are required; missing: " + ", ".join(missing))
    return result, {"expected_runs": list(SUBMISSION_RUNS), "complete_count": len(result),
                    "five_complete": not missing, "missing_runs": missing}


def build_summary(results_dir, data_dir, *, verify_checkpoints=False, require_complete=False,
                  require_five_submissions=False):
    pending = []

    def need(condition, field):
        if not condition:
            pending.append(field)

    training = read_json(data_dir / "training_manifest.json")
    split = training.get("split", {})
    source = training.get("source_provenance", {})
    if training:
        check(training.get("competition") == COMPETITION, "Training competition mismatch")
        check(training.get("source_type") == "official_kaggle_competition_training"
              and split.get("schema") == "train_dev_confirmation_v1", "Unknown official training/split schema")
        check(training.get("external_training_data_loaded") is False and training.get("official_test_labels_loaded") is False,
              "Training evidence must confirm official-only data and no hidden labels")
        counts = [split.get(key) for key in ("train_count", "validation_count", "confirmation_count")]
        check(counts == [40000, 5000, 5000] and sum(counts) == training.get("images", {}).get("count") == 50000,
              "Expected the fixed 40k/5k/5k partition of 50k official images")
    need(bool(training), "official_training_manifest")
    split_sha = digest(split.get("indices_sha256"))
    archive_sha = digest(source.get("official_train_npz_sha256", source.get("ordered_training_png_sha256")))
    data = {
        "source": training.get("source_type"), "official_image_count": training.get("images", {}).get("count"),
        "external_training_data_loaded": training.get("external_training_data_loaded"),
        "hidden_test_labels_loaded": training.get("official_test_labels_loaded"),
        "train_labels_csv_sha256": digest(source.get("train_labels_csv_sha256")),
        "ordered_training_png_sha256": digest(source.get("ordered_training_png_sha256")),
        "official_train_archive_sha256": digest(source.get("official_train_npz_sha256")),
        "source_manifest_sha256": digest(source.get("official_source_manifest_sha256")),
        "split": {key: split.get(key) for key in ("schema", "seed", "train_count", "validation_count", "confirmation_count")},
    }
    data["split"].update({key: digest(split.get(key)) for key in (
        "indices_sha256", "train_indices_sha256", "validation_indices_sha256", "confirmation_indices_sha256")})
    need(bool(data["train_labels_csv_sha256"] and data["ordered_training_png_sha256"] and split_sha
              and all(data["split"][key] for key in (
                  "train_indices_sha256", "validation_indices_sha256", "confirmation_indices_sha256"))),
         "official_source_and_split_fingerprints")
    actual = {}
    runs = {}
    for name in RUNS:
        folder = results_dir / name
        metrics = read_json(folder / "metrics.json")
        evaluation = read_json(folder / "evaluation.json")
        provenance = read_json(folder / "provenance.json")
        raw_config = read_json(folder / "config.json") or metrics.get("config", {})
        config = configuration(raw_config)
        if raw_config and metrics.get("config"):
            check(raw_config == metrics["config"], "Run configuration artifacts disagree: " + name)
        full = name == "final_full"
        if config:
            check(config.get("epochs") == 160 and config.get("train_all") is full,
                  "Run must use its declared 160-epoch candidate/full-data context: " + name)
            if split:
                check(config.get("split_seed") == split.get("seed"), "Run uses a different fold: " + name)
        if metrics:
            check(metrics.get("status") in ("running", "complete"), "Unknown training status")
            check(metrics.get("training_examples") == (50000 if full else 40000)
                  and metrics.get("validation_examples") == (0 if full else 5000), "Run partition counts mismatch: " + name)
            check(isinstance(metrics.get("epochs_completed"), int) and 0 <= metrics["epochs_completed"] <= 160,
                  "Invalid completed epoch count: " + name)
            if metrics.get("status") == "complete":
                check(metrics["epochs_completed"] == 160, "Completed run has fewer than 160 epochs: " + name)
        params = metrics.get("parameter_count", evaluation.get("parameter_count"))
        if params is not None:
            check(isinstance(params, int) and 0 < params <= 5_000_000, "Parameter budget violated: " + name)
        for record in (evaluation, provenance):
            if "parameter_count" in record and params is not None:
                check(record["parameter_count"] == params, "Parameter counts disagree: " + name)
            signature = record.get("data_signature", {})
            if signature and split_sha:
                check(signature.get("split_sha256") == split_sha and signature.get("archive_sha256") == archive_sha,
                      "Run source/fold fingerprint mismatch: " + name)
        code = code_provenance(provenance.get("code", evaluation.get("provenance", {})))
        if provenance.get("code") and evaluation.get("provenance"):
            check(code == code_provenance(evaluation["provenance"]), "Checkpoint/code provenance mismatch: " + name)
        checkpoint_sha = digest(evaluation.get("checkpoint_sha256"))
        checkpoint_verified = False
        if evaluation:
            check(metrics.get("status") == "complete", "Evaluation belongs to an unfinished run: " + name)
            check(evaluation.get("validation_examples") == (0 if full else 5000), "Evaluation partition mismatch: " + name)
            check(isinstance(evaluation.get("epoch"), int) and 1 <= evaluation["epoch"] <= 160,
                  "Invalid selected checkpoint epoch: " + name)
            check(evaluation.get("variant") in ("model", "ema"), "Invalid selected variant: " + name)
            check(isinstance(evaluation.get("flip_tta"), bool), "Inference policy is missing: " + name)
            if verify_checkpoints or require_complete:
                check((folder / "best.pt").is_file(), "Checkpoint bytes are unavailable: " + name)
                check(sha256(folder / "best.pt") == checkpoint_sha, "Stale checkpoint/evaluation hash: " + name)
                checkpoint_verified = True
        development = {key: number(evaluation.get(field), accuracy="accuracy" in field) for key, field in (
            ("plain_accuracy", "plain_validation_accuracy"), ("flip_accuracy", "flip_validation_accuracy"),
            ("chosen_accuracy", "validation_accuracy"), ("chosen_loss", "validation_loss"))}
        if evaluation and not full:
            expected_accuracy = development["flip_accuracy"] if evaluation["flip_tta"] else development["plain_accuracy"]
            check(development["chosen_accuracy"] == expected_accuracy, "Development score/TTA policy mismatch: " + name)
            need(all(value is not None for value in development.values()), f"runs.{name}.measured_development_scores")
        if full and metrics:
            check(metrics.get("best_validation_accuracy") is None and metrics.get("best_validation_loss") is None,
                  "Full-data model cannot claim a held-out development score")
            check(all(value is None for value in development.values()), "Full-data evaluation leaked an overlapping score")
            if evaluation:
                check(evaluation["epoch"] == 160 and evaluation["variant"] == config["final_variant"]
                      and evaluation["flip_tta"] == config["fixed_flip_tta"], "Final fixed epoch/variant/TTA policy mismatch")
        need(metrics.get("status") == "complete", f"runs.{name}.training_complete")
        need(bool(evaluation), f"runs.{name}.evaluation")
        need(bool(checkpoint_sha), f"runs.{name}.checkpoint_fingerprint")
        need(len(config) == len(CONFIG_KEYS) and params is not None and len(code["source_sha256"]) == len(CORE_FILES),
             f"runs.{name}.configuration_and_provenance")
        need(code["git_commit"] is not None and metrics.get("elapsed_seconds") is not None,
             f"runs.{name}.source_commit_and_runtime")
        runs[name] = {
            "status": metrics.get("status", "pending"), "config": config or None,
            "epochs_completed": metrics.get("epochs_completed"), "selected_epoch": evaluation.get("epoch"),
            "parameter_count": params, "training_examples": metrics.get("training_examples"),
            "development_examples": metrics.get("validation_examples"), "elapsed_seconds": number(metrics.get("elapsed_seconds")),
            "best_checkpoint_selection_accuracy": number(metrics.get("best_validation_accuracy"), accuracy=True),
            "development": development, "variant": evaluation.get("variant"), "flip_tta": evaluation.get("flip_tta"),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes_verified": checkpoint_verified, "code": code,
        }
        actual[name] = {"evaluation": evaluation, "config": config}
    seeds = {runs[name]["config"]["seed"] for name in CANDIDATES if runs[name]["config"]}
    check(len(seeds) <= 1, "The three candidate comparisons must share the same training seed")

    selection = read_json(results_dir / "selection.json")
    public_selection = {"status": "pending"}
    if selection:
        chosen = selection.get("run")
        check(chosen in CANDIDATES and selection.get("confirmation_used_for_selection") is False, "Invalid frozen recipe selection")
        selected = actual[chosen]["evaluation"]
        check(bool(selected) and all(actual[name]["evaluation"] for name in CANDIDATES), "Selection is missing candidate evaluations")
        winner = max(CANDIDATES, key=lambda name: (actual[name]["evaluation"]["validation_accuracy"],
                                                  -actual[name]["evaluation"]["validation_loss"]))
        check(winner == chosen, "Selected recipe is not the recorded development winner")
        for key, source_key in (("checkpoint_sha256", "checkpoint_sha256"), ("variant", "variant"),
                                ("flip_tta", "flip_tta"), ("validation_accuracy", "validation_accuracy")):
            check(selection.get(key) == selected.get(source_key), "Frozen selection disagrees with development evidence")
        check(selection.get("all_development_accuracies") == {
            name: actual[name]["evaluation"]["validation_accuracy"] for name in CANDIDATES}, "Frozen candidate scores mismatch")
        public_selection = {key: selection.get(key) for key in (
            "run", "validation_accuracy", "flip_tta", "checkpoint_sha256", "variant", "confirmation_used_for_selection")}
        public_selection["status"] = "frozen"
        final_config = actual["final_full"]["config"]
        if final_config:
            expected = dict(actual[chosen]["config"], train_all=True, final_variant=selection["variant"], fixed_flip_tta=selection["flip_tta"])
            check(final_config == expected, "Full-data retrain changed the frozen selected recipe")
    need(bool(selection), "frozen_selection")

    confirmation = read_json(results_dir / "confirmation.json")
    public_confirmation = {"status": confirmation.get("status", "pending")}
    if confirmation:
        check(confirmation.get("status") in ("started", "complete"), "Unknown confirmation status")
        check(bool(selection) and confirmation.get("run") == selection.get("run")
              and confirmation.get("checkpoint_sha256") == selection.get("checkpoint_sha256")
              and confirmation.get("flip_tta") == selection.get("flip_tta"), "Confirmation is for another model/policy")
        check(confirmation.get("examples") == 5000, "Confirmation must use the independent 5000 examples")
        check(confirmation.get("selection_file_sha256") == sha256(results_dir / "selection.json"), "Confirmation selection receipt is stale")
        if confirmation.get("status") == "complete":
            check(confirmation.get("split_sha256") == split_sha, "Confirmation fold fingerprint mismatch")
            score = number(confirmation.get("accuracy"), accuracy=True)
            check(score is not None, "Completed confirmation is missing its measured score")
            public_confirmation.update(
                accuracy=score, loss=number(confirmation.get("loss")), examples=5000,
                checkpoint_sha256=digest(confirmation.get("checkpoint_sha256")),
                selection_file_sha256=digest(confirmation.get("selection_file_sha256")),
                split_sha256=split_sha, flip_tta=confirmation.get("flip_tta"),
                confusion_matrix=aggregate_confusion(confirmation.get("confusion_matrix"), 5000, score),
            )
    need(confirmation.get("status") == "complete", "independent_confirmation")

    stability = read_json(results_dir / "stability.json")
    public_stability = {"status": stability.get("status", "pending")}
    if stability:
        check(stability.get("status") in ("waiting", "training", "evaluating", "failed", "complete"), "Unknown stability status")
        check(stability.get("purpose") == "independent_seed_stability_only" and stability.get("used_for_selection") is False
              and stability.get("confirmation_evaluated") is False and stability.get("hidden_predictions_generated") is False,
              "Stability run must remain excluded from selection, confirmation and hidden prediction")
        check(stability.get("split_seed") == split.get("seed") and stability.get("comparison_run") == "regularized",
              "Stability run uses another recipe/fold")
        public_stability.update({key: stability.get(key) for key in (
            "purpose", "used_for_selection", "seed", "split_seed", "comparison_run", "flip_tta", "epoch", "variant",
            "confirmation_evaluated", "hidden_predictions_generated")})
        for key in ("validation_accuracy", "plain_validation_accuracy", "flip_validation_accuracy"):
            public_stability[key] = number(stability.get(key), accuracy=True)
        public_stability["training_and_evaluation_seconds"] = number(stability.get("training_and_evaluation_seconds"))
        frozen_code = code_provenance(stability)
        public_stability["frozen_source_commit"] = frozen_code["git_commit"]
        public_stability["code"] = {"git_commit": None, "source_sha256": frozen_code["source_sha256"]}
        public_stability["parameter_count"] = stability.get("parameter_count")
        comparison_code = runs["regularized"]["code"]["source_sha256"]
        if comparison_code:
            check(public_stability["code"]["source_sha256"] == comparison_code, "Stability source differs from the declared recipe")
        if seeds:
            check(stability.get("seed") not in seeds, "Stability run did not use an independent seed")
        if stability.get("status") == "complete":
            check(public_stability["validation_accuracy"] is not None and isinstance(stability.get("parameter_count"), int)
                  and 0 < stability["parameter_count"] <= 5_000_000, "Completed stability metrics are invalid")
            repeat_metrics = read_json(results_dir / "stability_seed2027" / "metrics.json")
            repeat_evaluation = read_json(results_dir / "stability_seed2027" / "evaluation.json")
            check(repeat_metrics.get("epochs_completed") == 160 and repeat_metrics.get("status") == "complete"
                  and repeat_evaluation.get("validation_accuracy") == stability["validation_accuracy"], "Stability receipt lacks matching completed evidence")
            expected = dict(actual["regularized"]["config"], seed=stability["seed"])
            check(configuration(repeat_metrics.get("config", {})) == expected, "Stability recipe changed more than its seed")
            check(repeat_evaluation.get("data_signature", {}).get("split_sha256") == split_sha,
                  "Stability evaluation used another fold")
            measured_code = code_provenance(repeat_evaluation.get("provenance", {}))
            check(measured_code["source_sha256"] == frozen_code["source_sha256"], "Stability measured source changed")
            public_stability["code"] = measured_code
            public_stability["checkpoint_sha256"] = digest(repeat_evaluation.get("checkpoint_sha256"))
            if verify_checkpoints or require_complete:
                check(sha256(results_dir / "stability_seed2027" / "best.pt") == public_stability["checkpoint_sha256"],
                      "Stale stability checkpoint/evaluation hash")
            public_stability["epochs_completed"] = 160

    final_evaluation = actual["final_full"]["evaluation"]
    local_submission = final_evaluation.get("submission", {})
    csv_path = results_dir / "final_full" / "submission.csv"
    submission = None
    if csv_path.is_file():
        submission = validate_csv(csv_path, data_dir / "sample_submission.csv")
        check(local_submission.get("sha256") == submission["sha256"] and local_submission.get("rows") == 2000,
              "Final submission differs from the validated evaluation artifact")
    need(submission is not None, "validated_final_submission")
    kaggle = read_json(results_dir / "kaggle_submission.json")
    public_kaggle = {"status": kaggle.get("status", "pending"), "public_score": None,
                     "private_score": None, "private_score_status": "pending"}
    if kaggle:
        check(kaggle.get("status") in ("pending", "submitted", "complete", "failed"), "Unknown Kaggle receipt status")
        check(kaggle.get("run") == "final_full" and kaggle.get("competition") == COMPETITION and kaggle.get("team") == "Saber",
              "Kaggle receipt is for another run/competition/team")
        check(submission is not None and kaggle.get("submission_sha256") == submission["sha256"], "Kaggle receipt submission hash mismatch")
        ref = kaggle.get("submission_ref")
        check(ref is None or (isinstance(ref, (int, str)) and re.fullmatch(r"[0-9]+", str(ref))), "Invalid Kaggle submission reference")
        timestamp = kaggle.get("submitted_at")
        check(timestamp is None or (isinstance(timestamp, str) and re.fullmatch(r"[0-9TtZz:.+ -]{10,50}", timestamp)), "Invalid submission timestamp")
        public_kaggle.update(run="final_full", competition=COMPETITION, team="Saber", submission_ref=ref,
                             submission_sha256=digest(kaggle.get("submission_sha256")), submitted_at=timestamp,
                             public_score=number(kaggle.get("public_score"), accuracy=True),
                             private_score=number(kaggle.get("private_score"), accuracy=True))
        if public_kaggle["private_score"] is not None:
            public_kaggle["private_score_status"] = "available"
    need(kaggle.get("status") == "complete" and public_kaggle.get("public_score") is not None
         and public_kaggle.get("submission_ref") is not None, "official_kaggle_public_score")
    benchmark_receipts, submission_campaign = official_submissions(results_dir, data_dir,
                                                                  require_five=require_five_submissions)
    robust = robust_evidence(results_dir, split_sha, archive_sha, verify_checkpoints or require_complete)
    summary = {
        "schema_version": 1, "status": "pending" if pending else "complete", "competition": COMPETITION, "team": "Saber",
        "metric": "classification_accuracy_fraction", "official_data": data, "runs": runs,
        "selection": public_selection, "confirmation": public_confirmation, "stability": public_stability,
        "validated_submission": submission, "kaggle_submission": public_kaggle, "pending_fields": pending,
        "official_submissions": benchmark_receipts, "submission_campaign": submission_campaign,
        "robust_exploratory": robust,
    }
    if require_complete:
        check(not pending, "Evidence is incomplete: " + ", ".join(pending))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-five-submissions", action="store_true")
    args = parser.parse_args()
    summary = build_summary(args.results_dir, args.data_dir, verify_checkpoints=args.verify_checkpoints,
                            require_complete=args.require_complete,
                            require_five_submissions=args.require_five_submissions)
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    check(not re.search(r"[A-Za-z]:\\\\|KGAT_|sk-proj-", serialized), "Private path or credential pattern in public summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": summary["status"], "bytes": len(serialized.encode("utf-8")),
                      "pending_fields": summary["pending_fields"]}, indent=2))


if __name__ == "__main__":
    main()
