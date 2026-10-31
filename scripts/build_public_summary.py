"""Publishable scalar evidence from private artifacts; never copy private inputs.

The output contains allowlisted configurations, aggregate metrics and hashes.
It excludes images, weights, prediction rows, gold-label rows, credentials,
environment paths and arbitrary error messages. Missing measurements stay null
and appear in pending_fields. --require-complete also verifies checkpoint bytes.

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
CORE_FILES = ("train.py", "model.py", "data.py", "runtime.py")
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


def build_summary(results_dir, data_dir, *, verify_checkpoints=False, require_complete=False):
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
    summary = {
        "schema_version": 1, "status": "pending" if pending else "complete", "competition": COMPETITION, "team": "Saber",
        "metric": "classification_accuracy_fraction", "official_data": data, "runs": runs,
        "selection": public_selection, "confirmation": public_confirmation, "stability": public_stability,
        "validated_submission": submission, "kaggle_submission": public_kaggle, "pending_fields": pending,
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
    args = parser.parse_args()
    summary = build_summary(args.results_dir, args.data_dir, verify_checkpoints=args.verify_checkpoints,
                            require_complete=args.require_complete)
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
