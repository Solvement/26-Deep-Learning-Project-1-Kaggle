"""Version-2 primary 40k delivery, including independently audited ResizeMix.

The original version-1 delivery and pairing files remain byte-frozen.
No upload, confirmation evaluation, ensemble or full-50k refit path is provided.
All paths and generated predictions belong in ignored private result folders.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import seed_all
import torch
from data import COMPETITION, CSV_SHA256, load_kaggle_test, load_training_arrays
from model import build_model, count_parameters
from train import code_provenance, training_source
from research import benchmark as suite
from research import paired_compare_v2 as paired
from scripts.submit_verified import validate_csv

FREEZE_SCHEMA = "research_candidate_freeze_v2"
PREDICTION_SCHEMA = "research_prediction_delivery_v2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Evidence timestamps must include a timezone")
    return result


def utc_now():
    return datetime.now(timezone.utc)


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def snapshots(paths):
    return {str(Path(path).resolve()): suite.sha256(path) for path in set(map(Path, paths))}


def verify_snapshots(record):
    require(bool(record), "Immutable evidence fingerprints are missing")
    require(all(suite.sha256(path) == digest for path, digest in record.items()),
            "An immutable artifact or source fingerprint changed")


def data_context(data_dir):
    images, labels, train_ids, dev_ids = load_training_arrays(data_dir)
    del images, labels
    manifest = read(data_dir/"training_manifest.json")
    signature = {"archive_sha256": training_source(data_dir)["archive_sha256"],
                 "split_sha256": manifest["split"]["indices_sha256"]}
    return signature, code_provenance()["source_sha256"], train_ids, dev_ids


def artifact_paths(checkpoint_path, benchmark_path, checkpoint):
    folder = checkpoint_path.parent
    paths = [checkpoint_path, benchmark_path, *(folder/name for name in
             ("config.json", "metrics.json", "history.json", "provenance.json", "environment.json"))]
    marker = checkpoint.get("provenance", {}).get("research_training")
    if marker:
        _, validator = paired.training_validator(marker["schema"])
        paths += [validator, benchmark_path.with_name(benchmark_path.stem+".core.json")]
        paths += [ROOT/name for name in marker["source_sha256"]]
    return paths


def audit_experiment(checkpoint_path, benchmark_path, data, context):
    checkpoint_path, benchmark_path = Path(checkpoint_path), Path(benchmark_path)
    require("stability_seed2027" not in checkpoint_path.parts, "Original stability repeat is excluded")
    signature, core, train_ids, dev_ids = data
    checkpoint, document = paired.load_experiment(checkpoint_path, benchmark_path, signature, context,
                                                  core, train_ids, dev_ids)
    config = checkpoint["config"]
    require(config.get("num_classes") == 10 and config.get("train_all") is False,
            "Only official ten-class held-out 40k models are supported; full refits are unsupported")
    require(read(checkpoint_path.parent/"config.json") == config, "Saved configuration differs from checkpoint")
    provenance = read(checkpoint_path.parent/"provenance.json")
    require(provenance.get("code", {}).get("source_sha256") == core and
            provenance.get("data_signature") == signature, "Training provenance differs from official data/core")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    count = count_parameters(model)
    require(type(count) is int and 0 < count <= 5000000 and count_parameters(model, True) == count
            == checkpoint["parameter_count"], "Single-model parameter budget differs")
    require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()
                if value.is_floating_point()), "Checkpoint contains nonfinite weights or buffers")
    return checkpoint, document


def passed_policy(document, baseline, policy):
    require(policy in suite.POLICIES and policy in document.get("policies", {}), "Selected policy is missing")
    report = document["policies"][policy]
    suite.validate_baseline(baseline, {policy: report}, document["data_signature"], document["batch_size"],
                            document["evaluation_context"], suite.sha256(Path(suite.__file__)))
    gate = suite.promotion_gate(report, baseline["policies"]["flip"])
    require(document.get("promotion", {}).get(policy) == gate and gate["passed"] is True,
            "The selected policy did not actually pass the declared complete benchmark gate")
    return gate


def validate_pair(document, primary, baseline, checkpoint_path, benchmark_path, baseline_path, policy):
    expected = {"schema": "exploratory_paired_image_bootstrap_v2", "status": "complete",
        "candidate_policy": policy, "baseline_policy": "flip", "data_signature": primary["data_signature"],
        "evaluation_context": primary["evaluation_context"],
        "candidate_checkpoint_sha256": suite.sha256(checkpoint_path),
        "baseline_checkpoint_sha256": suite.BASELINE_CHECKPOINT_SHA,
        "candidate_benchmark_sha256": suite.sha256(benchmark_path),
        "baseline_benchmark_sha256": suite.sha256(baseline_path),
        "source_sha256": suite.sha256(Path(paired.__file__)),
        "paired_helper_source_sha256": suite.sha256(paired.PAIRED_HELPER_SOURCE_PATH),
        "benchmark_source_sha256": suite.sha256(Path(suite.__file__)),
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
        "automatic_promotion": False}
    require(all(document.get(key) == value for key, value in expected.items()), "Paired evidence source/hash/policy mismatch")
    boot = document.get("bootstrap", {})
    require(all(boot.get(key) == value for key, value in {"original_images": 5000,
        "replicates": paired.REPLICATES, "seed": paired.SEED, "confidence_level": .95,
        "sampling_unit": "original_image_with_all_five_stress_views"}.items()), "Paired bootstrap protocol differs")
    selected, reference = primary["policies"][policy], baseline["policies"]["flip"]
    for name, total in (("clean", 5000), ("stress", 25000)):
        row = boot[name]
        counts = [row.get(key) for key in ("candidate_only_correct", "baseline_only_correct", "both_correct", "both_wrong")]
        require(all(type(value) is int and value >= 0 for value in counts) and sum(counts) == total,
                "Paired correctness counts are incomplete")
        delta = selected[name+"_accuracy"]-reference[name+"_accuracy"]
        require(math.isclose(row.get("delta_accuracy", float("nan")), delta, abs_tol=1e-12) and
                math.isclose((counts[0]-counts[1])/total, delta, abs_tol=1e-12) and
                math.isclose((counts[0]+counts[2])/total, selected[name+"_accuracy"], abs_tol=1e-12) and
                math.isclose((counts[1]+counts[2])/total, reference[name+"_accuracy"], abs_tol=1e-12),
                "Paired point deltas/counts differ from the completed benchmark")
        interval = row.get("percentile_ci95", [])
        require(len(interval) == 2 and all(type(v) in (float, int) and math.isfinite(v) and -1 <= v <= 1 for v in interval)
                and interval[0] <= interval[1], "Paired confidence interval is invalid")
    gate = passed_policy(primary, baseline, policy)
    clean_path = gate["clean_delta"] >= .001-1e-12 and gate["stress_delta"] >= -.002-1e-12
    require(clean_path or boot["stress"]["percentile_ci95"][0] > 0,
            "A robustness-driven promotion requires a positive paired stress confidence interval")


def primary_bundle(checkpoint_path, benchmark_path, baseline_path, paired_path, policy, data_dir):
    data = data_context(data_dir)
    baseline_checkpoint = ROOT/"results/regularized/best.pt"
    require(suite.sha256(baseline_checkpoint) == suite.BASELINE_CHECKPOINT_SHA, "Trusted original baseline changed")
    baseline = read(baseline_path)
    context = baseline.get("evaluation_context", {})
    require(context.get("training_core_sha256") == data[1], "Baseline core source differs")
    original, baseline = audit_experiment(baseline_checkpoint, baseline_path, data, context)
    checkpoint, document = audit_experiment(checkpoint_path, benchmark_path, data, context)
    require(type(checkpoint["config"].get("seed")) is int and checkpoint["config"]["seed"] == 2026,
            "Only the preselected primary seed 2026 may generate predictions")
    require(document.get("baseline_sha256") == suite.sha256(baseline_path), "Candidate uses an arbitrary or stale baseline")
    paired_document = read(paired_path)
    validate_pair(paired_document, document, baseline, checkpoint_path, benchmark_path, baseline_path, policy)
    paths = artifact_paths(checkpoint_path, benchmark_path, checkpoint)
    paths += artifact_paths(baseline_checkpoint, baseline_path, original)
    paths += [paired_path, data_dir/"training_manifest.json", Path(__file__), Path(paired.__file__),
              paired.PAIRED_HELPER_SOURCE_PATH, Path(suite.__file__)]
    paths += [ROOT/name for name in data[1]]
    return {"checkpoint": checkpoint, "benchmark": document, "baseline": baseline, "data": data, "paths": paths}


def freeze(checkpoint_path, benchmark_path, paired_path, policy, manifest_path, data_dir=None,
           baseline_path=None, decision_path=None, repeat_seed=2027):
    data_dir = Path(data_dir or ROOT/"data").resolve()
    checkpoint_path, benchmark_path, paired_path, manifest_path = map(Path, (checkpoint_path, benchmark_path, paired_path, manifest_path))
    require(not manifest_path.exists(), "Frozen selection already exists")
    baseline_path = Path(baseline_path or ROOT/read(ROOT/"research/campaign.json")["baseline_benchmark"]).resolve()
    bundle = primary_bundle(checkpoint_path, benchmark_path, baseline_path, paired_path, policy, data_dir)
    frozen_at = utc_now().isoformat()
    decision = {"schema": "research_primary_decision_v1", "decided_at": frozen_at,
                "checkpoint_sha256": suite.sha256(checkpoint_path), "benchmark_sha256": suite.sha256(benchmark_path),
                "paired_sha256": suite.sha256(paired_path), "policy": policy, "seed": 2026,
                "planned_repeat_seed": repeat_seed}
    if decision_path:
        recorded = read(decision_path)
        require(all(recorded.get(k) == v for k, v in decision.items() if k != "decided_at"), "Primary decision metadata differs")
        require(stamp(recorded["decided_at"]) <= stamp(frozen_at), "Primary decision timestamp is in the future")
        decision = recorded
        bundle["paths"].append(Path(decision_path))
    require(type(repeat_seed) is int and repeat_seed != 2026, "A different independent repeat seed is required")
    value = {"schema": FREEZE_SCHEMA, "status": "frozen", "frozen_at": frozen_at, "competition": COMPETITION,
        "primary_decision": decision, "primary_decision_path": str(Path(decision_path).resolve()) if decision_path else None,
        "checkpoint_path": str(checkpoint_path.resolve()), "benchmark_path": str(benchmark_path.resolve()),
        "baseline_path": str(baseline_path), "paired_path": str(paired_path.resolve()), "data_dir": str(data_dir),
        "config": bundle["checkpoint"]["config"], "policy": policy, "parameter_count": bundle["benchmark"]["parameter_count"],
        "model_config": bundle["checkpoint"]["model_config"], "selected_epoch": bundle["checkpoint"].get("epoch"),
        "weight_variant": bundle["checkpoint"].get("variant"), "data_signature": bundle["benchmark"]["data_signature"],
        "source_git_commit": bundle["checkpoint"].get("provenance", {}).get("git_commit"),
        "evaluation_context": bundle["benchmark"]["evaluation_context"], "evidence_sha256": snapshots(bundle["paths"]),
        "confirmation_evaluated": False, "repeat_prediction_authorized": False, "full_refit_supported": False}
    exclusive_json(manifest_path, value)
    return value


def validate_frozen(manifest_path):
    document = read(manifest_path)
    require(document.get("schema") == FREEZE_SCHEMA and document.get("status") == "frozen" and
            document.get("competition") == COMPETITION and document.get("confirmation_evaluated") is False and
            document.get("repeat_prediction_authorized") is False and document.get("full_refit_supported") is False,
            "Valid immutable primary-only freeze is required")
    verify_snapshots(document.get("evidence_sha256"))
    bundle = primary_bundle(Path(document["checkpoint_path"]), Path(document["benchmark_path"]),
        Path(document["baseline_path"]), Path(document["paired_path"]), document["policy"], Path(document["data_dir"]))
    expected_paths = bundle["paths"]+[Path(document["primary_decision_path"])] if document.get("primary_decision_path") else bundle["paths"]
    require(document["evidence_sha256"] == snapshots(expected_paths), "Frozen evidence fingerprint coverage differs")
    decision = document["primary_decision"]
    expected = {"checkpoint_sha256": suite.sha256(document["checkpoint_path"]),
                "benchmark_sha256": suite.sha256(document["benchmark_path"]), "paired_sha256": suite.sha256(document["paired_path"]),
                "policy": document["policy"], "seed": 2026}
    require(all(decision.get(k) == v for k, v in expected.items()) and
            document.get("config") == bundle["checkpoint"]["config"] and
            document.get("parameter_count") == bundle["benchmark"]["parameter_count"] and
            document.get("evaluation_context") == bundle["benchmark"]["evaluation_context"] and
            document.get("model_config") == bundle["checkpoint"]["model_config"] and
            document.get("selected_epoch") == bundle["checkpoint"].get("epoch") and
            document.get("weight_variant") == bundle["checkpoint"].get("variant") and
            document.get("data_signature") == bundle["benchmark"]["data_signature"] and
            document.get("source_git_commit") == bundle["checkpoint"].get("provenance", {}).get("git_commit"), "Frozen primary metadata differs")
    require(stamp(decision["decided_at"]) <= stamp(document["frozen_at"]) <= utc_now(), "Frozen timestamps are invalid")
    if document.get("primary_decision_path"):
        require(read(document["primary_decision_path"]) == decision, "Original primary decision differs")
    return document, bundle


def validate_repeat(frozen, bundle, checkpoint_path, benchmark_path, receipt_path):
    require(Path(checkpoint_path).resolve() != Path(frozen["checkpoint_path"]).resolve(), "The primary model cannot serve as its own repeat")
    checkpoint, benchmark = audit_experiment(Path(checkpoint_path), Path(benchmark_path), bundle["data"], frozen["evaluation_context"])
    primary = bundle["checkpoint"]
    config, original = checkpoint["config"], primary["config"]
    require(config.get("seed") == frozen["primary_decision"]["planned_repeat_seed"] and config["seed"] != 2026 and
            {k:v for k,v in config.items() if k != "seed"} == {k:v for k,v in original.items() if k != "seed"},
            "Independent repeat must match the frozen recipe except for its predeclared seed")
    require(suite.sha256(checkpoint_path) != suite.sha256(frozen["checkpoint_path"]), "Repeat checkpoint is identical to primary")
    def trainer_marker(value):
        return {k:v for k,v in value.get("provenance", {}).get("research_training", {}).items() if k != "training_seed"}
    require(trainer_marker(checkpoint) == trainer_marker(primary), "Repeat training algorithm/precision/source differs")
    require(benchmark.get("baseline_sha256") == suite.sha256(frozen["baseline_path"]), "Repeat baseline differs")
    passed_policy(benchmark, bundle["baseline"], frozen["policy"])
    receipt = read(receipt_path)
    require(receipt.get("status") == "complete" and receipt.get("confirmation_evaluated") is False and
            receipt.get("submitted") is False and receipt.get("benchmark_sha256") == suite.sha256(benchmark_path),
            "A complete validation-only repeat supervisor receipt is required")
    decision = frozen["primary_decision"]
    require(stamp(decision["decided_at"]) < stamp(receipt["started_at"]) <= stamp(receipt["finished_at"]) <= utc_now() and
            stamp(frozen["frozen_at"]) < stamp(receipt["finished_at"]), "Primary selection/freeze occurred after repeat outcomes, or timestamps are invalid")
    if decision.get("planned_repeat_config_sha256"):
        require(receipt.get("config_sha256") == decision["planned_repeat_config_sha256"], "Repeat was not the predeclared configuration")
    return artifact_paths(Path(checkpoint_path), Path(benchmark_path), checkpoint)+[Path(receipt_path)]


def validate_test_files(data_dir):
    paths = [data_dir/name for name in ("sample_submission.csv", "test.csv", "official_test.npz", "official_test_source_manifest.json")]
    require(all(path.is_file() for path in paths), "This delivery path requires the audited official private test NPZ export")
    require(suite.sha256(paths[0]) == CSV_SHA256["sample_submission.csv"] and
            suite.sha256(paths[1]) == CSV_SHA256["test.csv"], "Official test/sample CSV fingerprints differ")
    source = read(paths[3])
    require(source.get("competition") == COMPETITION and source.get("source") == "official_kaggle_competition"
            and source.get("no_external_data") is True and source.get("hidden_labels_loaded") is False
            and source.get("test_csv_sha256") == CSV_SHA256["test.csv"]
            and source.get("sample_submission_csv_sha256") == CSV_SHA256["sample_submission.csv"]
            and source.get("test_labels_in_archive") is False and source.get("archive_sha256") == suite.sha256(paths[2]),
            "Official unlabeled test archive/source differs")
    return paths


def validate_delivery(manifest_path, require_prediction=True):
    """Revalidate private delivery immediately before an external uploader acts."""
    manifest_path = Path(manifest_path)
    delivery = read(manifest_path)
    if delivery.get("schema") == FREEZE_SCHEMA:
        require(not require_prediction, "Independent repeat and completed prediction delivery are required")
        frozen_path = manifest_path
    else:
        require(delivery.get("schema") == PREDICTION_SCHEMA and delivery.get("status") == "complete", "Unsupported delivery manifest")
        frozen_path = Path(delivery["frozen_manifest_path"])
        require(suite.sha256(frozen_path) == delivery.get("frozen_manifest_sha256"), "Frozen manifest changed")
    frozen, bundle = validate_frozen(frozen_path)
    result = {"run": Path(frozen["checkpoint_path"]).parent.name, "competition": COMPETITION,
              "checkpoint_sha256": suite.sha256(frozen["checkpoint_path"]), "parameter_count": frozen["parameter_count"],
              "policy": frozen["policy"], "seed": 2026, "training_examples": 40000,
              "manifest_sha256": suite.sha256(manifest_path), "frozen_manifest_sha256": suite.sha256(frozen_path)}
    if delivery.get("schema") == FREEZE_SCHEMA:
        return result
    repeat = delivery["repeat"]
    repeat_paths = validate_repeat(frozen, bundle, Path(repeat["checkpoint_path"]), Path(repeat["benchmark_path"]), Path(repeat["receipt_path"]))
    verify_snapshots(delivery.get("repeat_evidence_sha256"))
    require(delivery["repeat_evidence_sha256"] == snapshots(repeat_paths), "Repeat evidence fingerprint coverage differs")
    require(all(delivery.get(k) == v for k,v in result.items() if k not in {"manifest_sha256", "frozen_manifest_sha256"}) and
            delivery.get("confirmation_evaluated") is False and delivery.get("hidden_test_labels_loaded") is False and
            delivery.get("generator_source_sha256") == suite.sha256(Path(__file__)) and
            delivery.get("evaluation_context") == frozen["evaluation_context"], "Prediction source or primary selection differs")
    data_dir = Path(frozen["data_dir"])
    test_paths = validate_test_files(data_dir)
    verify_snapshots(delivery.get("test_files_sha256"))
    require(delivery["test_files_sha256"] == snapshots([*test_paths, Path(delivery["test_manifest_path"])]),
            "Official test fingerprint coverage differs")
    test_manifest = read(delivery["test_manifest_path"])
    require(test_manifest.get("competition") == COMPETITION and test_manifest.get("count") == 2000 and
            test_manifest.get("labels_loaded") is False and
            test_manifest.get("test_csv_sha256") == CSV_SHA256["test.csv"] and
            test_manifest.get("sample_submission_sha256") == CSV_SHA256["sample_submission.csv"] and
            test_manifest.get("source_provenance", {}).get("archive_sha256") == suite.sha256(data_dir/"official_test.npz") and
            test_manifest.get("source_provenance", {}).get("source_manifest_sha256") == suite.sha256(data_dir/"official_test_source_manifest.json"),
            "Official test manifest is invalid")
    digest = validate_csv(Path(delivery["csv_path"]), data_dir/"sample_submission.csv")
    require(digest == delivery.get("submission_sha256") and delivery.get("rows") == 2000 and
            delivery.get("columns") == ["id", "label"] and delivery.get("ids_match_official_order") is True,
            "Prediction CSV metadata differs")
    result.update(csv_path=str(Path(delivery["csv_path"]).resolve()), submission_sha256=digest)
    return result


def predict(manifest_path, repeat_checkpoint, repeat_benchmark, repeat_receipt, output_dir, device="cuda"):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    frozen, bundle = validate_frozen(manifest_path)
    repeat_paths = validate_repeat(frozen, bundle, Path(repeat_checkpoint), Path(repeat_benchmark), Path(repeat_receipt))
    repeat_hashes = snapshots(repeat_paths)
    csv_path, delivery_path = output_dir/"submission.csv", output_dir/"delivery.json"
    require(not csv_path.exists() and not delivery_path.exists(), "Existing prediction evidence is immutable")
    device = torch.device(device)
    context = {"device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)),
               "torch_version": torch.__version__, "training_core_sha256": code_provenance()["source_sha256"]}
    require(context == frozen["evaluation_context"], "Prediction backend/precision must match the paired benchmark")
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    data_dir = Path(frozen["data_dir"])
    test_paths = validate_test_files(data_dir)
    seed_all(2026)
    model = build_model(bundle["checkpoint"]["model_config"])
    model.load_state_dict(bundle["checkpoint"]["model"], strict=True)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    ids, images = load_kaggle_test(data_dir)
    test_manifest_path = data_dir/"kaggle_test_manifest.json"
    test_manifest = read(test_manifest_path)
    require(len(ids) == 2000 and len(set(ids)) == 2000 and test_manifest.get("count") == 2000
            and test_manifest.get("labels_loaded") is False, "Official test IDs/manifest are invalid")
    test_hashes = snapshots([*test_paths, test_manifest_path])
    predictions = []
    batch_size = bundle["benchmark"]["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.as_tensor(images[start:start+batch_size], device=device)
            with torch.autocast(device.type, dtype=suite.benchmark_dtype(device), enabled=device.type == "cuda"):
                logits = suite.infer(model, batch, frozen["policy"])
            require(torch.isfinite(logits).all().item(), "Nonfinite hidden-image inference logits")
            predictions.extend(logits.argmax(1).cpu().tolist())
    verify_snapshots(frozen["evidence_sha256"])
    verify_snapshots(repeat_hashes)
    verify_snapshots(test_hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, predictions))
    digest = validate_csv(csv_path, data_dir/"sample_submission.csv")
    delivery = {"schema": PREDICTION_SCHEMA, "status": "complete", "created_at": utc_now().isoformat(),
        "run": Path(frozen["checkpoint_path"]).parent.name, "competition": COMPETITION,
        "frozen_manifest_path": str(manifest_path.resolve()), "frozen_manifest_sha256": suite.sha256(manifest_path),
        "checkpoint_sha256": suite.sha256(frozen["checkpoint_path"]), "parameter_count": frozen["parameter_count"],
        "policy": frozen["policy"], "seed": 2026, "training_examples": 40000,
        "repeat": {"checkpoint_path": str(Path(repeat_checkpoint).resolve()), "benchmark_path": str(Path(repeat_benchmark).resolve()),
                   "receipt_path": str(Path(repeat_receipt).resolve())}, "repeat_evidence_sha256": repeat_hashes,
        "csv_path": str(csv_path.resolve()), "submission_sha256": digest, "rows": 2000, "columns": ["id", "label"],
        "ids_match_official_order": True, "test_manifest_path": str(test_manifest_path), "test_files_sha256": test_hashes,
        "generator_source_sha256": suite.sha256(Path(__file__)), "confirmation_evaluated": False,
        "hidden_test_labels_loaded": False, "evaluation_context": context}
    exclusive_json(delivery_path, delivery)
    return validate_delivery(delivery_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "predict", "validate"):
        action.add_argument("--"+name, action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    for name in ("checkpoint", "benchmark", "paired", "decision", "baseline-benchmark", "data-dir",
                 "repeat-checkpoint", "repeat-benchmark", "repeat-receipt", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--policy", choices=suite.POLICIES)
    parser.add_argument("--repeat-seed", type=int, default=2027)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.freeze:
        if not all((args.checkpoint, args.benchmark, args.paired, args.policy)):
            parser.error("--freeze requires --checkpoint --benchmark --paired --policy")
        freeze(args.checkpoint, args.benchmark, args.paired, args.policy, args.manifest, args.data_dir,
               args.baseline_benchmark, args.decision, args.repeat_seed)
        result = validate_delivery(args.manifest, require_prediction=False)
    elif args.predict:
        if not all((args.repeat_checkpoint, args.repeat_benchmark, args.repeat_receipt, args.output_dir)):
            parser.error("--predict requires explicit repeat checkpoint/benchmark/receipt and --output-dir")
        result = predict(args.manifest, args.repeat_checkpoint, args.repeat_benchmark, args.repeat_receipt,
                         args.output_dir, args.device)
    else:
        result = validate_delivery(args.manifest)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
