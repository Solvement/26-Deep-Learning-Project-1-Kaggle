"""Public report projection of measured research and canonical scored receipts.

This standard-library reader loads only JSON aggregates. It does not train,
predict, upload, read images/checkpoints, or treat a prepared CSV as a score.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "research_report_followup_v1"
BENCHMARK_SCHEMA = "official_development_stress_v3"
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
REPOSITORY = "https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/tree/feat/five-day-autoresearch"
POLICY = "center80_shift20"
POLICIES = {"plain", "flip", POLICY}
CASES = {"clean", "noise_004", "boxblur3", "contrast065", "brightness080", "resize24"}
THRESHOLDS = {"clean_improvement": .001, "stress_noninferiority": -.002,
              "stress_improvement": .005, "clean_noninferiority": -.002}
HASH = re.compile(r"[0-9a-f]{64}\Z")
RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{0,70}\Z")
PENDING = {"upload_attempt_started", "submitted", "recovered", "unknown", "scoring_pending", "scoring_error"}
ROWS = (
    ("wrn34_160", "WRN34-3 / 160", "wrn34_3_primary", "wrn34_3_paired", "wrn34_3_repeat", 160, "bf16"),
    ("sam", "SAM / WRN22-4", "SAM_local_primary", "SAM_paired", "SAM_local_repeat", 160, "bf16"),
    ("resizemix", "ResizeMix / WRN22-4", "resizemix_local_primary", "resizemix_local_paired", "resizemix_local_repeat", 160, "fp16"),
    ("wrn34_300", "WRN34-3 / 300", "WRN34_300_local_primary", "WRN34_300_local_paired", "WRN34_300_local_repeat", 300, "fp16"),
    ("swa40", "SWA / WRN34-3", "SWA_fixed40_primary", None, None, 200, "bf16"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value, label):
    require(isinstance(value, str) and HASH.fullmatch(value), f"Invalid {label} fingerprint")
    return value


def _number(value, label, low=0, high=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= low and
            (high is None or value <= high), f"Invalid {label}")
    return float(value)


def _integer(value, label, low=1, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), f"Invalid {label}")
    return value


def _stamp(value):
    require(isinstance(value, str), "A timezone-aware evidence timestamp is required")
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid evidence timestamp") from error
    require(stamp.tzinfo is not None, "Evidence timestamps must have timezones")
    return stamp


def _close(left, right):
    return type(left) in (int, float) and math.isfinite(left) and math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def _gate(report, reference):
    clean = report["clean_accuracy"]-reference["clean_accuracy"]
    stress = report["stress_accuracy"]-reference["stress_accuracy"]
    # Match the frozen benchmark's numerical tolerance at its declared edges.
    passed = ((clean >= .001-1e-12 and stress >= -.002-1e-12) or
              (stress >= .005-1e-12 and clean >= -.002-1e-12))
    return {"passed": passed, "clean_delta": clean, "stress_delta": stress, "thresholds": THRESHOLDS}


def _check_gate(saved, expected):
    require(isinstance(saved, dict) and saved.get("passed") is expected["passed"] and
            saved.get("thresholds") == THRESHOLDS and
            all(_close(saved.get(key), expected[key]) for key in ("clean_delta", "stress_delta")),
            "Saved promotion does not match the fixed gate")


def _benchmark(document, baseline, pixels):
    require(isinstance(document, dict) and document.get("schema") == BENCHMARK_SCHEMA and
            document.get("status") == "complete" and document.get("examples") == 5000 and
            document.get("training_examples") == 40000, "A complete independent 40k/5k v3 benchmark is required")
    require(document.get("evaluation_context") == baseline["evaluation_context"] and
            document.get("data_signature") == baseline["data_signature"], "Benchmark context or official split differs")
    _integer(document.get("parameter_count"), "parameter count", high=5000000)
    _hash(document.get("checkpoint_sha256"), "checkpoint")
    if "baseline_sha256" in document:
        require(document["baseline_sha256"] == baseline["benchmark_sha256"], "Benchmark baseline fingerprint differs")
    require(set(document.get("policies", {})) == POLICIES, "Every fixed inference policy is required")
    for policy, report in document["policies"].items():
        require(set(report.get("cases", {})) == CASES, "Every development stress case is required")
        for name, case in report["cases"].items():
            require(case.get("examples") == 5000, "A benchmark case is partial")
            _number(case.get("accuracy"), "case accuracy", high=1)
            _number(case.get("loss"), "case NLL")
            digest = _hash(case.get("input_pixel_sha256"), "development pixels")
            if pixels is not None:
                require(digest == pixels[name], "Development pixels differ between comparisons")
        clean = _number(report.get("clean_accuracy"), "clean accuracy", high=1)
        nll = _number(report.get("clean_loss"), "clean NLL")
        stress = _number(report.get("stress_accuracy"), "stress accuracy", high=1)
        require(_close(clean, report["cases"]["clean"]["accuracy"]) and
                _close(nll, report["cases"]["clean"]["loss"]) and
                _close(stress, sum(report["cases"][case]["accuracy"] for case in CASES-{"clean"})/5),
                "Benchmark aggregate differs from its six cases")
        _check_gate(document.get("promotion", {}).get(policy), _gate(report, baseline["flip"]))
    return document["policies"][POLICY]


def _pair(document, benchmark, baseline, benchmark_hash):
    require(document.get("schema") in {"exploratory_paired_image_bootstrap_v1", "exploratory_paired_image_bootstrap_v2"} and
            document.get("status") == "complete" and document.get("candidate_policy") == POLICY and
            document.get("baseline_policy") == "flip" and
            document.get("candidate_checkpoint_sha256") == benchmark["checkpoint_sha256"] and
            document.get("baseline_checkpoint_sha256") == baseline["checkpoint_sha256"] and
            document.get("candidate_benchmark_sha256") == benchmark_hash and
            document.get("baseline_benchmark_sha256") == baseline["benchmark_sha256"] and
            document.get("data_signature") == baseline["data_signature"] and
            document.get("evaluation_context") == baseline["evaluation_context"], "Paired evidence is not bound to this comparison")
    boot = document.get("bootstrap", {})
    require(boot.get("original_images") == 5000 and boot.get("replicates") == 10000 and
            boot.get("sampling_unit") == "original_image_with_all_five_stress_views" and
            boot.get("confidence_level") == .95, "Original-image bootstrap protocol differs")
    result = {}
    selected = benchmark["policies"][POLICY]
    for name, total in (("clean", 5000), ("stress", 25000)):
        row = boot.get(name, {})
        counts = [_integer(row.get(key), "paired correctness count", low=0) for key in
                  ("candidate_only_correct", "baseline_only_correct", "both_correct", "both_wrong")]
        delta = selected[name+"_accuracy"]-baseline["flip"][name+"_accuracy"]
        require(sum(counts) == total and _close(row.get("delta_accuracy"), delta) and
                _close((counts[0]-counts[1])/total, delta) and
                _close((counts[0]+counts[2])/total, selected[name+"_accuracy"]) and
                _close((counts[1]+counts[2])/total, baseline["flip"][name+"_accuracy"]), "Paired counts differ from benchmark points")
        interval = row.get("percentile_ci95")
        require(isinstance(interval, list) and len(interval) == 2, "Paired interval is missing")
        low, high = [_number(value, "paired delta interval", low=-1, high=1) for value in interval]
        require(low <= delta <= high, "Paired interval does not bracket its point")
        result[name] = [low, high]
    return result


def _legacy(path):
    document = _json(path)
    records = document.get("official_submissions")
    require(isinstance(records, list) and records, "Original scored submissions are missing")
    primary = document.get("kaggle_submission", {})
    require(primary.get("run") == "final_full", "The original primary study identity changed")
    for record in [*records, primary]:
        require(record.get("status") == "complete" and record.get("competition") == COMPETITION and
                record.get("team") == "Saber", "Original receipt is incomplete or belongs elsewhere")
        _number(record.get("public_score"), "original public score", high=1)
        _integer(record.get("submission_ref"), "original submission reference")
        _hash(record.get("submission_sha256"), "original submission")
    matched = [r for r in records if r["submission_ref"] == primary["submission_ref"]]
    require(len(matched) == 1 and all(matched[0].get(k) == primary.get(k) for k in
            ("run", "public_score", "submission_sha256")), "Original primary receipt is inconsistent")
    return max(r["public_score"] for r in records), primary["public_score"], records


def _receipt(path):
    record = _json(path)
    require(record.get("schema") is None and record.get("composition") in (None, False),
            "A benchmark, intent or composed proof is not a scored receipt")
    require(record.get("competition") == COMPETITION and record.get("team") == "Saber", "Research receipt belongs to another competition or team")
    require(isinstance(record.get("run"), str) and RUN.fullmatch(record["run"]) and not record["run"].startswith("stability"), "Invalid research run identity")
    ref = _integer(record.get("submission_ref"), "submission reference")
    for key in ("submission_sha256", "checkpoint_sha256", "manifest_sha256", "frozen_manifest_sha256"):
        _hash(record.get(key), key)
    _integer(record.get("parameter_count"), "receipt parameter count", high=5000000)
    require(type(record.get("seed")) is int and record["seed"] == 2026 and record.get("policy") in POLICIES and
            type(record.get("training_examples")) is int and record["training_examples"] in (40000, 50000), "Research receipt recipe metadata differs")
    require(_stamp(record.get("submitted_at")) <= _stamp(record.get("checked_at")), "Receipt timestamps are out of order")
    if "delivery_schema" in record:
        require(record["delivery_schema"] in {"research_prediction_delivery_v1", "research_prediction_delivery_v2", "research_refit_delivery_v1"},
                "Unsupported research delivery schema")
    if record["training_examples"] == 50000:
        require(type(record.get("validation_examples")) is int and record["validation_examples"] == 0 and
                record.get("independent_development_evaluated") is False and "development_accuracy" in record and
                record["development_accuracy"] is None, "Full refit must not claim independent development")
        _hash(record.get("parent_delivery_sha256"), "qualified refit parent")
    elif "validation_examples" in record:
        require(record["validation_examples"] in (None, 5000), "Held-out receipt validation count differs")
    status = record.get("status")
    require(status == "complete" or status in PENDING, "Unknown research submission state")
    public = _number(record.get("public_score"), "official public score", high=1) if status == "complete" else None
    private = record.get("private_score")
    if private is not None:
        private = _number(private, "official private score", high=1)
    immutable = tuple(record.get(key) for key in ("run", "submission_sha256", "checkpoint_sha256", "parameter_count", "policy",
        "seed", "training_examples", "manifest_sha256", "frozen_manifest_sha256", "submitted_at", "validation_examples",
        "independent_development_evaluated", "development_accuracy", "parent_delivery_sha256"))
    projected = {key: record[key] for key in ("run", "submission_sha256", "checkpoint_sha256", "policy", "training_examples")}
    projected.update(submission_ref=ref, public_score=public, private_score=private, receipt_sha256=_sha(path),
                     validation_examples=record.get("validation_examples"))
    return record, immutable, projected


def _receipts(directory, legacy):
    directory = Path(directory)
    require(directory.name not in {"submission_intents", "submission_receipt_history"}, "Use the canonical research receipts directory")
    if not directory.exists():
        return [], 0, {}
    require(directory.is_dir(), "Use the canonical research receipts directory")
    old_refs = {r["submission_ref"] for r in legacy}
    old_csv = {r["submission_sha256"] for r in legacy}
    seen, by_ref, by_csv = {}, {}, {}
    for path in sorted(directory.glob("*.json")):
        require(not path.is_symlink(), "Receipt symlinks are unsupported")
        raw, immutable, public = _receipt(path)
        ref, digest = public["submission_ref"], public["submission_sha256"]
        require(ref not in old_refs and digest not in old_csv, "An original submission cannot be relabeled as new research")
        require(by_ref.get(ref, digest) == digest and by_csv.get(digest, ref) == ref, "Submission reference/CSV fingerprints conflict")
        by_ref[ref], by_csv[digest] = digest, ref
        key = (ref, digest)
        if key in seen:
            previous = seen[key]
            require(previous[1] == immutable, "Duplicate receipt model metadata conflicts")
            if raw["status"] == previous[0]["status"] == "complete":
                require((public["public_score"], public["private_score"]) ==
                        (previous[2]["public_score"], previous[2]["private_score"]), "Duplicate receipt scores conflict")
            if previous[0]["status"] == "complete" and raw["status"] != "complete":
                continue
            if previous[0]["status"] == raw["status"] and _stamp(raw["checked_at"]) < _stamp(previous[0]["checked_at"]):
                continue
        seen[key] = (raw, immutable, public)
    scored = [value[2] for value in seen.values() if value[0]["status"] == "complete"]
    parameters = {value[2]["receipt_sha256"]: value[0]["parameter_count"] for value in seen.values()
                  if value[0]["status"] == "complete"}
    return sorted(scored, key=lambda r: r["submission_ref"]), len(seen)-len(scored), parameters


def load_followup(summary_path, receipts_dir):
    """Return the audited public snapshot consumed by report and notebook code."""
    summary_path = Path(summary_path)
    summary = _json(summary_path)
    _stamp(summary.get("recorded_at"))
    baseline = summary.get("baseline", {})
    require(baseline.get("schema") == BENCHMARK_SCHEMA and baseline.get("status") == "complete" and
            baseline.get("examples") == 5000 and baseline.get("training_examples") == 40000 and
            baseline.get("evaluation_context", {}).get("device_type") == "cuda" and
            baseline["evaluation_context"].get("amp_dtype") == "torch.bfloat16", "Complete native-BF16 v3 baseline is required")
    for key in ("checkpoint_sha256", "benchmark_sha256"):
        _hash(baseline.get(key), "baseline "+key)
    require(set(baseline.get("data_signature", {})) == {"archive_sha256", "split_sha256"}, "Official baseline data signature is missing")
    for value in baseline["data_signature"].values():
        _hash(value, "official data")
    for key in ("clean_accuracy", "stress_accuracy"):
        _number(baseline.get("flip", {}).get(key), "baseline "+key, high=1)
    _number(baseline["flip"].get("clean_loss"), "baseline NLL")
    old_best, old_primary, legacy = _legacy(summary_path.with_name("results.json"))
    official, unscored, receipt_parameters = _receipts(receipts_dir, legacy)
    rows, pixels = [], None
    for identity, label, key, pair_key, repeat_key, epochs, precision in ROWS:
        if key not in summary:
            continue
        document = summary[key]
        report = _benchmark(document, baseline, pixels)
        if pixels is None:
            pixels = {name: report["cases"][name]["input_pixel_sha256"] for name in CASES}
        config = document.get("config")
        if config is not None:
            require(config.get("epochs") == epochs and config.get("seed") == 2026 and config.get("train_all") is False,
                    "Primary configuration does not match the declared training protocol")
        elif identity == "wrn34_300":
            training = summary.get("WRN34_300_T4_primary", {})
            require(training.get("epochs_completed") == 300 and training.get("checkpoint_sha256") == document["checkpoint_sha256"] and
                    training.get("successful_updates", -1)+training.get("skipped_updates", -1) == 300*313,
                    "WRN300 total training length is not documented")
        epoch = _integer(document.get("checkpoint_epoch"), "selected checkpoint epoch", high=epochs)
        variant = document.get("checkpoint_variant")
        require(variant in {"model", "ema", "swa"}, "Unknown checkpoint variant")
        gate = _gate(report, baseline["flip"])
        eligible = gate["passed"]
        if identity == "swa40":
            parent = summary["wrn34_3_primary"]
            require(document.get("fixed_primary_policy") == POLICY and document.get("parent_checkpoint_sha256") == parent["checkpoint_sha256"] and
                    document.get("parent_benchmark_sha256") == parent["benchmark_sha256"], "Fixed SWA parent reference differs")
            for policy in POLICIES:
                expected = _gate(document["policies"][policy], parent["policies"][policy])
                _check_gate(document.get("promotion_vs_parent", {}).get(policy), expected)
            eligible = eligible and document["promotion_vs_parent"][POLICY]["passed"]
            require(document.get("candidate_eligible") is eligible and epoch == 200 and variant == "swa", "Fixed SWA eligibility differs")
        paired = None
        if pair_key and pair_key in summary:
            evidence = summary[pair_key]
            digest = document.get("benchmark_sha256", evidence.get("candidate_benchmark_sha256"))
            paired = _pair(evidence, document, baseline, _hash(digest, "candidate benchmark"))
        clean_path = gate["clean_delta"] >= .001-1e-12 and gate["stress_delta"] >= -.002-1e-12
        pair_passed = paired is not None and (clean_path or paired["stress"][0] > 0)
        pending_delivery = "not_promoted"
        if eligible:
            pending_delivery = "awaiting_pair" if paired is None else "awaiting_repeat" if pair_passed else "not_promoted"
        row = {"id": identity, "label": label, "parameters": document["parameter_count"], "epochs": epochs,
            "precision": precision, "policy": POLICY, "checkpoint_epoch": epoch, "checkpoint_variant": variant,
            "clean_accuracy": report["clean_accuracy"], "clean_nll": report["clean_loss"], "stress_accuracy": report["stress_accuracy"],
            "gate_passed": eligible, "clean_delta": gate["clean_delta"], "stress_delta": gate["stress_delta"],
            "clean_ci95": paired["clean"] if paired else None, "stress_ci95": paired["stress"] if paired else None,
            "repeat_status": "not_verified" if eligible else "not_promoted", "repeat_clean_accuracy": None,
            "repeat_stress_accuracy": None, "repeat_epochs_completed": None, "repeat_planned_epochs": epochs,
            "delivery_status": pending_delivery, "checkpoint_sha256": document["checkpoint_sha256"]}
        if repeat_key and repeat_key in summary:
            repeat = summary[repeat_key]
            repeated = _benchmark(repeat, baseline, pixels)
            repeat_config = repeat.get("config", {})
            require(repeat.get("checkpoint_sha256") != document["checkpoint_sha256"] and repeat.get("parameter_count") == document["parameter_count"] and
                    repeat_config.get("seed") == 2027 and repeat_config.get("epochs") == epochs,
                    "An independent seed2027 repeat is required")
            require(config is not None and {k:v for k,v in config.items() if k != "seed"} ==
                    {k:v for k,v in repeat_config.items() if k != "seed"}, "Repeat recipe differs from the primary")
            repeat_passed = _gate(repeated, baseline["flip"])["passed"]
            row.update(repeat_status="complete_passed" if repeat_passed else "complete_failed",
                       repeat_clean_accuracy=repeated["clean_accuracy"], repeat_stress_accuracy=repeated["stress_accuracy"],
                       repeat_epochs_completed=epochs,
                       delivery_status=("qualified_unscored" if pair_passed else pending_delivery)
                       if repeat_passed and eligible else "not_promoted")
        elif identity == "resizemix" and "resizemix_repeat_progress" in summary:
            progress = summary["resizemix_repeat_progress"]
            done = _integer(progress.get("epochs_completed"), "repeat progress", high=160)
            require(progress.get("planned_epochs") == 160 and progress.get("training_seed") == 2027 and
                    progress.get("configuration_and_sources_verified") is True and progress.get("all_recorded_metrics_finite") is True and
                    type(progress.get("successful_updates")) is int and type(progress.get("skipped_updates")) is int and
                    progress["successful_updates"] >= 0 and progress["skipped_updates"] >= 0 and
                    progress["successful_updates"]+progress["skipped_updates"] == done*313,
                    "Partial repeat progress is inconsistent")
            row.update(repeat_status="in_progress", repeat_epochs_completed=done)
        if any(r["checkpoint_sha256"] == row["checkpoint_sha256"] and r["policy"] == row["policy"] and
               r["training_examples"] == 40000 and receipt_parameters[r["receipt_sha256"]] == row["parameters"]
               for r in official):
            row["delivery_status"] = "scored"
        rows.append(row)
    require(rows, "No complete local v3 research benchmark is available")
    new_best = max((r["public_score"] for r in official), default=None)
    engineering = []
    se = summary.get("SE_model_only")
    if se is not None:
        require(se.get("total_parameters") == se.get("trainable_parameters") == 4332654 and
                se.get("standalone_CPU_tests", 0) > 0, "SE model-only parameter evidence differs")
        engineering.append("SE-WRN22-4/r16 has 4,332,654 total/trainable parameters in CPU construction; no SE accuracy is inferred from model tests.")
    if any(row["id"] == "swa40" for row in rows):
        engineering.append("The fixed averaged SWA candidate is reported with its measured gate outcome; raw-tail control is not substituted for it.")
    return {"schema": SCHEMA, "recorded_at": summary["recorded_at"], "repository": REPOSITORY,
        "source_summary_sha256": _sha(summary_path), "baseline": {
            "clean_accuracy": baseline["flip"]["clean_accuracy"], "clean_nll": baseline["flip"]["clean_loss"],
            "stress_accuracy": baseline["flip"]["stress_accuracy"], "policy": "flip", "examples": 5000, "training_examples": 40000},
        "rows": rows, "official_submissions": official, "unscored_receipts_count": unscored,
        "original_best_public": old_best, "original_primary_public": old_primary, "best_new_public": new_best,
        "best_observed_public": max(old_best, new_best) if new_best is not None else old_best,
        "limitations": ["Research uses the fixed 5,000 development images; the original one-time confirmation belongs only to the original regularized 40k model.",
            "Stress is the mean of five synthetic views, not hidden-test accuracy. Paired intervals are conditional on fitted models and reused/postselected development.",
            "All comparison rows use the same local v3 native-BF16 inference context. Training recipes and backends differ; epoch-count causality is not established.",
            "Only complete canonical scored receipts establish new Kaggle scores; a benchmark, prediction CSV, upload intent or partial repeat does not.",
            "Full-50k refits have no independent development score; original primary public accuracy and overall observed best remain separate."],
        "engineering": engineering}
