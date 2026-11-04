"""Compare recorded complete local v3 benchmarks with the frozen incumbent.

Standard-library only: this reads aggregate JSON, never tensors or images, and
does not create a promotion, repeat, prediction or submission authorization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "recorded_incumbent_benchmark_comparison_v1"
BENCHMARK_SCHEMA = "official_development_stress_v3"
CASES = ("clean", "noise_004", "boxblur3", "contrast065", "brightness080", "resize24")
POLICIES = ("plain", "flip", "center80_shift20")
POLICY = "center80_shift20"
INCUMBENT_PATH = ROOT / "results/research/wrn34_3_300_s2026_local_v3/benchmark.json"
INCUMBENT_BENCHMARK_SHA = "e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b"
INCUMBENT_CHECKPOINT_SHA = "53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
BENCHMARK_SOURCE_SHA = "d30acccd66a18f8de198824a57b565cb8e16cd7044fa1cec6608007c80d37f16"
CORE = {
    "train.py": "4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py": "79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py": "4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py": "75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
}
DATA = {
    "archive_sha256": "d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
    "split_sha256": "73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e",
}
CONTEXT = {"device_type": "cuda", "amp_dtype": "torch.bfloat16",
           "torch_version": "2.12.0+cu130", "training_core_sha256": CORE}
PLAN = ROOT / "research/incumbent-comparison-plan.md"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_digest(value) -> str:
    """Effective parsed configuration hash, explicitly not a raw-file hash."""
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value, *, minimum=0.0, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def read_document(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= 4 * 1024 * 1024, "Benchmark JSON exceeds aggregate-only size limit")
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON constant")))
    require(type(document) is dict, "Benchmark must be a JSON object")
    return document, digest(raw)


def validate_document(document):
    """Check recorded completeness/context; do not claim a tensor audit."""
    require(type(document) is dict, "Benchmark must be an object")
    fixed = {"schema": BENCHMARK_SCHEMA, "status": "complete", "bn_recalibrated": False,
             "confirmation_evaluated": False, "hidden_images_loaded": False,
             "external_data_loaded": False, "data_signature": DATA,
             "evaluation_context": CONTEXT, "benchmark_source_sha256": BENCHMARK_SOURCE_SHA,
             "stress_cases": list(CASES)}
    require(all(document.get(key) == value and type(document.get(key)) is type(value)
                for key, value in fixed.items()), "Complete fixed local native v3 context/source/data required")
    require(type(document.get("examples")) is int and document["examples"] == 5000
            and type(document.get("training_examples")) is int and document["training_examples"] == 40000,
            "Exactly 40,000 train and 5,000 independent development examples required")
    require(type(document.get("parameter_count")) is int and 0 < document["parameter_count"] <= 5_000_000,
            "Parameter budget must be explicitly recorded")
    require(type(document.get("batch_size")) is int and document["batch_size"] == 128,
            "Fixed native v3 batch size 128 required")
    config = document.get("config", {})
    require(type(config) is dict and config.get("train_all") is False
            and type(config.get("seed")) is int and config["seed"] == 2026
            and type(config.get("split_seed")) is int and config["split_seed"] == 2026
            and type(config.get("epochs")) is int and config["epochs"] > 0,
            "Primary seed2026 held-out configuration required; repeats/full50k are excluded")
    json_digest(config)  # Reject NaN/Inf anywhere in an effective configuration.
    require(document.get("model_config") == {key: config.get(key) for key in
            ("depth", "widen_factor", "dropout", "num_classes")}, "Constructor/config mismatch")
    require(type(document.get("checkpoint_epoch")) is int
            and 1 <= document["checkpoint_epoch"] <= config["epochs"]
            and document.get("checkpoint_variant") in ("model", "ema"), "Selected checkpoint metadata required")
    for key in ("checkpoint_sha256", "training_metrics_sha256", "training_history_sha256"):
        require(is_sha(document.get(key)), "Recorded checkpoint/history/metrics SHA-256 required")
    source = document.get("source", {})
    require(type(source) is dict and source.get("source_sha256") == CORE
            and isinstance(source.get("git_commit"), str) and len(source["git_commit"]) == 40
            and all(c in "0123456789abcdef" for c in source["git_commit"]), "Frozen core/source revision required")
    require(number(document.get("elapsed_seconds")), "Benchmark elapsed time required")
    reports = document.get("policies", {})
    require(type(reports) is dict and set(reports) == set(POLICIES), "All three complete policies required")
    for report in reports.values():
        require(type(report) is dict and type(report.get("cases")) is dict
                and set(report["cases"]) == set(CASES), "All six cases required for every policy")
        cases = report["cases"]
        for case in cases.values():
            require(type(case) is dict and type(case.get("examples")) is int and case["examples"] == 5000
                    and number(case.get("accuracy"), maximum=1.0) and number(case.get("loss"))
                    and is_sha(case.get("input_pixel_sha256")), "Incomplete case count/accuracy/NLL/pixel hash")
        require(number(report.get("clean_accuracy"), maximum=1.0)
                and number(report.get("clean_loss")) and number(report.get("stress_accuracy"), maximum=1.0)
                and report["clean_accuracy"] == cases["clean"]["accuracy"]
                and report["clean_loss"] == cases["clean"]["loss"]
                and math.isclose(report["stress_accuracy"], sum(cases[c]["accuracy"] for c in CASES[1:])/5,
                                 rel_tol=0, abs_tol=1e-12), "Case/summary accuracy or NLL mismatch")


def evidence_summary(document):
    selected = document["policies"][POLICY]
    summary = {key: document[key] for key in ("checkpoint_sha256", "checkpoint_epoch", "checkpoint_variant",
               "training_metrics_sha256", "training_history_sha256", "parameter_count", "config", "model_config")}
    summary.update(effective_config_sha256=json_digest(document["config"]),
                   core_source_sha256=document["source"]["source_sha256"],
                   benchmark_source_git_commit=document["source"]["git_commit"],
                   benchmark_source_sha256=document["benchmark_source_sha256"],
                   benchmark_elapsed_seconds=document["elapsed_seconds"],
                   clean_accuracy=selected["clean_accuracy"], clean_nll=selected["clean_loss"],
                   mean_stress_accuracy=selected["stress_accuracy"])
    integrity = document.get("research_training_integrity")
    if integrity is not None:
        require(type(integrity) is dict and type(integrity.get("training")) is dict,
                "Research integrity must identify the recorded training method")
        marker = integrity["training"]
        require(type(marker.get("source_sha256")) is dict and marker["source_sha256"]
                and all(isinstance(k, str) and is_sha(v) for k, v in marker["source_sha256"].items())
                and isinstance(marker.get("schema"), str), "Research training source/schema hashes required")
        summary["recorded_method_source"] = {"schema": marker["schema"], "source_sha256": marker["source_sha256"]}
        summary["recorded_method_integrity_sha256"] = json_digest(integrity)
    return summary


def compare_documents(candidate, incumbent):
    for document in (candidate, incumbent):
        validate_document(document)
    require(incumbent["checkpoint_sha256"] == INCUMBENT_CHECKPOINT_SHA,
            "The frozen WRN34-3/300 primary incumbent is required")
    for policy in POLICIES:
        for case in CASES:
            require(candidate["policies"][policy]["cases"][case]["input_pixel_sha256"]
                    == incumbent["policies"][policy]["cases"][case]["input_pixel_sha256"],
                    "Candidate/incumbent development pixels differ")
    a, b = candidate["policies"][POLICY], incumbent["policies"][POLICY]
    cases = {}
    for case in CASES:
        x, y = a["cases"][case], b["cases"][case]
        cases[case] = {"examples": 5000, "input_pixel_sha256": x["input_pixel_sha256"],
                       "candidate_accuracy": x["accuracy"], "incumbent_accuracy": y["accuracy"],
                       "accuracy_delta": x["accuracy"]-y["accuracy"],
                       "candidate_nll": x["loss"], "incumbent_nll": y["loss"],
                       "nll_delta": x["loss"]-y["loss"]}
    return {"schema": SCHEMA, "status": "complete", "policy": POLICY,
            "comparison_scope": "recorded_benchmark_scalars_only", "development_examples": 5000,
            "training_examples": 40000, "data_signature": DATA, "evaluation_context": CONTEXT,
            "candidate": evidence_summary(candidate), "incumbent": evidence_summary(incumbent),
            "same_policy_deltas": {"clean_accuracy": a["clean_accuracy"]-b["clean_accuracy"],
                                   "clean_nll": a["clean_loss"]-b["clean_loss"],
                                   "mean_stress_accuracy": a["stress_accuracy"]-b["stress_accuracy"]},
            "cases": cases, "stress_case_regressions": [c for c in CASES[1:] if cases[c]["accuracy_delta"] < 0],
            "noise_accuracy_regressed": cases["noise_004"]["accuracy_delta"] < 0,
            "tensor_checkpoint_audit_performed": False, "benchmark_reexecuted": False,
            "bootstrap_performed": False, "automatic_promotion": False,
            "submission_authorized_by_this_report": False,
            "interpretation": "Matched recorded aggregate comparison on reused development; no paired confidence interval, independent confirmation, seed-variance estimate or hidden-test guarantee. The old baseline gate and failed-candidate decisions remain unchanged. No promotion or delivery qualification is created."}


def frozen_sources():
    paths = {ROOT / name: checksum for name, checksum in CORE.items()}
    paths[ROOT / "research/benchmark.py"] = BENCHMARK_SOURCE_SHA
    for path, checksum in paths.items():
        require(digest(path.read_bytes()) == checksum, "Frozen native v3 source bytes differ")
    return {str(path): checksum for path, checksum in paths.items()}


def run(candidate_path, output_path, incumbent_path=INCUMBENT_PATH):
    candidate_path, output_path, incumbent_path = map(Path, (candidate_path, output_path, incumbent_path))
    if output_path.exists():
        raise FileExistsError("Existing comparison evidence cannot be overwritten")
    sources = frozen_sources()
    candidate, candidate_sha = read_document(candidate_path)
    incumbent, incumbent_sha = read_document(incumbent_path)
    require(incumbent_sha == INCUMBENT_BENCHMARK_SHA, "Frozen incumbent benchmark bytes differ")
    result = compare_documents(candidate, incumbent)
    self_path = Path(__file__)
    snapshots = {**sources, str(candidate_path): candidate_sha, str(incumbent_path): incumbent_sha,
                 str(self_path): digest(self_path.read_bytes()), str(PLAN): digest(PLAN.read_bytes())}
    result.update(recorded_at=datetime.now(timezone.utc).isoformat(), candidate_benchmark_sha256=candidate_sha,
                  incumbent_benchmark_sha256=incumbent_sha, comparator_source_sha256=snapshots[str(self_path)],
                  comparison_plan_sha256=snapshots[str(PLAN)])
    if any(digest(Path(path).read_bytes()) != checksum for path, checksum in snapshots.items()):
        raise RuntimeError("Comparison source/evidence changed before publication")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-benchmark", type=Path, required=True)
    parser.add_argument("--incumbent-benchmark", type=Path, default=INCUMBENT_PATH,
                        help="Relocated identical pinned local incumbent benchmark only")
    parser.add_argument("--output", type=Path, required=True)
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        parser.print_help()
        return None
    args = parser.parse_args(values)
    result = run(args.candidate_benchmark, args.output, args.incumbent_benchmark)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "policy": result["policy"], "same_policy_deltas": result["same_policy_deltas"],
                      "stress_case_regressions": result["stress_case_regressions"],
                      "automatic_promotion": False}))
    return result


if __name__ == "__main__":
    main()
