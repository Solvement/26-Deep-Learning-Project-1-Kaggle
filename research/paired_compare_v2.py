"""Versioned paired evidence with a source-bound ResizeMix training audit.

The frozen v1 module supplies its unchanged pixel, aggregate and image-cluster
bootstrap helpers. This entrypoint independently dispatches method validators;
no frozen module/global is patched. Both this verifier and the v1 helper hashes
are recorded and must be bound by any v2 delivery consumer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import seed_all, write_json
import torch
from data import load_training_arrays
from model import build_model, count_parameters
from research import benchmark as suite
from research import paired_compare as legacy
from train import code_provenance, tensor_images, training_source

SCHEMA = "exploratory_paired_image_bootstrap_v2"
PAIRED_HELPER_SOURCE_PATH = Path(legacy.__file__)
SEED, REPLICATES, STABILITY_SHA = legacy.SEED, legacy.REPLICATES, legacy.STABILITY_SHA
paired_bootstrap = legacy.paired_bootstrap
validate_document = legacy.validate_document
collect_correctness = legacy.collect_correctness


def training_validator(schema):
    if schema == "research_resizemix_training_v1":
        from research.benchmark_resizemix import validate_resizemix_training
        return validate_resizemix_training, ROOT/"research/benchmark_resizemix.py"
    return legacy.training_validator(schema)


def load_experiment(checkpoint_path, evidence_path, signature, context, core_source, train_ids, val_ids):
    """Read and validate actual method/history/scaler evidence before inference."""
    checkpoint_path, evidence_path = Path(checkpoint_path), Path(evidence_path)
    checkpoint_sha = suite.sha256(checkpoint_path)
    if checkpoint_sha == STABILITY_SHA:
        raise ValueError("The validation-only stability checkpoint is excluded even if renamed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    folder = checkpoint_path.parent
    metrics, history = [json.loads((folder/name).read_text(encoding="utf-8")) for name in ("metrics.json", "history.json")]
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    fingerprints = {"checkpoint_sha256": checkpoint_sha,
                    "training_metrics_sha256": suite.sha256(folder/"metrics.json"),
                    "training_history_sha256": suite.sha256(folder/"history.json")}
    validate_document(document, checkpoint, fingerprints, signature, context, core_source, suite.sha256(Path(suite.__file__)))
    suite.validate_checkpoint(checkpoint, train_ids, val_ids, signature, metrics, core_source, history)
    if ("research_training" in checkpoint.get("provenance", {}) or
            "research_training" in metrics or "research_training_integrity" in document):
        provenance, environment = [json.loads((folder/name).read_text(encoding="utf-8")) for name in ("provenance.json", "environment.json")]
        schema = checkpoint.get("provenance", {}).get("research_training", {}).get("schema")
        validator, validator_path = training_validator(schema)
        integrity = validator(checkpoint, metrics, history, provenance, environment)
        recorded = document.get("research_training_integrity", {})
        if (any(recorded.get(key) != value for key, value in integrity.items()) or
                recorded.get("validator_source_sha256") != suite.sha256(validator_path) or
                recorded.get("provenance_document_sha256") != suite.sha256(folder/"provenance.json") or
                recorded.get("environment_document_sha256") != suite.sha256(folder/"environment.json")):
            raise ValueError("Method benchmark integrity/source records differ")
        core = evidence_path.with_name(evidence_path.stem+".core.json")
        if (recorded.get("core_benchmark_sha256") != suite.sha256(core) or
                json.loads(core.read_text(encoding="utf-8")) != {key: value for key, value in document.items() if key != "research_training_integrity"}):
            raise ValueError("Method benchmark core evidence differs")
    return checkpoint, document


def run(args):
    paths = (args.candidate_checkpoint, args.baseline_checkpoint)
    if any("stability_seed2027" in path.parts for path in paths):
        raise ValueError("The validation-only stability repeat is excluded")
    if args.output.exists():
        raise FileExistsError("Existing paired v2 evidence cannot be overwritten")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    core_source = code_provenance()["source_sha256"]
    context = {"device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)),
               "torch_version": torch.__version__, "training_core_sha256": core_source}
    seed_all(2026)
    images, labels, train_ids, val_ids = load_training_arrays(args.data_dir)
    manifest = json.loads((args.data_dir/"training_manifest.json").read_text(encoding="utf-8"))
    signature = {"archive_sha256": training_source(args.data_dir)["archive_sha256"],
                 "split_sha256": manifest["split"]["indices_sha256"]}
    c, candidate = load_experiment(paths[0], args.candidate_benchmark, signature, context, core_source, train_ids, val_ids)
    b, baseline = load_experiment(paths[1], args.baseline_benchmark, signature, context, core_source, train_ids, val_ids)
    evidence_paths = [*paths, args.candidate_benchmark, args.baseline_benchmark,
                      Path(__file__), PAIRED_HELPER_SOURCE_PATH, Path(suite.__file__), *(ROOT/name for name in core_source)]
    for path, checkpoint, evidence in zip(paths, (c, b), (args.candidate_benchmark, args.baseline_benchmark)):
        names = ["metrics.json", "history.json"]
        if "research_training" in checkpoint.get("provenance", {}):
            names += ["provenance.json", "environment.json"]
            marker = checkpoint["provenance"]["research_training"]
            _, validator_path = training_validator(marker["schema"])
            evidence_paths += [validator_path, *(ROOT/name for name in marker["source_sha256"])]
            evidence_paths += [evidence.with_name(evidence.stem+".core.json")]
        evidence_paths += [path.parent/name for name in names]
    snapshots = {path: suite.sha256(path) for path in evidence_paths}
    if (suite.sha256(paths[1]) != suite.BASELINE_CHECKPOINT_SHA or
            candidate.get("baseline_sha256") != suite.sha256(args.baseline_benchmark) or
            args.policy not in candidate.get("policies", {})):
        raise ValueError("Trusted original baseline, recorded pairing and selected policy are required")
    reports = {args.policy: candidate["policies"][args.policy]}
    suite.validate_baseline(baseline, reports, signature, candidate["batch_size"], context, suite.sha256(Path(suite.__file__)))
    models = []
    for checkpoint in (c, b):
        model = build_model(checkpoint["model_config"])
        if count_parameters(model) != checkpoint["parameter_count"]:
            raise ValueError("Constructed model parameter count differs from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model.eval())
    dev_images = tensor_images(images[val_ids], device)
    dev_labels = torch.as_tensor(labels[val_ids], device=device)
    vectors = [collect_correctness(model.to(device=device, memory_format=torch.channels_last), dev_images, dev_labels,
               policy, candidate["batch_size"], device, document["policies"][policy])
               for model, policy, document in zip(models, (args.policy, "flip"), (candidate, baseline))]
    if any(suite.sha256(path) != checksum for path, checksum in snapshots.items()):
        raise RuntimeError("Source/checkpoint/evidence changed during paired v2 evaluation")
    bootstrap = paired_bootstrap(*vectors, seed=SEED, replicates=REPLICATES)
    if bootstrap.get("original_images") != 5000:
        raise ValueError("Complete paired v2 evidence requires exactly 5,000 original images")
    # Also guard the CPU bootstrap interval: source/evidence must remain frozen
    # until the output is written, not just until GPU inference finishes.
    if any(suite.sha256(path) != checksum for path, checksum in snapshots.items()):
        raise RuntimeError("Source/checkpoint/evidence changed during paired v2 bootstrap")
    result = {"schema": SCHEMA, "status": "complete", "candidate_policy": args.policy, "baseline_policy": "flip",
              "data_signature": signature, "evaluation_context": context,
              "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
              "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
              "candidate_benchmark_sha256": snapshots[args.candidate_benchmark],
              "baseline_benchmark_sha256": snapshots[args.baseline_benchmark],
              "source_sha256": snapshots[Path(__file__)],
              "paired_helper_source_sha256": snapshots[PAIRED_HELPER_SOURCE_PATH],
              "benchmark_source_sha256": snapshots[Path(suite.__file__)], "bootstrap": bootstrap,
              "confirmation_evaluated": False, "hidden_images_loaded": False,
              "external_data_loaded": False, "automatic_promotion": False,
              "interpretation": "Exploratory reused/postselected development; image-cluster percentile intervals are conditional on these fitted checkpoints, not independent confirmation or hidden-test guarantees."}
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-checkpoint", "candidate-benchmark", "baseline-checkpoint", "baseline-benchmark", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--policy", choices=suite.POLICIES, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "schema": result["schema"], "bootstrap": result["bootstrap"]}))


if __name__ == "__main__":
    main()
