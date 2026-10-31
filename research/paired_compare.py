"""Paired exploratory development evidence; never changes a promotion decision.

The bootstrap samples original images, carrying all five stress views together.
Only aggregate results are saved. Reused/postselected development is not an
independent confirmation set, and these intervals do not predict Kaggle scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import seed_all, write_json
import numpy as np
import torch
from data import load_training_arrays
from model import build_model, count_parameters
from research import benchmark as suite
from research.benchmark_shadow import validate_shadow
from train import code_provenance, tensor_images, training_source

SEED, REPLICATES = 20261031, 10000
STABILITY_SHA = "258c4f4714f1382ace8864141a817a14e629d15b01628ef0c94497247cacd049"


def paired_bootstrap(candidate, baseline, seed=SEED, replicates=REPLICATES, chunk_size=100):
    """Accept private [original image, clean + five stress cases] correctness."""
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if (candidate.shape != baseline.shape or candidate.ndim != 2 or
            candidate.shape[1] != len(suite.CASES) or len(candidate) < 2 or
            not np.isin(candidate, [0, 1]).all() or not np.isin(baseline, [0, 1]).all()):
        raise ValueError("Matched binary correctness matrices with six views are required")
    if type(replicates) is not int or replicates < 2 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Positive chunk size and at least two bootstrap replicates required")
    differences = candidate.astype(np.float64)-baseline.astype(np.float64)
    # Average stress within each original image BEFORE sampling image rows.
    paired = np.column_stack((differences[:, 0], differences[:, 1:].mean(axis=1)))
    draws = np.empty((replicates, 2))
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates-start)
        indices = rng.integers(0, len(paired), size=(size, len(paired)), dtype=np.int32)
        draws[start:start+size] = paired[indices].mean(axis=1)
    result = {"original_images": len(paired), "replicates": replicates, "seed": seed,
              "sampling_unit": "original_image_with_all_five_stress_views", "confidence_level": .95}
    for column, name in enumerate(("clean", "stress")):
        x = differences[:, :1] if column == 0 else differences[:, 1:]
        a, b = (candidate[:, :1], baseline[:, :1]) if column == 0 else (candidate[:, 1:], baseline[:, 1:])
        result[name] = {"delta_accuracy": float(paired[:, column].mean()),
            "percentile_ci95": np.quantile(draws[:, column], [.025, .975]).tolist(),
            "candidate_only_correct": int(((a == 1) & (b == 0)).sum()),
            "baseline_only_correct": int(((a == 0) & (b == 1)).sum()),
            "both_correct": int(((a == 1) & (b == 1)).sum()),
            "both_wrong": int(((a == 0) & (b == 0)).sum()),
            "original_image_net_wins": int((x.sum(axis=1) > 0).sum()),
            "original_image_net_losses": int((x.sum(axis=1) < 0).sum())}
    return result


def validate_document(document, checkpoint, fingerprints, signature, context, core_source, source_hash):
    """Reject incompatible/partial evidence before model construction or inference."""
    if checkpoint.get("config", {}).get("train_all") is not False:
        raise ValueError("Paired development rejects full-data checkpoints")
    expected = {"schema": suite.SCHEMA, "status": "complete", "examples": 5000,
                "training_examples": 40000, "bn_recalibrated": False,
                "confirmation_evaluated": False, "hidden_images_loaded": False,
                "external_data_loaded": False, "data_signature": signature,
                "evaluation_context": context, "benchmark_source_sha256": source_hash,
                "stress_cases": list(suite.CASES), **fingerprints}
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("Benchmark source/hash/context, completeness or permitted inputs differ")
    if (document.get("source", {}).get("source_sha256") != core_source or
            document.get("config") != checkpoint["config"] or
            document.get("model_config") != checkpoint.get("model_config") or
            document.get("checkpoint_epoch") != checkpoint.get("epoch") or
            document.get("checkpoint_variant") != checkpoint.get("variant") or
            document.get("parameter_count") != checkpoint.get("parameter_count") or
            type(document.get("batch_size")) is not int or document["batch_size"] < 1):
        raise ValueError("Benchmark and checkpoint training/source metadata differ")
    if checkpoint["model_config"] != {key: checkpoint["config"][key] for key in ("depth", "widen_factor", "dropout", "num_classes")}:
        raise ValueError("Model constructor and training configuration differ")
    if not document.get("policies") or not set(document["policies"]) <= set(suite.POLICIES):
        raise ValueError("Complete supported policy reports are required")
    for report in document["policies"].values():
        cases = report.get("cases", {})
        if set(cases) != set(suite.CASES):
            raise ValueError("Every fixed stress case must be present")
        for case in cases.values():
            accuracy, pixels = case.get("accuracy"), case.get("input_pixel_sha256", "")
            if (case.get("examples") != 5000 or not isinstance(accuracy, (int, float)) or
                    not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
                    len(pixels) != 64 or any(c not in "0123456789abcdef" for c in pixels)):
                raise ValueError("Incomplete case counts, accuracy or pixel fingerprint")
        if (report.get("clean_accuracy") != cases["clean"]["accuracy"] or
                not math.isclose(report.get("stress_accuracy", float("nan")),
                    sum(cases[case]["accuracy"] for case in suite.CASES[1:])/5, abs_tol=1e-12)):
            raise ValueError("Case and summary accuracies differ")


def load_experiment(checkpoint_path, evidence_path, signature, context, core_source, train_ids, val_ids):
    if suite.sha256(checkpoint_path) == STABILITY_SHA:
        raise ValueError("The validation-only stability checkpoint is excluded even if renamed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    folder = checkpoint_path.parent
    metrics, history = [json.loads((folder/name).read_text(encoding="utf-8")) for name in ("metrics.json", "history.json")]
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    fingerprints = {"checkpoint_sha256": suite.sha256(checkpoint_path),
                    "training_metrics_sha256": suite.sha256(folder/"metrics.json"),
                    "training_history_sha256": suite.sha256(folder/"history.json")}
    validate_document(document, checkpoint, fingerprints, signature, context, core_source, suite.sha256(Path(suite.__file__)))
    suite.validate_checkpoint(checkpoint, train_ids, val_ids, signature, metrics, core_source, history)
    if ("research_training" in checkpoint.get("provenance", {}) or
            "research_training" in metrics or "research_training_integrity" in document):
        provenance, environment = [json.loads((folder/name).read_text(encoding="utf-8")) for name in ("provenance.json", "environment.json")]
        integrity = validate_shadow(checkpoint, metrics, history, provenance, environment)
        recorded = document.get("research_training_integrity", {})
        if (any(recorded.get(key) != value for key, value in integrity.items()) or
                recorded.get("validator_source_sha256") != suite.sha256(ROOT/"research/benchmark_shadow.py") or
                recorded.get("provenance_document_sha256") != suite.sha256(folder/"provenance.json") or
                recorded.get("environment_document_sha256") != suite.sha256(folder/"environment.json")):
            raise ValueError("Shadow benchmark integrity/source records differ")
        core = evidence_path.with_name(evidence_path.stem+".core.json")
        if (recorded.get("core_benchmark_sha256") != suite.sha256(core) or
                json.loads(core.read_text(encoding="utf-8")) != {k: v for k, v in document.items() if k != "research_training_integrity"}):
            raise ValueError("Shadow benchmark core evidence differs")
    return checkpoint, document


def collect_correctness(model, images, labels, policy, batch_size, device, reference):
    """Reproduce the saved suite exactly; keep vectors in private process memory."""
    vectors = np.empty((len(labels), len(suite.CASES)), dtype=np.bool_)
    with torch.inference_mode():
        for column, case in enumerate(suite.CASES):
            pixels = hashlib.sha256()
            for start in range(0, len(labels), batch_size):
                batch = suite.stress(images[start:start+batch_size], case, start)
                pixels.update(batch.cpu().numpy().tobytes())
                with torch.autocast(device.type, dtype=suite.benchmark_dtype(device), enabled=device.type == "cuda"):
                    logits = suite.infer(model, batch, policy)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite paired-development logits")
                vectors[start:start+len(batch), column] = logits.argmax(1).eq(labels[start:start+len(batch)]).cpu().numpy()
            expected = reference["cases"][case]
            if (pixels.hexdigest() != expected["input_pixel_sha256"] or
                    abs(vectors[:, column].mean()-expected["accuracy"]) > 1e-12):
                raise ValueError("Recomputed development pixels/accuracy differ from frozen benchmark")
    return vectors


def run(args):
    paths = (args.candidate_checkpoint, args.baseline_checkpoint)
    if any("stability_seed2027" in path.parts for path in paths):
        raise ValueError("The validation-only stability repeat is excluded")
    if args.output.exists():
        raise FileExistsError("Existing paired evidence cannot be overwritten")
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
                      Path(__file__), Path(suite.__file__), *(ROOT/name for name in core_source)]
    for path, checkpoint, evidence in zip(paths, (c, b), (args.candidate_benchmark, args.baseline_benchmark)):
        names = ["metrics.json", "history.json"]
        if "research_training" in checkpoint.get("provenance", {}):
            names += ["provenance.json", "environment.json"]
            evidence_paths += [ROOT/"research/benchmark_shadow.py", ROOT/"research/train_shadow.py", ROOT/"research/augmentations.py"]
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
    dev_images, dev_labels = tensor_images(images[val_ids], device), torch.as_tensor(labels[val_ids], device=device)
    vectors = [collect_correctness(model.to(device=device, memory_format=torch.channels_last), dev_images, dev_labels,
               policy, candidate["batch_size"], device, doc["policies"][policy])
               for model, policy, doc in zip(models, (args.policy, "flip"), (candidate, baseline))]
    if any(suite.sha256(path) != checksum for path, checksum in snapshots.items()):
        raise RuntimeError("Source/checkpoint/evidence changed during paired evaluation")
    result = {"schema": "exploratory_paired_image_bootstrap_v1", "status": "complete",
              "candidate_policy": args.policy, "baseline_policy": "flip", "data_signature": signature,
              "evaluation_context": context, "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
              "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
              "candidate_benchmark_sha256": suite.sha256(args.candidate_benchmark),
              "baseline_benchmark_sha256": suite.sha256(args.baseline_benchmark),
              "source_sha256": suite.sha256(Path(__file__)), "benchmark_source_sha256": suite.sha256(Path(suite.__file__)),
              "bootstrap": paired_bootstrap(*vectors), "confirmation_evaluated": False,
              "hidden_images_loaded": False, "external_data_loaded": False, "automatic_promotion": False,
              "interpretation": "Exploratory reused/postselected development; percentile intervals are conditional on these fitted checkpoints, not independent confirmation or hidden-test guarantees."}
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
    print(json.dumps({"status": result["status"], "bootstrap": result["bootstrap"]}))


if __name__ == "__main__":
    main()
