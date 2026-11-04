"""Audit the independent SAM training source before the common v3 benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import write_json
import torch
from research import benchmark
from research.train_sam import research_provenance, validate_config


def validate_sam(checkpoint, metrics, history, provenance, environment):
    config = checkpoint["config"]
    validate_config(config)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    precision = marker.get("precision")
    if precision not in ("bf16", "fp32"):
        raise ValueError("SAM requires its verified native BF16 or CPU FP32 path")
    expected = research_provenance(precision, config)
    if (marker != expected or metrics.get("research_training") != expected or
            provenance.get("code", {}).get("research_training") != expected):
        raise ValueError("SAM checkpoint/metrics/provenance differ from the independent training source")
    if (environment.get("sam_training_precision") != precision or
            environment.get("grad_scaler_enabled") is not False):
        raise ValueError("SAM environment or scaler provenance differs")
    steps = math.ceil(40000/config["batch_size"])
    if len(history) != config["epochs"]:
        raise ValueError("SAM requires a complete training history")
    for epoch, row in enumerate(history, 1):
        if (row.get("epoch") != epoch or type(row.get("successful_updates")) is not int or
                row["successful_updates"] != steps or row.get("skipped_updates") != 0 or
                type(row.get("sam_forward_passes")) is not int or row["sam_forward_passes"] != 2*steps):
            raise ValueError("SAM history must prove two passes per successful update")
        loss, norm = row.get("sam_loss"), row.get("mean_perturbation_norm")
        if (any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in (loss, norm)) or
                loss < 0 or not 0 <= norm <= expected["rho"]+1e-6):
            raise ValueError("SAM loss or actual perturbation norm is invalid")
    updates = steps*config["epochs"]
    if (metrics.get("successful_updates") != updates or metrics.get("skipped_updates") != 0 or
            metrics.get("sam_forward_passes") != 2*updates):
        raise ValueError("SAM complete update/pass totals differ from history")
    return {"schema": "research_sam_integrity_v1", "training": expected,
            "successful_updates": updates, "skipped_updates": 0, "sam_forward_passes": 2*updates,
            "confirmation_evaluated": False, "hidden_images_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    core_output = args.output.with_name(args.output.stem+".core.json")
    if args.output.exists() or core_output.exists():
        raise FileExistsError("Use new output paths; SAM evidence is immutable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    folder = args.checkpoint.parent
    documents = [json.loads((folder/name).read_text(encoding="utf-8")) for name in
                 ("metrics.json", "history.json", "provenance.json", "environment.json")]
    integrity = validate_sam(checkpoint, *documents)
    original_sha = benchmark.sha256(args.checkpoint)
    del checkpoint
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT/"research/benchmark.py"), "--checkpoint", str(args.checkpoint),
                    "--output", str(core_output), "--baseline", str(args.baseline), *remainder]
        benchmark.main()
    finally:
        sys.argv = original_argv
    if benchmark.sha256(args.checkpoint) != original_sha:
        raise RuntimeError("SAM checkpoint changed after source validation")
    result = json.loads(core_output.read_text(encoding="utf-8"))
    result["research_training_integrity"] = {**integrity,
        "validator_source_sha256": benchmark.sha256(Path(__file__)),
        "core_benchmark_sha256": benchmark.sha256(core_output),
        "provenance_document_sha256": benchmark.sha256(folder/"provenance.json"),
        "environment_document_sha256": benchmark.sha256(folder/"environment.json")}
    write_json(args.output, result)
    print(json.dumps({"event": "sam_integrity_and_benchmark_complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
