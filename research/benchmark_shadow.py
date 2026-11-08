"""Validate the independent trainer, then run the frozen development benchmark.

The original benchmark source and original-checkpoint baseline stay unchanged.
New training source, precision and successful/skipped SGD updates are checked
before a shadow candidate can enter the complete benchmark or promotion gate.
"""
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
from research.train_shadow import research_provenance, validate_config

MAX_SKIPPED_FRACTION = .01


def validate_shadow(checkpoint, metrics, history, provenance, environment):
    config = checkpoint["config"]
    validate_config(config)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    precision = marker.get("precision")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Shadow checkpoint must record its actual training precision")
    expected = research_provenance(precision, config)
    if (marker != expected or metrics.get("research_training") != expected or
            provenance.get("code", {}).get("research_training") != expected):
        raise ValueError("Shadow checkpoint/metrics/provenance do not match the current independent training source")
    if (environment.get("shadow_training_precision") != precision or
            environment.get("grad_scaler_enabled") is not (precision == "fp16")):
        raise ValueError("Training environment and native GradScaler provenance differ")
    if len(history) != config["epochs"]:
        raise ValueError("Shadow history must contain every completed training epoch")
    steps = math.ceil(40000/config["batch_size"])
    successful = skipped = 0
    for epoch, row in enumerate(history, 1):
        count, missed = row.get("successful_updates"), row.get("skipped_updates")
        if (row.get("epoch") != epoch or type(count) is not int or type(missed) is not int or
                count < 1 or missed < 0 or count+missed != steps):
            raise ValueError("Shadow update history is incomplete or inconsistent")
        scale = row.get("grad_scaler_scale")
        if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Shadow history contains an invalid GradScaler scale")
        successful += count
        skipped += missed
    if metrics.get("successful_updates") != successful or metrics.get("skipped_updates") != skipped:
        raise ValueError("Shadow total update counts do not match complete history")
    fraction = skipped/(successful+skipped)
    if fraction > MAX_SKIPPED_FRACTION or (precision != "fp16" and skipped):
        raise ValueError("Too many overflow-skipped updates, or skipped updates without FP16 scaling")
    return {"schema": "research_shadow_integrity_v1", "training": expected,
            "successful_updates": successful, "skipped_updates": skipped,
            "skipped_fraction": fraction, "maximum_skipped_fraction": MAX_SKIPPED_FRACTION,
            "confirmation_evaluated": False, "hidden_images_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    core_output = args.output.with_name(args.output.stem+".core.json")
    if args.output.exists() or core_output.exists():
        raise FileExistsError("Use new output paths; existing benchmark evidence is immutable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    folder = args.checkpoint.parent
    documents = [json.loads((folder/name).read_text(encoding="utf-8")) for name in
                 ("metrics.json", "history.json", "provenance.json", "environment.json")]
    integrity = validate_shadow(checkpoint, *documents)
    checkpoint_sha = benchmark.sha256(args.checkpoint)
    del checkpoint
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT/"research/benchmark.py"), "--checkpoint", str(args.checkpoint),
                    "--output", str(core_output), "--baseline", str(args.baseline), *remainder]
        benchmark.main()
    finally:
        sys.argv = original_argv
    if benchmark.sha256(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed after its shadow-source validation")
    result = json.loads(core_output.read_text(encoding="utf-8"))
    result["research_training_integrity"] = {**integrity,
        "validator_source_sha256": benchmark.sha256(Path(__file__)),
        "core_benchmark_sha256": benchmark.sha256(core_output),
        "provenance_document_sha256": benchmark.sha256(folder/"provenance.json"),
        "environment_document_sha256": benchmark.sha256(folder/"environment.json")}
    write_json(args.output, result)
    print(json.dumps({"event": "shadow_integrity_and_benchmark_complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
