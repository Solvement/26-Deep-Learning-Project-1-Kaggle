"""Verify a source-bound ResizeMix run before the unchanged v3 dev benchmark."""
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
from research.train_resizemix import research_provenance, validate_config


def validate_resizemix_training(checkpoint, metrics, history, provenance, environment):
    config = checkpoint["config"]
    validate_config(config)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    precision = marker.get("precision")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("ResizeMix must record actual training precision")
    expected = research_provenance(precision, config)
    if (marker != expected or metrics.get("research_training") != expected or
            provenance.get("code", {}).get("research_training") != expected):
        raise ValueError("ResizeMix checkpoint/metrics/provenance source differs")
    if (environment.get("resizemix_training_precision") != precision or
            environment.get("grad_scaler_enabled") is not (precision == "fp16")):
        raise ValueError("ResizeMix environment/scaler differs")
    steps = math.ceil(40000/config["batch_size"])
    if len(history) != config["epochs"]:
        raise ValueError("ResizeMix requires complete training history")
    updates = skipped = mixed = 0
    for epoch, row in enumerate(history, 1):
        count, misses, batches = row.get("successful_updates"), row.get("skipped_updates"), row.get("resizemix_batches")
        if (row.get("epoch") != epoch or any(type(v) is not int for v in (count, misses, batches)) or
                count < 1 or misses < 0 or count+misses != steps or not 0 <= batches <= steps):
            raise ValueError("ResizeMix step/augmentation counts are invalid")
        scale = row.get("grad_scaler_scale")
        if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("ResizeMix scaler history is invalid")
        updates += count
        skipped += misses
        mixed += batches
    if (metrics.get("successful_updates") != updates or metrics.get("skipped_updates") != skipped or
            metrics.get("resizemix_batches") != mixed):
        raise ValueError("ResizeMix history and metric totals differ")
    if skipped/(updates+skipped) > .01 or (precision != "fp16" and skipped):
        raise ValueError("ResizeMix overflow skips exceed the verified protocol")
    if config["resizemix_probability"] > 0 and mixed == 0:
        raise ValueError("Enabled ResizeMix never ran")
    return {"schema": "research_resizemix_integrity_v1", "training": expected,
            "successful_updates": updates, "skipped_updates": skipped,
            "resizemix_batches": mixed, "confirmation_evaluated": False, "hidden_images_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    core_output = args.output.with_name(args.output.stem+".core.json")
    if args.output.exists() or core_output.exists():
        raise FileExistsError("ResizeMix benchmark evidence cannot be overwritten")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    folder = args.checkpoint.parent
    documents = [json.loads((folder/name).read_text(encoding="utf-8")) for name in
                 ("metrics.json", "history.json", "provenance.json", "environment.json")]
    integrity = validate_resizemix_training(checkpoint, *documents)
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
        raise RuntimeError("ResizeMix checkpoint changed after source validation")
    result = json.loads(core_output.read_text(encoding="utf-8"))
    result["research_training_integrity"] = {**integrity,
        "validator_source_sha256": benchmark.sha256(Path(__file__)),
        "core_benchmark_sha256": benchmark.sha256(core_output),
        "provenance_document_sha256": benchmark.sha256(folder/"provenance.json"),
        "environment_document_sha256": benchmark.sha256(folder/"environment.json")}
    write_json(args.output, result)
    print(json.dumps({"event": "resizemix_integrity_and_benchmark_complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
