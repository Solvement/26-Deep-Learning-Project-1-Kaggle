"""Deep-audit only the independent AugMix160 recipe, then run full frozen v3.

This typed route does not accept SAM/shadow metadata, change their schemas,
recalibrate BN, select another policy, predict hidden images or submit files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import write_json
import torch
from research import benchmark as core
from research.train_augmix import (SCHEMA, PARAMETER_COUNT, OFFICIAL_SIGNATURE, raw_config, research_provenance,
    validate_config, validate_history, validate_saved_state)
from train import code_provenance


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_augmix(checkpoint, metrics, history, provenance, environment, input_config_bytes, last):
    config = checkpoint.get("config", {})
    validate_config(config)
    raw = raw_config(config, input_config_bytes)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    precision = marker.get("precision")
    expected = research_provenance(precision, config, hashlib.sha256(raw).hexdigest())
    require(all(doc.get("schema") == SCHEMA for doc in (checkpoint, metrics, provenance, environment, last)),
            "AugMix requires its independent typed checkpoint/metrics/provenance/environment/last state")
    require(marker == expected and metrics.get("research_training") == expected and
            provenance.get("code", {}).get("research_training") == expected and environment.get("research_training") == expected and
            last.get("provenance", {}).get("research_training") == expected,
            "AugMix source/raw/effective config/RNG/precision identity differs")
    core_sources = code_provenance()["source_sha256"]
    require(checkpoint.get("provenance", {}).get("source_sha256") == core_sources and
            provenance.get("code", {}).get("source_sha256") == core_sources and
            last.get("provenance", {}).get("source_sha256") == core_sources,
            "AugMix frozen core source differs")
    require(environment.get("augmix_training_precision") == precision and environment.get("grad_scaler_enabled") is (precision == "fp16") and
            environment.get("amp_dtype") == ("torch.float16" if precision == "fp16" else "torch.float32"),
            "AugMix actual environment or native scaler path differs")
    model_config = {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}
    require(checkpoint.get("model_config") == last.get("model_config") == model_config and
            checkpoint.get("parameter_count") == last.get("parameter_count") == metrics.get("parameter_count") == provenance.get("parameter_count") == PARAMETER_COUNT,
            "AugMix actual WRN22-4 parameter count/model identity differs")
    require(metrics.get("status") == "complete" and metrics.get("epochs_completed") == last.get("epoch") == 160 and
            metrics.get("training_examples") == provenance.get("train_examples") == 40000 and
            metrics.get("validation_examples") == provenance.get("validation_examples") == 5000 and
            metrics.get("config") == last.get("config") == config and last.get("history") == history and
            checkpoint.get("data_signature") == last.get("data_signature") == provenance.get("data_signature"),
            "AugMix promotion requires actual complete official40k/5k/160 training")
    require(checkpoint.get("data_signature") == OFFICIAL_SIGNATURE and type(checkpoint.get("epoch")) is int and
            1 <= checkpoint["epoch"] <= 160 and all(type(last.get(k)) is int for k in
            ("updates","skipped_updates","attempted_updates","source_examples_seen","model_views_seen")),
            "AugMix exact official signature, selected epoch or typed final ledger differs")
    successful, skipped = validate_history(history, 160)
    require(type(metrics.get("successful_updates")) is int and metrics["successful_updates"] == last.get("updates") == successful and
            type(metrics.get("skipped_updates")) is int and metrics["skipped_updates"] == last.get("skipped_updates") == skipped and
            type(metrics.get("attempted_updates")) is int and metrics["attempted_updates"] == last.get("attempted_updates") == 50080 and
            successful+skipped == 50080 and metrics.get("forward_calls") == 50080 and
            metrics.get("source_examples_seen") == last.get("source_examples_seen") == 6400000 and
            metrics.get("model_views_seen") == last.get("model_views_seen") == 19200000,
            "AugMix cumulative attempts/source/view/success/skip ledger differs")
    fraction = skipped/50080
    require(fraction <= .01 and (precision == "fp16" or skipped == 0), "AugMix has excessive overflow skips or skips without native FP16")
    validate_saved_state(last, config, precision)
    return {"schema":"research_augmix_integrity_v1", "training":expected, "successful_updates":successful,
        "skipped_updates":skipped, "attempted_updates":50080, "source_examples_seen":6400000, "model_views_seen":19200000,
        "forward_calls":50080, "skipped_fraction":fraction, "maximum_skipped_fraction":.01,
        "confirmation_evaluated":False, "hidden_images_loaded":False, "external_data_loaded":False}


def validate_pool_document(document, expected_sources=None):
    require(document.get("schema") == "research_augmix_worker_pool_v1" and document.get("status") == "closed" and
            document.get("all_workers_stopped") is True and document.get("num_workers") == 2 and document.get("prefetch_factor") == 2 and
            document.get("context") == "spawn" and document.get("worker_cuda_initialized") is False and
            type(document.get("owner_pid")) is int and document["owner_pid"] > 0 and
            type(document.get("owner_created_at")) in (int,float) and math.isfinite(document["owner_created_at"]) and document["owner_created_at"] > 0 and
            len(document.get("workers", [])) == 2, "AugMix needs complete source-bound CPU worker teardown")
    for worker in document["workers"]:
        require(type(worker.get("pid")) is int and worker["pid"] > 0 and type(worker.get("created_at")) in (int,float) and
                math.isfinite(worker["created_at"]) and worker["created_at"] > 0, "Actual worker PID/creation identity required")
    require(len({worker["pid"] for worker in document["workers"]}) == 2, "Two distinct worker identities required")
    for key in ("source_ids_sha256", "sample_seeds_sha256"):
        require(isinstance(document.get(key),str) and len(document[key]) == 64 and all(c in "0123456789abcdef" for c in document[key]),
                "AugMix worker source/seed fingerprints required")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    core_output = args.output.with_name(args.output.stem+".core.json")
    if args.output.exists() or core_output.exists(): raise FileExistsError("Use fresh AugMix benchmark output paths")
    folder = args.checkpoint.parent
    paths = [args.checkpoint, *(folder/name for name in ("metrics.json","history.json","provenance.json","environment.json","input_config.json","last.pt","config.json"))]
    hashes = {str(path.resolve()):core.sha256(path) for path in paths}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics, history, provenance, env = [json.loads((folder/name).read_text()) for name in ("metrics.json","history.json","provenance.json","environment.json")]
    last = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
    integrity = validate_augmix(checkpoint, metrics, history, provenance, env, (folder/"input_config.json").read_bytes(), last)
    require(json.loads((folder/"config.json").read_text()) == checkpoint["config"], "AugMix exact effective config artifact differs")
    require(integrity["training"]["precision"] == "fp16", "Actual AugMix promotion requires the registered native FP16 training path")
    for epoch, row in enumerate(history,1):
        pool_path = folder/f"worker_pool_epoch{epoch:04d}.json"
        validate_pool_document(json.loads(pool_path.read_text()))
        require(row.get("worker_pool_receipt_sha256") == core.sha256(pool_path), "Epoch worker receipt differs from its recorded history")
        hashes[str(pool_path.resolve())] = core.sha256(pool_path)
    del checkpoint, last
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT/"research/benchmark.py"), "--checkpoint",str(args.checkpoint), "--output",str(core_output),
            "--baseline",str(args.baseline), "--data-dir",str(args.data_dir), "--device",args.device, "--batch-size","128"]
        core.main()
    finally: sys.argv = original_argv
    require(all(core.sha256(path) == digest for path,digest in hashes.items()), "AugMix evidence changed during the complete benchmark")
    result = json.loads(core_output.read_text())
    require(result.get("status") == "complete" and result.get("examples") == 5000 and set(result.get("policies",{})) == set(core.POLICIES),
            "Only the complete5000/all-three-policy v3 suite can qualify AugMix")
    result["fixed_primary_policy"] = "center80_shift20"
    result["research_training_integrity"] = {**integrity, "validator_source_sha256":core.sha256(Path(__file__)),
        "core_benchmark_sha256":core.sha256(core_output), "evidence_sha256":hashes,
        "provenance_document_sha256":core.sha256(folder/"provenance.json"), "environment_document_sha256":core.sha256(folder/"environment.json")}
    write_json(args.output,result)
    print(json.dumps({"event":"augmix_integrity_and_benchmark_complete","output":str(args.output),
        "fixed_policy_passed":result["promotion"]["center80_shift20"]["passed"]}),flush=True)


if __name__ == "__main__": main()
