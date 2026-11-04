"""Full v3 SE-SAM benchmark with original and fixed-SAM incremental gates.

This independent builder never modifies the frozen benchmark or SAM modules.
The primary policy is always center80_shift20. No confirmation/hidden loader,
partial benchmark, BN recalibration, prediction or submission mode is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import seed_all, write_json
import torch
from data import load_training_arrays
from model import count_parameters
from research import benchmark as core
from research.benchmark_sam import validate_sam
from research.se_model import ARCHITECTURE, PARAMETER_COUNT, build_model
from research.train_se_sam import (SCHEMA, input_config, model_configuration,
    research_provenance, validate_config, validate_saved_state)
from train import code_provenance, tensor_images, training_source

SAM_CHECKPOINT_SHA256 = "5a9771fe26e5b6bd6253a1ddfb2a3187e2087c3348b1569e3585fb6fa6bf07b2"
SAM_BENCHMARK_SHA256 = "d647ac6bbc734020767c8c3b99f8b1e7e3db842191780c7c8029db9c0554a211"
BASELINE_BENCHMARK_SHA256 = "eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
SAM_FREEZE_SHA256 = "d505c367a78970ddb56dc036dcec97b1851f52230fd0bf1f49188692c6c0c943"
SAM_BENCHMARK = ROOT/"results/research/wrn22_sam_s2026/benchmark.json"
SAM_CHECKPOINT = ROOT/"results/research/wrn22_sam_s2026/best.pt"
POLICY = "center80_shift20"


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_training(checkpoint, metrics, history, provenance, environment, input_config_bytes, last=None):
    """Bind complete fixed training, exact raw config, schema and every update."""
    config = checkpoint.get("config", {})
    validate_config(config)
    raw = input_config(config, input_config_bytes)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    expected = research_provenance(marker.get("precision"), config, hashlib.sha256(raw).hexdigest())
    require(all(document.get("schema") == SCHEMA for document in (checkpoint, metrics, provenance, environment)),
            "SE typed checkpoint/metrics/provenance/environment schema differs")
    require(checkpoint.get("model_config") == model_configuration(config) and
            checkpoint.get("parameter_count") == metrics.get("parameter_count") == PARAMETER_COUNT,
            "SE architecture or exact total/trainable parameter count differs")
    require(marker == expected and metrics.get("research_training") == expected and
            provenance.get("code", {}).get("research_training") == expected and
            environment.get("research_training") == expected,
            "SE source, raw/effective configuration, precision or seed provenance differs")
    current_core = code_provenance()["source_sha256"]
    require(checkpoint.get("provenance", {}).get("source_sha256") == current_core and
            provenance.get("code", {}).get("source_sha256") == current_core,
            "SE frozen core source differs")
    require(environment.get("se_sam_training_precision") == expected["precision"] and
            environment.get("grad_scaler_enabled") is False,
            "SE environment/scaler path differs")
    require(metrics.get("status") == "complete" and metrics.get("config") == config and
            metrics.get("epochs_completed") == 160 and metrics.get("training_examples") == 40000 and
            metrics.get("validation_examples") == 5000 and len(history) == 160 and
            provenance.get("train_examples") == 40000 and provenance.get("validation_examples") == 5000 and
            provenance.get("parameter_count") == PARAMETER_COUNT and
            provenance.get("data_signature") == checkpoint.get("data_signature"),
            "SE promotion requires complete official40k/5k and all160 epochs")
    for epoch, row in enumerate(history, 1):
        require(row.get("epoch") == epoch and type(row.get("successful_updates")) is int and
                row["successful_updates"] == 313 and type(row.get("sam_forward_passes")) is int and
                row["sam_forward_passes"] == 626 and type(row.get("skipped_updates")) is int and row["skipped_updates"] == 0,
                "SE history must prove every successful two-pass update")
        for key in ("train_loss", "sam_loss", "mean_first_grad_norm", "mean_perturbation_norm",
                    "validation_loss", "ema_validation_loss", "epoch_seconds", "elapsed_seconds", "lr"):
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "SE history contains nonfinite loss/norm/time")
        require(row["mean_perturbation_norm"] <= .050001, "SE SAM perturbation exceeds fixed rho")
        for key in ("train_accuracy", "validation_accuracy", "ema_validation_accuracy"):
            require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and 0 <= row[key] <= 1,
                    "SE history has invalid accuracy")
    require(type(metrics.get("successful_updates")) is int and metrics["successful_updates"] == 50080 and
            type(metrics.get("skipped_updates")) is int and metrics["skipped_updates"] == 0 and type(metrics.get("sam_forward_passes")) is int and
            metrics["sam_forward_passes"] == 100160, "SE complete update/pass totals differ")
    if last is not None:
        require(last.get("schema") == SCHEMA and last.get("config") == config and last.get("epoch") == 160 and
                last.get("model_config") == model_configuration(config) and last.get("parameter_count") == PARAMETER_COUNT and
                last.get("provenance") == checkpoint.get("provenance") and last.get("history") == history and
                last.get("data_signature") == checkpoint.get("data_signature") and
                last.get("updates") == 50080 and last.get("sam_forward_passes") == 100160 and
                last.get("skipped_updates") == 0 and "grad_scaler" not in last,
                "SE final checkpoint ledger differs from complete training")
        require(all(key in last for key in ("model", "ema", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")),
                "SE final state lacks exact model/EMA/SGD/RNG")
        require(isinstance(last["torch_rng"], torch.Tensor) and last["torch_rng"].dtype == torch.uint8 and
                isinstance(last["cuda_rng"], list) and (expected["precision"] != "bf16" or bool(last["cuda_rng"])),
                "SE final state RNG is incomplete")
        groups = last["optimizer"].get("param_groups", [])
        require(len(groups) == 1 and groups[0].get("momentum") == .9 and groups[0].get("weight_decay") == .0005 and
                groups[0].get("nesterov") is True and bool(groups[0].get("params")), "SE final SGD state differs")
        validate_saved_state(last, config)
    return {"schema": "research_se_sam_integrity_v1", "training": expected,
            "successful_updates": 50080, "skipped_updates": 0, "sam_forward_passes": 100160,
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def dual_gate(reports, baseline, sam_reference, signature, batch_size, context):
    require(set(reports) == set(core.POLICIES), "SE requires all three fixed inference policies")
    core.validate_baseline(baseline, reports, signature, batch_size, context, core.sha256(ROOT/"research/benchmark.py"))
    require(sam_reference.get("schema") == core.SCHEMA and sam_reference.get("status") == "complete" and
            sam_reference.get("examples") == 5000 and sam_reference.get("training_examples") == 40000 and
            sam_reference.get("checkpoint_sha256") == SAM_CHECKPOINT_SHA256 and
            sam_reference.get("parameter_count") == 4298970 and sam_reference.get("bn_recalibrated") is False and
            sam_reference.get("batch_size") == batch_size and sam_reference.get("data_signature") == signature and
            sam_reference.get("evaluation_context") == context and sam_reference.get("stress_cases") == list(core.CASES) and
            sam_reference.get("benchmark_source_sha256") == core.sha256(ROOT/"research/benchmark.py") and
            set(sam_reference.get("policies", {})) == set(core.POLICIES),
            "Fixed SAM reference must use the complete identical v3 context and source")
    for policy in core.POLICIES:
        for case in core.CASES:
            reference = sam_reference["policies"][policy].get("cases", {}).get(case, {})
            target = reports[policy].get("cases", {}).get(case, {})
            require(reference.get("examples") == target.get("examples") == 5000 and
                    reference.get("input_pixel_sha256") == target.get("input_pixel_sha256") and
                    isinstance(reference.get("input_pixel_sha256"), str) and len(reference["input_pixel_sha256"]) == 64,
                    "SE and fixed SAM cases must contain the same complete5000 pixels")
    original = {policy:core.promotion_gate(report, baseline["policies"]["flip"]) for policy, report in reports.items()}
    incremental = {policy:core.promotion_gate(report, sam_reference["policies"][POLICY]) for policy, report in reports.items()}
    return {"policy": POLICY, "passed": original[POLICY]["passed"] and incremental[POLICY]["passed"],
            "against_original": original, "against_fixed_sam": incremental,
            "alternative_policies_promotable": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("SE benchmark evidence is immutable")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError("SE evaluation requires the same native BF16 local context")
    seed_all(2026)
    started = time.perf_counter()
    folder = args.checkpoint.parent
    paths = [args.checkpoint, *(folder/name for name in ("metrics.json", "history.json", "provenance.json", "environment.json", "input_config.json", "last.pt")),
             args.baseline, SAM_CHECKPOINT, SAM_BENCHMARK, ROOT/"results/research/wrn22_sam_delivery/frozen.json",
             *(SAM_CHECKPOINT.parent/name for name in ("metrics.json", "history.json", "provenance.json", "environment.json"))]
    hashes = {str(path.resolve()):core.sha256(path) for path in paths}
    require(core.sha256(SAM_CHECKPOINT) == SAM_CHECKPOINT_SHA256 and core.sha256(SAM_BENCHMARK) == SAM_BENCHMARK_SHA256 and
            core.sha256(args.baseline) == BASELINE_BENCHMARK_SHA256 and
            core.sha256(ROOT/"results/research/wrn22_sam_delivery/frozen.json") == SAM_FREEZE_SHA256,
            "The predeclared fixed SAM checkpoint or benchmark changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics, history, provenance, environment = [json.loads((folder/name).read_text()) for name in
        ("metrics.json", "history.json", "provenance.json", "environment.json")]
    last = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
    integrity = validate_training(checkpoint, metrics, history, provenance, environment, (folder/"input_config.json").read_bytes(), last)
    del last
    images, labels, train_ids, val_ids = load_training_arrays(args.data_dir)
    manifest = json.loads((args.data_dir/"training_manifest.json").read_text())
    signature = {"archive_sha256": training_source(args.data_dir)["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    core_source = code_provenance()["source_sha256"]
    core.validate_checkpoint(checkpoint, train_ids, val_ids, signature, metrics, core_source, history)
    # Deeply audit the actual fixed SAM training artifacts, not just its scalar
    # benchmark or the constants printed in the experiment plan.
    sam = torch.load(SAM_CHECKPOINT, map_location="cpu", weights_only=False)
    sm, sh, sp, se = [json.loads((SAM_CHECKPOINT.parent/name).read_text()) for name in
        ("metrics.json", "history.json", "provenance.json", "environment.json")]
    validate_sam(sam, sm, sh, sp, se)
    core.validate_checkpoint(sam, train_ids, val_ids, signature, sm, core_source, sh)
    del sam
    model = build_model(checkpoint["model_config"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"], strict=True)
    require(count_parameters(model) == count_parameters(model, True) == PARAMETER_COUNT, "SE actual parameter count differs")
    model.eval()
    val_images, val_labels = tensor_images(images[val_ids], device), torch.as_tensor(labels[val_ids], device=device)
    del images, labels
    reports = {policy:core.score(model, val_images, val_labels, device, policy, 128) for policy in core.POLICIES}
    context = {"device_type": device.type, "amp_dtype": str(core.benchmark_dtype(device)),
               "torch_version": torch.__version__, "training_core_sha256": core_source}
    baseline, sam_reference = json.loads(args.baseline.read_text()), json.loads(SAM_BENCHMARK.read_text())
    gates = dual_gate(reports, baseline, sam_reference, signature, 128, context)
    require(all(core.sha256(path) == digest for path, digest in hashes.items()), "SE or fixed reference artifacts changed during evaluation")
    result = {"schema": core.SCHEMA, "status": "complete", "examples": 5000, "training_examples": 40000,
        "parameter_count": PARAMETER_COUNT, "checkpoint_sha256": core.sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_variant": checkpoint["variant"],
        "model_config": checkpoint["model_config"], "config": checkpoint["config"], "data_signature": signature,
        "stress_cases": list(core.CASES), "batch_size": 128, "evaluation_context": context, "bn_recalibrated": False,
        "policies": reports, "promotion": gates["against_original"], "incremental_promotion": gates["against_fixed_sam"],
        "fixed_primary_policy": POLICY, "dual_promotion": gates, "baseline_sha256": core.sha256(args.baseline),
        "fixed_sam_checkpoint_sha256": SAM_CHECKPOINT_SHA256, "fixed_sam_benchmark_sha256": SAM_BENCHMARK_SHA256,
        "training_metrics_sha256": core.sha256(folder/"metrics.json"), "training_history_sha256": core.sha256(folder/"history.json"),
        "elapsed_seconds": time.perf_counter()-started, "source": code_provenance(),
        "benchmark_source_sha256": core.sha256(ROOT/"research/benchmark.py"),
        "se_benchmark_source_sha256": core.sha256(Path(__file__)),
        "research_training_integrity": integrity, "evidence_sha256": hashes,
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
        "interpretation": "Exploratory reused development and synthetic stress only; fixed weighted policy must pass both references. No hidden-score guarantee."}
    write_json(args.output, result)
    print(json.dumps({"event": "se_sam_benchmark_complete", "dual_gate_passed": gates["passed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__": main()
