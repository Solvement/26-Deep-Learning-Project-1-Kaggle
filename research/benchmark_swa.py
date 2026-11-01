"""Audit the fixed epoch-200 SWA recipe and score it with the unchanged v3 suite.

SWA is selected before tail outcomes. The raw-tail endpoint is a control only;
neither variant is interpreted as a legacy best-development checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import seed_all, write_json
import numpy as np
import torch
from data import load_training_arrays
from model import build_model, count_parameters
from train import code_provenance, tensor_images, training_source
from research import benchmark as suite

FIXED_POLICY = "center80_shift20"
PARENT_BENCHMARK_SHA = "cf8cfa9fbb103db3ad705a5971b993d187e055dff6e6d0a0a3c1ac5964168690"
PARENT_CHECKPOINT_SHA = "1787237fb5f4566043843f9008516e911b9a6d867bc6ac4720c52448871c7b9a"


def candidate_eligible(variant, regularized_gates, parent_gates):
    return (variant == "swa" and regularized_gates[FIXED_POLICY]["passed"] is True and
        parent_gates[FIXED_POLICY]["passed"] is True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_rng(last, cuda):
    try:
        random.Random().setstate(last["python_rng"])
        np.random.RandomState().set_state(last["numpy_rng"])
        torch.Generator(device="cpu").set_state(last["torch_rng"])
        states = last["cuda_rng"]
        require(isinstance(states, list) and (not cuda or bool(states)) and
            all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.dtype == torch.uint8 and v.numel() > 0
                for v in states), "SWA CUDA RNG state is incomplete")
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("SWA final RNG state is incomplete") from error


def validate_bn(proof, precision, helper_sha, parameter_count, cuda):
    expected = {"schema": "research_swa_bn_recompute_v1", "training_examples": 40000,
        "batch_size": 128, "batches": 313, "last_batch_examples": 64, "seed": 2026, "precision": precision,
        "shuffle": False, "augmentation": "frozen_crop_flip_cutout0", "mixup": False, "cutout": 0,
        "momentum": "sample_weighted_batch_size_over_seen", "parameters_unchanged": True,
        "training_modes_restored": True, "momenta_restored": True, "rng_restored": True,
        "helper_source_sha256": helper_sha, "grad_enabled": False, "parameter_count": parameter_count,
        "device_type": "cuda" if cuda else "cpu"}
    require(isinstance(proof, dict) and all(proof.get(k) == v for k, v in expected.items()) and
        isinstance(proof.get("augmented_input_sha256"), str) and len(proof["augmented_input_sha256"]) == 64,
        "SWA BN proof must use only 40k augmented training images with restored RNG/modes")


def validate_swa_training(checkpoint, metrics, history, provenance, environment, last, parent_metadata=None, prefix_history=None):
    from research import train_swa
    config = checkpoint["config"]
    train_swa.validate_config(config)
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    precision = marker.get("precision")
    require(precision in {"bf16", "fp32"}, "SWA training precision must be explicit")
    expected = train_swa.research_provenance(precision, config)
    core = code_provenance()["source_sha256"]
    require(marker == expected and metrics.get("research_training") == expected and
        provenance.get("code", {}).get("research_training") == expected and
        checkpoint.get("provenance", {}).get("source_sha256") == core and
        provenance.get("code", {}).get("source_sha256") == core and
        environment.get("swa_training_precision") == precision and environment.get("grad_scaler_enabled") is False and
        environment.get("research_training") == expected, "SWA source/precision provenance differs")
    variant = checkpoint.get("variant")
    selection = {"swa": "fixed_swa_final", "raw_tail_control": "fixed_raw_tail_control"}.get(variant)
    require(checkpoint.get("schema") == "research_swa_checkpoint_v1" and selection is not None and
        checkpoint.get("epoch") == 200 and checkpoint.get("selection") == selection and
        checkpoint.get("no_best_selection") is True and metrics.get("selection") == "fixed_swa_final" and
        metrics.get("no_best_selection") is True and not any(k.startswith("best_") for k in metrics) and
        "validation_accuracy" not in checkpoint and "validation_loss" not in checkpoint,
        "SWA requires fixed epoch-200 selection, never a legacy best epoch")
    require(metrics.get("status") == "complete" and metrics.get("config") == config and
        metrics.get("epochs_completed") == 200 and metrics.get("prefix_epochs") == 160 and metrics.get("tail_epochs") == 40 and
        metrics.get("training_examples") == provenance.get("train_examples") == 40000 and
        metrics.get("validation_examples") == provenance.get("validation_examples") == 5000 and
        metrics.get("ema_tracked") is True and metrics.get("ema_selected") is False and
        metrics.get("confirmation_evaluated") is False and metrics.get("hidden_images_loaded") is False and
        metrics.get("external_data_loaded") is False and
        checkpoint.get("data_signature") == provenance.get("data_signature"), "SWA completion/data counts differ")
    parent = checkpoint.get("parent_training")
    require(isinstance(parent, dict) and bool(parent) and (parent_metadata is None or parent == parent_metadata) and
        metrics.get("parent_training") == parent and provenance.get("parent_training") == parent and
        last.get("parent_training") == parent, "SWA parent prefix evidence differs")
    require(isinstance(history, dict) and set(history) == {"prefix", "tail"} and
        [row.get("epoch") for row in history["prefix"]] == list(range(1, 161)) and
        (prefix_history is None or history["prefix"] == prefix_history) and len(history["tail"]) == 40,
        "SWA requires complete original prefix and fixed 40-epoch tail histories")
    steps = math.ceil(40000/config["batch_size"])
    for index, row in enumerate(history["tail"], 1):
        require(row.get("epoch") == 160+index and row.get("lr") == .01 and
            type(row.get("successful_updates")) is int and row["successful_updates"] == steps and
            row.get("skipped_updates") == 0 and row.get("swa_count") == index and
            all(row.get(key) is None for key in ("validation_accuracy", "validation_loss", "ema_validation_accuracy", "ema_validation_loss")) and
            finite(row.get("train_loss")) and finite(row.get("train_accuracy")) and 0 <= row["train_accuracy"] <= 1,
            "SWA tail update/average/LR/no-selection history differs")
    counts = {"swa_count": 40, "prefix_updates": 160*steps, "tail_updates": 40*steps,
        "successful_updates": 200*steps, "skipped_updates": 0}
    require(all(checkpoint.get(k) == v and metrics.get(k) == v for k, v in counts.items()),
        "SWA actual update and averaging totals differ")
    require(last.get("schema") == "research_swa_last_v1" and last.get("epoch") == 200 and
        last.get("config") == config and last.get("data_signature") == checkpoint["data_signature"] and
        last.get("provenance") == checkpoint["provenance"] and last.get("history") == history and
        last.get("updates") == counts["successful_updates"] and last.get("tail_updates") == counts["tail_updates"] and
        last.get("swa_count") == 40, "SWA final trajectory state is incomplete")
    model = build_model(checkpoint["model_config"])
    require(checkpoint["model_config"] == {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")},
        "SWA model configuration differs from its frozen recipe")
    parameters = dict(model.named_parameters())
    count = count_parameters(model)
    require(count == count_parameters(model, True) == checkpoint.get("parameter_count") == metrics.get("parameter_count")
        == provenance.get("parameter_count") and 0 < count <= 5000000, "SWA single-model parameter budget differs")
    for state in (checkpoint["model"], last["model"], last["swa_model"], last["ema"]):
        model.load_state_dict(state, strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()),
            "SWA state contains nonfinite parameters/buffers")
    trajectory = last["swa_model" if variant == "swa" else "model"]
    require(all(torch.equal(checkpoint["model"][key], trajectory[key]) for key in parameters),
        "SWA checkpoint parameters differ from their fixed averaging/raw trajectory")
    optimizer = last.get("optimizer", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    require(len(groups) == 1 and groups[0].get("lr") == .01 and groups[0].get("momentum") == config["momentum"] and
        groups[0].get("nesterov") is True and groups[0].get("dampening") == 0 and
        groups[0].get("weight_decay") == config["weight_decay"] and len(groups[0].get("params", [])) == len(parameters) and
        len(set(groups[0]["params"])) == len(parameters) and set(states) == set(groups[0]["params"]),
        "SWA final SGD momentum state is incomplete")
    for key, parameter in zip(groups[0]["params"], parameters.values()):
        momentum = states[key].get("momentum_buffer")
        require(isinstance(momentum, torch.Tensor) and momentum.shape == parameter.shape and torch.isfinite(momentum).all().item(),
            "SWA final SGD momentum is invalid")
    validate_rng(last, environment.get("device") == "cuda")
    for name in ("swa", "raw_tail_control"):
        final = metrics.get("final_validation", {}).get(name, {})
        require(final.get("examples") == 5000 and finite(final.get("accuracy")) and 0 <= final["accuracy"] <= 1 and
            finite(final.get("loss")) and final["loss"] >= 0, "SWA fixed-final validation record is incomplete")
        validate_bn(metrics.get("bn_recomputation", {}).get(name), precision, expected["source_sha256"]["research/swa.py"],
            count, environment.get("device") == "cuda")
    require(checkpoint.get("final_validation") == metrics["final_validation"][variant] and
        checkpoint.get("bn_recomputation") == metrics["bn_recomputation"][variant] and
        metrics["bn_recomputation"]["swa"]["augmented_input_sha256"] == metrics["bn_recomputation"]["raw_tail_control"]["augmented_input_sha256"],
        "SWA/control final metrics or shared BN augmentation pixels differ")
    return {"schema": "research_swa_integrity_v1", "training": expected, "parent_training": parent,
        "selection": selection, "no_best_selection": True, "epoch": 200, **counts,
        "bn_recomputation": checkpoint["bn_recomputation"], "training_examples": 40000, "validation_examples": 5000,
        "raw_control_prediction_authorized": False, "confirmation_evaluated": False, "hidden_images_loaded": False}


def validate_run(checkpoint_path):
    from research import train_swa
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    folder = checkpoint_path.parent
    documents = {name: json.loads((folder/(name+".json")).read_text()) for name in
        ("config", "metrics", "history", "provenance", "environment")}
    require(documents["config"] == checkpoint["config"], "Saved SWA configuration differs")
    parent = train_swa.validate_parent(Path(checkpoint["parent_training"]["checkpoint_path"]), checkpoint["config"], checkpoint["data_signature"])
    last = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
    integrity = validate_swa_training(checkpoint, documents["metrics"], documents["history"], documents["provenance"],
        documents["environment"], last, parent["metadata"], parent["prefix_history"])
    paths = [checkpoint_path, folder/"last.pt", *(folder/(name+".json") for name in documents), *parent["paths"]]
    paths += [ROOT/name for name in integrity["training"]["source_sha256"]]
    return checkpoint, integrity, {str(Path(path).resolve()): suite.sha256(path) for path in paths}


def validate_parent_benchmark(signature, context, train_ids, val_ids, reports):
    from research import paired_compare
    folder = ROOT/"results/research/wrn34_3_s2026"
    checkpoint, path = folder/"best.pt", folder/"benchmark.json"
    require(suite.sha256(checkpoint) == PARENT_CHECKPOINT_SHA and suite.sha256(path) == PARENT_BENCHMARK_SHA,
        "The predeclared WRN34 seed2026 parent benchmark/checkpoint changed")
    saved, document = paired_compare.load_experiment(checkpoint, path, signature, context,
        code_provenance()["source_sha256"], train_ids, val_ids)
    provenance = json.loads((folder/"provenance.json").read_text())
    environment = json.loads((folder/"environment.json").read_text())
    require(json.loads((folder/"config.json").read_text()) == saved["config"] and
        provenance.get("code", {}).get("source_sha256") == context["training_core_sha256"] and
        provenance.get("data_signature") == signature and provenance.get("train_examples") == 40000 and
        provenance.get("validation_examples") == 5000 and environment.get("device") == context["device_type"] and
        environment.get("amp_dtype") == context["amp_dtype"] and environment.get("packages", {}).get("torch") == context["torch_version"],
        "The fixed parent's source/data/native evaluation context differs")
    require(saved["config"]["seed"] == 2026 and set(document["policies"]) == set(suite.POLICIES) and
        document["policies"][FIXED_POLICY]["clean_accuracy"] == .9632 and
        document["policies"][FIXED_POLICY]["stress_accuracy"] == .78592,
        "The fixed parent weighted-policy reference differs")
    for report in reports.values():
        require(all(report["cases"][case]["examples"] == 5000 and report["cases"][case]["input_pixel_sha256"] ==
            document["policies"][FIXED_POLICY]["cases"][case]["input_pixel_sha256"] for case in suite.CASES),
            "SWA and its fixed parent must score identical development pixels")
    paths = [checkpoint, path, *(folder/name for name in ("config.json", "metrics.json", "history.json", "provenance.json", "environment.json")),
        ROOT/"research/paired_compare.py"]
    return document, {str(path.resolve()): suite.sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("SWA benchmark evidence cannot be overwritten")
    if "stability_seed2027" in args.checkpoint.parts:
        raise ValueError("The original validation-only stability repeat is excluded")
    checkpoint, integrity, hashes = validate_run(args.checkpoint)
    images, labels, train_ids, val_ids = load_training_arrays(args.data_dir)
    manifest = json.loads((args.data_dir/"training_manifest.json").read_text())
    signature = {"archive_sha256": training_source(args.data_dir)["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(checkpoint["data_signature"] == signature and len(train_ids) == len(set(train_ids.tolist())) == 40000 and
        len(val_ids) == len(set(val_ids.tolist())) == 5000 and not (set(train_ids.tolist()) & set(val_ids.tolist())),
        "SWA benchmark requires the exact disjoint official 40k/5k split")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_all(2026)
    started, device = time.perf_counter(), torch.device(args.device)
    model = build_model(checkpoint["model_config"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    val_images, val_labels = tensor_images(images[val_ids], device), torch.as_tensor(labels[val_ids], device=device)
    reports = {policy: suite.score(model, val_images, val_labels, device, policy, 128) for policy in suite.POLICIES}
    core = code_provenance()
    context = {"device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)), "torch_version": torch.__version__,
        "training_core_sha256": core["source_sha256"]}
    baseline = json.loads(args.baseline.read_text())
    suite.validate_baseline(baseline, reports, signature, 128, context, suite.sha256(Path(suite.__file__)))
    parent_document, parent_hashes = validate_parent_benchmark(signature, context, train_ids, val_ids, reports)
    hashes.update(parent_hashes)
    require(all(suite.sha256(path) == digest for path, digest in hashes.items()), "SWA evidence changed during scoring")
    gates = {policy: suite.promotion_gate(report, baseline["policies"]["flip"]) for policy, report in reports.items()}
    parent_gates = {policy: suite.promotion_gate(report, parent_document["policies"][policy]) for policy, report in reports.items()}
    result = {"schema": suite.SCHEMA, "status": "complete", "examples": 5000, "training_examples": 40000,
        "parameter_count": count_parameters(model), "checkpoint_sha256": suite.sha256(args.checkpoint),
        "checkpoint_epoch": 200, "checkpoint_variant": checkpoint["variant"], "model_config": checkpoint["model_config"],
        "config": checkpoint["config"], "data_signature": signature, "stress_cases": list(suite.CASES), "batch_size": 128,
        "evaluation_context": context, "bn_recalibrated": True, "bn_recalibration_stage": "fixed_swa_training_finalization",
        "policies": reports, "promotion": gates, "promotion_vs_parent": parent_gates,
        "fixed_primary_policy": FIXED_POLICY, "parent_benchmark_sha256": PARENT_BENCHMARK_SHA,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "candidate_eligible": candidate_eligible(checkpoint["variant"], gates, parent_gates),
        "baseline_sha256": suite.sha256(args.baseline), "training_metrics_sha256": suite.sha256(args.checkpoint.parent/"metrics.json"),
        "training_history_sha256": suite.sha256(args.checkpoint.parent/"history.json"), "source": core,
        "benchmark_source_sha256": suite.sha256(Path(suite.__file__)), "swa_benchmark_source_sha256": suite.sha256(Path(__file__)),
        "research_swa_integrity": {**integrity, "validator_source_sha256": suite.sha256(Path(__file__)), "evidence_sha256": hashes},
        "elapsed_seconds": time.perf_counter()-started, "confirmation_evaluated": False, "hidden_images_loaded": False,
        "external_data_loaded": False, "raw_control_prediction_authorized": False,
        "interpretation": "Fixed SWA epoch200 or raw-tail control; reused development proxies, no original confirmation or hidden-score guarantee."}
    write_json(args.output, result)
    print(json.dumps({"event": "swa_benchmark_complete", "variant": checkpoint["variant"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
