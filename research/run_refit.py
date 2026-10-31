"""Prepare, then explicitly supervise one scratch 50k refit of a qualified WRN.

The frozen 40k primary/seed-repeat delivery authorizes the recipe and policy.
Full refits never select a development checkpoint, evaluate confirmation, load
hidden images, initialize from parent weights or perform a Kaggle upload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import write_json
import psutil
import torch
from model import build_model, count_parameters
from train import DEFAULT_CONFIG, code_provenance
from research import delivery
from research.run_local import acquire_lease, now, sha256, supervised_command, used_seconds

SCHEMA = "research_full_refit_plan_v1"
SOURCE_FILES = ("train.py", "model.py", "data.py", "runtime.py", "research/run_refit.py",
                "research/run_local.py", "research/watchdog.py", "research/delivery.py")
VALIDATION_KEYS = ("validation_loss", "validation_accuracy", "ema_validation_loss", "ema_validation_accuracy")
ARTIFACT_NAMES = ("best.pt", "last.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json")


def derive_refit_config(frozen):
    """Pure derivation: change only data coverage and the preselected variant."""
    config = dict(frozen["config"])
    delivery.require(set(config) == set(DEFAULT_CONFIG), "Only original core-compatible WRN recipes support refit")
    delivery.require(config.get("train_all") is False and type(config.get("seed")) is int and
        config["seed"] == 2026 and type(config.get("epochs")) is int and config["epochs"] > 0 and
        config.get("split_seed") == 2026 and config.get("num_classes") == 10,
        "Refit requires the complete qualified core seed-2026 recipe")
    delivery.require(frozen.get("weight_variant") in {"model", "ema"}, "Parent weight variant is invalid")
    config.update(train_all=True, final_variant=frozen["weight_variant"])
    return config


def load_parent(parent_manifest_path):
    parent_manifest_path = Path(parent_manifest_path).resolve()
    summary = delivery.validate_delivery(parent_manifest_path, require_prediction=True)
    parent = delivery.read(parent_manifest_path)
    delivery.require(parent.get("schema") == delivery.PREDICTION_SCHEMA and summary["training_examples"] == 40000,
                     "Parent must be the completed qualified 40k primary delivery")
    frozen = delivery.read(parent["frozen_manifest_path"])
    config = derive_refit_config(frozen)
    checkpoint = torch.load(frozen["checkpoint_path"], map_location="cpu", weights_only=False)
    delivery.require("research_training" not in checkpoint.get("provenance", {}),
                     "SAM, CutMix, ResizeMix and other independent trainers are unsupported for this refit")
    delivery.require(frozen["evaluation_context"]["device_type"] == "cuda" and
        frozen["evaluation_context"]["amp_dtype"] == "torch.bfloat16", "Frozen core refit requires the native BF16 parent protocol")
    return summary, parent, frozen, config


def prepare(run, parent_manifest_path):
    delivery.require(run.replace("_", "").isalnum() and not run.startswith("stability"), "Use a unique exploratory refit ID")
    deadline = datetime.fromisoformat(delivery.read(ROOT/"research/campaign.json")["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    summary, parent, frozen, config = load_parent(parent_manifest_path)
    folder = ROOT/"results/research"/run
    folder.mkdir(parents=True, exist_ok=False)
    config_path, plan_path = folder/"input_config.json", folder/"refit_plan.json"
    delivery.exclusive_json(config_path, config)
    value = {"schema": SCHEMA, "status": "prepared", "run": run, "prepared_at": now().isoformat(),
        "parent_delivery_path": str(Path(parent_manifest_path).resolve()), "parent_delivery_sha256": summary["manifest_sha256"],
        "frozen_manifest_path": parent["frozen_manifest_path"], "frozen_manifest_sha256": summary["frozen_manifest_sha256"],
        "parent_checkpoint_sha256": summary["checkpoint_sha256"], "policy": summary["policy"],
        "config": config, "input_config_path": str(config_path), "input_config_sha256": sha256(config_path),
        "core_source_sha256": code_provenance()["source_sha256"],
        "source_sha256": {name: sha256(ROOT/name) for name in SOURCE_FILES},
        "parameter_count": summary["parameter_count"], "data_signature": frozen["data_signature"],
        "data_dir": frozen["data_dir"], "evaluation_context": frozen["evaluation_context"],
        "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "training_examples": 50000, "validation_examples": 0, "confirmation_evaluated": False,
        "hidden_images_loaded": False, "overlapping_development_benchmark": False}
    delivery.exclusive_json(plan_path, value)
    return plan_path


def validate_plan(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = delivery.read(plan_path)
    delivery.require(plan.get("schema") == SCHEMA and plan.get("status") == "prepared" and
        plan.get("run") == plan_path.parent.name and plan.get("initialization") == "scratch_seeded" and
        plan.get("parent_weights_loaded") is False and plan.get("confirmation_evaluated") is False and
        plan.get("hidden_images_loaded") is False and plan.get("overlapping_development_benchmark") is False,
        "A valid immutable scratch-refit plan is required")
    summary, parent, frozen, config = load_parent(plan["parent_delivery_path"])
    expected = {"parent_delivery_sha256": summary["manifest_sha256"], "parent_checkpoint_sha256": summary["checkpoint_sha256"],
        "frozen_manifest_path": parent["frozen_manifest_path"], "frozen_manifest_sha256": summary["frozen_manifest_sha256"],
        "policy": summary["policy"], "config": config, "parameter_count": summary["parameter_count"],
        "data_signature": frozen["data_signature"], "data_dir": frozen["data_dir"],
        "evaluation_context": frozen["evaluation_context"], "training_examples": 50000, "validation_examples": 0}
    delivery.require(all(plan.get(k) == v for k,v in expected.items()), "Refit plan differs from the immutable approved parent chain")
    config_path = plan_path.parent/"input_config.json"
    delivery.require(Path(plan["input_config_path"]).resolve() == config_path and
        sha256(config_path) == plan.get("input_config_sha256") and delivery.read(config_path) == config,
        "Immutable refit input configuration changed")
    delivery.require(plan.get("core_source_sha256") == code_provenance()["source_sha256"] and
        plan.get("source_sha256") == {name: sha256(ROOT/name) for name in SOURCE_FILES}, "Refit core or supervisor source changed")
    delivery.require(delivery.stamp(plan["prepared_at"]) <= now(), "Refit preparation timestamp is invalid")
    return plan


def validate_records(plan, checkpoint, last, metrics, history, provenance, environment):
    """Audit full-data evidence without creating a development accuracy claim."""
    config, core, signature = plan["config"], plan["core_source_sha256"], plan["data_signature"]
    epochs, variant = config["epochs"], config["final_variant"]
    delivery.require(metrics.get("status") == "complete" and metrics.get("epochs_completed") == epochs and
        metrics.get("config") == config and metrics.get("training_examples") == 50000 and metrics.get("validation_examples") == 0 and
        all(key in metrics and metrics[key] is None for key in ("best_validation_accuracy", "best_validation_loss")),
        "Full refit must complete 50k training with zero validation and no development score")
    delivery.require(len(history) == epochs and [row.get("epoch") for row in history] == list(range(1, epochs+1)),
                     "Full refit history is incomplete")
    for row in history:
        delivery.require(row.get("selected_variant") == variant and all(key in row and row[key] is None for key in VALIDATION_KEYS),
                         "Full refit history contains development evaluation or a changed variant")
        delivery.require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for key in
                             ("train_loss", "train_accuracy", "lr", "elapsed_seconds")), "Nonfinite full-refit training history")
    expected_model = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    delivery.require(checkpoint.get("config") == config and checkpoint.get("model_config") == expected_model and
        checkpoint.get("epoch") == epochs and checkpoint.get("variant") == variant and
        all(key in checkpoint and checkpoint[key] is None for key in ("validation_accuracy", "validation_loss")),
        "Full refit checkpoint must be the fixed final epoch/variant with no validation score")
    for record in (checkpoint, last):
        delivery.require(record.get("data_signature") == signature and record.get("config") == config and
            record.get("provenance", {}).get("source_sha256") == core and
            "research_training" not in record.get("provenance", {}), "Full refit checkpoint source/configuration differs")
    delivery.require(last.get("epoch") == epochs and last.get("history") == history and
        last.get("updates") == math.ceil(50000/config["batch_size"])*epochs and
        last.get("best_accuracy") is None and last.get("best_loss") is None,
        "Final resumable refit state differs from completed history")
    final_state = last.get("ema" if variant == "ema" else "model", {})
    delivery.require(set(final_state) == set(checkpoint["model"]) and
        all(torch.equal(value, final_state[key]) for key,value in checkpoint["model"].items()),
        "Fixed final checkpoint weights differ from the final resumable state")
    delivery.require(provenance.get("code", {}).get("source_sha256") == core and
        "research_training" not in provenance.get("code", {}) and provenance.get("data_signature") == signature and
        provenance.get("train_examples") == 50000 and provenance.get("validation_examples") == 0,
        "Full refit training provenance/counts differ")
    delivery.require(environment.get("device") == "cuda" and environment.get("amp_dtype") == "torch.bfloat16" and
        environment.get("packages", {}).get("torch") == plan["evaluation_context"]["torch_version"],
        "Full refit training backend/precision differs from the approved parent protocol")
    model = build_model(expected_model)
    model.load_state_dict(checkpoint["model"], strict=True)
    count = count_parameters(model)
    delivery.require(0 < count <= 5000000 and count_parameters(model, True) == count == checkpoint.get("parameter_count")
        == metrics.get("parameter_count") == provenance.get("parameter_count") == plan["parameter_count"], "Full refit parameter budget differs")
    delivery.require(all(torch.isfinite(value).all().item() for value in model.state_dict().values() if value.is_floating_point()),
                     "Full refit checkpoint contains nonfinite weights")


def audit_artifacts(plan_path, plan):
    folder = Path(plan_path).parent
    delivery.require(not (folder/"benchmark.json").exists(), "Full refit cannot claim an overlapping development benchmark")
    checkpoint, last = [torch.load(folder/name, map_location="cpu", weights_only=False) for name in ("best.pt", "last.pt")]
    metrics, history, provenance, environment = [delivery.read(folder/name) for name in
        ("metrics.json", "history.json", "provenance.json", "environment.json")]
    delivery.require(delivery.read(folder/"config.json") == plan["config"], "Training output configuration differs")
    validate_records(plan, checkpoint, last, metrics, history, provenance, environment)
    return {"run": plan["run"], "config": plan["config"], "policy": plan["policy"],
        "plan_path": str(Path(plan_path).resolve()), "plan_sha256": sha256(plan_path),
        "parent_manifest_path": plan["parent_delivery_path"], "parent_manifest_sha256": plan["parent_delivery_sha256"],
        "parent_delivery_path": plan["parent_delivery_path"], "parent_delivery_sha256": plan["parent_delivery_sha256"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"], "checkpoint_path": str((folder/"best.pt").resolve()),
        "checkpoint_sha256": sha256(folder/"best.pt"), "parameter_count": plan["parameter_count"],
        "data_signature": plan["data_signature"], "source_sha256": plan["core_source_sha256"],
        "training_examples": 50000, "validation_examples": 0}


def validate_refit(plan_path):
    plan = validate_plan(plan_path)
    receipt = delivery.read(Path(plan_path).parent/"supervisor.json")
    delivery.require(receipt.get("status") == "complete" and receipt.get("role") == "full_refit" and
        receipt.get("backend") == "local" and receipt.get("trainer") == "frozen" and receipt.get("precision") == "bf16" and
        receipt.get("plan_sha256") == sha256(plan_path) and receipt.get("config_sha256") == plan["input_config_sha256"] and
        receipt.get("parent_delivery_sha256") == plan["parent_delivery_sha256"] and
        receipt.get("parent_delivery_path") == plan["parent_delivery_path"] and
        receipt.get("parent_checkpoint_sha256") == plan["parent_checkpoint_sha256"] and receipt.get("policy") == plan["policy"] and
        receipt.get("source_sha256") == plan["source_sha256"] and receipt.get("initialization") == "scratch_seeded" and
        receipt.get("parent_weights_loaded") is False and receipt.get("confirmation_evaluated") is False and
        receipt.get("hidden_images_loaded") is False and receipt.get("submitted") is False and
        receipt.get("training_examples") == 50000 and receipt.get("validation_examples") == 0,
        "A complete source-bound scratch-refit supervisor receipt is required")
    delivery.require(all(type(receipt.get(key)) is int and receipt[key] > 0 for key in ("pid", "child_pid", "watchdog_pid")) and
        all(type(receipt.get(key)) in (float, int) and math.isfinite(receipt[key]) and receipt[key] > 0
            for key in ("process_created_at", "child_created_at")), "Actual supervised process identity evidence is missing")
    delivery.require(delivery.stamp(plan["prepared_at"]) <= delivery.stamp(receipt["started_at"]) <=
        delivery.stamp(receipt["finished_at"]) <= now(), "Refit plan must predate the actual bounded training attempt")
    result = audit_artifacts(plan_path, plan)
    delivery.require(receipt.get("checkpoint_sha256") == result["checkpoint_sha256"] and
        receipt.get("artifact_sha256") == {name: sha256(Path(plan_path).parent/name) for name in ARTIFACT_NAMES},
        "Completed supervisor artifact hashes differ from the full refit")
    return result


def launch(plan_path, minimum_seconds=60):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path)
    policy = delivery.read(ROOT/"research/campaign.json")
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    folder, base = plan_path.parent, plan_path.parent.parent
    receipt_path = folder/"supervisor.json"
    if receipt_path.exists() or any((folder/name).exists() for name in ("best.pt", "last.pt", "metrics.json", "history.json")):
        raise FileExistsError("Existing full-refit attempt is immutable; no implicit restart or resume")
    acquire_lease(base/"local_lease.json")
    started = time.perf_counter()
    receipt = {"run": plan["run"], "status": "starting", "role": "full_refit", "backend": "local", "trainer": "frozen",
        "precision": "bf16", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat(),
        "plan_sha256": sha256(plan_path), "config_sha256": plan["input_config_sha256"], "source_sha256": plan["source_sha256"],
        "parent_delivery_path": plan["parent_delivery_path"], "parent_delivery_sha256": plan["parent_delivery_sha256"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"], "policy": plan["policy"], "initialization": "scratch_seeded",
        "parent_weights_loaded": False, "confirmation_evaluated": False, "hidden_images_loaded": False, "submitted": False}
    try:
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=receipt_path)
        budget = min(policy["local_job_timeout_seconds"], policy["local_gpu_seconds_per_utc_day"]-used, int((deadline-now()).total_seconds()))
        if budget < max(60, minimum_seconds):
            raise RuntimeError("Insufficient local budget for the requested full-refit duration")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Original frozen training requires native CUDA BF16")
        delivery.require(torch.__version__ == plan["evaluation_context"]["torch_version"], "Refit torch runtime differs from parent")
        receipt.update(status="training", reserved_seconds=budget, utc_budget_day=day.date().isoformat())
        write_json(receipt_path, receipt)
        command = [sys.executable, "train.py", "--config", str(folder/"input_config.json"), "--output-dir", str(folder),
                   "--data-dir", plan["data_dir"], "--device", "cuda"]
        remaining = budget-(time.perf_counter()-started)-30
        if remaining < 1:
            raise TimeoutError("Full-refit setup exhausted the available runtime")
        supervised_command(command, folder/"training.log", remaining, receipt_path, receipt)
        validate_plan(plan_path)
        result = audit_artifacts(plan_path, plan)
        receipt.update(status="complete", checkpoint_sha256=result["checkpoint_sha256"], training_examples=50000, validation_examples=0,
                       artifact_sha256={name: sha256(folder/name) for name in ARTIFACT_NAMES})
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
        write_json(receipt_path, receipt)
        lease = base/"local_lease.json"
        if lease.exists() and delivery.read(lease).get("pid") == os.getpid():
            lease.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--parent-delivery", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--minimum-seconds", type=int, default=60, help="Required remaining local budget before starting")
    args = parser.parse_args()
    delivery.require(args.run.replace("_", "").isalnum() and not args.run.startswith("stability"), "Use a unique refit ID")
    delivery.require(60 <= args.minimum_seconds <= 10800, "Minimum duration must be from 60 through 10800 seconds")
    plan_path = ROOT/"results/research"/args.run/"refit_plan.json"
    if args.launch:
        delivery.require(plan_path.is_file(), "Prepare and inspect an immutable full-refit plan before --launch")
        delivery.require(Path(delivery.read(plan_path)["parent_delivery_path"]).resolve() == args.parent_delivery.resolve(), "Prepared parent differs")
        result = launch(plan_path, args.minimum_seconds)
        print(json.dumps({"run": args.run, "status": result["status"]}))
    else:
        path = prepare(args.run, args.parent_delivery)
        print(json.dumps({"run": args.run, "status": "prepared", "plan_path": str(path), "gpu_started": False}))


if __name__ == "__main__":
    main()
