"""Native, guarded 50k ResizeMix refit of the fixed scored seed-2026 parent."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import preflight_core_control as control
from research.process_tree_guard import matching_process, run_guarded

BASE = ROOT / "results/research"
LEASE = BASE / "local_lease.json"
ENTRY = Path(__file__).resolve()
GUARD = ROOT / "research/process_tree_guard.py"
PLAN = ROOT / "research/resizemix-full-refit-plan.md"
POLICY_PATH = ROOT / "research/campaign.json"
PARENT = BASE / "wrn22_resizemix_primary_predictions/delivery.json"
FROZEN = BASE / "wrn22_resizemix_delivery/frozen.json"
SCORED = BASE / "submission_receipts/57854107.json"
PARENT_SHA = "7376bed6c3789c4f78df6ec37ae3da738656826ec72a5a95fc37e88714522313"
FROZEN_SHA = "d78a40defda363ec2e66a5e81e841885711ca1940c2508176d7384d85883f2a8"
SCORED_SHA = "9b85f6a3b1a54f6ffb81d5b54d423b5d2aa80b72f61f7b62f4b2c4a4bf0addec"
CHECKPOINT_SHA = "c39436412b454f37f030e49beb977ab8e4a6dc60f8027a0e9942c3c3ad534894"
SCHEMA = "research_resizemix_full_refit_supervisor_v1"
INPUT_SCHEMA = "research_resizemix_full_refit_inputs_v1"
REPORT_SCHEMAS = {"preflight": "research_resizemix_full_refit_actual_preflight_v1",
                  "train": "research_resizemix_full_refit_train_result_v1"}
REPORTS = {"preflight": "native_preflight.json", "train": "train_result.json"}
TRAIN_ARTIFACTS = ("best.pt", "last.pt", "config.json", "metrics.json", "history.json",
                   "provenance.json", "environment.json")
STOP = datetime(2026, 11, 3, 23, 55, tzinfo=timezone.utc)
require, sha, read, write = control.require, control.sha256, control.read, control.write


def source_snapshot(attestation):
    require(all(sha(ROOT / path) == digest for path, digest in control.PINS.items()),
            "Frozen core and guard sources changed")
    names = (*control.PINS, *attestation.get("source_sha256", {}), "research/preflight_core_control.py",
             "research/train_resizemix.py", "research/resizemix.py", "research/benchmark_resizemix.py",
             "research/run_resizemix300_primary.py", "research/preflight_batched_core_control.py",
             "research/batched_finite_proof.py", "research/delivery_v2.py", "research/paired_compare_v2.py",
             "research/paired_compare.py", "research/benchmark.py", "scripts/submit_verified.py",
             "research/submit_candidate.py", "research/repeat_attestation.py",
             "research/train_resizemix_full_refit.py")
    paths = {ROOT / name for name in names} | {ENTRY, PLAN, POLICY_PATH}
    require(all(path.resolve().is_relative_to(ROOT) for path in paths), "Local bound source required")
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def evidence():
    """Bind the fixed qualified parent and official cache without importing Torch."""
    require(sha(PARENT) == PARENT_SHA and sha(FROZEN) == FROZEN_SHA and sha(SCORED) == SCORED_SHA,
            "Exact scored ResizeMix parent bytes required")
    parent, frozen, scored = read(PARENT), read(FROZEN), read(SCORED)
    require(parent["schema"] == "research_prediction_delivery_v2" and parent["status"] == "complete"
            and Path(parent["frozen_manifest_path"]).resolve() == FROZEN and parent["frozen_manifest_sha256"] == FROZEN_SHA
            and parent["checkpoint_sha256"] == sha(frozen["checkpoint_path"]) == CHECKPOINT_SHA
            and parent["policy"] == frozen["policy"] == "center80_shift20"
            and parent["parameter_count"] == frozen["parameter_count"] == 4298970
            and parent["seed"] == frozen["config"]["seed"] == 2026 and parent["training_examples"] == 40000
            and frozen["selected_epoch"] == 150 and frozen["weight_variant"] == "ema"
            and frozen["config"]["train_all"] is False and frozen["config"]["epochs"] == 160
            and scored["status"] == "complete" and scored["submission_ref"] == 57854107 and scored["public_score"] == .875
            and scored["manifest_sha256"] == PARENT_SHA and scored["checkpoint_sha256"] == CHECKPOINT_SHA,
            "Fixed scored v2 parent lineage differs")
    attestation_path = Path(parent["repeat"]["receipt_path"]).resolve()
    attestation = read(attestation_path)
    require(attestation["schema"] == "research_cloud_repeat_attestation_v1"
            and all(sha(ROOT / path) == digest for path, digest in attestation["source_sha256"].items()),
            "Frozen ResizeMix repeat attestation source changed")
    paths, identity = control.official_inputs(Path(frozen["data_dir"]))
    data_signature = {key: identity[key] for key in ("archive_sha256", "split_sha256")}
    require(frozen["data_signature"] == data_signature, "Exact official parent data signature required")
    snapshots = (frozen["evidence_sha256"], parent["repeat_evidence_sha256"], parent["test_files_sha256"],
                 attestation.get("evidence_sha256", {}))
    require(all(sha(path) == digest for mapping in snapshots for path, digest in mapping.items()),
            "A frozen parent evidence file changed")
    paths.extend((PARENT, FROZEN, SCORED, attestation_path, Path(parent["csv_path"])))
    for mapping in snapshots:
        paths.extend(Path(path) for path in mapping)
    return {
        "parent_delivery_path": str(PARENT), "parent_delivery_sha256": PARENT_SHA,
        "frozen_manifest_path": str(FROZEN), "frozen_manifest_sha256": FROZEN_SHA,
        "parent_checkpoint_sha256": CHECKPOINT_SHA, "parent_submission_ref": 57854107,
        "plan_path": str(PLAN), "plan_sha256": sha(PLAN),
        "data_dir": str(Path(frozen["data_dir"]).resolve()), "data_signature": data_signature,
        "evaluation_context": frozen["evaluation_context"], "policy": frozen["policy"],
        "weight_variant": "ema", "config": {**frozen["config"], "train_all": True},
        "source_sha256": source_snapshot(attestation),
        "base_input_sha256": {str(path.resolve()): sha(path) for path in set(paths)},
    }


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def guard_seconds(report):
    median = report.get("median_step_seconds")
    require(positive(median), "Positive measured full-batch median required")
    return 1.5 * median * 391 * 160 + 300


def budget_seconds(policy, current, used, preflight=None):
    require(policy.get("concurrent_local_gpu_jobs") == 1 and policy.get("local_gpu_seconds_per_utc_day") == 21600
            and policy.get("local_job_timeout_seconds") == 10800
            and datetime.fromisoformat(policy["stops_at"]) == datetime(2026, 11, 5, 5, tzinfo=timezone.utc)
            and type(used) in (float, int) and math.isfinite(used) and used >= 0,
            "Fixed local resource rules required")
    budget = min(10800, 21600 - used, (STOP - current).total_seconds())
    required = guard_seconds(preflight) if preflight is not None else 180
    require(required <= budget, "Measured full refit or complete preflight does not fit fixed quota and 23:55 UTC stop")
    return budget if preflight is not None else 180


def guard_records(folder, receipt, *, stopped=True):
    proofs = receipt.get("process_tree_guard_results", [])
    require(receipt.get("process_tree_cleanup_verified") is True and len(proofs) == 1,
            "Exactly one cleaned native process tree required")
    proof = proofs[0]
    nonce = proof.get("nonce", "")
    directory = folder / ("process_tree_" + nonce)
    require(re.fullmatch(r"[0-9a-f]{32}", nonce) and Path(proof["guard_directory"]).resolve() == directory,
            "Native guard directory identity differs")
    paths = [directory / name for name in ("spec.json", "ready.json", "completion.json", "host_result.json")]
    spec, ready, completion, host = map(read, paths)
    require(host == proof and proof["method"] == "windows_job_v1" and proof["cleanup_verified"] is True
            and proof["completion_authenticated"] is True and proof["timed_out"] is False
            and proof["child_returncode"] == proof["completion_exit_code"] == completion["exit_code"] == 0
            and spec["nonce"] == ready["nonce"] == completion["nonce"] == nonce
            and spec["script"] == str(ENTRY) and spec["cwd"] == str(ROOT)
            and spec["guard_source_sha256"] == sha(GUARD) == control.PINS["research/process_tree_guard.py"]
            and spec["script_sha256"] == ready["script_sha256"] == completion["script_sha256"] == sha(ENTRY)
            and spec["arguments"] == worker_arguments(folder, receipt["input_manifest_sha256"])
            and spec["host_pid"] == receipt["pid"] and spec["host_created_at"] == receipt["process_created_at"]
            and proof["child_pid"] == ready["pid"] == completion["pid"] == receipt["child_pid"]
            and proof["child_created_at"] == ready["created_at"] == completion["created_at"] == receipt["child_created_at"]
            and spec["deadline_unix"] <= receipt["deadline_unix"], "Authenticated native guard proof differs")
    require(matching_process(receipt["child_pid"], receipt["child_created_at"]) is None
            and (not stopped or matching_process(receipt["pid"], receipt["process_created_at"]) is None),
            "Recorded refit process still running")
    return paths


def worker_arguments(folder, manifest_sha):
    return ["--_worker", str(folder), "--_manifest-sha", manifest_sha]


def validate_worker(folder, manifest_sha):
    from research.profile_augmix import verify_os_boundary

    process = psutil.Process()
    argv = process.cmdline()
    require(folder.parent == BASE and len(argv) == 4 and Path(argv[1]).resolve() == GUARD
            and argv[2] == "--bootstrap", "Own native worker bootstrap required")
    spec_path = Path(argv[3]).resolve()
    require(spec_path.name == "spec.json" and spec_path.parent.parent == folder, "Own guard directory required")
    spec, ready = read(spec_path), read(spec_path.with_name("ready.json"))
    receipt, lease = read(folder / "supervisor.json"), read(LEASE)
    parent = process.parent()
    manifest = read(folder / "inputs.json")
    nonce = spec_path.parent.name.removeprefix("process_tree_")
    require(parent is not None and receipt["schema"] == SCHEMA and receipt["run"] == folder.name
            and manifest["schema"] == INPUT_SCHEMA and manifest["run"] == folder.name
            and receipt["mode"] == manifest["mode"] in REPORTS and receipt["status"] in {"benchmarking", "training"}
            and receipt["input_manifest_sha256"] == manifest_sha == sha(folder / "inputs.json")
            and parent.pid == receipt["pid"] == lease["pid"] == spec["host_pid"]
            and all(abs(parent.create_time() - stamp) < .01 for stamp in (
                receipt["process_created_at"], lease["process_created_at"], spec["host_created_at"]))
            and process.pid == receipt["child_pid"] == ready["pid"]
            and all(abs(process.create_time() - stamp) < .01 for stamp in (receipt["child_created_at"], ready["created_at"]))
            and lease.get("cleanup_requires_reconciliation") is not True
            and spec["schema"] == ready["schema"] == "owned_python_process_tree_v1"
            and re.fullmatch(r"[0-9a-f]{32}", nonce) and spec["nonce"] == ready["nonce"] == nonce
            and spec["job_name"] == "Local\\SaberAugMix_" + nonce and ready["method"] == "windows_job_v1"
            and Path(spec["script"]).resolve() == ENTRY and Path(spec["cwd"]).resolve() == ROOT
            and spec["guard_source_sha256"] == sha(GUARD) == control.PINS["research/process_tree_guard.py"]
            and spec["script_sha256"] == ready["script_sha256"] == sha(ENTRY)
            and spec["arguments"] == worker_arguments(folder, manifest_sha)
            and 0 < receipt["reserved_seconds"] <= (180 if manifest["mode"] == "preflight" else 10800)
            and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"] <= STOP.timestamp(),
            "Refit worker source, receipt, deadline or native owner differs")
    verify_os_boundary(spec, ready, process)
    verify_bindings(manifest)
    return manifest


def verify_bindings(manifest):
    require(manifest["source_sha256"] and manifest["base_input_sha256"]
            and all(sha(ROOT / path) == digest for path, digest in manifest["source_sha256"].items())
            and all(sha(path) == digest for mapping in (manifest["base_input_sha256"], manifest["preflight_evidence_sha256"])
                    for path, digest in mapping.items()), "Bound refit inputs or source changed")


def preflight_binding(folder, receipt):
    paths = (folder / REPORTS["preflight"], folder / "inputs.json", folder / "supervisor.json",
             *guard_records(folder, receipt))
    return {str(path): sha(path) for path in paths}


def lineage(manifest):
    return {key: manifest[key] for key in ("parent_delivery_sha256", "frozen_manifest_sha256",
        "parent_checkpoint_sha256", "parent_submission_ref", "data_signature", "source_sha256",
        "config_sha256", "plan_sha256", "preflight_artifact_sha256")}


def validate_step(row, batch_size):
    bn_before, bn_after = row["bn_counters_before"], row["bn_counters_after"]
    require(row["source_examples"] == batch_size and row["logits_dtype"] == "torch.float16"
            and row["all_state_finite"] is True and row["geometry_verified"] is True
            and row["gradient_count"] == 62 and row["state_tensor_count"] in (238, 300)
            and row["bn_single_forward"] is True and len(bn_before) == len(bn_after) == 19
            and all(after == before + 1 for before, after in zip(bn_before, bn_after, strict=True))
            and type(row["mixed"]) is bool and row["rectangle_count"] == (batch_size if row["mixed"] else 0)
            and row["retained_weight"] == 1 - row["patch_area"] / 1024
            and positive(row["loss"]) and type(row["updated"]) is bool
            and positive(row["scale_before"]) and positive(row["scale_after"]),
            "Actual native FP16 geometry, gradient, state and BN proof differs")
    if row["updated"]:
        require(row["state_tensor_count"] == 300 and positive(row["gradient_norm"])
                and positive(row["parameter_probe_max_delta"]) and positive(row["ema_probe_max_delta"])
                and row["scale_after"] >= row["scale_before"], "Actual FP16 update proof differs")
    else:
        require(row["gradient_norm"] is None and row["parameter_probe_max_delta"] == row["ema_probe_max_delta"] == 0
                and row["scale_after"] < row["scale_before"] and row["skip_momentum_probe_preserved"] is True,
                "Skipped FP16 update failed to preserve optimizer and EMA probes")


def validate_report(report, manifest, folder):
    from research import train_resizemix_full_refit as engine

    mode = manifest["mode"]
    require(report["schema"] == REPORT_SCHEMAS[mode] and report["status"] == "complete"
            and report["training_examples"] == 50000 and report["validation_examples"] == 0
            and report["precision"] == "fp16" and report["parameter_count"] == 4298970
            and report["environment"]["device"] == "cuda"
            and report["environment"]["resizemix_training_precision"] == "fp16"
            and report["environment"]["actual_autocast_dtype"] == "torch.float16"
            and report["environment"]["grad_scaler_enabled"] is True
            and all(report[key] is False for key in ("hidden_images_loaded", "confirmation_evaluated", "external_data_loaded")),
            "Exact complete 50k/0-dev refit report required")
    attempts = 23 if mode == "preflight" else 62560
    success, skips = report["successful_updates"], report["skipped_updates"]
    require(report["attempted_updates"] == attempts and type(success) is int and type(skips) is int
            and 0 < success <= attempts and 0 <= skips and success + skips == attempts,
            "Actual refit optimizer accounting differs")
    if mode == "preflight":
        seconds = report["step_seconds"]
        require(report["batch_size"] == 128 and report["steps_per_epoch"] == 391
                and len(report["steps"]) == len(seconds) == 23 and all(map(positive, seconds))
                and report["median_step_seconds"] == statistics.median(seconds[3:])
                and report["source_examples_seen"] == 2944 and report["no_weights_saved"] is True
                and report["initial_grad_scaler_scale"] == 65536.0 and report["warmup_attempts"] == 3
                and report["timed_attempts"] == 20 and report["bn_modules"] == 19
                and report["full_state_inventory_count"] == 300 and positive(report["tail_step_seconds"])
                and not any((folder / name).exists() for name in TRAIN_ARTIFACTS),
                "Complete native actual-image timing and step proofs required")
        for row in report["steps"]:
            validate_step(row, 128)
        validate_step(report["tail_step"], 80)
        overflow = report["overflow_smoke"]
        validate_step(overflow, 128)
        require(sum(row["updated"] for row in report["steps"]) == success
                and {row["mixed"] for row in report["steps"]} == {False, True}
                and overflow["updated"] is False and overflow["status"] == "passed"
                and all(overflow[key] is True for key in ("infinite_gradient_injected",
                    "all_parameters_unchanged", "ema_parameters_unchanged", "optimizer_state_unchanged")),
                "Actual mixed/unmixed updates and forced overflow proof required")
        return
    history = read(folder / "history.json")
    metrics = read(folder / "metrics.json")
    require(report["epochs_completed"] == len(history) == 160 and report["source_examples_seen"] == 8000000
            and report["final_variant"] == "ema" and report["lineage"] == lineage(manifest)
            and engine.validate_history(history) == (success, skips, report["resizemix_batches"])
            and skips <= 625 and read(folder / "config.json") == manifest["config"]
            and all(metrics[key] == report[key] for key in ("status", "epochs_completed", "training_examples",
                "validation_examples", "parameter_count", "attempted_updates", "successful_updates",
                "skipped_updates", "resizemix_batches", "source_examples_seen"))
            and metrics["research_training"]["lineage"] == lineage(manifest)
            and read(folder / "provenance.json")["data_signature"] == manifest["data_signature"]
            and all(report[key] is False for key in (
                "independent_development_evaluated", "parent_weights_loaded", "repeat_weights_loaded")),
            "Fixed final EMA, immutable config and at most 1% FP16 skips required")


def validate_final_tensors(folder, manifest, report):
    import torch

    selected, last = (torch.load(folder / name, map_location="cpu", weights_only=False) for name in ("best.pt", "last.pt"))
    require(selected["epoch"] == last["epoch"] == 160 and selected["variant"] == "ema"
            and selected["parameter_count"] == 4298970 and selected["validation_accuracy"] is None
            and selected["validation_loss"] is None and last["history"] == read(folder / "history.json")
            and last["updates"] == report["successful_updates"] and last["skipped_updates"] == report["skipped_updates"]
            and last["attempted_updates"] == 62560 and last["source_examples_seen"] == 8000000
            and selected["config"] == last["config"] == manifest["config"]
            and selected["data_signature"] == last["data_signature"] == manifest["data_signature"]
            and selected["provenance"] == last["provenance"]
            and selected["provenance"]["research_training"]["lineage"] == lineage(manifest)
            and selected["model"].keys() == last["ema"].keys()
            and all(torch.equal(tensor, last["ema"][key]) for key, tensor in selected["model"].items()),
            "Actual final EMA tensors differ from the fixed resumable state")


def worker(folder, manifest_sha):
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # DLL isolation before first Torch import.
    import torch
    from research.submit_candidate import validate_manifest
    from research import train_resizemix_full_refit as engine

    summary = validate_manifest(PARENT, ROOT)
    require(summary["manifest_sha256"] == PARENT_SHA and summary["frozen_manifest_sha256"] == FROZEN_SHA
            and summary["checkpoint_sha256"] == CHECKPOINT_SHA and summary["policy"] == "center80_shift20"
            and summary["training_examples"] == 40000 and summary["seed"] == 2026
            and summary["repeat_receipt_schema"] == "research_cloud_repeat_attestation_v1"
            and manifest["config"] == engine.fixed_config(), "Typed scored parent and exact refit recipe required")
    require(torch.cuda.is_available(), "Native CUDA FP16 required")
    if manifest["mode"] == "preflight":
        report = engine.preflight(manifest["config"], Path(manifest["data_dir"]))
    else:
        prior = read(manifest["preflight_path"])
        actual, measured = runtime.environment(), prior["environment"]
        require(all(actual[key] == measured[key] for key in ("gpu", "cuda_runtime", "cudnn"))
                and actual["packages"]["torch"] == measured["packages"]["torch"], "Measured refit runtime changed")
        report = engine.train(manifest["config"], folder, Path(manifest["data_dir"]), lineage(manifest))
        validate_final_tensors(folder, manifest, report)
    validate_report(report, manifest, folder)
    verify_bindings(manifest)
    write(folder / REPORTS[manifest["mode"]], report)


def validate_completed(folder):
    """Pure revalidation of an authenticated, stopped, immutable attempt."""
    folder = Path(folder).resolve()
    if folder.name == "supervisor.json":
        folder = folder.parent
    require(folder.parent == BASE, "Own controlled refit directory required")
    receipt, manifest = read(folder / "supervisor.json"), read(folder / "inputs.json")
    mode = manifest["mode"]
    started = datetime.fromisoformat(receipt["started_at"])
    finished = datetime.fromisoformat(receipt["finished_at"])
    expected_names = {REPORTS[mode], *(TRAIN_ARTIFACTS if mode == "train" else ())}
    require(receipt["schema"] == SCHEMA and receipt["status"] == "complete"
            and receipt["run"] == manifest["run"] == folder.name and receipt["mode"] == mode
            and receipt["input_manifest_sha256"] == sha(folder / "inputs.json")
            and manifest["schema"] == INPUT_SCHEMA and set(receipt["artifact_sha256"]) == expected_names
            and all(sha(folder / name) == digest for name, digest in receipt["artifact_sha256"].items())
            and started < finished <= STOP
            and 0 < receipt["elapsed_seconds"] <= receipt["reserved_seconds"] + .1
            and abs(receipt["elapsed_seconds"] - (finished - started).total_seconds()) < .01
            and abs(receipt["deadline_unix"] - started.timestamp() - receipt["reserved_seconds"]) < .01
            and all(receipt[key] is False for key in ("confirmation_evaluated", "hidden_images_loaded", "submitted")),
            "Complete immutable refit receipt and artifacts required")
    guard_records(folder, receipt)
    verify_bindings(manifest)
    require(manifest["config_sha256"] == control.json_sha(manifest["config"]),
            "Refit effective configuration hash differs")
    preflight = None
    if mode == "train":
        path = Path(manifest["preflight_path"]).resolve()
        require(path.parent.parent == BASE and path.name == REPORTS["preflight"]
                and read(path.parent / "inputs.json")["mode"] == "preflight", "Own native preflight required")
        prior, prior_manifest, preflight = validate_completed(path.parent)
        require(datetime.fromisoformat(prior["finished_at"]) < started
                and all(prior_manifest[key] == manifest[key] for key in evidence())
                and manifest["preflight_evidence_sha256"] == preflight_binding(path.parent, prior)
                and manifest["preflight_artifact_sha256"] == sha(path),
                "Immutable same-plan preflight and chronology required")
    else:
        require(manifest["preflight_path"] is None and manifest["preflight_artifact_sha256"] is None
                and not manifest["preflight_evidence_sha256"], "Fresh preflight cannot inherit any measured state")
        require(all(manifest[key] == value for key, value in evidence().items()), "Actual scored parent or source changed")
    require(receipt["reserved_seconds"] == budget_seconds(read(POLICY_PATH), started,
                receipt["utc_day_used_before_seconds"], preflight), "Fixed measured reservation differs")
    report = read(folder / REPORTS[mode])
    validate_report(report, manifest, folder)
    return receipt, manifest, report


def parent(run, mode, preflight):
    require(os.name == "nt" and re.fullmatch(r"resizemix_full_refit_[A-Za-z0-9_]+", run)
            and (mode == "train") == bool(preflight), "Fresh named Windows refit and matching preflight required")
    folder = BASE / run
    require(not folder.exists() and not LEASE.exists(), "Existing run or lease requires review")
    binding = evidence()
    preflight_report, preflight_files, preflight_path = None, {}, None
    if preflight:
        prior_receipt, prior_manifest, preflight_report = validate_completed(preflight)
        require(prior_manifest["mode"] == "preflight" and all(prior_manifest[key] == value for key, value in binding.items()),
                "Completed identical source/data/config preflight required")
        preflight_folder = BASE / prior_receipt["run"]
        preflight_path = str(preflight_folder / REPORTS["preflight"])
        preflight_files = preflight_binding(preflight_folder, prior_receipt)
    policy = read(POLICY_PATH)
    account = control.accounting()
    control.claim_lease(LEASE)
    receipt = {}
    try:
        started = account["now"]()
        day = started.replace(hour=0, minute=0, second=0, microsecond=0)
        used = account["used_seconds"](BASE, day, day + timedelta(days=1))
        budget = budget_seconds(policy, started, used, preflight_report)
        if preflight:
            require(datetime.fromisoformat(prior_receipt["finished_at"]) < started, "Preflight must finish first")
        folder.mkdir(exist_ok=False)
        config_sha = control.json_sha(binding["config"])
        manifest = {**binding, "schema": INPUT_SCHEMA, "run": run, "mode": mode, "config_sha256": config_sha,
                    "preflight_path": preflight_path, "preflight_evidence_sha256": preflight_files,
                    "preflight_artifact_sha256": preflight_files.get(preflight_path)}
        manifest_path = folder / "inputs.json"
        write(manifest_path, manifest)
        receipt_path = folder / "supervisor.json"
        receipt = {"schema": SCHEMA, "backend": "local", "run": run, "mode": mode, "role": "full_refit",
                   "precision": "fp16", "status": "training" if mode == "train" else "benchmarking",
                   "started_at": started.isoformat(), "pid": os.getpid(),
                   "process_created_at": psutil.Process().create_time(), "reserved_seconds": budget,
                   "deadline_unix": started.timestamp() + budget, "input_manifest_sha256": sha(manifest_path),
                   "utc_day_used_before_seconds": used,
                   "confirmation_evaluated": False, "hidden_images_loaded": False, "submitted": False}
        write(receipt_path, receipt)
        remaining = receipt["deadline_unix"] - time.time() - (300 if mode == "train" else 2)
        require(remaining > 0, "Setup exceeded the bounded refit reservation")
        run_guarded([sys.executable, str(ENTRY), *worker_arguments(folder, receipt["input_manifest_sha256"])],
                    folder / "refit.log", remaining, receipt_path, receipt, cwd=ROOT)
        verify_bindings(manifest)
        validate_report(read(folder / REPORTS[mode]), manifest, folder)
        guard_records(folder, receipt, stopped=False)
        require(time.time() <= receipt["deadline_unix"], "Fixed refit runtime cap exceeded")
        names = (REPORTS[mode], *(TRAIN_ARTIFACTS if mode == "train" else ()))
        receipt.update(status="complete", artifact_sha256={name: sha(folder / name) for name in names})
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if "schema" in receipt:
            finished = account["now"]()
            receipt.update(finished_at=finished.isoformat(), elapsed_seconds=(finished - started).total_seconds())
            if receipt.get("process_tree_cleanup_verified") is False:
                receipt["status"] = "cleanup_unverified"
            write(folder / "supervisor.json", receipt)
        owner = read(LEASE)
        if owner.get("pid") == os.getpid() and abs(owner["process_created_at"] - psutil.Process().create_time()) < .01:
            if control.lease_can_release(owner, receipt):
                LEASE.unlink()
            else:
                owner.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False,
                             supervisor=str(folder / "supervisor.json"))
                write(LEASE, owner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--mode", choices=tuple(REPORTS))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        require(args._manifest_sha and not any((args.run, args.mode, args.preflight)), "Only guarded worker flags required")
        worker(args._worker.resolve(), args._manifest_sha)
    else:
        require(args.run and args.mode and not args._manifest_sha, "Specify a bounded run and mode")
        parent(args.run, args.mode, args.preflight)


if __name__ == "__main__":
    main()
