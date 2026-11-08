"""Predict only with the completed, fixed-final EMA ResizeMix full-data refit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import preflight_core_control as control
from research.process_tree_guard import _valid_child_record, may_release_lease, run_guarded
from research.profile_augmix import verify_os_boundary
from research.trivialaugment_delivery import official_test_paths
from scripts.submit_verified import COMPETITION, validate_csv

BASE = ROOT / "results/research"
LEASE = BASE / "local_lease.json"
CAMPAIGN = ROOT / "research/campaign.json"
HARD_STOP = datetime(2026, 11, 5, 5, tzinfo=timezone.utc)
SCHEMA = "research_resizemix_full_refit_delivery_v1"
INPUT_SCHEMA = "research_resizemix_full_refit_prediction_inputs_v1"
RECEIPT_SCHEMA = "research_resizemix_full_refit_prediction_supervisor_v1"
NATIVE_SCHEMA = "owned_python_process_tree_v1"
AUDIT_SCHEMA = "research_resizemix_full_refit_final_ema_tensor_audit_v1"
POLICY = "center80_shift20"
PARAMETERS = 4_298_970
PARENT_DELIVERY = ROOT / "results/research/wrn22_resizemix_primary_predictions/delivery.json"
PARENT_DELIVERY_SHA = "7376bed6c3789c4f78df6ec37ae3da738656826ec72a5a95fc37e88714522313"
PARENT_FREEZE = ROOT / "results/research/wrn22_resizemix_delivery/frozen.json"
PARENT_FREEZE_SHA = "d78a40defda363ec2e66a5e81e841885711ca1940c2508176d7384d85883f2a8"
PARENT_CHECKPOINT_SHA = "c39436412b454f37f030e49beb977ab8e4a6dc60f8027a0e9942c3c3ad534894"
GUARD = ROOT / "research/process_tree_guard.py"
SOURCES = (
    "research/resizemix_full_refit_delivery.py", "research/run_resizemix_full_refit.py",
    "research/preflight_core_control.py", "research/profile_augmix.py",
    "research/process_tree_guard.py", "research/run_local.py",
    "research/trivialaugment_delivery.py", "scripts/submit_verified.py",
    "research/benchmark.py", "data.py", "runtime.py", "train.py", "model.py", "research/campaign.json",
)
require, sha, read, write = control.require, control.sha256, control.read, control.write


def stamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Artifact timestamps must be timezone-aware")
    return parsed


def snapshot(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def fixed_directory(value):
    path = Path(value).resolve()
    require(path.parent == BASE.resolve(), "Use a direct local research run directory")
    return path


def lineage(refit_dir):
    """Revalidate actual completed refit, frozen parent and source/data closure."""
    from research.run_resizemix_full_refit import guard_records, validate_completed

    folder = fixed_directory(refit_dir)
    receipt, manifest, report = validate_completed(folder)
    checkpoint = folder / "best.pt"
    last = folder / "last.pt"
    fixed_report = {
        "status": "complete", "precision": "fp16", "epochs_completed": 160,
        "training_examples": 50000, "validation_examples": 0, "parameter_count": PARAMETERS,
        "final_variant": "ema", "independent_development_evaluated": False,
        "confirmation_evaluated": False, "hidden_images_loaded": False,
        "external_data_loaded": False, "parent_weights_loaded": False, "repeat_weights_loaded": False,
    }
    require(receipt.get("status") == "complete" and receipt.get("mode") == manifest.get("mode") == "train"
            and all(type(report.get(key)) is type(value) and report[key] == value
                    for key, value in fixed_report.items()), "Completed typed full-data ResizeMix refit required")
    require(Path(manifest["parent_delivery_path"]).resolve() == PARENT_DELIVERY.resolve()
            and manifest["parent_delivery_sha256"] == sha(PARENT_DELIVERY) == PARENT_DELIVERY_SHA
            and Path(manifest["frozen_manifest_path"]).resolve() == PARENT_FREEZE.resolve()
            and manifest["frozen_manifest_sha256"] == sha(PARENT_FREEZE) == PARENT_FREEZE_SHA
            and manifest["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA,
            "Original scored ResizeMix parent delivery and freeze required")
    frozen = read(PARENT_FREEZE)
    config = {**frozen["config"], "train_all": True, "final_variant": "ema"}
    require(control.json_sha(manifest["config"]) == control.json_sha(config)
            and config["seed"] == 2026 and config["epochs"] == 160
            and manifest["evaluation_context"] == frozen["evaluation_context"]
            and manifest["data_signature"] == frozen["data_signature"]
            and manifest["policy"] == frozen["policy"] == POLICY
            and manifest["weight_variant"] == frozen["weight_variant"] == "ema",
            "Fixed original 50k refit configuration or inference policy differs")
    data_dir = Path(manifest["data_dir"]).resolve()
    require(data_dir == Path(frozen["data_dir"]).resolve(), "Refit must use the frozen official data cache")
    artifacts = receipt["artifact_sha256"]
    require(sha(checkpoint) == artifacts["best.pt"] != PARENT_CHECKPOINT_SHA
            and sha(last) == artifacts["last.pt"], "Only the actual final refit checkpoint is eligible")
    sources = snapshot(ROOT / name for name in {*SOURCES, *manifest["source_sha256"]})
    evidence = snapshot((
        folder / "inputs.json", folder / "supervisor.json", PARENT_DELIVERY, PARENT_FREEZE,
        *(folder / name for name in artifacts), *guard_records(folder, receipt),
        *manifest["base_input_sha256"], *manifest["preflight_evidence_sha256"],
        *official_test_paths(data_dir),
    ))
    return {
        "refit_dir": str(folder), "refit_input_manifest_sha256": sha(folder / "inputs.json"),
        "refit_supervisor_sha256": sha(folder / "supervisor.json"), "checkpoint_sha256": artifacts["best.pt"],
        "last_sha256": artifacts["last.pt"], "parent_delivery_sha256": PARENT_DELIVERY_SHA,
        "frozen_manifest_sha256": PARENT_FREEZE_SHA, "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "evaluation_context": manifest["evaluation_context"], "data_signature": manifest["data_signature"],
        "data_dir": str(data_dir), "config": config, "sources_sha256": sources, "input_sha256": evidence,
    }, receipt


def worker_arguments(folder, digest):
    return ["--_worker", str(folder), "--_manifest-sha", digest]


def verify_guard(folder, receipt, digest, *, active):
    dirs = sorted(folder.glob("process_tree_*"))
    require(len(dirs) == 1 and re.fullmatch(r"process_tree_[0-9a-f]{32}", dirs[0].name)
            and dirs[0].resolve().parent == folder, "Exactly one owned native prediction boundary required")
    directory = dirs[0]
    spec, ready = (read(directory / name) for name in ("spec.json", "ready.json"))
    nonce = directory.name.removeprefix("process_tree_")
    pid, created = receipt["child_pid"], receipt["child_created_at"]
    start = stamp(receipt["started_at"])
    method = "windows_job_v1" if os.name == "nt" else "posix_session_v1"
    require(spec["schema"] == NATIVE_SCHEMA and spec["nonce"] == nonce
            and spec["host_pid"] == receipt["pid"] and spec["host_created_at"] == receipt["process_created_at"]
            and spec["script"] == str(Path(__file__).resolve()) and spec["script_sha256"] == sha(__file__)
            and spec["guard_source_sha256"] == sha(GUARD) and spec["cwd"] == str(ROOT.resolve())
            and spec["arguments"] == worker_arguments(folder, digest)
            and spec["job_name"] == ("Local\\SaberAugMix_" + nonce if os.name == "nt" else None)
            and _valid_child_record(ready, spec, pid, created) and ready["method"] == method
            and type(receipt["reserved_seconds"]) is int and receipt["reserved_seconds"] == 180
            and start.timestamp() < spec["deadline_unix"] <= receipt["deadline_unix"] + .1
            and receipt["deadline_unix"] <= min(start.timestamp() + 180.1, HARD_STOP.timestamp()),
            "Guard script, arguments, process identity or budget differs")
    if active:
        child = psutil.Process()
        parent = child.parent()
        require(parent is not None and parent.pid == receipt["pid"]
                and abs(parent.create_time() - receipt["process_created_at"]) < .01
                and child.pid == pid and abs(child.create_time() - created) < .01
                and time.time() < spec["deadline_unix"], "Live prediction worker identity differs")
        lease = read(LEASE)
        require(lease["pid"] == parent.pid
                and abs(lease["process_created_at"] - parent.create_time()) < .01
                and control.lease_can_release(lease, {}), "Canonical local lease is foreign or unresolved")
        argv = child.cmdline()
        require(len(argv) == 4 and Path(argv[0]).resolve() == Path(sys.executable).resolve()
                and Path(argv[1]).resolve() == GUARD.resolve() and argv[2] == "--bootstrap"
                and Path(argv[3]).resolve() == (directory / "spec.json").resolve(),
                "Exact native bootstrap command required")
        verify_os_boundary(spec, ready, child)
        return

    completion, proof = (read(directory / name) for name in ("completion.json", "host_result.json"))
    require(receipt["process_tree_guard_results"] == [proof]
            and receipt["process_tree_cleanup_verified"] is True
            and proof["schema"] == NATIVE_SCHEMA and proof["nonce"] == nonce
            and proof["method"] == method and proof["guard_directory"] == str(directory)
            and proof["guard_source_sha256"] == sha(GUARD)
            and proof["child_pid"] == pid and proof["child_created_at"] == created
            and proof["cleanup_verified"] is True and proof["timed_out"] is False
            and proof["completion_authenticated"] is True
            and type(proof["completion_exit_code"]) is int and proof["completion_exit_code"] == 0
            and type(proof["child_returncode"]) is int and proof["child_returncode"] == 0
            and _valid_child_record(completion, spec, pid, created) and completion["method"] == method
            and type(completion["exit_code"]) is int and completion["exit_code"] == 0,
            "Complete authenticated prediction cleanup required")
    identities = [(receipt["pid"], receipt["process_created_at"]), (pid, created)]
    if ready.get("watchdog_pid") is not None:
        identities.append((ready["watchdog_pid"], ready["watchdog_created_at"]))
    for process_id, creation_time in identities:
        require(type(process_id) is int and process_id > 0, "Invalid owned process identity")
        try:
            require(abs(psutil.Process(process_id).create_time() - creation_time) >= .01,
                    "A matching owned prediction process is still active")
        except psutil.NoSuchProcess:
            pass


def delivery_fields(folder, binding):
    refit_dir = Path(binding["refit_dir"])
    return {
        "schema": SCHEMA, "status": "complete", "run": refit_dir.name, "competition": COMPETITION,
        "refit_dir": str(refit_dir), "prediction_input_manifest_sha256": sha(folder / "inputs.json"),
        "refit_input_manifest_sha256": binding["refit_input_manifest_sha256"],
        "refit_supervisor_sha256": binding["refit_supervisor_sha256"],
        "frozen_manifest_sha256": PARENT_FREEZE_SHA, "parent_delivery_sha256": PARENT_DELIVERY_SHA,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "checkpoint_sha256": binding["checkpoint_sha256"], "last_sha256": binding["last_sha256"],
        "parameter_count": PARAMETERS, "weight_variant": "ema", "epochs_completed": 160,
        "checkpoint_epoch": 160, "training_precision": "fp16", "seed": 2026, "policy": POLICY,
        "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
        "independent_development_evaluated": False, "evaluation_context": binding["evaluation_context"],
        "confirmation_evaluated": False, "hidden_test_labels_loaded": False, "external_data_loaded": False,
        "parent_weights_loaded": False, "bn_recalibrated": False, "rows": 2000,
        "csv_path": str(folder / "submission.csv"),
    }


def tensor_audit(binding):
    return {
        "schema": AUDIT_SCHEMA, "checkpoint_sha256": binding["checkpoint_sha256"],
        "last_sha256": binding["last_sha256"], "epoch": 160, "variant": "ema",
        "selected_state_equals_terminal_ema": True, "parameters": PARAMETERS,
        "model_state_unchanged_by_inference": True, "prediction_context": binding["evaluation_context"],
    }


def worker(folder, digest):
    folder = fixed_directory(folder)
    receipt, inputs = read(folder / "supervisor.json"), read(folder / "inputs.json")
    require(receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "benchmarking"
            and receipt["backend"] == "local" and receipt["mode"] == "predict"
            and receipt["submitted"] is False and receipt["confirmation_evaluated"] is False
            and receipt["run"] == inputs["run"] == folder.name and inputs["schema"] == INPUT_SCHEMA
            and receipt["input_manifest_sha256"] == digest == sha(folder / "inputs.json"),
            "Actual bounded refit prediction supervisor required")
    verify_guard(folder, receipt, digest, active=True)
    binding, _ = lineage(inputs["binding"]["refit_dir"])
    require(control.json_sha(inputs) == control.json_sha({"schema": INPUT_SCHEMA, "run": folder.name, "binding": binding}),
            "Refit prediction source or inputs differ")

    import runtime
    import torch
    from data import load_kaggle_test
    from model import build_model, count_parameters
    from research import benchmark as suite
    from train import code_provenance

    device = torch.device("cuda")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
            "Original native BF16 prediction context required")
    runtime.seed_all(2026)
    context = {
        "device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)),
        "torch_version": torch.__version__, "training_core_sha256": code_provenance()["source_sha256"],
    }
    require(context == binding["evaluation_context"], "Frozen parent prediction context differs")
    refit = Path(binding["refit_dir"])
    selected = torch.load(refit / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(refit / "last.pt", map_location="cpu", weights_only=False)
    require(selected["epoch"] == last["epoch"] == 160 and selected["variant"] == "ema"
            and selected["config"] == last["config"] == binding["config"]
            and selected["data_signature"] == last["data_signature"] == binding["data_signature"]
            and selected["validation_accuracy"] is None and selected["validation_loss"] is None
            and selected["parameter_count"] == PARAMETERS and last["best_accuracy"] is None
            and last["best_loss"] is None and selected["provenance"] == last["provenance"],
            "Fixed terminal EMA and zero-validation artifact required")
    model_config = {key: binding["config"][key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    require(selected["model_config"] == model_config, "Selected refit architecture differs")
    state, ema = selected["model"], last["ema"]
    require(state.keys() == ema.keys() and all(
        tensor.dtype == ema[key].dtype and tensor.shape == ema[key].shape
        and torch.isfinite(tensor).all().item() and torch.equal(tensor, ema[key])
        for key, tensor in state.items()
    ), "Fixed selected state is not the terminal EMA")
    del last, ema
    model = build_model(model_config)
    require(count_parameters(model) == count_parameters(model, True) == PARAMETERS,
            "Wrong refit parameter count")
    model.load_state_dict(state, strict=True)
    model.to(device=device, memory_format=torch.channels_last).eval()

    data_dir = Path(binding["data_dir"])
    ids, pixels = load_kaggle_test(data_dir)
    require(len(ids) == len(set(ids)) == len(pixels) == 2000, "Official unlabeled test identity differs")
    predictions = []
    with torch.inference_mode():
        for start in range(0, 2000, 128):
            batch = torch.as_tensor(pixels[start:start + 128], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = suite.infer(model, batch, POLICY)
            require(logits.shape == (len(batch), 10) and torch.isfinite(logits).all().item(),
                    "Invalid refit logits")
            predictions.extend(logits.argmax(1).cpu().tolist())
    require(all(torch.equal(tensor.cpu(), state[key]) for key, tensor in model.state_dict().items()),
            "Refit prediction changed model parameters or buffers")
    require(all(sha(path) == value for path, value in binding["input_sha256"].items())
            and all(sha(path) == value for path, value in binding["sources_sha256"].items()),
            "Bound refit prediction inputs changed")
    write(folder / "tensor_audit.json", tensor_audit(binding))
    path = folder / "submission.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "label"))
        writer.writerows(zip(ids, predictions, strict=True))
    submission_hash = validate_csv(path, data_dir / "sample_submission.csv")
    write(folder / "delivery.json", {
        **delivery_fields(folder, binding), "created_at": datetime.now(timezone.utc).isoformat(),
        "tensor_audit_sha256": sha(folder / "tensor_audit.json"), "submission_sha256": submission_hash,
    })


def verify_output(folder, binding):
    path = folder / "delivery.json"
    document = read(path)
    audit_path = folder / "tensor_audit.json"
    expected = {
        **delivery_fields(folder, binding), "created_at": document["created_at"],
        "tensor_audit_sha256": sha(audit_path), "submission_sha256": validate_csv(
            folder / "submission.csv", Path(binding["data_dir"]) / "sample_submission.csv"),
    }
    require(control.json_sha(document) == control.json_sha(expected)
            and control.json_sha(read(audit_path)) == control.json_sha(tensor_audit(binding)),
            "Refit delivery or guarded tensor audit differs")
    return document


def validate_refit_delivery(manifest_path, root=ROOT):
    """Read-only validated uploader summary; no Torch import or test decode."""
    require(Path(root).resolve() == ROOT.resolve(), "Prediction must use the bound repository")
    path = Path(manifest_path).resolve()
    folder = fixed_directory(path.parent)
    require(path.name == "delivery.json", "Use the completed guarded delivery.json")
    inputs, receipt = read(folder / "inputs.json"), read(folder / "supervisor.json")
    binding, refit = lineage(inputs["binding"]["refit_dir"])
    digest = sha(folder / "inputs.json")
    require(control.json_sha(inputs) == control.json_sha({"schema": INPUT_SCHEMA, "run": folder.name, "binding": binding})
            and receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "complete"
            and receipt["run"] == folder.name and receipt["input_manifest_sha256"] == digest
            and receipt["backend"] == "local" and receipt["mode"] == "predict"
            and receipt["submitted"] is False and receipt["confirmation_evaluated"] is False
            and receipt["artifact_sha256"] == sha(path),
            "Actual complete guarded refit prediction evidence required")
    verify_guard(folder, receipt, digest, active=False)
    require(not LEASE.exists(), "Reconcile the canonical local lease before submission validation")
    document = verify_output(folder, binding)
    start, end = stamp(receipt["started_at"]), stamp(receipt["finished_at"])
    elapsed = (end - start).total_seconds()
    require(stamp(refit["finished_at"]) <= start <= stamp(document["created_at"]) <= end <= HARD_STOP
            and end <= datetime.now(timezone.utc) and 0 < elapsed <= 180.1
            and 0 < receipt["elapsed_seconds"] <= 180.1
            and abs(elapsed - receipt["elapsed_seconds"]) <= 1
            and end.timestamp() <= receipt["deadline_unix"] + .1,
            "Actual bounded prediction chronology differs")
    return {**document, "manifest_sha256": sha(path), "delivery_schema": SCHEMA}


def launch(run, refit_dir):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}", run) and not run.startswith("stability"),
            "Fresh unique prediction run required")
    folder = fixed_directory(BASE / run)
    require(not folder.exists() and not LEASE.exists(), "Existing run or lease requires review")
    binding, _ = lineage(refit_dir)
    policy = read(CAMPAIGN)
    require(stamp(policy["stops_at"]) == HARD_STOP, "Campaign hard stop differs")
    account = control.accounting()
    control.claim_lease(LEASE)
    owner = psutil.Process()
    receipt = {}
    owns = False
    try:
        start, began = account["now"](), time.perf_counter()
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        reserved = control.bounded_seconds(policy, start, account["used_seconds"](BASE, day, day + timedelta(days=1)))
        folder.mkdir(exist_ok=False)
        owns = True
        write(folder / "inputs.json", {"schema": INPUT_SCHEMA, "run": run, "binding": binding})
        digest = sha(folder / "inputs.json")
        receipt = {
            "schema": RECEIPT_SCHEMA, "backend": "local", "mode": "predict", "run": run,
            "status": "benchmarking", "pid": owner.pid, "process_created_at": owner.create_time(),
            "started_at": start.isoformat(), "reserved_seconds": reserved,
            "deadline_unix": min(start.timestamp() + reserved, HARD_STOP.timestamp()),
            "input_manifest_sha256": digest, "submitted": False, "confirmation_evaluated": False,
        }
        remaining = receipt["deadline_unix"] - time.time() - 2
        require(remaining > 0, "Insufficient remaining prediction time")
        run_guarded(
            [sys.executable, str(Path(__file__).resolve()), *worker_arguments(folder, digest)],
            folder / "prediction.log", remaining, folder / "supervisor.json", receipt, cwd=ROOT,
        )
        verify_output(folder, binding)
        require(time.time() <= receipt["deadline_unix"], "Prediction exceeded the reserved budget")
        receipt.update(status="complete", artifact_sha256=sha(folder / "delivery.json"))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns and "schema" in receipt:
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            receipt.update(finished_at=account["now"]().isoformat(), elapsed_seconds=time.perf_counter() - began)
            write(folder / "supervisor.json", receipt)
        if LEASE.exists():
            lease = read(LEASE)
            if lease["pid"] == owner.pid and abs(lease["process_created_at"] - owner.create_time()) < .01:
                if control.lease_can_release(lease, receipt):
                    LEASE.unlink()
                else:
                    lease.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False,
                                 supervisor=str(folder / "supervisor.json"))
                    write(LEASE, lease)
    return folder / "delivery.json"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_worker":
        require(len(argv) == 4 and argv[2] == "--_manifest-sha", "Exact guarded worker arguments required")
        worker(Path(argv[1]), argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--predict", action="store_true")
    actions.add_argument("--validate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--refit-dir", type=Path)
    args = parser.parse_args(argv)
    if args.predict:
        if args.manifest is not None or args.run is None or args.refit_dir is None:
            parser.error("--predict requires --run and --refit-dir")
        print(launch(args.run, args.refit_dir))
    else:
        if args.manifest is None or args.run is not None or args.refit_dir is not None:
            parser.error("--validate requires --manifest")
        document = validate_refit_delivery(args.manifest)
        print(f"Verified {document['rows']} official prediction rows for {document['run']}")


if __name__ == "__main__":
    main()
