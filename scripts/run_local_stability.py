"""Queue one predeclared independent-seed repeat after the primary GPU campaign.

This run measures seed sensitivity only. It cannot choose the recipe, inspect
confirmation data, or predict hidden evaluation images. It has no retry path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "3862ffe"
CORE_FILES = ("train.py", "model.py", "data.py", "runtime.py")
RECIPE = "configs/wrn_final.json"
OUTPUT = ROOT / "results" / "stability_seed2027"
STATUS = ROOT / "results" / "stability.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def frozen_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{FROZEN_COMMIT}:{path}"], cwd=ROOT)


def verify_sources() -> dict:
    hashes = {}
    for path in CORE_FILES:
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        expected = hashlib.sha256(frozen_bytes(path)).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen training source changed: {path}")
        hashes[path] = actual
    if read_json(ROOT / RECIPE) != json.loads(frozen_bytes(RECIPE)):
        raise RuntimeError("Predeclared regularized recipe changed")
    return hashes


def campaign_process():
    matches = []
    expected = ROOT / "run_campaign.py"
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"]):
        try:
            for argument in process.info["cmdline"] or []:
                if Path(argument).name != "run_campaign.py":
                    continue
                path = Path(argument)
                if not path.is_absolute():
                    path = Path(process.info["cwd"]) / path
                if path.resolve() == expected:
                    matches.append((process.pid, process.info["create_time"]))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, TypeError):
            continue
    if len(matches) > 1:
        raise RuntimeError("Multiple primary campaign processes found; GPU ownership is ambiguous")
    return matches[0] if matches else None


def process_alive(identity) -> bool:
    if identity is None:
        return False
    try:
        process = psutil.Process(identity[0])
        return process.create_time() == identity[1] and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def primary_complete() -> bool:
    path = ROOT / "results" / "campaign_status.json"
    if not path.exists():
        return False
    status = read_json(path)
    if status.get("status") != "complete":
        return False
    if status.get("selected_run") not in {"baseline", "architecture", "regularized"}:
        raise RuntimeError("Completed campaign artifact has no valid selected recipe")
    metrics = read_json(ROOT / "results" / "final_full" / "metrics.json")
    if metrics.get("status") != "complete" or metrics.get("epochs_completed") != 160 or not metrics["config"].get("train_all"):
        raise RuntimeError("Campaign completion is not backed by a completed 160-epoch full-data fit")
    submission = status.get("final_submission", {})
    path = ROOT / "results" / "final_full" / "submission.csv"
    if submission.get("rows") != 2000 or submission.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("Campaign completion has an invalid final submission artifact")
    return True


def emit(event: dict):
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
    line = json.dumps(event)
    print(line, flush=True)
    with (OUTPUT / "scheduler.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_command(arguments: list[str], log_path: Path, deadline: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Stability GPU time budget exhausted")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=remaining, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=10800)
    parser.add_argument("--gpu-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if min(args.wait_seconds, args.gpu_seconds, args.poll_seconds) < 1:
        raise ValueError("All time limits must be positive")
    if STATUS.exists() or OUTPUT.exists():
        raise FileExistsError("Stability run already exists; automatic retries and overwrites are forbidden")
    source_sha = verify_sources()
    config = json.loads(frozen_bytes(RECIPE))
    config.update(seed=2027, split_seed=2026)
    if config.get("epochs") != 160:
        raise RuntimeError("The frozen regularized recipe must have exactly 160 epochs")
    identity = campaign_process()
    if identity is None and not primary_complete():
        raise RuntimeError("No live primary campaign and no valid completion artifact; refusing to queue")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "config_requested.json", config)
    started = time.monotonic()
    state = {
        "status": "waiting", "purpose": "independent_seed_stability_only",
        "used_for_selection": False, "seed": 2027, "split_seed": 2026,
        "comparison_run": "regularized", "validation_accuracy": None,
        "flip_tta": None, "parameter_count": None, "source_sha": source_sha,
        "frozen_commit": FROZEN_COMMIT, "elapsed_seconds": 0,
        "pid": os.getpid(), "campaign_pid": identity[0] if identity else None,
        "wait_limit_seconds": args.wait_seconds, "gpu_limit_seconds": args.gpu_seconds,
        "confirmation_evaluated": False, "hidden_predictions_generated": False,
    }
    write_json(STATUS, state)
    emit({"event": "stability_queued", "pid": os.getpid(), "campaign_pid": state["campaign_pid"],
          "wait_limit_seconds": args.wait_seconds, "gpu_limit_seconds": args.gpu_seconds})
    try:
        wait_deadline = started + args.wait_seconds
        while True:
            complete = primary_complete()
            alive = process_alive(identity)
            if complete and not alive:
                # Also reject another live campaign before claiming the GPU.
                if campaign_process() is not None:
                    raise RuntimeError("A primary campaign is still running")
                break
            if not alive and not complete:
                raise RuntimeError("Primary campaign exited without valid completion; stability run cancelled")
            if time.monotonic() >= wait_deadline:
                raise TimeoutError("Primary campaign wait exceeded its wall-time cap")
            time.sleep(min(args.poll_seconds, max(0.01, wait_deadline - time.monotonic())))
        if verify_sources() != source_sha:
            raise RuntimeError("Frozen sources changed while waiting")
        primary_config = read_json(ROOT / "results" / "regularized" / "config.json")
        for key, value in config.items():
            if key != "seed" and primary_config.get(key) != value:
                raise RuntimeError(f"Regularized comparison recipe differs for {key}")
        gpu_started = time.monotonic()
        gpu_deadline = gpu_started + args.gpu_seconds
        state.update(status="training", elapsed_seconds=time.monotonic() - started,
                     waited_seconds=gpu_started - started)
        write_json(STATUS, state)
        emit({"event": "stability_training_started"})
        run_command(["train.py", "--config", str(OUTPUT / "config_requested.json"),
                     "--output-dir", str(OUTPUT), "--device", "cuda"], OUTPUT / "train.log", gpu_deadline)
        verify_sources()
        metrics = read_json(OUTPUT / "metrics.json")
        if metrics.get("status") != "complete" or metrics.get("epochs_completed") != 160 or metrics["config"]["seed"] != 2027:
            raise RuntimeError("Stability training did not complete the predeclared 160-epoch repeat")
        if metrics.get("training_examples") != 40000 or metrics.get("validation_examples") != 5000 or metrics["config"].get("train_all"):
            raise RuntimeError("Stability run used an unexpected training/development partition")
        state.update(status="evaluating", elapsed_seconds=time.monotonic() - started)
        write_json(STATUS, state)
        emit({"event": "stability_development_evaluation_started"})
        run_command(["predict.py", "--checkpoint", str(OUTPUT / "best.pt"), "--output-dir", str(OUTPUT),
                     "--validation-only", "--device", "cuda"], OUTPUT / "evaluation.log", gpu_deadline)
        evaluation = read_json(OUTPUT / "evaluation.json")
        if evaluation.get("validation_examples") != 5000 or evaluation.get("submitted_to_kaggle") is not False:
            raise RuntimeError("Stability evaluation metadata is invalid")
        if not 0 <= evaluation["validation_accuracy"] <= 1 or evaluation["parameter_count"] > 5_000_000:
            raise RuntimeError("Stability score or parameter count is invalid")
        if (OUTPUT / "submission.csv").exists():
            raise RuntimeError("Stability run unexpectedly generated hidden-test predictions")
        state.update(status="complete", validation_accuracy=evaluation["validation_accuracy"],
                     plain_validation_accuracy=evaluation["plain_validation_accuracy"],
                     flip_validation_accuracy=evaluation["flip_validation_accuracy"],
                     flip_tta=evaluation["flip_tta"], parameter_count=evaluation["parameter_count"],
                     epoch=evaluation["epoch"], variant=evaluation["variant"],
                     elapsed_seconds=time.monotonic() - started,
                     training_and_evaluation_seconds=time.monotonic() - gpu_started)
        write_json(STATUS, state)
        emit({"event": "stability_complete", "validation_accuracy": state["validation_accuracy"],
              "used_for_selection": False})
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error),
                     elapsed_seconds=time.monotonic() - started)
        write_json(STATUS, state)
        emit({"event": "stability_failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
