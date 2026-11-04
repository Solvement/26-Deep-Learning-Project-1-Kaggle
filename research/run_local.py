"""One supervised, budgeted local experiment followed by the full dev benchmark.

Launch separately from the agent; no Kaggle uploads or confirmation evaluation.
The persistent lease prevents another heartbeat from launching overlapping work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc)


def trainer_identity(trainer, precision):
    """Bind resume to the actual trainer; frozen core always uses CUDA BF16."""
    aliases = {"bfloat16": "bf16", "torch.bfloat16": "bf16"}
    precision = aliases.get(precision, precision)
    if trainer not in {"frozen", "shadow", "sam"} or precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("Unknown trainer or precision")
    if trainer == "frozen":
        if precision not in {"auto", "bf16"}:
            raise ValueError("Frozen trainer hardcodes BF16; precision overrides require shadow or SAM")
        precision = "bf16"
    return trainer, precision


def validate_resume_identity(previous, trainer, precision):
    # Historical local receipts predate these fields and called frozen train.py.
    prior = trainer_identity(previous.get("trainer", "frozen"), previous.get("precision", "auto"))
    requested = trainer_identity(trainer, precision)
    if prior != requested:
        raise ValueError("Resume trainer or precision changed; use the original trainer/precision")


def build_commands(trainer, precision, config_path, folder, baseline, resume=False):
    trainer_identity(trainer, precision)
    training_entry, benchmark_entry = {
        "frozen": ("train.py", "research/benchmark.py"),
        "shadow": ("research/train_shadow.py", "research/benchmark_shadow.py"),
        "sam": ("research/train_sam.py", "research/benchmark_sam.py"),
    }[trainer]
    command = [sys.executable, training_entry, "--config", str(config_path),
               "--output-dir", str(folder), "--device", "cuda"]
    if trainer != "frozen":
        command.extend(["--precision", precision])
    if resume:
        command.append("--resume")
    benchmark = [sys.executable, benchmark_entry, "--checkpoint", str(folder/"best.pt"),
                 "--output", str(folder/"benchmark.json"), "--baseline", str(baseline), "--device", "cuda"]
    return command, benchmark


def assert_stopped(record):
    """Never resume while any recorded owner or training child is still alive."""
    for key, stamp in (("pid", "process_created_at"), ("child_pid", "child_created_at")):
        if not record.get(key):
            continue
        try:
            process = psutil.Process(record[key])
        except psutil.NoSuchProcess:
            continue
        expected = record.get(stamp)
        if expected is None or abs(process.create_time()-expected) < .01:
            raise RuntimeError("Recorded experiment process is still alive; resume refused")


def close_interrupted_attempt(receipt_path):
    """Preserve an orphan receipt and conservatively charge its reservation."""
    original_bytes = receipt_path.read_bytes()
    record = json.loads(original_bytes)
    if record.get("status") == "complete":
        raise RuntimeError("Completed experiments cannot be resumed")
    assert_stopped(record)
    if not record.get("finished_at"):
        began = datetime.fromisoformat(record["started_at"])
        end = min(now(), began+timedelta(seconds=record["reserved_seconds"]))
        record.update(original_status=record["status"], status="interrupted", finished_at=end.isoformat(),
                      elapsed_seconds=(end-began).total_seconds(),
                      budget_accounting="Conservative full reservation, capped by reconciliation time; actual interruption time unknown.",
                      reconciled_at=now().isoformat())
    number = 1
    while ((receipt_path.parent/f"supervisor_attempt_{number}.json").exists() or
           (receipt_path.parent/f"supervisor_raw_{number}.json").exists()):
        number += 1
    archived = receipt_path.parent/f"supervisor_attempt_{number}.json"
    with (receipt_path.parent/f"supervisor_raw_{number}.json").open("xb") as stream:
        stream.write(original_bytes)
    for name in ("environment", "provenance"):
        source = receipt_path.parent/f"{name}.json"
        if source.exists():
            with (receipt_path.parent/f"{name}_attempt_{number}.json").open("xb") as stream:
                stream.write(source.read_bytes())
    record["original_receipt_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    with archived.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return archived


def used_seconds(base, day_start, day_end, exclude=None):
    used = 0.0
    paths = list(base.glob("*/supervisor.json"))+list(base.glob("*/supervisor_attempt_*.json"))
    for path in paths:
        if path == exclude:
            continue
        record = json.loads(path.read_text())
        if (record.get("backend") == "colab" or
                record.get("resource_policy") == "existing_Colab_entitlement_only_no_purchases" or
                ("session" in record and "assignment_attempts" in record)):
            # Colab and local GPUs have independent leases and time budgets.
            continue
        if record.get("status") in {"starting", "training", "benchmarking"}:
            raise RuntimeError("An unfinished supervisor needs reconciliation before more GPU work")
        began = datetime.fromisoformat(record["started_at"])
        finished = datetime.fromisoformat(record["finished_at"])
        used += max(0, (min(finished, day_end)-max(began, day_start)).total_seconds())
    return used


def acquire_lease(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lease = json.loads(path.read_text())
        try:
            process = psutil.Process(lease["pid"])
            if abs(process.create_time()-lease["process_created_at"]) < .01:
                raise RuntimeError("A local research job already holds the GPU lease")
        except psutil.NoSuchProcess:
            pass
        # Preserve stale/crashed evidence rather than silently discarding it.
        path.rename(path.with_name(f"local_lease_stale_{time.time_ns()}.json"))
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
                   "started_at": now().isoformat()}, stream)


def supervised_command(command, log_path, timeout, receipt_path, receipt):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        guard = None
        try:
            receipt.update(child_pid=process.pid, child_created_at=psutil.Process(process.pid).create_time())
            guard = subprocess.Popen([sys.executable, str(ROOT/"research/watchdog.py"),
                "--parent", str(os.getpid()), "--parent-created", str(psutil.Process().create_time()),
                "--child", str(process.pid), "--child-created", str(receipt["child_created_at"]),
                "--seconds", str(timeout)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            receipt.update(watchdog_pid=guard.pid)
            write_json(receipt_path, receipt)
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            if guard is not None and guard.poll() is None:
                guard.terminate()
                guard.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume an inactive interrupted run from its exact last.pt")
    parser.add_argument("--trainer", choices=("frozen", "shadow", "sam"), default="frozen")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    _, bound_precision = trainer_identity(args.trainer, args.precision)
    if not args.run.replace("_", "").isalnum() or args.run.startswith("stability"):
        parser.error("Use a unique alphanumeric exploratory run ID")
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("train_all") is not False or config.get("split_seed") != 2026:
        raise ValueError("Local candidate must preserve independent 40k/5k development")
    base = ROOT/"results/research"
    folder = base/args.run
    folder.mkdir(parents=True, exist_ok=args.resume)
    receipt_path = folder/"supervisor.json"
    prior_attempt = None
    if args.resume:
        if not receipt_path.exists() or not (folder/"last.pt").exists():
            raise FileNotFoundError("Resume requires an existing receipt and last checkpoint")
        previous = json.loads(receipt_path.read_text())
        validate_resume_identity(previous, args.trainer, args.precision)
        if previous.get("config_sha256") != sha256(config_path) or (folder/"benchmark.json").exists():
            raise ValueError("Resume configuration changed or benchmark already exists")
        assert_stopped(previous)
    elif receipt_path.exists():
        raise FileExistsError("Existing supervised experiment: inspect/resume deliberately, never restart blindly")
    lease_path = base/"local_lease.json"
    acquire_lease(lease_path)
    started = time.perf_counter()
    receipt = {"run": args.run, "status": "starting", "pid": os.getpid(),
               "started_at": now().isoformat(), "config_sha256": sha256(config_path),
               "confirmation_evaluated": False, "submitted": False,
               "backend": "local", "trainer": args.trainer, "precision": bound_precision,
               "precision_requested": args.precision,
               "process_created_at": psutil.Process().create_time(), "resumed": args.resume}
    owns_receipt = not args.resume
    try:
        if args.resume:
            prior_attempt = close_interrupted_attempt(receipt_path)
            owns_receipt = True
            receipt.update(previous_attempt=prior_attempt.name, resumed_checkpoint_sha256=sha256(folder/"last.pt"))
        day = now().date().isoformat()
        day_start = now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start+timedelta(days=1)
        used = used_seconds(base, day_start, day_end, exclude=receipt_path)
        budget = min(policy["local_job_timeout_seconds"], policy["local_gpu_seconds_per_utc_day"]-used,
                     int((deadline-now()).total_seconds()))
        if budget < 60:
            raise RuntimeError("No local runtime budget remains")
        receipt.update(reserved_seconds=budget, utc_budget_day=day, status="training")
        write_json(receipt_path, receipt)
        command, benchmark = build_commands(args.trainer, args.precision, config_path, folder,
                                             ROOT/policy["baseline_benchmark"], args.resume)
        log_name = "training_resume_"+str(time.time_ns())+".log" if args.resume else "training.log"
        receipt["training_log"] = log_name
        supervised_command(command, folder/log_name, budget, receipt_path, receipt)
        receipt["status"] = "benchmarking"
        write_json(receipt_path, receipt)
        remaining = budget-(time.perf_counter()-started)
        if remaining < 60:
            raise TimeoutError("No supervised runtime remains for the required benchmark")
        supervised_command(benchmark, folder/"benchmark.log", remaining, receipt_path, receipt)
        evidence = json.loads((folder/"benchmark.json").read_text())
        receipt.update(status="complete", benchmark_sha256=sha256(folder/"benchmark.json"),
                       benchmark_passed=any(g["passed"] for g in evidence["promotion"].values()))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
        if owns_receipt:
            write_json(receipt_path, receipt)
        # Delete only our own small lease file, never any experiment directory.
        if lease_path.exists() and json.loads(lease_path.read_text()).get("pid") == os.getpid():
            lease_path.unlink()


if __name__ == "__main__":
    main()
