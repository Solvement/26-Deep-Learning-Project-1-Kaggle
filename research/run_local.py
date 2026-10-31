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
    args = parser.parse_args()
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
    folder.mkdir(parents=True, exist_ok=False)
    receipt_path = folder/"supervisor.json"
    if receipt_path.exists():
        raise FileExistsError("Existing supervised experiment: inspect/resume deliberately, never restart blindly")
    lease_path = base/"local_lease.json"
    acquire_lease(lease_path)
    started = time.perf_counter()
    receipt = {"run": args.run, "status": "starting", "pid": os.getpid(),
               "started_at": now().isoformat(), "config_sha256": sha256(config_path),
               "confirmation_evaluated": False, "submitted": False}
    try:
        day = now().date().isoformat()
        day_start = now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start+timedelta(days=1)
        used = 0.0
        for path in base.glob("*/supervisor.json"):
            record = json.loads(path.read_text())
            if record.get("status") in {"starting", "training", "benchmarking"}:
                raise RuntimeError("An unfinished supervisor needs reconciliation before more GPU work")
            began = datetime.fromisoformat(record["started_at"])
            finished = datetime.fromisoformat(record["finished_at"])
            used += max(0, (min(finished, day_end)-max(began, day_start)).total_seconds())
        budget = min(policy["local_job_timeout_seconds"], policy["local_gpu_seconds_per_utc_day"]-used,
                     int((deadline-now()).total_seconds()))
        if budget < 60:
            raise RuntimeError("No local runtime budget remains")
        receipt.update(reserved_seconds=budget, utc_budget_day=day, status="training")
        write_json(receipt_path, receipt)
        command = [sys.executable, "train.py", "--config", str(config_path), "--output-dir", str(folder), "--device", "cuda"]
        supervised_command(command, folder/"training.log", budget, receipt_path, receipt)
        receipt["status"] = "benchmarking"
        write_json(receipt_path, receipt)
        remaining = budget-(time.perf_counter()-started)
        if remaining < 60:
            raise TimeoutError("No supervised runtime remains for the required benchmark")
        benchmark = [sys.executable, "research/benchmark.py", "--checkpoint", str(folder/"best.pt"),
                     "--output", str(folder/"benchmark.json"), "--baseline", str(ROOT/policy["baseline_benchmark"]), "--device", "cuda"]
        supervised_command(benchmark, folder/"benchmark.log", remaining, receipt_path, receipt)
        evidence = json.loads((folder/"benchmark.json").read_text())
        receipt.update(status="complete", benchmark_sha256=sha256(folder/"benchmark.json"),
                       benchmark_passed=any(g["passed"] for g in evidence["promotion"].values()))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
        write_json(receipt_path, receipt)
        # Delete only our own small lease file, never any experiment directory.
        if lease_path.exists() and json.loads(lease_path.read_text()).get("pid") == os.getpid():
            lease_path.unlink()


if __name__ == "__main__":
    main()
