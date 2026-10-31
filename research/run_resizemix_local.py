"""One independent ResizeMix seed repeat, supervised on the shared local GPU.

No resume, synthetic profiling, confirmation evaluation, hidden prediction or
upload is available. Existing/frozen supervisors and trainer sources are reused
without modification. FP16 must be explicit for a repeat of the T4 protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
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
from research.run_local import acquire_lease, now, sha256, supervised_command, used_seconds
from research.train_resizemix import validate_config
from research import benchmark as suite

SOURCE_FILES = ("train.py", "model.py", "data.py", "runtime.py", "research/train_resizemix.py",
    "research/resizemix.py", "research/train_shadow.py", "research/augmentations.py", "research/benchmark.py",
    "research/benchmark_resizemix.py", "research/run_local.py", "research/watchdog.py", "research/run_resizemix_local.py")


def build_commands(config_path, folder, baseline, precision):
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("ResizeMix repeat precision must be explicitly fp16, bf16 or fp32")
    training = [sys.executable, "research/train_resizemix.py", "--config", str(config_path),
                "--output-dir", str(folder), "--data-dir", str(ROOT/"data"), "--device", "cuda", "--precision", precision]
    benchmark = [sys.executable, "research/benchmark_resizemix.py", "--checkpoint", str(folder/"best.pt"),
                 "--output", str(folder/"benchmark.json"), "--baseline", str(baseline), "--device", "cuda"]
    return training, benchmark


def run(run_id, config_path, precision, minimum_seconds=2400):
    if not run_id.replace("_", "").isalnum() or run_id.startswith("stability"):
        raise ValueError("Use a unique independent-repeat run ID")
    if type(minimum_seconds) is not int or not 60 <= minimum_seconds <= 10800:
        raise ValueError("Minimum runtime must be from 60 through 10800 seconds")
    config_path = Path(config_path).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config.decode("utf-8"))
    validate_config(config)
    if config["seed"] == 2026:
        raise ValueError("This entrypoint only repeats a frozen primary using a different seed")
    source = {name: sha256(ROOT/name) for name in SOURCE_FILES}
    manifest_path = ROOT/"data/training_manifest.json"
    manifest_hash = sha256(manifest_path)
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    base = ROOT/"results/research"
    folder = base/run_id
    if folder.exists():
        raise FileExistsError("Repeat directory is immutable; no implicit restart or resume")
    baseline_path = ROOT/policy["baseline_benchmark"]
    training, benchmark = build_commands(folder/"input_config.json", folder, baseline_path, precision)
    if not baseline_path.is_file():
        raise FileNotFoundError("The current complete local baseline is required")
    baseline = json.loads(baseline_path.read_text())
    if (baseline.get("schema") != suite.SCHEMA or baseline.get("status") != "complete" or
        baseline.get("examples") != 5000 or baseline.get("checkpoint_sha256") != suite.BASELINE_CHECKPOINT_SHA):
        raise ValueError("A completed v3 trusted-original baseline is required before a repeat")
    baseline_hash = sha256(baseline_path)
    lease = base/"local_lease.json"
    acquire_lease(lease)
    started, receipt, owns_folder = time.perf_counter(), {}, False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        owns_folder = True
        receipt = {"run": run_id, "role": "development_repeat", "backend": "local", "trainer": "resizemix",
            "precision": precision, "status": "starting", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
            "started_at": now().isoformat(), "seed": config["seed"], "config_sha256": hashlib.sha256(raw_config).hexdigest(),
            "source_config_path": str(config_path), "input_config_path": str(folder/"input_config.json"),
            "source_sha256": source, "training_manifest_sha256": manifest_hash,
            "baseline_benchmark_sha256": baseline_hash,
            "confirmation_evaluated": False, "submitted": False, "hidden_images_loaded": False}
        with (folder/"input_config.json").open("xb") as stream:
            stream.write(raw_config)
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=folder/"supervisor.json")
        budget = min(policy["local_job_timeout_seconds"], policy["local_gpu_seconds_per_utc_day"]-used,
                     int((deadline-now()).total_seconds()))
        if budget < minimum_seconds:
            raise RuntimeError("Insufficient local budget for a complete ResizeMix repeat")
        if not torch.cuda.is_available() or (precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False)):
            raise RuntimeError("Requested native CUDA precision is unavailable")
        if source != {name: sha256(ROOT/name) for name in SOURCE_FILES} or sha256(manifest_path) != manifest_hash:
            raise RuntimeError("Repeat source or official data manifest changed before launch")
        receipt.update(status="training", reserved_seconds=budget, utc_budget_day=day.date().isoformat())
        write_json(folder/"supervisor.json", receipt)
        remaining = budget-(time.perf_counter()-started)-60
        if remaining < 1:
            raise TimeoutError("No supervised training time remains after reserving the full benchmark")
        supervised_command(training, folder/"training.log", remaining, folder/"supervisor.json", receipt)
        receipt["status"] = "benchmarking"
        write_json(folder/"supervisor.json", receipt)
        remaining = budget-(time.perf_counter()-started)
        if remaining < 1:
            raise TimeoutError("No supervised runtime remains for the full benchmark")
        supervised_command(benchmark, folder/"benchmark.log", remaining, folder/"supervisor.json", receipt)
        if (sha256(folder/"input_config.json") != receipt["config_sha256"] or
            source != {name: sha256(ROOT/name) for name in SOURCE_FILES} or sha256(manifest_path) != manifest_hash or
            sha256(baseline_path) != baseline_hash):
            raise RuntimeError("Immutable repeat inputs/source changed")
        evidence = json.loads((folder/"benchmark.json").read_text())
        if (evidence.get("schema") != policy["benchmark_schema"] or evidence.get("status") != "complete" or
            evidence.get("examples") != 5000 or evidence.get("training_examples") != 40000 or
            evidence.get("config") != config or evidence.get("confirmation_evaluated") is not False):
            raise ValueError("Repeat did not produce the complete fixed official development benchmark")
        receipt.update(status="complete", benchmark_sha256=sha256(folder/"benchmark.json"),
                       benchmark_passed=any(g["passed"] for g in evidence["promotion"].values()),
                       training_examples=40000, validation_examples=5000)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        if lease.exists() and json.loads(lease.read_text()).get("pid") == os.getpid():
            lease.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), required=True)
    parser.add_argument("--minimum-seconds", type=int, default=2400)
    args = parser.parse_args()
    result = run(args.run, args.config, args.precision, args.minimum_seconds)
    print(json.dumps({"run": args.run, "status": result["status"], "benchmark_passed": result["benchmark_passed"]}))


if __name__ == "__main__":
    main()
