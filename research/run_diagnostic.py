"""Run one allowlisted GPU diagnostic under the shared local lease and budget.

No training campaign, submission, confirmation access or arbitrary shell command
is available. Every invocation requires a new immutable run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import psutil
from research.run_local import acquire_lease, now, sha256, supervised_command, used_seconds
from runtime import write_json

MODES = ("sam-speed", "shadow-benchmark", "sam-benchmark", "paired")
POLICIES = ("plain", "flip", "center80_shift20")


def build_command(args, folder, baseline):
    """Validate required files before acquiring a GPU lease or creating output."""
    inputs = []
    if args.mode == "sam-speed":
        if args.config is None:
            raise ValueError("sam-speed requires --config")
        config = args.config.resolve()
        inputs.append(config)
        command = [sys.executable, "research/train_sam.py", "--config", str(config),
                   "--output-dir", str(folder), "--device", "cuda", "--precision", "bf16",
                   "--benchmark-steps", "20"]
        output = folder/"engineering_benchmark.json"
    else:
        if args.checkpoint is None:
            raise ValueError("This diagnostic requires --checkpoint")
        checkpoint = args.checkpoint.resolve()
        if "stability_seed2027" in checkpoint.parts:
            raise ValueError("The validation-only stability repeat is excluded")
        inputs += [checkpoint, baseline]
        if args.mode == "paired":
            if args.candidate_benchmark is None or args.policy is None:
                raise ValueError("paired requires --candidate-benchmark and --policy")
            candidate = args.candidate_benchmark.resolve()
            trusted = ROOT/"results/regularized/best.pt"
            inputs += [candidate, trusted]
            output = folder/"paired.json"
            command = [sys.executable, "research/paired_compare.py", "--candidate-checkpoint", str(checkpoint),
                       "--candidate-benchmark", str(candidate), "--baseline-checkpoint", str(trusted),
                       "--baseline-benchmark", str(baseline), "--policy", args.policy, "--output", str(output),
                       "--device", "cuda"]
        else:
            entry = "research/benchmark_shadow.py" if args.mode == "shadow-benchmark" else "research/benchmark_sam.py"
            output = folder/"benchmark.json"
            command = [sys.executable, entry, "--checkpoint", str(checkpoint), "--baseline", str(baseline),
                       "--output", str(output), "--device", "cuda"]
    inputs.append(ROOT/command[1])
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("A required diagnostic input or entrypoint is missing")
    return command, output, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate-benchmark", type=Path)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.run.replace("_", "").isalnum() or args.run.startswith("stability"):
        parser.error("Use a unique alphanumeric exploratory diagnostic ID")
    if not 1 <= args.seconds <= 10800:
        parser.error("--seconds must be from 1 through 10800")
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    base = ROOT/"results/research"
    folder = base/args.run
    if folder.exists():
        raise FileExistsError("Use a new diagnostic directory; existing evidence is immutable")
    command, output, inputs = build_command(args, folder, ROOT/policy["baseline_benchmark"])
    lease = base/"local_lease.json"
    acquire_lease(lease)
    started, receipt, owns_folder = time.perf_counter(), {}, False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        owns_folder = True
        receipt = {"run": args.run, "mode": args.mode, "role": "diagnostic", "backend": "local",
                   "status": "starting", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
                   "started_at": now().isoformat(), "confirmation_evaluated": False, "submitted": False,
                   "supervisor_source_sha256": sha256(Path(__file__))}
        receipt["input_sha256"] = {str(path): sha256(path) for path in inputs}
        receipt_path = folder/"supervisor.json"
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=receipt_path)
        budget = min(args.seconds, policy["local_job_timeout_seconds"],
                     policy["local_gpu_seconds_per_utc_day"]-used, int((deadline-now()).total_seconds()))
        if budget < 1:
            raise RuntimeError("No bounded local diagnostic runtime remains")
        receipt.update(status="benchmarking", reserved_seconds=budget, utc_budget_day=day.date().isoformat())
        write_json(receipt_path, receipt)
        remaining = budget-(time.perf_counter()-started)
        if remaining < 1:
            raise TimeoutError("Diagnostic setup exhausted the remaining runtime")
        supervised_command(command, folder/"diagnostic.log", remaining, receipt_path, receipt)
        result = json.loads(output.read_text())
        if result.get("status") != "complete":
            raise ValueError("Diagnostic did not produce complete evidence")
        receipt.update(status="complete", artifact=output.name, artifact_sha256=sha256(output))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        if lease.exists() and json.loads(lease.read_text()).get("pid") == os.getpid():
            lease.unlink()


if __name__ == "__main__":
    main()
