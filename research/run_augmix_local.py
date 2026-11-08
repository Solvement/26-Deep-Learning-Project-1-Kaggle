"""Explicit, budgeted local AugMix-JSD preflight, training or complete benchmark.

Only this fixed candidate can run. The frozen local supervisor, shared lease,
daily accounting and PID-bound watchdog are imported without changing them.
An audited native preflight is mandatory before formal training. No action is
taken with no arguments; no prediction, upload, paid resource or cloud path.
Formal training currently admits only primary seed2026; a later seed2027 needs
its independently audited paired/frozen-primary route before it can run.
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import psutil
from runtime import environment, write_json
from research.run_local import (acquire_lease, assert_stopped, close_interrupted_attempt,
    now, sha256, supervised_command, used_seconds)
from research.train_augmix import (SOURCE_KEYS, validate_config, validate_preflight, cleanup_workers)
from research.process_tree_guard import run_guarded, TreeCleanupError, may_release_lease
from train import DEFAULT_CONFIG

SCHEMA = "research_augmix_local_supervisor_v1"


def bound_sources():
    keys = tuple(dict.fromkeys((*SOURCE_KEYS, "research/benchmark_augmix.py", "research/run_augmix_local.py",
        "research/benchmark.py", "research/run_local.py", "research/watchdog.py", "research/process_tree_guard.py",
        "train.py", "model.py", "data.py", "runtime.py")))
    return {key:sha256(ROOT/key) for key in keys}


def require(condition, message):
    if not condition: raise ValueError(message)


def admission_budget(policy, current, used, preflight=None):
    require(isinstance(current, datetime) and current.tzinfo is not None, "Aware current time is required")
    require(type(used) in (int, float) and math.isfinite(used) and used >= 0, "Used local seconds must be finite")
    deadline = datetime.fromisoformat(policy["stops_at"])
    require(deadline.tzinfo is not None and current < deadline, "Five-day research hard stop has passed")
    require(policy.get("concurrent_local_gpu_jobs") == 1 and policy.get("local_job_timeout_seconds") == 10800 and
            policy.get("local_gpu_seconds_per_utc_day") == 21600, "Fixed local concurrency/day/job limits changed")
    budget = min(10800, 21600-used, (deadline-current).total_seconds())
    require(budget >= 1, "No local GPU budget remains")
    if preflight is not None:
        median = preflight.get("median_step_seconds")
        require(type(median) in (int, float) and math.isfinite(median) and median > 0, "Finite measured pipeline duration required")
        required = 1.5*median*313*160+300
        require(required <= budget, "AugMix native throughput cannot fit fixed 1.5 margin plus300 finalization within remaining budget")
    return budget


def inputs(config_path, baseline, data_dir):
    manifest = data_dir/"training_manifest.json"
    document = json.loads(manifest.read_text())
    paths = [config_path, baseline, manifest, *(data_dir/name for name in document["files_sha256"])]
    require(all(path.is_file() for path in paths), "A trusted baseline or official AugMix input is missing")
    return {str(path.resolve()):sha256(path) for path in paths}


def cleanup_owned_pools(folder, receipt):
    if not may_release_lease(receipt):
        return False
    owner, created = receipt.get("child_pid"), receipt.get("child_created_at")
    if owner is None or created is None: return True
    # A partial spawn may not have reached the trainer's first pool receipt.
    # Only the independent OS tree boundary proves those children are gone.
    if receipt.get("process_tree_cleanup_verified") is not True:
        return False
    closed = True
    for path in Path(folder).glob("worker_pool_*.json"):
        record = json.loads(path.read_text())
        if record.get("owner_pid") == owner and abs(record.get("owner_created_at",0)-created) < .01:
            closed = cleanup_workers(record.get("workers",[])) and closed
    return closed


def execute(command, log, seconds, folder, receipt):
    try:
        run_guarded(command, log, seconds, folder/"supervisor.json", receipt, cwd=ROOT)
    finally:
        if not cleanup_owned_pools(folder, receipt):
            receipt["process_tree_cleanup_verified"] = False
            raise TreeCleanupError("Owned AugMix worker cleanup is unverified; retain the local lease")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", required=True, choices=("preflight", "train", "benchmark"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--resume", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    require(args.run.replace("_", "").isalnum() and len(args.run) <= 70 and not args.run.startswith("stability"), "Unique AugMix run ID required")
    require(not args.resume or args.mode == "train", "Resume is only allowed for the exact training run")
    require((args.mode == "train") == (args.preflight is not None), "Formal AugMix training requires its explicit native preflight")
    require((args.mode == "benchmark") == (args.checkpoint is not None), "Only standalone benchmark takes an existing checkpoint")
    config_path = args.config.resolve()
    config = {**DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    validate_config(config)
    require(args.mode != "train" or config["seed"] == 2026,
            "AugMix seed2027 formal repeat requires a future verified primary pairing/freeze route; only primary2026 is admitted")
    config_sha = sha256(config_path)
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    admission_budget(policy, now(), 0)
    baseline = ROOT/policy["baseline_benchmark"]
    binding = inputs(config_path, baseline, args.data_dir.resolve())
    if args.mode == "benchmark":
        candidate = args.checkpoint.resolve()
        candidate_paths = [candidate, *(candidate.parent/name for name in
            ("last.pt", "config.json", "input_config.json", "metrics.json", "history.json", "provenance.json", "environment.json"))]
        require(all(path.is_file() for path in candidate_paths), "Standalone AugMix benchmark needs its complete candidate artifacts")
        require(json.loads((candidate.parent/"config.json").read_text()) == config, "Standalone AugMix benchmark configuration differs")
        binding.update({str(path):sha256(path) for path in candidate_paths})
    sources = bound_sources()
    preflight = None
    if args.preflight:
        preflight_path = args.preflight.resolve()
        preflight_receipt_path = preflight_path.parent/"supervisor.json"
        preflight_receipt = json.loads(preflight_receipt_path.read_text())
        require(preflight_receipt.get("schema") == SCHEMA and preflight_receipt.get("mode") == "preflight" and
                preflight_receipt.get("status") == "complete" and preflight_receipt.get("source_sha256") == sources and
                preflight_receipt.get("artifact_sha256") == sha256(preflight_path), "Complete source-bound AugMix preflight supervisor required")
        assert_stopped(preflight_receipt)
        preflight = validate_preflight(json.loads(preflight_path.read_text()), config, config_sha)
        binding.update({str(preflight_path):sha256(preflight_path), str(preflight_receipt_path):sha256(preflight_receipt_path)})
    base = ROOT/"results/research"
    folder = base/args.run
    previous = None
    if args.resume:
        require((folder/"supervisor.json").is_file() and (folder/"last.pt").is_file(), "AugMix resume needs original supervisor and last state")
        previous = json.loads((folder/"supervisor.json").read_text())
        require(previous.get("schema") == SCHEMA and previous.get("mode") == "train" and
                previous.get("source_sha256") == sources and previous.get("input_sha256") == binding and
                previous.get("config_sha256") == config_sha and not (folder/"benchmark.json").exists(),
                "AugMix resume source/input/config changed or training already evaluated")
        assert_stopped(previous)
    else:
        require(not folder.exists(), "Use a fresh AugMix run directory; existing artifacts are immutable")
    lease = base/"local_lease.json"
    if lease.exists():
        existing_lease = json.loads(lease.read_text())
        require(existing_lease.get("cleanup_requires_reconciliation") is not True and
                existing_lease.get("process_tree_cleanup_verified") is not False,
                "The previous local process tree requires explicit cleanup reconciliation")
    acquire_lease(lease)
    started, owns_folder, receipt = time.perf_counter(), False, {}
    try:
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=folder/"supervisor.json" if args.resume else None)
        budget = admission_budget(policy, now(), used, preflight)
        if args.mode != "train": budget = min(budget, 300 if args.mode == "benchmark" else 180)
        if args.resume:
            archived = close_interrupted_attempt(folder/"supervisor.json")
            receipt = {"schema": SCHEMA, "run": args.run, "mode": "train", "backend": "local", "status": "starting",
                "started_at": now().isoformat(), "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
                "trainer": "augmix", "precision": "fp16", "resumed": True, "previous_attempt": archived.name,
                "resumed_checkpoint_sha256": sha256(folder/"last.pt"), "config_sha256": config_sha,
                "source_sha256": sources, "input_sha256": binding, "reserved_seconds": 0,
                "utc_budget_day": day.date().isoformat(), "confirmation_evaluated": False, "submitted": False}
            owns_folder = True
            # Recalculate after preserving the conservative prior reservation.
            used = used_seconds(base, day, day+timedelta(days=1), exclude=folder/"supervisor.json")
            budget = admission_budget(policy, now(), used, preflight)
        else:
            folder.mkdir(parents=True, exist_ok=False)
            owns_folder = True
        receipt = {"schema": SCHEMA, "run": args.run, "mode": args.mode, "backend": "local", "trainer": "augmix",
            "status": "benchmarking" if args.mode != "train" else "training", "precision": "fp16", "pid": os.getpid(),
            "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat(), "resumed": args.resume,
            "reserved_seconds": budget, "utc_budget_day": day.date().isoformat(), "config_sha256": config_sha,
            "source_sha256": sources, "input_sha256": binding, "confirmation_evaluated": False, "submitted": False}
        if args.resume: receipt.update(previous_attempt=archived.name, resumed_checkpoint_sha256=sha256(folder/"last.pt"))
        write_json(folder/"supervisor.json", receipt)
        if preflight:
            actual = environment()
            expected = preflight["environment"]
            require(all(actual.get(k) == expected.get(k) for k in ("gpu", "cuda_runtime")) and
                    actual["packages"]["torch"] == expected["packages"]["torch"], "Actual local GPU/precision/Torch differs from preflight")
        train_command = [sys.executable, "research/train_augmix.py", "--config", str(config_path),
            "--output-dir", str(folder), "--data-dir", str(args.data_dir.resolve()), "--device", "cuda", "--precision", "fp16"]
        benchmark_command = [sys.executable, "research/benchmark_augmix.py", "--checkpoint", str(args.checkpoint or folder/"best.pt"),
            "--output", str(folder/"benchmark.json"), "--baseline", str(baseline), "--data-dir", str(args.data_dir.resolve()), "--device", "cuda"]
        if args.mode == "preflight":
            command, artifact = train_command+["--preflight"], folder/"engineering_benchmark.json"
        elif args.mode == "benchmark":
            command, artifact = benchmark_command, folder/"benchmark.json"
        else:
            command, artifact = train_command+(["--resume"] if args.resume else []), folder/"benchmark.json"
        log = folder/("resume_"+str(time.time_ns())+".log" if args.resume else args.mode+".log")
        execute(command, log, budget-(time.perf_counter()-started), folder, receipt)
        if args.mode == "train":
            require(all(sha256(path) == digest for path, digest in binding.items()) and sources == bound_sources(), "Bound AugMix inputs or source changed during training")
            remaining = budget-(time.perf_counter()-started)
            if remaining < 1: raise TimeoutError("No bounded runtime remains for full AugMix benchmark")
            receipt["status"] = "benchmarking"
            write_json(folder/"supervisor.json", receipt)
            execute(benchmark_command, folder/"benchmark.log", remaining, folder, receipt)
        result = json.loads(artifact.read_text())
        require(result.get("status") == "complete", "AugMix task did not produce complete evidence")
        require(all(sha256(path) == digest for path, digest in binding.items()) and sources == bound_sources(), "Bound AugMix inputs or source changed during execution")
        if args.mode == "preflight": validate_preflight(result, config, config_sha)
        receipt.update(status="complete", artifact=artifact.name, artifact_sha256=sha256(artifact))
        if args.mode != "preflight": receipt["benchmark_passed"] = result["promotion"]["center80_shift20"]["passed"]
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder and receipt:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        # Do not infer cleanup from an empty or missing pool-record directory.
        # The Job Object/private session also owns not-yet-recorded workers.
        can_release = may_release_lease(receipt)
        if not can_release and owns_folder and receipt:
            receipt["status"] = "cleanup_unverified"
            write_json(folder/"supervisor.json", receipt)
            if lease.exists():
                current_lease = json.loads(lease.read_text())
                if (current_lease.get("pid") == os.getpid() and
                        abs(current_lease.get("process_created_at", 0)-psutil.Process().create_time()) < .01):
                    current_lease.update(cleanup_requires_reconciliation=True,
                        process_tree_cleanup_verified=False, supervisor=str(folder/"supervisor.json"))
                    write_json(lease, current_lease)
        if can_release and lease.exists():
            own = json.loads(lease.read_text())
            if own.get("pid") == os.getpid() and abs(own.get("process_created_at", 0)-psutil.Process().create_time()) < .01:
                lease.unlink()


if __name__ == "__main__": main()
