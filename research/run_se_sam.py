"""Explicit, budgeted local SE-SAM preflight, training or complete benchmark.

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
from research.train_se_sam import (ENGINEERING_SCHEMA, SOURCE_KEYS, fixed_config,
    model_configuration, research_provenance, validate_config)
from research.benchmark_se_sam import (SAM_BENCHMARK, SAM_BENCHMARK_SHA256,
    SAM_CHECKPOINT, SAM_CHECKPOINT_SHA256, BASELINE_BENCHMARK_SHA256, SAM_FREEZE_SHA256)
from train import DEFAULT_CONFIG

SCHEMA = "research_se_sam_supervisor_v1"


def bound_sources():
    keys = tuple(dict.fromkeys((*SOURCE_KEYS, "research/benchmark_se_sam.py", "research/run_se_sam.py",
        "research/benchmark_sam.py", "research/benchmark.py", "research/run_local.py", "research/watchdog.py",
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
        require(type(median) in (int, float) and math.isfinite(median) and median > 0, "Finite measured SAM duration required")
        required = 1.5*median*313*160+300
        require(required <= budget, "SE native throughput cannot fit fixed 1.5 margin plus300 finalization within remaining budget")
    return budget


def validate_preflight(report, config, config_sha256, source_sha256):
    validate_config(config)
    expected = research_provenance("bf16", config, config_sha256)
    require(report.get("schema") == ENGINEERING_SCHEMA and report.get("status") == "complete" and
            report.get("synthetic_only") is True and report.get("precision") == "bf16" and
            report.get("parameters") == 4332654 and report.get("model_config") == model_configuration(config) and
            report.get("config") == config and report.get("input_config_sha256") == config_sha256 and
            report.get("research_training") == expected, "SE preflight native/source/config/architecture identity differs")
    require(source_sha256 == bound_sources(), "SE preflight supervising method source differs")
    require(report.get("batch_size") == 128 and report.get("warmup_steps") == 3 and report.get("timed_steps") == 20 and
            report.get("successful_updates") == 23 and report.get("sam_forward_passes") == 46 and
            report.get("skipped_updates") == 0 and report.get("gate_count") == 9 and
            report.get("gate_changed_steps") == report.get("gate_gradient_verified_steps") == 23 and
            report.get("second_pass_bn_verified") is True, "SE preflight lacks every real gate update or two-pass BN proof")
    gate_reports = report.get("gate_evidence", [])
    require(len(gate_reports) == 23, "Every preflight update needs per-gate evidence")
    expected_gates = {f"stage{stage}.{block}" for stage in (1, 2, 3) for block in range(3)}
    for step in gate_reports:
        require(set(step) == expected_gates, "Nine fixed SE gates must be present in each step")
        for item in step.values():
            require(all(type(item.get(k)) in (int, float) and math.isfinite(item[k]) and item[k] > 0
                        for k in ("gradient_l2", "update_l2")), "SE preflight gate gradient/update must be positive and finite")
    median = report.get("median_step_seconds")
    require(type(median) in (int, float) and math.isfinite(median) and median > 0 and
            type(report.get("p90_step_seconds")) in (int, float) and math.isfinite(report["p90_step_seconds"]) and
            report["p90_step_seconds"] >= median and
            math.isclose(report.get("required_training_and_benchmark_seconds", -1), 1.5*median*313*160+300, rel_tol=0, abs_tol=1e-9),
            "SE preflight did not retain the predeclared runtime guard")
    env = report.get("environment", {})
    require(env.get("device") == "cuda" and env.get("amp_dtype") == "torch.bfloat16" and bool(env.get("gpu")) and
            env.get("packages", {}).get("torch"), "SE preflight must be actual native CUDA BF16")
    require(all(report.get(k) is False for k in ("training_data_loaded", "confirmation_evaluated", "hidden_images_loaded", "external_data_loaded")),
            "SE engineering preflight must be synthetic only")
    require(all(type(report.get(k)) in (int, float) and math.isfinite(report[k]) and report[k] >= 0
                for k in ("mean_first_loss", "mean_sam_loss", "mean_perturbation_norm")) and
            report["mean_perturbation_norm"] <= .050001,
            "SE preflight losses and perturbation norm must be finite and within fixed rho")
    return report


def inputs(config_path, baseline, data_dir):
    require(sha256(SAM_CHECKPOINT) == SAM_CHECKPOINT_SHA256 and sha256(SAM_BENCHMARK) == SAM_BENCHMARK_SHA256 and
            sha256(baseline) == BASELINE_BENCHMARK_SHA256 and
            sha256(ROOT/"results/research/wrn22_sam_delivery/frozen.json") == SAM_FREEZE_SHA256,
            "The predeclared fixed SAM reference artifacts changed")
    manifest = data_dir/"training_manifest.json"
    document = json.loads(manifest.read_text())
    paths = [config_path, baseline, SAM_CHECKPOINT, SAM_BENCHMARK, manifest,
             ROOT/"results/research/wrn22_sam_delivery/frozen.json"]
    paths.extend(data_dir/name for name in document["files_sha256"])
    require(all(path.is_file() for path in paths), "A fixed reference or official input is missing")
    return {str(path.resolve()):sha256(path) for path in paths}


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
    require(args.run.replace("_", "").isalnum() and len(args.run) <= 70 and not args.run.startswith("stability"), "Unique SE run ID required")
    require(not args.resume or args.mode == "train", "Resume is only allowed for the exact training run")
    require((args.mode == "train") == (args.preflight is not None), "Formal SE training requires its explicit native preflight")
    require((args.mode == "benchmark") == (args.checkpoint is not None), "Only standalone benchmark takes an existing checkpoint")
    config_path = args.config.resolve()
    config = {**DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    validate_config(config)
    require(args.mode != "train" or config["seed"] == 2026,
            "SE seed2027 formal repeat requires a future verified primary pairing/freeze route; only primary2026 is admitted")
    config_sha = sha256(config_path)
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    admission_budget(policy, now(), 0)
    baseline = ROOT/policy["baseline_benchmark"]
    binding = inputs(config_path, baseline, args.data_dir.resolve())
    if args.mode == "benchmark":
        candidate = args.checkpoint.resolve()
        candidate_paths = [candidate, *(candidate.parent/name for name in
            ("last.pt", "config.json", "input_config.json", "metrics.json", "history.json", "provenance.json", "environment.json"))]
        require(all(path.is_file() for path in candidate_paths), "Standalone SE benchmark needs its complete candidate artifacts")
        require(json.loads((candidate.parent/"config.json").read_text()) == config, "Standalone SE benchmark configuration differs")
        binding.update({str(path):sha256(path) for path in candidate_paths})
    sources = bound_sources()
    preflight = None
    if args.preflight:
        preflight_path = args.preflight.resolve()
        preflight_receipt_path = preflight_path.parent/"supervisor.json"
        preflight_receipt = json.loads(preflight_receipt_path.read_text())
        require(preflight_receipt.get("schema") == SCHEMA and preflight_receipt.get("mode") == "preflight" and
                preflight_receipt.get("status") == "complete" and preflight_receipt.get("source_sha256") == sources and
                preflight_receipt.get("artifact_sha256") == sha256(preflight_path), "Complete source-bound SE preflight supervisor required")
        assert_stopped(preflight_receipt)
        preflight = validate_preflight(json.loads(preflight_path.read_text()), config, config_sha, sources)
        binding.update({str(preflight_path):sha256(preflight_path), str(preflight_receipt_path):sha256(preflight_receipt_path)})
    base = ROOT/"results/research"
    folder = base/args.run
    previous = None
    if args.resume:
        require((folder/"supervisor.json").is_file() and (folder/"last.pt").is_file(), "SE resume needs original supervisor and last state")
        previous = json.loads((folder/"supervisor.json").read_text())
        require(previous.get("schema") == SCHEMA and previous.get("mode") == "train" and
                previous.get("source_sha256") == sources and previous.get("input_sha256") == binding and
                previous.get("config_sha256") == config_sha and not (folder/"benchmark.json").exists(),
                "SE resume source/input/config changed or training already evaluated")
        assert_stopped(previous)
    else:
        require(not folder.exists(), "Use a fresh SE run directory; existing artifacts are immutable")
    lease = base/"local_lease.json"
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
                "trainer": "se_sam", "precision": "bf16", "resumed": True, "previous_attempt": archived.name,
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
        receipt = {"schema": SCHEMA, "run": args.run, "mode": args.mode, "backend": "local", "trainer": "se_sam",
            "status": "benchmarking" if args.mode != "train" else "training", "precision": "bf16", "pid": os.getpid(),
            "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat(), "resumed": args.resume,
            "reserved_seconds": budget, "utc_budget_day": day.date().isoformat(), "config_sha256": config_sha,
            "source_sha256": sources, "input_sha256": binding, "confirmation_evaluated": False, "submitted": False}
        if args.resume: receipt.update(previous_attempt=archived.name, resumed_checkpoint_sha256=sha256(folder/"last.pt"))
        write_json(folder/"supervisor.json", receipt)
        if preflight:
            actual = environment()
            expected = preflight["environment"]
            require(all(actual.get(k) == expected.get(k) for k in ("device", "gpu", "cuda_runtime", "amp_dtype")) and
                    actual["packages"]["torch"] == expected["packages"]["torch"], "Actual local GPU/precision/Torch differs from preflight")
        train_command = [sys.executable, "research/train_se_sam.py", "--config", str(config_path),
            "--output-dir", str(folder), "--data-dir", str(args.data_dir.resolve()), "--device", "cuda", "--precision", "bf16"]
        benchmark_command = [sys.executable, "research/benchmark_se_sam.py", "--checkpoint", str(args.checkpoint or folder/"best.pt"),
            "--output", str(folder/"benchmark.json"), "--baseline", str(baseline), "--data-dir", str(args.data_dir.resolve()), "--device", "cuda"]
        if args.mode == "preflight":
            command, artifact = train_command+["--benchmark-steps", "20"], folder/"engineering_benchmark.json"
        elif args.mode == "benchmark":
            command, artifact = benchmark_command, folder/"benchmark.json"
        else:
            command, artifact = train_command+(["--resume"] if args.resume else []), folder/"benchmark.json"
        log = folder/("resume_"+str(time.time_ns())+".log" if args.resume else args.mode+".log")
        supervised_command(command, log, budget-(time.perf_counter()-started), folder/"supervisor.json", receipt)
        if args.mode == "train":
            require(all(sha256(path) == digest for path, digest in binding.items()) and sources == bound_sources(), "Bound SE inputs or source changed during training")
            remaining = budget-(time.perf_counter()-started)
            if remaining < 1: raise TimeoutError("No bounded runtime remains for full SE benchmark")
            receipt["status"] = "benchmarking"
            write_json(folder/"supervisor.json", receipt)
            supervised_command(benchmark_command, folder/"benchmark.log", remaining, folder/"supervisor.json", receipt)
        result = json.loads(artifact.read_text())
        require(result.get("status") == "complete", "SE task did not produce complete evidence")
        require(all(sha256(path) == digest for path, digest in binding.items()) and sources == bound_sources(), "Bound SE inputs or source changed during execution")
        if args.mode == "preflight": validate_preflight(result, config, config_sha, sources)
        receipt.update(status="complete", artifact=artifact.name, artifact_sha256=sha256(artifact))
        if args.mode != "preflight": receipt["benchmark_passed"] = result["dual_promotion"]["passed"]
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder and receipt:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        if lease.exists():
            own = json.loads(lease.read_text())
            if own.get("pid") == os.getpid() and abs(own.get("process_created_at", 0)-psutil.Process().create_time()) < .01:
                lease.unlink()


if __name__ == "__main__": main()
