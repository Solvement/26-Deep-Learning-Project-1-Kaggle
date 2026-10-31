"""Explicit, <=180-second engineering profile of the unchanged AugMix preflight.

This diagnostic never admits training, measures accuracy, predicts or uploads.
The worker runs the exact source-bound 23-update preflight, plus its separate
overflow check. In-memory record_function wrappers preserve callable results
and restore every patched boundary. Only operator/range aggregates are saved;
worker CPU/PIL execution is outside the main-process profiler and cannot be
separated from its queue/IPC wait by this measurement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import functools
import hashlib
import json
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
SCHEMA = "research_augmix_profile_supervisor_v1"
SUMMARY_SCHEMA = "research_augmix_instrumented_profile_v1"
CONFIG = "research/configs/wrn22_augmix_jsd.json"
ANCHOR = "results/research/wrn22_augmix_jsd_native_fp16_s2026_v1"
ANCHOR_SHA = "dd8b866e4efdb1b0f3bcc3aaf573b63e104067b363df6818be2727fdf9faa56e"
CONFIG_SHA = "c7ba91cb25ff128dfe816b303cde43cd4678673e0847a9f5b5bca7c4c4541df8"
PINNED_SOURCES = {
    "train.py": "4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py": "79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py": "4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py": "75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
    "research/train_augmix.py": "4b975f8132f0608d7088a1d66a13008b702dac0387abe9ed8ff8cc15735689b0",
    "research/augmix.py": "9786de01bcbe170d99f40d7681220732946348e933297b7bd600be8932e04f5f",
    "research/train_shadow.py": "9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
    "research/augmentations.py": "718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
    "research/run_augmix_local.py": "61aee851bb6f2bbff757b83d80c8710c1437c330810f8408505227b31d9fe776",
    "research/process_tree_guard.py": "5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_seconds(policy, current, used):
    require(isinstance(current, datetime) and current.tzinfo is not None,
            "Aware current time required")
    require(type(used) in (int, float) and math.isfinite(used) and used >= 0,
            "Finite nonnegative local accounting required")
    require(policy.get("concurrent_local_gpu_jobs") == 1 and
            policy.get("local_gpu_seconds_per_utc_day") == 21600 and
            policy.get("local_job_timeout_seconds") == 10800,
            "Fixed campaign budget changed")
    stop = datetime.fromisoformat(policy["stops_at"])
    require(stop.tzinfo is not None, "Aware campaign stop required")
    seconds = min(180, 21600-used, (stop-current).total_seconds())
    require(seconds > 0, "No bounded diagnostic time remains")
    return seconds


def source_snapshot(root=ROOT):
    root = Path(root)
    values = {name: sha256(root/name) for name in PINNED_SOURCES}
    require(values == PINNED_SOURCES, "The already-bound AugMix pipeline changed")
    for name in ("research/profile_augmix.py", "research/run_local.py", "research/campaign.json"):
        values[name] = sha256(root/name)
    require(sha256(root/CONFIG) == CONFIG_SHA, "The fixed AugMix configuration changed")
    return values


@contextmanager
def instrument_boundaries(engine, model_type, range_factory, clock=time.perf_counter):
    """Instrument existing call boundaries without replacing their work/results.

    Queue wait is observed in the main process; augmentation runs in separate
    workers. Forward, loss and backward+step ranges are nested inside train_step.
    The outer engine proof and transfer have no separate callable boundaries;
    their operator costs remain visible only in profiler aggregates.
    """
    originals, samples = [], []
    state = {"phase": "normal", "step": 0}

    def patch(owner, attribute, label, *, queue=False, overflow=False):
        original = getattr(owner, attribute)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            prior = state["phase"]
            if overflow:
                state["phase"] = "overflow"
            if queue and state["phase"] == "normal":
                state["step"] += 1
            phase, step = state["phase"], state["step"]
            began = clock()
            try:
                with range_factory("augmix_profile."+phase+"."+label):
                    return original(*args, **kwargs)
            finally:
                samples.append({"phase": phase, "step": step, "stage": label,
                                "wall_seconds": max(0., clock()-began)})
                state["phase"] = prior
        originals.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)

    try:
        patch(engine.EpochPipeline, "__next__", "queue_wait", queue=True)
        patch(model_type, "forward", "model_forward")
        patch(engine, "augmix_loss", "loss")
        patch(engine, "optimizer_step", "backward_and_optimizer")
        patch(engine, "train_step", "train_step_inclusive")
        patch(engine, "overflow_smoke", "separate_overflow", overflow=True)
        yield samples
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def wall_summary(samples):
    groups = {}
    for row in samples:
        require(row["phase"] in ("normal", "overflow") and type(row["step"]) is int and
                type(row["wall_seconds"]) in (int, float) and
                math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0,
                "Invalid aggregate wall timing")
        # Match the original preflight's three warmups for normal ranges only.
        if row["phase"] == "normal" and row["step"] <= 3:
            continue
        key = row["phase"]+"."+row["stage"]
        groups.setdefault(key, []).append(row["wall_seconds"])
    return {key: {"calls": len(values), "total_seconds": sum(values),
                  "median_seconds": statistics.median(values),
                  "min_seconds": min(values), "max_seconds": max(values)}
            for key, values in sorted(groups.items())}


def operator_summary(events, limit=40):
    """Bounded aggregate names/counts/times only, never tensor/shape/stack data."""
    rows = []
    for event in events:
        def metric(name):
            value = float(getattr(event, name, 0.) or 0.)
            require(math.isfinite(value) and value >= 0, "Invalid profiler aggregate")
            return value/1000  # profiler microseconds -> milliseconds
        row = {"name": str(event.key)[:256], "calls": int(event.count),
               "cpu_total_ms": metric("cpu_time_total"),
               "cpu_self_ms": metric("self_cpu_time_total"),
               "device_total_ms": metric("device_time_total"),
               "device_self_ms": metric("self_device_time_total")}
        require(row["calls"] >= 0, "Invalid profiler count")
        rows.append(row)
    cpu = sorted(rows, key=lambda r: r["cpu_self_ms"], reverse=True)[:limit]
    device = sorted(rows, key=lambda r: r["device_self_ms"], reverse=True)[:limit]
    sync = [r for r in rows if any(p in r["name"].lower() for p in
            ("local_scalar", "synchronize", "memcpy", "aten::to", "aten::_to_copy"))]
    ranges = [r for r in rows if r["name"].startswith("augmix_profile.")]
    return {"top_self_cpu": cpu, "top_self_device": device,
            "synchronization_and_copy_operators": sync[:limit], "instrumented_ranges": ranges}


def cuda_activity_summary(events):
    """Count observed positive-duration CUDA activities, not supported capability."""
    count = 0
    for event in events:
        kind = str(getattr(event, "device_type", "")).rsplit(".", 1)[-1]
        duration = float(getattr(event, "device_time_total", 0.) or 0.)
        if kind == "CUDA" and math.isfinite(duration) and duration > 0:
            count += 1
    return {"cuda_activity_observed": count > 0, "positive_device_event_count": count,
            "cuda_decomposition_available": count > 0}


def verify_os_boundary(spec, ready, process, *, platform=None):
    """Read-only check of the current native Job/session before importing Torch."""
    platform = os.name if platform is None else platform
    if platform == "nt":
        import ctypes
        from ctypes import wintypes
        from research.process_tree_guard import _kernel
        kernel = _kernel()
        handle = kernel.OpenJobObjectW(0x0004, False, spec["job_name"])  # QUERY only
        require(bool(handle), "The private native Job Object does not exist")
        try:
            member = wintypes.BOOL()
            require(bool(kernel.IsProcessInJob(kernel.GetCurrentProcess(), handle, ctypes.byref(member))) and
                    bool(member.value), "This worker is not inside its exact native Job Object")
        finally:
            kernel.CloseHandle(handle)
    else:
        require(os.getpgrp() == process.pid and os.getsid(0) == process.pid,
                "The worker is not its private POSIX group/session leader")
        watcher = psutil.Process(ready["watchdog_pid"])
        require(abs(watcher.create_time()-ready["watchdog_created_at"]) < .01 and
                watcher.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD) and
                os.getpgid(watcher.pid) == process.pid and os.getsid(watcher.pid) == process.pid,
                "The private POSIX watchdog identity/session is not active")


def validate_active_guard(folder, receipt, process, parent, manifest_sha):
    """Bind canonical lease, runtime guard records, bootstrap command and OS."""
    lease_path = ROOT/"results/research/local_lease.json"
    require(lease_path.is_file(), "The canonical local GPU lease is missing")
    lease = json.loads(lease_path.read_text())
    require(lease.get("pid") == parent.pid and
            abs(lease.get("process_created_at", 0)-parent.create_time()) < .01 and
            lease.get("cleanup_requires_reconciliation") is not True and
            lease.get("process_tree_cleanup_verified") is not False and
            lease.get("status") not in ("cleanup_unverified", "unknown", "failed"),
            "The canonical GPU lease is foreign or requires cleanup reconciliation")
    matches = []
    for directory in folder.glob("process_tree_*"):
        try:
            spec = json.loads((directory/"spec.json").read_text())
            ready = json.loads((directory/"ready.json").read_text())
        except (OSError, ValueError):
            continue
        if (isinstance(ready, dict) and ready.get("pid") == process.pid and
                type(ready.get("created_at")) in (int, float) and
                abs(ready["created_at"]-process.create_time()) < .01):
            matches.append((directory.resolve(), spec, ready))
    require(len(matches) == 1, "Exactly one active guard for this worker is required")
    directory, spec, ready = matches[0]
    nonce = directory.name.removeprefix("process_tree_")
    guard_path = (ROOT/"research/process_tree_guard.py").resolve()
    script = Path(__file__).resolve()
    expected_method = "windows_job_v1" if os.name == "nt" else "posix_session_v1"
    require(isinstance(spec, dict) and re.fullmatch(r"[0-9a-f]{32}", nonce) and
            spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
            spec.get("nonce") == ready.get("nonce") == nonce and
            spec.get("host_pid") == parent.pid and
            abs(spec.get("host_created_at", 0)-parent.create_time()) < .01 and
            Path(spec.get("cwd", "")).resolve() == ROOT.resolve() and
            Path(spec.get("script", "")).resolve() == script and
            spec.get("script_sha256") == ready.get("script_sha256") == sha256(script) and
            spec.get("guard_source_sha256") == sha256(guard_path) == PINNED_SOURCES["research/process_tree_guard.py"] and
            spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha] and
            ready.get("method") == expected_method and
            type(spec.get("deadline_unix")) in (int, float) and
            time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
            spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None),
            "The active guard nonce/source/script/owner/deadline contract differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
            Path(command[1]).resolve() == guard_path and command[2] == "--bootstrap" and
            Path(command[3]).resolve() == directory/"spec.json",
            "The worker was not launched by the exact process-tree bootstrap")
    verify_os_boundary(spec, ready, process)


def validate_worker_owner(folder, manifest_sha):
    require(sha256(folder/"profile_inputs.json") == manifest_sha, "Diagnostic input manifest changed")
    receipt = json.loads((folder/"supervisor.json").read_text())
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "profiling" and
            receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
            type(receipt.get("reserved_seconds")) in (int, float) and
            math.isfinite(receipt["reserved_seconds"]) and 0 < receipt["reserved_seconds"] <= 180 and
            type(receipt.get("deadline_unix")) in (int, float) and
            math.isfinite(receipt["deadline_unix"]) and
            receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
            receipt.get("pid") == parent.pid and
            abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
            receipt.get("child_pid") == process.pid and
            abs(receipt.get("child_created_at", 0)-process.create_time()) < .01 and
            time.time() < receipt.get("deadline_unix", 0),
            "Worker requires its live exact guarded supervisor")
    validate_active_guard(folder, receipt, process, parent, manifest_sha)
    manifest = json.loads((folder/"profile_inputs.json").read_text())
    require(manifest["source_sha256"] == source_snapshot(), "Diagnostic source binding changed")
    require(all(sha256(path) == digest for path, digest in manifest["input_sha256"].items()),
            "Diagnostic official input binding changed")
    return manifest


def worker(folder, manifest_sha):
    manifest = validate_worker_owner(folder, manifest_sha)
    import torch
    from model import WideResNet
    from research import train_augmix as engine
    from runtime import write_json
    require(torch.cuda.is_available() and torch.profiler.ProfilerActivity.CUDA in
            torch.profiler.supported_activities(), "CUDA activity profiling is unavailable")
    config = engine.fixed_config()
    raw = (ROOT/CONFIG).read_bytes()
    began = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
            profile_memory=False, with_stack=False) as profiler:
        with instrument_boundaries(engine, WideResNet, torch.profiler.record_function) as samples:
            report = engine.preflight(config, folder, ROOT/"data", torch.device("cuda"), "fp16", raw)
    elapsed = time.perf_counter()-began
    engine.validate_preflight(report, config, CONFIG_SHA)
    require(manifest["source_sha256"] == source_snapshot(), "Bound source changed during diagnostic")
    result = {"schema": SUMMARY_SCHEMA, "status": "complete", "instrumented": True,
        "formal_admission_evidence": False, "accuracy_measured": False,
        "official_training_only": True, "confirmation_evaluated": False, "hidden_images_loaded": False,
        "attempted_normal_updates": 23, "successful_normal_updates": report["successful_updates"],
        "separate_overflow_passed": report["overflow_smoke"]["status"] == "passed",
        "preflight_artifact_sha256": sha256(folder/"engineering_benchmark.json"),
        "source_sha256": manifest["source_sha256"], "config_sha256": CONFIG_SHA,
        "canonical_lease_and_active_native_guard_verified": True,
        "profile_elapsed_seconds": elapsed,
        "instrumented_preflight_median_seconds": report["median_step_seconds"],
        "timed_wall_boundaries": wall_summary(samples),
        "operators": operator_summary(profiler.key_averages()),
        **cuda_activity_summary(profiler.events()),
        "interpretation": "Instrumented same-callable engineering profile; overhead and overlapping inclusive ranges prevent additive or formal runtime claims. Normal wall ranges omit three warmups; profiler operators include setup, all23 updates and the separately labeled overflow. CUDA capability does not prove captured CUDA activity: if cuda_activity_observed is false, device decomposition is unavailable and reported zero device fields do not mean zero GPU work; only CPU/queue evidence is usable. CPU/PIL workers are not profiled, so queue/IPC wait cannot separate their CPU cost. Transfer and outer finite/update proofs have no separate wrapper and appear only as aggregate operators. No tensor values, shapes, IDs, images, weights or raw traces are exported."}
    write_json(folder/"profile_summary.json", result)
    print(json.dumps({"profile_complete": True, "accuracy_measured": False}))


def parent(run):
    from runtime import write_json
    from research.run_local import acquire_lease, assert_stopped, now, used_seconds
    from research.run_augmix_local import inputs, bound_sources, cleanup_owned_pools
    from research.process_tree_guard import run_guarded, may_release_lease, TreeCleanupError
    from research import train_augmix as engine
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}", run or "") and
            not run.startswith("stability"), "A fresh diagnostic run ID is required")
    source = source_snapshot()
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    budget_seconds(policy, now(), 0)
    anchor = ROOT/ANCHOR
    require(sha256(anchor/"engineering_benchmark.json") == ANCHOR_SHA,
            "The actual uninstrumented native anchor differs")
    previous = json.loads((anchor/"supervisor.json").read_text())
    require(previous.get("status") == "complete" and previous.get("artifact_sha256") == ANCHOR_SHA and
            previous.get("process_tree_cleanup_verified") is True and
            previous.get("source_sha256") == bound_sources(), "Complete unchanged native anchor required")
    assert_stopped(previous)
    engine.validate_preflight(json.loads((anchor/"engineering_benchmark.json").read_text()),
                              engine.fixed_config(), CONFIG_SHA)
    binding = inputs(ROOT/CONFIG, ROOT/policy["baseline_benchmark"], ROOT/"data")
    binding.update({str(anchor/"engineering_benchmark.json"): ANCHOR_SHA,
                    str(anchor/"supervisor.json"): sha256(anchor/"supervisor.json")})
    base, folder = ROOT/"results/research", ROOT/"results/research"/run
    require(not folder.exists(), "Diagnostic artifacts are immutable; use a fresh run")
    lease = base/"local_lease.json"
    # A diagnostic has no need to reclaim a stale owner. A dead owner may have
    # failed before annotating its lease; explicit reconciliation must precede
    # any new diagnostic rather than relying on acquire_lease's stale rename.
    require(not lease.exists(), "Existing local GPU lease needs completion or explicit reconciliation")
    acquire_lease(lease)
    began, receipt, owns_folder = time.perf_counter(), {}, False
    try:
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        budget = budget_seconds(policy, now(), used_seconds(base, day, day+timedelta(days=1)))
        folder.mkdir(exist_ok=False)
        owns_folder = True
        write_json(folder/"profile_inputs.json", {"source_sha256": source, "input_sha256": binding})
        receipt = {"schema": SCHEMA, "backend": "local", "run": run, "mode": "profile",
            "status": "profiling", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
            "started_at": now().isoformat(), "reserved_seconds": budget,
            "deadline_unix": time.time()+budget, "source_sha256": source,
            "input_manifest_sha256": sha256(folder/"profile_inputs.json"),
            "confirmation_evaluated": False, "submitted": False}
        write_json(folder/"supervisor.json", receipt)
        command = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(folder),
                   "--_manifest-sha", receipt["input_manifest_sha256"]]
        try:
            run_guarded(command, folder/"profile.log", max(.001, budget-(time.perf_counter()-began)),
                        folder/"supervisor.json", receipt, cwd=ROOT)
        finally:
            if not cleanup_owned_pools(folder, receipt):
                receipt["process_tree_cleanup_verified"] = False
                raise TreeCleanupError("Diagnostic worker cleanup is unverified")
        require(source == source_snapshot() and all(sha256(p) == d for p,d in binding.items()),
                "Bound source or inputs changed during profile")
        result = json.loads((folder/"profile_summary.json").read_text())
        require(result.get("schema") == SUMMARY_SCHEMA and result.get("status") == "complete" and
                result.get("formal_admission_evidence") is False and result.get("accuracy_measured") is False,
                "Complete engineering-only profile summary required")
        receipt.update(status="complete", artifact="profile_summary.json",
                       artifact_sha256=sha256(folder/"profile_summary.json"))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder:
            receipt.update(elapsed_seconds=time.perf_counter()-began, finished_at=now().isoformat())
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            write_json(folder/"supervisor.json", receipt)
        if lease.exists():
            own = json.loads(lease.read_text())
            if own.get("pid") == os.getpid() and abs(own.get("process_created_at", 0)-psutil.Process().create_time()) < .01:
                if may_release_lease(receipt):
                    lease.unlink()
                else:
                    own.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False,
                               supervisor=str(folder/"supervisor.json"))
                    write_json(lease, own)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Fresh engineering diagnostic ID; never formal training")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        require(args.run is None and args._manifest_sha is not None,
                "Only a controlled worker manifest is allowed")
        folder = args._worker.resolve()
        require(folder.parent == (ROOT/"results/research").resolve(), "Controlled diagnostic directory required")
        worker(folder, args._manifest_sha)
    elif args.run:
        require(args._manifest_sha is None, "Worker-only argument")
        parent(args.run)
    else:
        require(args._manifest_sha is None, "Worker-only argument")
        parser.print_help()


if __name__ == "__main__":
    main()
