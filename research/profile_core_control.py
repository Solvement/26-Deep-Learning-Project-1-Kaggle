"""One guarded <=180s diagnostic of unchanged WRN300/BF16 preflight costs.

No arguments show help. Profiler timings never admit formal training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import preflight_core_control as c
from research import profile_augmix as p

SCHEMA = "research_core_control_profile_supervisor_v1"
INPUT_SCHEMA = "research_core_control_profile_inputs_v1"
RESULT_SCHEMA = "research_core_control_profile_v1"
RECIPE = "wrn34_3_300_bf16"
CORE_SHA = "85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761"
PLAN = "research/core-control-profile-plan.md"
PLAN_SHA = "05bcf22ce267d1e867131b8271fe65b526e738eb407bbc28664c9ebfb329dd4c"
STAGES = ("probe_snapshots", "BN_before", "original_core_step", "state_and_probes", "BN_after")
INTERPRETATION = "Diagnostic only. Profiler overhead is included. All23 steps retain whole-step timing and every proof. Three complete warmups run outside capture; operator and wall aggregates cover exactly20 subsequent steps. Inclusive CPU ranges are not additive GPU costs. Missing observed CUDA activity means device decomposition unavailable, never zero GPU work. No training admission, traces, tensors or accuracy results are produced."


def source_snapshot():
    c.require(c.sha256(ROOT/"research/preflight_core_control.py") == CORE_SHA and
              c.sha256(ROOT/PLAN) == PLAN_SHA, "Frozen diagnostic dependency or plan changed")
    return {**c.source_snapshot(), PLAN: PLAN_SHA,
            "research/profile_core_control.py": c.sha256(__file__)}


def binding_for(data_dir, sources):
    paths, identity = c.official_inputs(data_dir)
    config_path = ROOT/c.WRN300_CONFIGS[2026][0]
    return {str(path.resolve()): c.sha256(path) for path in
            [*paths, config_path, *(ROOT/name for name in sources)]}, identity


def validate_worker(folder, manifest_sha):
    """Authenticate this entry's actual lease/OS boundary before Torch imports."""
    folder = Path(folder).resolve()
    c.require(folder.parent == (ROOT/"results/research").resolve() and
              re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
              c.sha256(folder/"profile_inputs.json") == manifest_sha, "Controlled profile manifest required")
    manifest, receipt = c.read(folder/"profile_inputs.json"), c.read(folder/"supervisor.json")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    c.require(receipt.get("schema") == SCHEMA and receipt.get("status") == "profiling" and
              receipt.get("backend") == "local" and receipt.get("mode") == "profile" and
              receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
              receipt.get("reserved_seconds") == 180 and type(receipt.get("reserved_seconds")) is int and
              type(receipt.get("deadline_unix")) in (int, float) and math.isfinite(receipt["deadline_unix"]) and
              started.timestamp() <= time.time()+.1 and time.time() < receipt["deadline_unix"] <= started.timestamp()+180.1 and
              receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
              receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at", 0)-process.create_time()) < .01,
              "Exact live bounded profile supervisor required")
    lease = c.read(ROOT/"results/research/local_lease.json")
    c.require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at", 0)-parent.create_time()) < .01 and
              lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False and
              lease.get("status") not in ("unknown", "failed", "cleanup_unverified"), "Canonical live local lease required")
    matches = []
    for directory in folder.glob("process_tree_*"):
        try:
            spec, ready = c.read(directory/"spec.json"), c.read(directory/"ready.json")
        except (OSError, ValueError):
            continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at", 0)-process.create_time()) < .01:
            matches.append((directory, spec, ready))
    c.require(len(matches) == 1, "Exactly one actual profile OS guard required")
    directory, spec, ready = matches[0]
    nonce, guard = directory.name.removeprefix("process_tree_"), ROOT/"research/process_tree_guard.py"
    c.require(re.fullmatch(r"[0-9a-f]{32}", nonce) and
              spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
              spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid and
              abs(spec.get("host_created_at", 0)-parent.create_time()) < .01 and
              Path(spec.get("cwd", "")).resolve() == ROOT.resolve() and
              Path(spec.get("script", "")).resolve() == Path(__file__).resolve() and
              spec.get("script_sha256") == ready.get("script_sha256") == c.sha256(__file__) and
              spec.get("guard_source_sha256") == c.sha256(guard) == c.PINS["research/process_tree_guard.py"] and
              spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha] and
              ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
              type(spec.get("deadline_unix")) in (int, float) and
              time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
              spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "Profile guard identity/source differs")
    command = process.cmdline()
    c.require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
              Path(command[1]).resolve() == guard.resolve() and command[2] == "--bootstrap" and
              Path(command[3]).resolve() == directory/"spec.json", "Actual profile bootstrap required")
    p.verify_os_boundary(spec, ready, process)
    sources = source_snapshot()
    config = c.configuration(2026, recipe=RECIPE)
    binding, identity = binding_for(Path(manifest["data_dir"]), sources)
    c.require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("recipe") == RECIPE and
              manifest.get("seed") == 2026 and type(manifest.get("seed")) is int and
              manifest.get("run") == receipt.get("run") == folder.name and
              manifest.get("effective_config") == config and manifest.get("effective_config_sha256") == c.json_sha(config) and
              manifest.get("config_sha256") == c.WRN300_CONFIGS[2026][1] and
              manifest.get("source_sha256") == receipt.get("source_sha256") == sources and
              manifest.get("data_identity") == identity and manifest.get("input_sha256") == receipt.get("input_sha256") == binding and
              receipt.get("confirmation_evaluated") is False and receipt.get("formal_admission_evidence") is False,
              "Exact fixed source/config/official input binding differs")
    c.require(isinstance(manifest.get("source_git_commit"),str) and re.fullmatch(r"[0-9a-f]{40}",manifest["source_git_commit"]), "Actual source commit required")
    return manifest


@contextmanager
def stage(samples, index, name, record_function, clock=time.perf_counter):
    began = clock()
    with record_function("core_profile."+name):
        yield
    samples.append((index, name, clock()-began))


def wall_aggregates(samples):
    c.require(len(samples) == 23*len(STAGES) and
              {(index, name) for index, name, _ in samples} == {(i, s) for i in range(23) for s in STAGES} and
              all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for _, _, t in samples), "Complete23-step stage timings required")
    return {name: {"calls":20, "total_seconds":sum(values), "median_seconds":statistics.median(values)}
            for name in STAGES for values in [[t for i, s, t in samples if i >= 3 and s == name]]}


def operators(events):
    events = list(events)
    result = p.operator_summary(events, limit=30)
    result["instrumented_ranges"] = [row for row in p.operator_summary(events, limit=10000)["top_self_cpu"]
                                     if row["name"].startswith("core_profile.")]
    return result


def validate_result(result, manifest):
    exact = {"schema": RESULT_SCHEMA, "status":"complete", "diagnostic_only":True,
             "formal_admission_evidence":False, "accuracy_measured":False, "confirmation_evaluated":False,
             "hidden_images_loaded":False, "external_data_loaded":False, "recipe":RECIPE, "seed":2026,
             "precision":"bf16", "grad_scaler_enabled":False, "grad_scaler_scale":1.,
             "parameter_count":4166298, "bn_modules":31, "batch_size":128, "source_examples_seen":2944,
             "unique_source_examples":2944,
             "warmup_updates":3, "timed_updates":20, "successful_updates":23, "skipped_updates":0,
             "captured_updates":20, "excluded_warmup_updates":3,
             "parameter_changed_steps":23, "ema_changed_steps":23, "finite_proof_steps":23,
             "bn_forward_counts":[23]*31, "ema_buffer_copies":0,
             "source_sha256":manifest["source_sha256"], "config_sha256":manifest["config_sha256"],
             "input_manifest_sha256":manifest["manifest_sha256"], "source_git_commit":manifest["source_git_commit"],
             "data_identity":manifest["data_identity"], "interpretation":INTERPRETATION}
    extra = {"wall_stages", "instrumented_whole_step", "operators", "cuda_activity", "environment", "environment_sha256"}
    c.require(set(result) == set(exact)|extra and all(type(result.get(k)) is type(v) and result[k] == v for k, v in exact.items()), "Typed diagnostic/proof-only result required")
    c.require(set(result["wall_stages"]) == set(STAGES) and all(set(r) == {"calls","total_seconds","median_seconds"} and type(r["calls"]) is int and r["calls"] == 20 and
              all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k] >= 0 for k in ("total_seconds","median_seconds"))
              for r in result["wall_stages"].values()), "Twenty timed wall ranges required")
    whole = result["instrumented_whole_step"]
    c.require(set(whole) == {"calls","median_seconds","total_seconds"} and whole["calls"] == 20 and
              all(type(whole[k]) in (int,float) and math.isfinite(whole[k]) and whole[k] > 0 for k in ("median_seconds","total_seconds")), "Instrumented full proof timings required")
    environment = result["environment"]
    c.require(set(environment) == {"device","gpu","cuda_runtime","cudnn","packages","amp_dtype","cudnn_benchmark","deterministic_algorithms"} and
              environment["device"] == "cuda" and isinstance(environment["gpu"],str) and bool(environment["gpu"]) and
              environment["amp_dtype"] == "torch.bfloat16" and result["environment_sha256"] == c.json_sha(environment), "Actual native hardware/runtime summary hash differs")
    c.require(set(result["operators"]) == {"top_self_cpu","top_self_device","synchronization_and_copy_operators","instrumented_ranges"}, "Only bounded profiler aggregates may be exported")
    for rows in result["operators"].values():
        c.require(isinstance(rows,list) and len(rows) <= 30, "Bounded aggregate lists required")
        for row in rows:
            c.require(set(row) == {"name","calls","cpu_total_ms","cpu_self_ms","device_total_ms","device_self_ms"} and
                      isinstance(row["name"],str) and len(row["name"]) <= 256 and type(row["calls"]) is int and row["calls"] >= 0 and
                      all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] >= 0 for k in ("cpu_total_ms","cpu_self_ms","device_total_ms","device_self_ms")), "Only scalar operator aggregates may be exported")
    ranges = result["operators"]["instrumented_ranges"]
    c.require(len(ranges) == len(STAGES) and {r["name"] for r in ranges} == {"core_profile."+s for s in STAGES} and
              all(r["calls"] == 20 for r in ranges), "All five actual captured CPU ranges must have exactly20 calls")
    activity = result["cuda_activity"]
    c.require(set(activity) == {"cuda_activity_observed","positive_device_event_count","cuda_decomposition_available"} and
              type(activity["positive_device_event_count"]) is int and activity["positive_device_event_count"] >= 0 and
              activity["cuda_activity_observed"] is (activity["positive_device_event_count"] > 0) and
              activity["cuda_decomposition_available"] is activity["cuda_activity_observed"], "Observed positive device activity required for decomposition")
    json.dumps(result, allow_nan=False)


def worker(folder, manifest_sha):
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # Process DLL isolation after authentication, before Torch.
    import copy
    import hashlib
    import numpy as np
    import torch
    import train as core
    from data import load_training_arrays
    from model import build_model, count_parameters
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) and
              torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(), "Actual native BF16 and profiler support required")
    config = manifest["effective_config"]
    runtime.seed_all(2026)
    images, labels, train_ids, _ = load_training_arrays(Path(manifest["data_dir"]))
    c.require(len(train_ids) == 40000 and hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest() == c.DATA_IDENTITY["train_indices_sha256"], "Actual official40k training IDs required")
    device = torch.device("cuda")
    train_x, train_y = core.tensor_images(images[train_ids], device), torch.as_tensor(labels[train_ids], device=device)
    model = build_model({k: config[k] for k in ("depth","widen_factor","dropout","num_classes")}).to(device=device, memory_format=torch.channels_last)
    c.require(count_parameters(model) == count_parameters(model, True) == 4166298, "Actual parameter count differs")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train()
    parameters, averages = list(model.parameters()), list(ema.parameters())
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    c.require(len(bns) == 31 and all(m.num_batches_tracked.item() == 0 for m in bns) and
              all(q.dtype == torch.float32 for q in [*parameters, *averages]), "Original scratch BN/master precision required")
    order = torch.randperm(len(train_y), device=device)
    selected = train_ids[order[:23*128].cpu().numpy()]
    c.require(len(np.unique(selected)) == 2944, "2944 unique actual training sources required")
    samples, durations = [], []
    def run_step(index):
        # One original body for both windows: proof, RNG and SGD/EMA order stay unchanged.
        torch.cuda.synchronize()
        began = time.perf_counter()
        with stage(samples, index, "probe_snapshots", torch.profiler.record_function):
            before, before_ema = parameters[0].detach().clone(), averages[0].detach().clone()
        with stage(samples, index, "BN_before", torch.profiler.record_function):
            counters = [m.num_batches_tracked.item() for m in bns]
        with stage(samples, index, "original_core_step", torch.profiler.record_function):
            row = c.core_step(core, torch, model, ema, optimizer, train_x, train_y, order, index, config, index, "shadow", scaler)
        with stage(samples, index, "state_and_probes", torch.profiler.record_function):
            state_values = [*model.state_dict().values(), *ema.state_dict().values(),
                            *(value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))]
            finite = bool(torch.stack([torch.isfinite(q).all() for q in state_values]).all().item())
            changed = (parameters[0]-before).abs().max().item()
            ema_changed = (averages[0]-before_ema).abs().max().item()
        with stage(samples, index, "BN_after", torch.profiler.record_function):
            c.require(finite and changed > 0 and ema_changed > 0 and row["updates"] == index+1 and
                      all(m.num_batches_tracked.item() == old+1 for m, old in zip(bns, counters)), "Original full finite/update/BN proofs required")
            row.update(parameter_probe_max_delta=changed, ema_probe_max_delta=ema_changed, all_parameters_finite=True, bn_single_forward=True)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-began)
        return row
    for index in range(3):
        row = run_step(index)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False) as profiler:
        for index in range(3,23):
            row = run_step(index)
    env = runtime.environment()
    hardware = {k: env[k] for k in ("device","gpu","cuda_runtime","cudnn","packages","amp_dtype","cudnn_benchmark","deterministic_algorithms")}
    result = {"schema": RESULT_SCHEMA, "status":"complete", "diagnostic_only":True,
        "formal_admission_evidence":False, "accuracy_measured":False, "confirmation_evaluated":False,
        "hidden_images_loaded":False, "external_data_loaded":False, "recipe":RECIPE, "seed":2026,
        "precision":"bf16", "grad_scaler_enabled":scaler.is_enabled(), "grad_scaler_scale":scaler.get_scale(),
        "parameter_count":count_parameters(model), "bn_modules":len(bns), "batch_size":128, "source_examples_seen":23*128,
        "unique_source_examples":len(np.unique(selected)),
        "warmup_updates":3, "timed_updates":20, "successful_updates":row["updates"], "skipped_updates":0,
        "captured_updates":20, "excluded_warmup_updates":3,
        "parameter_changed_steps":23, "ema_changed_steps":23, "finite_proof_steps":23,
        "bn_forward_counts":[int(m.num_batches_tracked.item()) for m in bns], "ema_buffer_copies":0,
        "source_sha256":manifest["source_sha256"], "config_sha256":manifest["config_sha256"], "input_manifest_sha256":manifest_sha,
        "source_git_commit":manifest["source_git_commit"],"data_identity":manifest["data_identity"],
        "wall_stages":wall_aggregates(samples), "instrumented_whole_step":{"calls":20,"median_seconds":statistics.median(durations[3:]),"total_seconds":sum(durations[3:])},
        "operators":operators(profiler.key_averages()), "cuda_activity":p.cuda_activity_summary(profiler.events()),
        "environment":hardware, "environment_sha256":c.json_sha(hardware),
        "interpretation":INTERPRETATION}
    validate_result(result, {**manifest, "manifest_sha256":manifest_sha})
    c.write(folder/"profile_summary.json", result)


def parent(run, data_dir):
    c.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}", run or "") and not run.startswith("stability"), "Fresh controlled run required")
    sources, config = source_snapshot(), c.configuration(2026, recipe=RECIPE)
    binding, identity = binding_for(data_dir, sources)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    c.require(re.fullmatch(r"[0-9a-f]{40}",commit), "Actual source revision required")
    account, policy = c.accounting(), c.read(ROOT/"research/campaign.json")
    base, folder = ROOT/"results/research", ROOT/"results/research"/run
    lease = base/"local_lease.json"
    c.require(not folder.exists() and not lease.exists(), "Existing artifacts/lease require explicit reconciliation")
    def budget():
        now = account["now"]()
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return c.bounded_seconds(policy, now, account["used_seconds"](base, day, day+timedelta(days=1)))
    budget()
    from research.process_tree_guard import run_guarded, may_release_lease
    c.claim_lease(lease)
    began, owns, receipt = time.perf_counter(), False, {}
    try:
        seconds = budget()
        folder.mkdir(parents=True, exist_ok=False)
        owns = True
        started = account["now"]()
        manifest = {"schema":INPUT_SCHEMA,"run":run,"recipe":RECIPE,"seed":2026,"data_dir":str(data_dir),
            "effective_config":config,"effective_config_sha256":c.json_sha(config),"config_sha256":c.WRN300_CONFIGS[2026][1],
            "source_sha256":sources,"input_sha256":binding,"data_identity":identity}
        manifest["source_git_commit"] = commit
        c.write(folder/"profile_inputs.json", manifest)
        digest = c.sha256(folder/"profile_inputs.json")
        receipt = {"schema":SCHEMA,"run":run,"backend":"local","mode":"profile","status":"profiling",
            "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":started.isoformat(),
            "reserved_seconds":int(seconds),"deadline_unix":started.timestamp()+seconds,
            "source_sha256":sources,"input_sha256":binding,"input_manifest_sha256":digest,
            "confirmation_evaluated":False,"formal_admission_evidence":False}
        c.write(folder/"supervisor.json", receipt)
        remaining = receipt["deadline_unix"]-time.time()-2
        c.require(remaining > 0, "Profile reservation exhausted")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",digest],
            folder/"profile.log", remaining, folder/"supervisor.json", receipt, cwd=ROOT)
        validate_result(c.read(folder/"profile_summary.json"), {**manifest,"manifest_sha256":digest})
        c.require(source_snapshot() == sources and all(c.sha256(path) == sha for path, sha in binding.items()) and
                  time.time() <= receipt["deadline_unix"], "Complete bounded unchanged profile required")
        receipt.update(status="complete",artifact="profile_summary.json",artifact_sha256=c.sha256(folder/"profile_summary.json"))
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__)
        raise
    finally:
        if owns:
            receipt.update(finished_at=account["now"]().isoformat(),elapsed_seconds=time.perf_counter()-began)
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            c.write(folder/"supervisor.json", receipt)
        if lease.exists():
            owner = c.read(lease)
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time()) < .01:
                if c.lease_can_release(owner, receipt):
                    lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"))
                    c.write(lease, owner)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        c.require(args.run is None and args._manifest_sha is not None, "Only guarded worker arguments allowed")
        worker(args._worker, args._manifest_sha)
    elif args.run:
        c.require(args._manifest_sha is None, "Worker-only argument")
        parent(args.run, args.data_dir.resolve())
    else:
        c.require(args._manifest_sha is None, "Worker-only argument")
        parser.print_help()


if __name__ == "__main__":
    main()
