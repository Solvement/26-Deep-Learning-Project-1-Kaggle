"""One <=180s real-image WRN300 preflight with only outer proofs rescheduled.

No arguments show help. No profiler, formal training, accuracy or upload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
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

SCHEMA = "research_batched_core_control_supervisor_v1"
INPUT_SCHEMA = "research_batched_core_control_inputs_v1"
RESULT_SCHEMA = "research_batched_core_control_preflight_v1"
RECIPE = "wrn34_3_300_bf16"
CORE_SHA = "85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761"
HELPER = "research/batched_finite_proof.py"
HELPER_SHA = "c7bb0c161fab8b3a0361c1179f44d59d991428dd9d7bc8485f28d7cd61ee59ff"
PLAN = "research/batched-core-control-plan.md"
PLAN_SHA = "b0a58f1922c4efdddb2a086787ef8c326cf57ba88e322b1291ecbe50ab190cec"
PROFILE = "results/research/wrn300_core_profile_20261102_v2/profile_summary.json"
PROFILE_SHA = "1aca18a6632bbecb20888ff0e2d1e6cb8b2c2f0495f65a11e12c9966c7455af4"
PROFILE_SOURCE = "research/profile_core_control_v2.py"
PROFILE_SOURCE_SHA = "deb61e7fbb24be9e83c6dfb956ef74fb5df074b560090533a841a7caab7ec060"
TIMING_SCOPE = "Uninstrumented real-image original core_step plus all original and batched outer finite/inventory/BN/update/probe proofs and allocations;3 warmup+20 timed full steps. Setup is charged by supervisor. No profiler or formal training."


def source_snapshot():
    pins = {"research/preflight_core_control.py": CORE_SHA, HELPER: HELPER_SHA,
            PLAN: PLAN_SHA, PROFILE_SOURCE: PROFILE_SOURCE_SHA}
    c.require(all(c.sha256(ROOT/name) == digest for name, digest in pins.items()),
              "Frozen core, helper, profile source or independent plan changed")
    return {**c.source_snapshot(), **pins,
            "research/preflight_batched_core_control.py": c.sha256(__file__)}


def profile_evidence():
    c.require(c.sha256(ROOT/PROFILE) == PROFILE_SHA, "Actual completed profile evidence changed")
    report = c.read(ROOT/PROFILE)
    c.require(report.get("schema") == "research_core_control_profile_v2" and
              report.get("status") == "complete" and report.get("diagnostic_only") is True and
              report.get("formal_admission_evidence") is False and report.get("successful_updates") == 23 and
              report.get("captured_updates") == 20 and report.get("recipe") == RECIPE,
              "Complete diagnostic-only profile required; it is not admission timing")


def binding_for(data_dir, sources):
    profile_evidence()
    paths, identity = c.official_inputs(data_dir)
    paths += [ROOT/PROFILE, ROOT/c.WRN300_CONFIGS[2026][0], *(ROOT/name for name in sources)]
    return {str(path.resolve()): c.sha256(path) for path in paths}, identity


def inventory_template(model, ema, torch):
    """Derive exact expected state and lazy-SGD momentum metadata at setup."""
    states = [model.state_dict(), ema.state_dict()]
    c.require(states[0] and list(states[0]) == list(states[1]), "Complete matched model/EMA state required")
    template = {}
    for role, state in zip(("model", "ema"), states):
        for name, value in state.items():
            dtype = torch.int64 if name.endswith("num_batches_tracked") else torch.float32
            c.require(torch.is_tensor(value) and value.dtype == dtype and
                      (dtype != torch.int64 or value.ndim == 0), "Original FP32/scalar-int64 state required")
            template[role+":"+name] = (str(value.dtype), tuple(value.shape), str(value.device))
    parameters = dict(model.named_parameters())
    c.require(parameters and set(parameters).issubset(states[0]), "Complete named parameter inventory required")
    for name, value in parameters.items():
        c.require(value.dtype == torch.float32, "Original FP32 momentum ownership required")
        template["momentum:"+name] = (str(value.dtype), tuple(value.shape), str(value.device))
    return template


def full_state_values(model, ema, optimizer, template, torch):
    """No omission/extra/shape/dtype/device changes are accepted after SGD."""
    parameters = dict(model.named_parameters())
    c.require(set(optimizer.state) == set(parameters.values()), "Every parameter needs its actual SGD momentum state")
    entries = [(role+":"+name, value) for role, state in
               (("model", model.state_dict()), ("ema", ema.state_dict())) for name, value in state.items()]
    for name, parameter in parameters.items():
        state = optimizer.state[parameter]
        c.require(set(state) == {"momentum_buffer"}, "Only the complete original SGD momentum state is supported")
        entries.append(("momentum:"+name, state["momentum_buffer"]))
    c.require(len(entries) == len(template) and {name for name, _ in entries} == set(template),
              "Full exact model/EMA/momentum inventory required")
    for name, value in entries:
        c.require(torch.is_tensor(value) and
                  (str(value.dtype), tuple(value.shape), str(value.device)) == template[name],
                  "Full state dtype/shape/device differs from initial inventory")
    return [value for _, value in entries]


def execute_step(core, torch, model, ema, optimizer, train_x, train_y, order,
                 index, config, scaler, parameters, averages, bns, template, proof):
    """Only this outer proof schedule differs; all work is inside worker timing."""
    before, before_ema = parameters[0].detach().clone(), averages[0].detach().clone()
    counters = proof.snapshot_bn_counters([m.num_batches_tracked for m in bns])
    row = c.core_step(core, torch, model, ema, optimizer, train_x, train_y, order,
                      index, config, index, "shadow", scaler)
    state_values = full_state_values(model, ema, optimizer, template, torch)
    finite = bool(proof.all_finite(state_values).item())
    changed = (parameters[0]-before).abs().max().item()
    ema_changed = (averages[0]-before_ema).abs().max().item()
    bn = proof.prove_bn_increment(counters, [m.num_batches_tracked for m in bns])
    c.require(finite and changed > 0 and ema_changed > 0 and row["updates"] == index+1 and
              bn["all_incremented"], "Original complete finite/real SGD/EMA/single-BN proofs required")
    row.update(parameter_probe_max_delta=changed, ema_probe_max_delta=ema_changed,
               all_parameters_finite=True, bn_single_forward=True,
               bn_counters_before=bn["before"], bn_counters_after=bn["after"],
               state_tensor_count=len(state_values))
    return row


def guard_seconds(median):
    c.require(type(median) in (int, float) and math.isfinite(median) and median > 0,
              "Finite positive actual median required")
    return 1.5*median*313*300+300


def validate_result(report, manifest):
    exact = {"schema": RESULT_SCHEMA, "status":"complete", "engineering_only":True,
             "formal_training_admitted":False, "accuracy_evaluated":False,
             "confirmation_evaluated":False, "hidden_images_loaded":False, "external_data_loaded":False,
             "recipe":RECIPE, "seed":2026, "training_precision":"bf16", "native_bf16_without_emulation":True,
             "grad_scaler_enabled":False, "disabled_scaler_scale":1.,
             "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
             "parameter_count":4166298, "trainable_parameter_count":4166298, "bn_modules":31,
             "epochs":300, "planned_formal_attempted_updates":93900, "training_pool_examples":40000,
             "evaluation_examples":0, "batch_size":128, "warmup_updates":3, "timed_updates":20,
             "successful_updates":23, "skipped_updates":0, "source_examples_seen":2944,
             "unique_source_examples":2944, "bn_forward_counts":[23]*31, "ema_buffer_copies":0,
             "state_tensor_count":480, "state_float32_count":418, "state_int64_count":62,
             "preflight_state_discarded":True, "profile_evidence_sha256":PROFILE_SHA,
             "source_sha256":manifest["source_sha256"], "config_sha256":manifest["config_sha256"],
             "effective_config_sha256":manifest["effective_config_sha256"],
             "input_manifest_sha256":manifest["manifest_sha256"], "source_git_commit":manifest["source_git_commit"],
             "data_identity":manifest["data_identity"], "timing_scope":TIMING_SCOPE}
    extra = {"steps", "step_seconds", "median_step_seconds", "full_step_seconds", "setup_seconds",
             "conservative_pipeline_seconds", "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes",
             "environment", "environment_sha256", "inventory_sha256",
             "ordered_training_indices_sha256", "ordered_official_ids_sha256"}
    c.require(set(report) == set(exact)|extra and
              all(type(report.get(k)) is type(v) and report[k] == v for k, v in exact.items()),
              "Exact engineering-only source-bound preflight report required")
    c.require(manifest["config_sha256"] == c.WRN300_CONFIGS[2026][1] and
              manifest["data_identity"] == c.DATA_IDENTITY and
              manifest["effective_config_sha256"] == c.json_sha(c.configuration(2026, recipe=RECIPE)),
              "Exact unchanged WRN300 configuration and official split required")
    rows, durations = report["steps"], report["step_seconds"]
    c.require(len(rows) == len(durations) == 23 and
              all(type(s) in (int,float) and math.isfinite(s) and s > 0 for s in durations),
              "23 timed complete successful updates required")
    for index, row in enumerate(rows):
        fields = {"loss", "gradient_norm", "lr", "mixed", "updates", "logits_dtype",
                  "parameter_probe_max_delta", "ema_probe_max_delta", "all_parameters_finite",
                  "bn_single_forward", "bn_counters_before", "bn_counters_after", "state_tensor_count"}
        c.require(set(row) == fields and type(row["updates"]) is int and row["updates"] == index+1 and
                  row["all_parameters_finite"] is True and row["bn_single_forward"] is True and
                  row["bn_counters_before"] == [index]*31 and row["bn_counters_after"] == [index+1]*31 and
                  all(type(v) is int for v in row["bn_counters_before"]+row["bn_counters_after"]) and
                  type(row["state_tensor_count"]) is int and row["state_tensor_count"] == 480 and
                  type(row["mixed"]) is bool and row["logits_dtype"] == "torch.bfloat16" and
                  all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] > 0 for k in
                      ("loss", "gradient_norm", "lr", "parameter_probe_max_delta", "ema_probe_max_delta")) and
                  row["lr"] == .1*(index+1)/(5*313), "Complete original gradient/update/probe/BN proofs required")
    median = statistics.median(durations[3:])
    c.require(all(type(report[key]) in (int,float) and math.isfinite(report[key]) and report[key] > 0
                  for key in ("median_step_seconds", "full_step_seconds", "conservative_pipeline_seconds", "setup_seconds")),
              "Finite positive numeric timing scalars required; booleans are forbidden")
    c.require(report["median_step_seconds"] == median and report["full_step_seconds"] == sum(durations) and
              report["conservative_pipeline_seconds"] == guard_seconds(median) and
              report["setup_seconds"] > 0, "Full uninstrumented timing and unchanged guard required")
    for key in ("peak_gpu_memory_bytes", "peak_gpu_reserved_bytes"):
        c.require(type(report[key]) is int and report[key] > 0, "Actual peak CUDA memory required")
    c.require(report["peak_gpu_reserved_bytes"] >= report["peak_gpu_memory_bytes"], "Reserved peak cannot be below allocated peak")
    for key in ("inventory_sha256", "ordered_training_indices_sha256", "ordered_official_ids_sha256"):
        c.require(isinstance(report[key],str) and re.fullmatch(r"[0-9a-f]{64}",report[key]), "Actual inventory and source identity hashes required")
    env = report["environment"]
    environment_keys = {"python", "python_version", "platform", "packages", "device", "cuda_runtime", "cudnn", "gpu",
                        "gpu_memory_GiB", "amp_dtype", "cudnn_benchmark", "deterministic_algorithms",
                        "process_local_cudnn_path_entries_removed"}
    package_keys = {"torch", "torchvision", "numpy", "Pillow", "matplotlib", "nbformat", "reportlab"}
    c.require(type(env) is dict and set(env) == environment_keys and
              all(isinstance(env[key],str) and env[key].strip() for key in ("python", "python_version", "platform", "gpu")) and
              type(env["packages"]) is dict and set(env["packages"]) == package_keys and
              all(value is None or isinstance(value,str) and value.strip() for value in env["packages"].values()) and
              all(isinstance(env["packages"][key],str) and env["packages"][key].strip() for key in ("torch", "numpy", "Pillow")) and
              re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:\+[A-Za-z0-9_.-]+)?",env["packages"]["torch"]) and
              env["device"] == "cuda" and env["amp_dtype"] == "torch.bfloat16" and
              isinstance(env["cuda_runtime"],str) and re.fullmatch(r"[0-9]+\.[0-9]+",env["cuda_runtime"]) and
              type(env["cudnn"]) is int and env["cudnn"] > 0 and
              type(env["gpu_memory_GiB"]) in (int,float) and math.isfinite(env["gpu_memory_GiB"]) and env["gpu_memory_GiB"] > 0 and
              env["cudnn_benchmark"] is True and env["deterministic_algorithms"] is False and
              type(env["process_local_cudnn_path_entries_removed"]) is int and env["process_local_cudnn_path_entries_removed"] >= 0 and
              report["environment_sha256"] == c.json_sha(env), "Complete actual native runtime/GPU/package context required")
    c.json_sha(report)  # Reject NaN/Inf anywhere in the final typed JSON.


def worker(folder, manifest_sha):
    setup_began = time.perf_counter()
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # Authentication and real OS containment precede Torch.
    import copy
    import numpy as np
    import torch
    import train as core
    from data import load_training_arrays
    from model import build_model, count_parameters
    from research import batched_finite_proof as proof
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
              "Actual native CUDA BF16 required; emulation/FP16/CPU forbidden")
    config = manifest["effective_config"]
    runtime.seed_all(2026)
    images, labels, train_ids, _ = load_training_arrays(Path(manifest["data_dir"]))
    c.require(len(train_ids) == 40000 and hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest() ==
              c.DATA_IDENTITY["train_indices_sha256"], "Actual complete official40k pool required")
    device = torch.device("cuda")
    train_x, train_y = core.tensor_images(images[train_ids], device), torch.as_tensor(labels[train_ids], device=device)
    model = build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}).to(device=device, memory_format=torch.channels_last)
    c.require(count_parameters(model) == count_parameters(model,True) == 4166298, "Actual parameter count differs")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"],weight_decay=config["weight_decay"],nesterov=True)
    scaler = torch.amp.GradScaler("cuda",enabled=False)
    model.train()
    parameters, averages = list(model.parameters()), list(ema.parameters())
    bns = [m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    c.require(len(parameters) == len(averages) == 98 and len(bns) == 31 and
              proof.snapshot_bn_counters([m.num_batches_tracked for m in bns]).cpu().tolist() == [0]*31 and
              all(q.dtype == torch.float32 for q in [*parameters,*averages]), "Original scratch BN/FP32 masters required")
    template = inventory_template(model,ema,torch)
    floats = sum(row[0] == "torch.float32" for row in template.values())
    integers = sum(row[0] == "torch.int64" for row in template.values())
    c.require(len(template) == 480 and floats == 418 and integers == 62, "Full original state inventory required")
    order = torch.randperm(len(train_y),device=device)
    selected = train_ids[order[:23*128].cpu().numpy()]
    original_ids = np.load(Path(manifest["data_dir"])/"official_train_ids.npy",allow_pickle=False)[selected]
    c.require(len(np.unique(selected)) == 2944, "2944 unique actual sources required")
    rows, durations = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_began
    for index in range(23):
        torch.cuda.synchronize()
        began = time.perf_counter()
        row = execute_step(core,torch,model,ema,optimizer,train_x,train_y,order,index,
                           config,scaler,parameters,averages,bns,template,proof)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-began)
        rows.append(row)
    env = runtime.environment()
    report = dict(schema=RESULT_SCHEMA,status="complete",engineering_only=True,formal_training_admitted=False,
        accuracy_evaluated=False,confirmation_evaluated=False,hidden_images_loaded=False,external_data_loaded=False,
        recipe=RECIPE,seed=2026,training_precision="bf16",native_bf16_without_emulation=True,
        grad_scaler_enabled=scaler.is_enabled(),disabled_scaler_scale=scaler.get_scale(),
        master_parameter_dtype="torch.float32",ema_parameter_dtype="torch.float32",
        parameter_count=count_parameters(model),trainable_parameter_count=count_parameters(model,True),bn_modules=len(bns),
        epochs=300,planned_formal_attempted_updates=93900,training_pool_examples=len(train_ids),evaluation_examples=0,
        batch_size=128,warmup_updates=3,timed_updates=20,successful_updates=rows[-1]["updates"],skipped_updates=0,
        source_examples_seen=2944,unique_source_examples=len(np.unique(selected)),bn_forward_counts=rows[-1]["bn_counters_after"],
        ema_buffer_copies=0,state_tensor_count=len(template),state_float32_count=floats,state_int64_count=integers,
        preflight_state_discarded=True,profile_evidence_sha256=PROFILE_SHA,source_sha256=manifest["source_sha256"],
        config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"],
        input_manifest_sha256=manifest_sha,source_git_commit=manifest["source_git_commit"],data_identity=manifest["data_identity"],
        timing_scope=TIMING_SCOPE,steps=rows,step_seconds=durations,median_step_seconds=statistics.median(durations[3:]),
        full_step_seconds=sum(durations),setup_seconds=setup_seconds,conservative_pipeline_seconds=guard_seconds(statistics.median(durations[3:])),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        environment=env,environment_sha256=c.json_sha(env),inventory_sha256=c.json_sha(template),
        ordered_training_indices_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        ordered_official_ids_sha256=hashlib.sha256(np.ascontiguousarray(original_ids).tobytes()).hexdigest())
    validate_result(report,{**manifest,"manifest_sha256":manifest_sha})
    c.write(Path(folder)/"batched_core_preflight.json",report)


def validate_worker(folder, manifest_sha):
    """Authenticate this entry's actual lease/OS boundary before Torch imports."""
    folder = Path(folder).resolve()
    c.require(folder.parent == (ROOT/"results/research").resolve() and
              re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
              c.sha256(folder/"inputs.json") == manifest_sha, "Controlled batched preflight manifest required")
    manifest, receipt = c.read(folder/"inputs.json"), c.read(folder/"supervisor.json")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    c.require(receipt.get("schema") == SCHEMA and receipt.get("status") == "benchmarking" and
              receipt.get("backend") == "local" and receipt.get("mode") == "preflight" and
              receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
              receipt.get("reserved_seconds") == 180 and type(receipt.get("reserved_seconds")) is int and
              type(receipt.get("deadline_unix")) in (int, float) and math.isfinite(receipt["deadline_unix"]) and
              started.timestamp() <= time.time()+.1 and time.time() < receipt["deadline_unix"] <= started.timestamp()+180.1 and
              receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
              receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at", 0)-process.create_time()) < .01,
              "Exact live bounded batched preflight supervisor required")
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
    c.require(len(matches) == 1, "Exactly one actual batched preflight OS guard required")
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
              spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "Batched preflight guard identity/source differs")
    command = process.cmdline()
    c.require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
              Path(command[1]).resolve() == guard.resolve() and command[2] == "--bootstrap" and
              Path(command[3]).resolve() == directory/"spec.json", "Actual batched preflight bootstrap required")
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
              receipt.get("confirmation_evaluated") is False and receipt.get("formal_training_admitted") is False,
              "Exact fixed source/config/official input binding differs")
    c.require(isinstance(manifest.get("source_git_commit"),str) and re.fullmatch(r"[0-9a-f]{40}",manifest["source_git_commit"]), "Actual source commit required")
    return manifest


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
        c.write(folder/"inputs.json", manifest)
        digest = c.sha256(folder/"inputs.json")
        receipt = {"schema":SCHEMA,"run":run,"backend":"local","mode":"preflight","status":"benchmarking",
            "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":started.isoformat(),
            "reserved_seconds":int(seconds),"deadline_unix":started.timestamp()+seconds,
            "source_sha256":sources,"input_sha256":binding,"input_manifest_sha256":digest,
            "confirmation_evaluated":False,"formal_training_admitted":False}
        c.write(folder/"supervisor.json", receipt)
        remaining = receipt["deadline_unix"]-time.time()-2
        c.require(remaining > 0, "Batched preflight reservation exhausted")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",digest],
            folder/"preflight.log", remaining, folder/"supervisor.json", receipt, cwd=ROOT)
        validate_result(c.read(folder/"batched_core_preflight.json"), {**manifest,"manifest_sha256":digest})
        c.require(source_snapshot() == sources and all(c.sha256(path) == sha for path, sha in binding.items()) and
                  time.time() <= receipt["deadline_unix"], "Complete bounded unchanged batched preflight required")
        receipt.update(status="complete",artifact="batched_core_preflight.json",artifact_sha256=c.sha256(folder/"batched_core_preflight.json"))
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
