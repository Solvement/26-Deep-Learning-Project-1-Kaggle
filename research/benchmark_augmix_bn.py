"""Guarded synthetic two-arm FP16 BN-backend diagnostic; never admission evidence.

The only arm change is a BatchNorm2d forward-local cuDNN-disabled context.
Convolutions, sourceB128/view384, model, loss, SGD, EMA and init_scale256 stay
unchanged. No official arrays, accuracy, predictions, weights or traces are saved.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import profile_augmix as shared

SCHEMA = "research_augmix_bn_supervisor_v1"
RESULT_SCHEMA = "research_augmix_bn_backend_diagnostic_v1"
PROFILE_SHA = "82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66"
REFERENCE = "results/research/wrn22_augmix_jsd_profile_s2026_v1/profile_summary.json"
REFERENCE_SHA = "fb288e65e77e9a75512b370bfd39bdf80aa9fde3f5a4aa5cc8f4105b7d458dc3"
require, sha256, budget_seconds = shared.require, shared.sha256, shared.budget_seconds


def source_snapshot():
    require(sha256(ROOT/"research/profile_augmix.py") == PROFILE_SHA,
            "The audited shared diagnostic helpers changed")
    result = shared.source_snapshot()
    result["research/benchmark_augmix_bn.py"] = sha256(__file__)
    return result


def native_bn_class():
    """Construct lazily, so direct worker rejection happens before Torch import."""
    import torch
    class NativeBatchNorm2d(torch.nn.BatchNorm2d):
        def forward(self, inputs):
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(inputs)
    return NativeBatchNorm2d


def convert_bn(model):
    """Replace before optimizer creation; preserve names, tensor bytes and mode."""
    import torch
    cls, count = native_bn_class(), 0
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            replacement = cls(child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats)
            state = child.state_dict()
            first = next(iter(state.values()), None)
            if first is not None:
                replacement = replacement.to(device=first.device,
                    dtype=first.dtype if first.is_floating_point() else None)
            replacement.load_state_dict(state, strict=True)
            replacement.train(child.training)
            for key, parameter in replacement.named_parameters():
                parameter.requires_grad_(dict(child.named_parameters())[key].requires_grad)
            setattr(model, name, replacement)
            count += 1
        else:
            count += convert_bn(child)
    return count


def state_digest(state):
    """Hash private in-memory tensor bytes; never serialize the actual state."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()+b"\0"+str(tensor.dtype).encode()+b"\0")
        digest.update(str(tuple(tensor.shape)).encode()+b"\0"+tensor.numpy().tobytes())
    return digest.hexdigest()


def scalar_difference(left, right):
    import torch
    require(set(left) == set(right), "Paired diagnostic tensor topology differs")
    maximum, square, count = 0., 0., 0
    for name in left:
        require(left[name].shape == right[name].shape, "Paired diagnostic shapes differ")
        delta = left[name].detach().cpu().double()-right[name].detach().cpu().double()
        require(bool(torch.isfinite(delta).all()), "Nonfinite backend difference")
        if delta.numel():
            maximum = max(maximum, float(delta.abs().max()))
            square += float(delta.square().sum())
            count += delta.numel()
    return {"max_absolute": maximum, "root_mean_square": math.sqrt(square/max(1,count)),
            "elements_compared": count, "bitwise_equivalence_claimed": False}


def synthetic_batch():
    """One fixed synthetic RGB source batch and the existing three-view transform."""
    import numpy as np
    import torch
    from research.train_augmix import prepare_batch
    pixels = np.random.Generator(np.random.PCG64(2026)).integers(
        0, 256, (128,32,32,3), dtype=np.uint8)
    labels = np.arange(128, dtype=np.int64) % 10
    return prepare_batch(pixels, labels, np.random.Generator(np.random.PCG64(9026)), torch.device("cpu"))


def run_arm(template, prepared, device, dtype, backend, rng_state):
    """Exactly23 original train_step calls; return aggregates plus private RAM probes."""
    import torch
    from research import train_augmix as engine
    from research.train_shadow import restore_rng
    require(backend in ("cudnn", "native_bn_only"), "Unknown BN diagnostic arm")
    require(torch.backends.cudnn.enabled, "The convolution cuDNN flag must start enabled")
    model = copy.deepcopy(template)
    initial = state_digest(model.state_dict())
    changed_modules = convert_bn(model) if backend == "native_bn_only" else 0
    require(initial == state_digest(model.state_dict()), "BN conversion changed initial state bytes")
    model = model.to(device=device, memory_format=torch.channels_last).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    # Conversion must precede this exact parameter-topology binding.
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005, nesterov=True)
    scaler = torch.amp.GradScaler(device.type, init_scale=256., enabled=device.type == "cuda")
    bns = [m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    require(bool(bns), "BN diagnostic needs actual BN modules")
    if backend == "native_bn_only": require(changed_modules == len(bns), "Every BN must be converted")
    restore_rng(rng_state, device)
    walls, cuda_ms, losses, first = [], [], [], {}
    counters_passed = changed_steps = finite_gradient_steps = 0
    bn_nodes, conv_calls = set(), [0]
    def observe(_module, _args, output):
        if "output" not in first: first["output"] = output.detach().cpu().clone()
    def observe_bn(_module, _args, output):
        node = type(output.grad_fn).__name__
        bn_nodes.add(node)
        if device.type == "cuda":
            require(("CudnnBatchNormBackward" in node) if backend == "cudnn" else ("NativeBatchNorm" in node),
                    "The actual autograd BN backend differs from the declared arm")
    def observe_conv(_module, _args):
        require(torch.backends.cudnn.enabled, "cuDNN was disabled outside BN during convolution")
        conv_calls[0] += 1
    handles = [model.register_forward_hook(observe)]
    handles.extend(module.register_forward_hook(observe_bn) for module in bns)
    handles.extend(module.register_forward_pre_hook(observe_conv) for module in model.modules()
                   if isinstance(module,torch.nn.Conv2d))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    try:
        for index in range(23):
            before = next(model.parameters()).detach().clone()
            counters = [m.num_batches_tracked.detach().clone() for m in bns]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            began = time.perf_counter()
            report = engine.train_step(model, optimizer, scaler, prepared, device, dtype,
                ema=ema, successful_updates=index)
            gradients = {name:p.grad for name,p in model.named_parameters()}
            require(all(value is not None for value in gradients.values()), "A model gradient is missing")
            norms = torch.stack(torch._foreach_norm(list(gradients.values()), float("inf")))
            maximum = float(norms.amax())
            require(math.isfinite(maximum) and maximum > 0, "Finite positive gradients required")
            finite_gradient_steps += 1
            require(report["updated"] is True and report["forward_calls"] == 1 and
                    report["model_views"] == 3*report["source_examples"], "A true single3B update is required")
            require(all(torch.equal(m.num_batches_tracked, old+1) for m,old in zip(bns,counters)),
                    "BN must update exactly once per original train_step")
            counters_passed += 1
            require(not torch.equal(next(model.parameters()).detach(), before), "The SGD probe did not change")
            changed_steps += 1
            require(math.isfinite(float(torch.stack(torch._foreach_norm([p.detach() for p in model.parameters()],float("inf"))).amax())),
                    "Nonfinite model parameter")
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-began
            if index >= 3:
                walls.append(elapsed)
                if device.type == "cuda":
                    milliseconds = float(start.elapsed_time(end))
                    require(math.isfinite(milliseconds) and milliseconds > 0, "Actual positive CUDA event timing required")
                    cuda_ms.append(milliseconds)
            losses.append(report["loss"])
            if index == 0:
                first["gradients"] = {k:v.detach().cpu().clone() for k,v in gradients.items()}
                first["bn_buffers"] = {name:v.detach().cpu().clone() for name,v in model.named_buffers()
                    if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    finally:
        for handle in handles: handle.remove()
    result = {"backend": backend, "precision":"fp16" if device.type == "cuda" else "cpu_fp32_test_only",
        "initial_state_sha256":initial, "parameters":sum(p.numel() for p in model.parameters()),
        "batchnorm_modules":len(bns), "converted_before_optimizer":backend == "native_bn_only",
        "observed_bn_backward_nodes":sorted(bn_nodes), "convolution_cudnn_enabled_calls":conv_calls[0],
        "warmup_steps":3, "timed_steps":20, "successful_updates":23, "skipped_updates":0,
        "single_forward_calls":23, "bn_single_update_verified_steps":counters_passed,
        "finite_positive_gradient_steps":finite_gradient_steps, "actual_parameter_changed_steps":changed_steps,
        "timed_step_wall_seconds":walls, "timed_step_cuda_ms":cuda_ms,
        "median_step_wall_seconds":__import__("statistics").median(walls),
        "mean_loss":sum(losses)/len(losses),
        "peak_allocated_gpu_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    return result, first


def validate_worker(folder, manifest_sha):
    require(sha256(folder/"diagnostic_inputs.json") == manifest_sha, "BN diagnostic manifest differs")
    receipt = json.loads((folder/"supervisor.json").read_text())
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "profiling" and
        receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
        type(receipt.get("reserved_seconds")) in (int,float) and math.isfinite(receipt["reserved_seconds"]) and
        0 < receipt["reserved_seconds"] <= 180 and type(receipt.get("deadline_unix")) in (int,float) and
        math.isfinite(receipt["deadline_unix"]) and time.time() < receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
        started.timestamp() <= time.time()+.1 and receipt.get("pid") == parent.pid and
        abs(receipt.get("process_created_at",0)-parent.create_time()) < .01 and
        receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at",0)-process.create_time()) < .01,
        "BN worker needs its exact bounded live supervisor")
    lease = json.loads((ROOT/"results/research/local_lease.json").read_text())
    require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at",0)-parent.create_time()) < .01 and
        lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False,
        "Canonical BN diagnostic lease is missing, foreign or unknown")
    matches = []
    for path in folder.glob("process_tree_*"):
        try:
            spec, ready = [json.loads((path/name).read_text()) for name in ("spec.json","ready.json")]
        except (OSError,ValueError): continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at",0)-process.create_time()) < .01:
            matches.append((path.resolve(),spec,ready))
    require(len(matches) == 1, "Exactly one active BN process-tree guard required")
    path,spec,ready = matches[0]
    nonce = path.name.removeprefix("process_tree_")
    guard = (ROOT/"research/process_tree_guard.py").resolve()
    require(re.fullmatch(r"[0-9a-f]{32}",nonce) and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
        spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid and
        abs(spec.get("host_created_at",0)-parent.create_time()) < .01 and
        Path(spec.get("cwd","")).resolve() == ROOT.resolve() and Path(spec.get("script","")).resolve() == Path(__file__).resolve() and
        spec.get("script_sha256") == ready.get("script_sha256") == sha256(__file__) and
        spec.get("guard_source_sha256") == sha256(guard) == shared.PINNED_SOURCES["research/process_tree_guard.py"] and
        spec.get("arguments") == ["--_worker",str(folder),"--_manifest-sha",manifest_sha] and
        ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
        type(spec.get("deadline_unix")) in (int,float) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
        spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "BN guard identity/source/deadline differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
        Path(command[1]).resolve() == guard and command[2] == "--bootstrap" and Path(command[3]).resolve() == path/"spec.json",
        "BN worker must be launched by the exact OS bootstrap")
    shared.verify_os_boundary(spec,ready,process)
    manifest = json.loads((folder/"diagnostic_inputs.json").read_text())
    require(manifest == {"source_sha256":source_snapshot(),"config_sha256":shared.CONFIG_SHA,
        "reference_profile_sha256":REFERENCE_SHA}, "BN diagnostic sources/config/reference changed")
    require(sha256(ROOT/REFERENCE) == REFERENCE_SHA, "Actual profile reference changed")
    return manifest


def worker(folder, manifest_sha):
    manifest = validate_worker(folder,manifest_sha)  # Before Torch or CUDA import.
    import torch
    from runtime import seed_all, environment, write_json
    from model import build_model, count_parameters
    from research.train_shadow import capture_rng
    from research.train_augmix import fixed_config
    require(torch.cuda.is_available() and torch.backends.cudnn.enabled, "Actual CUDA and original cuDNN are required")
    device = torch.device("cuda")
    config = fixed_config()
    seed_all(2026)
    template = build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
    require(count_parameters(template) == 4298970, "The fixed WRN22-4 total parameter count changed")
    cpu_batch = synthetic_batch()
    prepared = (cpu_batch[0].to(device=device,memory_format=torch.channels_last),cpu_batch[1].to(device))
    require(prepared[0].shape == (384,3,32,32) and len(prepared[1]) == 128, "Fixed synthetic sourceB128/view384 required")
    fingerprint = state_digest({"views":cpu_batch[0],"targets":cpu_batch[1]})
    rng = capture_rng(device)
    arms, snapshots = [], []
    for backend in ("cudnn","native_bn_only"):
        arm, snapshot = run_arm(template,prepared,device,torch.float16,backend,rng)
        arms.append(arm); snapshots.append(snapshot)
    require(arms[0]["initial_state_sha256"] == arms[1]["initial_state_sha256"] and torch.backends.cudnn.enabled,
        "Arms must share initial tensor bytes and restore cuDNN globally")
    result = {"schema":RESULT_SCHEMA,"status":"complete","synthetic_only":True,"official_data_loaded":False,
        "formal_admission_evidence":False,"accuracy_measured":False,"confirmation_evaluated":False,
        "hidden_images_loaded":False,"source_batch_size":128,"model_views_per_step":384,"precision":"fp16",
        "synthetic_unique_sources":128,"same_prepared_batch_reused":True,"rng_reset_before_each_arm":True,
        "numeric_protocol":"FP16+GradScaler_init256; only BatchNorm2d.forward uses cuDNN-disabled context in native arm",
        "convolution_cudnn_enabled":True,"input_sha256":fingerprint,"source_sha256":manifest["source_sha256"],
        "config_sha256":shared.CONFIG_SHA,"reference_profile_sha256":REFERENCE_SHA,"arms":arms,
        "first_step_differences":{name:scalar_difference(snapshots[0][name],snapshots[1][name]) for name in ("gradients","bn_buffers")},
        "first_step_output_difference":scalar_difference({"output":snapshots[0]["output"]},{"output":snapshots[1]["output"]}),
        "environment":{k:v for k,v in environment().items() if k in ("gpu","cudnn","cuda_runtime","amp_dtype")},
        "interpretation":"Synthetic GPU-step engineering only; input preparation/transfer, queue, epoch setup and official data are outside timing. Both arms include identical finite-gradient/update/BN-counter proof. CUDA event intervals include host submission gaps and are not exclusive kernel sums. Backend numerical differences are reported, not claimed bitwise equal. No accuracy or formal throughput admission inferred."}
    result["environment"]["amp_dtype"] = "torch.float16"
    require(manifest["source_sha256"] == source_snapshot(), "Bound source changed during BN diagnostic")
    write_json(folder/"bn_summary.json",result)


def parent(run):
    from runtime import write_json
    from research.run_local import now, used_seconds, acquire_lease
    from research.process_tree_guard import run_guarded, may_release_lease
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",run or "") and not run.startswith("stability"), "Fresh BN diagnostic ID required")
    source = source_snapshot()
    require(sha256(ROOT/REFERENCE) == REFERENCE_SHA, "The actual profile reference changed")
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    budget_seconds(policy,now(),0)
    base,folder = ROOT/"results/research",ROOT/"results/research"/run
    lease = base/"local_lease.json"
    require(not folder.exists() and not lease.exists(), "Fresh directory and fully reconciled free local lease required")
    acquire_lease(lease)
    began, receipt, owns = time.perf_counter(), {}, False
    try:
        day = now().replace(hour=0,minute=0,second=0,microsecond=0)
        budget = budget_seconds(policy,now(),used_seconds(base,day,day+timedelta(days=1)))
        folder.mkdir(exist_ok=False); owns = True
        write_json(folder/"diagnostic_inputs.json",{"source_sha256":source,"config_sha256":shared.CONFIG_SHA,"reference_profile_sha256":REFERENCE_SHA})
        receipt = {"schema":SCHEMA,"backend":"local","run":run,"mode":"bn_diagnostic","status":"profiling",
            "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":now().isoformat(),
            "reserved_seconds":budget,"deadline_unix":time.time()+budget,"source_sha256":source,
            "input_manifest_sha256":sha256(folder/"diagnostic_inputs.json"),"submitted":False,"confirmation_evaluated":False}
        write_json(folder/"supervisor.json",receipt)
        remaining = budget-(time.perf_counter()-began)
        require(remaining > 0,"No bounded BN diagnostic time remains")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder),"--_manifest-sha",receipt["input_manifest_sha256"]],
            folder/"diagnostic.log",remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        result = json.loads((folder/"bn_summary.json").read_text())
        require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "complete" and
            result.get("formal_admission_evidence") is False and result.get("accuracy_measured") is False and
            source == source_snapshot(), "Complete unchanged engineering-only BN result required")
        receipt.update(status="complete",artifact="bn_summary.json",artifact_sha256=sha256(folder/"bn_summary.json"))
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__)
        raise
    finally:
        if owns:
            receipt.update(elapsed_seconds=time.perf_counter()-began,finished_at=now().isoformat())
            if not may_release_lease(receipt): receipt["status"] = "cleanup_unverified"
            write_json(folder/"supervisor.json",receipt)
        if lease.exists():
            owner = json.loads(lease.read_text())
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time()) < .01:
                if may_release_lease(receipt): lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"))
                    write_json(lease,owner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        require(args.run is None and args._manifest_sha is not None,"Controlled worker manifest required")
        folder = args._worker.resolve()
        require(folder.parent == (ROOT/"results/research").resolve(),"Controlled BN diagnostic folder required")
        worker(folder,args._manifest_sha)
    elif args.run:
        require(args._manifest_sha is None,"Worker-only argument")
        parent(args.run)
    else:
        require(args._manifest_sha is None,"Worker-only argument")
        parser.print_help()


if __name__ == "__main__": main()
