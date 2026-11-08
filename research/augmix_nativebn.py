"""Explicit native-training-BN AugMix protocol built on the frozen engine.

Only training BatchNorm2d.forward changes backend. Evaluation keeps the original
forward. Patches exist only inside this process and always restore on exit.
No data, prediction, upload or unguarded command entry is provided here.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "research_augmix_nativebn_training_v1"
PREFLIGHT_SCHEMA = "research_augmix_nativebn_preflight_v1"
PROOF_SCHEMA = "research_augmix_nativebn_execution_v1"
DIAGNOSTIC = "results/research/wrn22_augmix_bn_backend_s2026_v1/bn_summary.json"
DIAGNOSTIC_SHA = "3b11a371655c6716c04c3115a125aff9285e56be0c60a32f9154da02dfc4d12c"
CONFIG_SHA = "c7ba91cb25ff128dfe816b303cde43cd4678673e0847a9f5b5bca7c4c4541df8"
NEW_SOURCES = ("research/augmix_nativebn.py", "research/run_augmix_nativebn.py",
               "research/benchmark_augmix_nativebn.py")


def require(condition, message):
    if not condition: raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_sources():
    from research.profile_augmix import source_snapshot
    sources = source_snapshot()
    for key in (*NEW_SOURCES, "research/benchmark_augmix.py", "research/benchmark.py"):
        sources[key] = sha256(ROOT/key)
    require(sha256(ROOT/DIAGNOSTIC) == DIAGNOSTIC_SHA, "Actual two-arm BN diagnostic changed")
    return sources


def extension():
    return {"schema":"research_augmix_nativebn_extension_v1",
        "training_batchnorm_backend":"aten_native_forward_local_cudnn_disabled",
        "evaluation_batchnorm_backend":"original_unpatched_forward",
        "convolution_cudnn_enabled":True, "module_state_topology_unchanged":True,
        "source_batch_size":128, "model_batch_size":384, "batchnorm_modules":19,
        "initial_grad_scaler_scale":256.0, "config_file_sha256":CONFIG_SHA,
        "source_sha256":bound_sources(), "actual_backend_diagnostic_sha256":DIAGNOSTIC_SHA}


def research_provenance(precision, config, input_config_sha256=None):
    from research import train_augmix as engine
    # This symbol is captured before the context replaces the engine reference.
    original = getattr(engine.research_provenance, "_nativebn_original", engine.research_provenance)
    marker = original(precision, config, input_config_sha256)
    return {**marker, "schema":SCHEMA, "method":"augmix_jsd_nativebn",
        "numeric_protocol":"explicit_FP16_init256_training_BN_native_eval_BN_original_v1",
        "bn_backend_extension":extension()}


def normalize_marker(marker, config, raw_sha):
    """Validate the whole new protocol before returning original math metadata."""
    from research import train_augmix as engine
    require(isinstance(marker,dict), "Explicit native-BN marker required")
    precision = marker.get("precision")
    expected = research_provenance(precision,config,raw_sha)
    require(marker == expected, "Native-BN source/config/backend protocol differs")
    original = getattr(engine.research_provenance, "_nativebn_original", engine.research_provenance)
    return original(precision,config,raw_sha)


def new_proof():
    return {"schema":PROOF_SCHEMA,"attempted_updates":0,"successful_updates":0,"skipped_updates":0,
        "training_bn_forward_calls":0,"bn_single_update_verified_calls":0,
        "finite_positive_gradient_steps":0,"actual_parameter_changed_steps":0,
        "skipped_probe_preserved_steps":0,
        "convolution_cudnn_enabled_calls":0,"evaluation_bn_forward_calls":0,
        "observed_training_bn_nodes":[],"real_cuda_backend_verified":False}


def validate_proof(proof, attempts, successes, skipped, *, real_cuda=True):
    require(isinstance(proof,dict) and set(proof) == set(new_proof()) and proof.get("schema") == PROOF_SCHEMA,
        "Exact native-BN execution proof required")
    expected = {"attempted_updates":attempts,"successful_updates":successes,"skipped_updates":skipped,
        "training_bn_forward_calls":19*attempts,"bn_single_update_verified_calls":19*attempts,
        "finite_positive_gradient_steps":successes,"actual_parameter_changed_steps":successes,
        "skipped_probe_preserved_steps":skipped}
    require(all(type(proof.get(k)) is int and proof[k] == v for k,v in expected.items()) and successes+skipped == attempts,
        "Native-BN actual step/BN/gradient/update counts differ")
    require(type(proof["convolution_cudnn_enabled_calls"]) is int and proof["convolution_cudnn_enabled_calls"] >= 22*attempts and
        type(proof["evaluation_bn_forward_calls"]) is int and proof["evaluation_bn_forward_calls"] >= 0,
        "Original convolution/evaluation scopes were not recorded")
    nodes = proof["observed_training_bn_nodes"]
    require(type(nodes) is list and nodes and nodes == sorted(set(nodes)) and
        all(type(node) is str and "NativeBatchNorm" in node for node in nodes), "Actual native BN autograd nodes required")
    require(type(proof["real_cuda_backend_verified"]) is bool and (not real_cuda or proof["real_cuda_backend_verified"] is True),
        "Actual CUDA native-BN execution proof required")
    return proof


@contextmanager
def native_training_context(config, raw_sha, *, initial_proof=None, strict=True):
    """Patch the frozen engine in RAM; every proof cost is inside train_step."""
    import torch
    from research import train_augmix as engine
    engine.validate_config(config)
    proof = copy.deepcopy(initial_proof) if initial_proof is not None else new_proof()
    original_bn, original_conv = torch.nn.BatchNorm2d.forward, torch.nn.Conv2d.forward
    original_marker, original_step = engine.research_provenance, engine.train_step
    original_schema, original_write, original_save = engine.SCHEMA, engine.write_json, engine.save_checkpoint
    require(not hasattr(original_marker,"_nativebn_original"), "Native-BN contexts cannot be nested")
    nodes = set(proof["observed_training_bn_nodes"])
    def bn_forward(module, inputs):
        require(torch.backends.cudnn.enabled, "cuDNN disabled outside the BN-local training scope")
        if not module.training:
            proof["evaluation_bn_forward_calls"] += 1
            return original_bn(module,inputs)
        require(module.track_running_stats and module.num_batches_tracked is not None,
            "Registered BN must retain running statistics")
        before = module.num_batches_tracked.detach().clone()
        with torch.backends.cudnn.flags(enabled=False):
            result = original_bn(module,inputs)
        require(torch.backends.cudnn.enabled and torch.equal(module.num_batches_tracked,before+1),
            "Each training BN forward must restore cuDNN and update BN exactly once")
        node = type(result.grad_fn).__name__
        require("NativeBatchNorm" in node, "Training did not execute the actual native BN backend")
        nodes.add(node)
        proof["observed_training_bn_nodes"] = sorted(nodes)
        proof["training_bn_forward_calls"] += 1
        proof["bn_single_update_verified_calls"] += 1
        proof["real_cuda_backend_verified"] = proof["real_cuda_backend_verified"] or inputs.device.type == "cuda"
        return result
    def conv_forward(module,inputs):
        require(torch.backends.cudnn.enabled, "Convolution must retain its original cuDNN scope")
        if module.training: proof["convolution_cudnn_enabled_calls"] += 1
        return original_conv(module,inputs)
    def step(model, optimizer, scaler, prepared, device, dtype, **kwargs):
        bns = [m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
        if strict: require(len(bns) == 19, "Fixed WRN22-4 must have19 BN modules")
        probe = next(model.parameters())
        before = probe.detach().clone()
        ema = kwargs.get("ema")
        ema_probe = next(ema.parameters()) if ema is not None else None
        ema_before = ema_probe.detach().clone() if ema_probe is not None else None
        momentum = optimizer.state.get(probe,{}).get("momentum_buffer")
        momentum_before = momentum.detach().clone() if momentum is not None else None
        calls = proof["training_bn_forward_calls"]
        report = original_step(model,optimizer,scaler,prepared,device,dtype,**kwargs)
        require(proof["training_bn_forward_calls"]-calls == len(bns) and report["forward_calls"] == 1 and
            report["model_views"] == 3*report["source_examples"], "Native-BN requires the original single3B forward")
        if strict: require(1 <= report["source_examples"] <= 128, "Native-BN source batch changed")
        updated = report["updated"]
        require(type(updated) is bool, "Actual optimizer update flag required")
        if updated:
            gradients = [p.grad for p in model.parameters()]
            require(all(g is not None for g in gradients), "Missing native-BN gradient")
            norm = float(torch.stack(torch._foreach_norm(gradients,float("inf"))).amax())
            require(math.isfinite(norm) and norm > 0 and not torch.equal(probe.detach(),before),
                "Successful native-BN update needs finite positive gradients and changed parameters")
            proof["finite_positive_gradient_steps"] += 1
            proof["actual_parameter_changed_steps"] += 1
        else:
            after_momentum = optimizer.state.get(probe,{}).get("momentum_buffer")
            require(report["scale_after"] < report["scale_before"] and torch.equal(probe.detach(),before) and
                (ema_probe is None or torch.equal(ema_probe.detach(),ema_before)) and
                ((momentum_before is None and after_momentum is None) or
                 (momentum_before is not None and after_momentum is not None and torch.equal(momentum_before,after_momentum))),
                "Overflow skip must back off the scaler and preserve SGD/EMA probes")
            proof["skipped_probe_preserved_steps"] += 1
        proof["attempted_updates"] += 1
        proof["successful_updates"] += int(updated)
        proof["skipped_updates"] += int(not updated)
        return report
    def marker(*args,**kwargs):
        return research_provenance(*args,**kwargs)
    marker._nativebn_original = original_marker
    def write(path, document):
        if Path(path).name == "metrics.json": document = {**document,"nativebn_execution_proof":copy.deepcopy(proof)}
        return original_write(path,document)
    def save(path,document):
        return original_save(path,{**document,"nativebn_execution_proof":copy.deepcopy(proof)})
    try:
        torch.nn.BatchNorm2d.forward,torch.nn.Conv2d.forward = bn_forward,conv_forward
        engine.research_provenance,engine.train_step,engine.SCHEMA = marker,step,SCHEMA
        engine.write_json,engine.save_checkpoint = write,save
        yield proof
    finally:
        torch.nn.BatchNorm2d.forward,torch.nn.Conv2d.forward = original_bn,original_conv
        engine.research_provenance,engine.train_step,engine.SCHEMA = original_marker,original_step,original_schema
        engine.write_json,engine.save_checkpoint = original_write,original_save


def validate_preflight(envelope, config, raw_sha):
    from research import train_augmix as engine
    require(envelope.get("schema") == PREFLIGHT_SCHEMA and envelope.get("status") == "complete" and
        envelope.get("source_sha256") == bound_sources() and envelope.get("config_file_sha256") == raw_sha == CONFIG_SHA and
        envelope.get("actual_backend_diagnostic_sha256") == DIAGNOSTIC_SHA,
        "Complete source-bound native-BN preflight envelope required")
    report = envelope.get("engine_result",{})
    normalized = copy.deepcopy(report)
    normalized["research_training"] = normalize_marker(report.get("research_training"),config,raw_sha)
    # Validation is outside the training patch context and uses original math.
    engine.validate_preflight(normalized,config,raw_sha)
    validate_proof(envelope.get("execution_proof"),24,23,1)
    require(envelope.get("engine_result_sha256") == hashlib.sha256(
        json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest(), "Native-BN inner preflight result changed")
    return report


def run_engine(mode, config, output, data_dir, device, raw, *, resume=False):
    import torch
    from runtime import write_json
    from research import train_augmix as engine
    require(mode in ("preflight","train") and config["seed"] == 2026,
        "Only primary2026 native-BN preflight/training is registered")
    raw_sha = hashlib.sha256(raw).hexdigest()
    require(raw_sha == CONFIG_SHA, "Exact registered raw configuration required")
    source = bound_sources()
    initial = None
    if resume:
        last = torch.load(Path(output)/"last.pt",map_location="cpu",weights_only=False)
        normalize_marker(last.get("provenance",{}).get("research_training"),config,raw_sha)
        initial = validate_proof(last.get("nativebn_execution_proof"),last.get("attempted_updates"),last.get("updates"),last.get("skipped_updates"))
        del last
    with native_training_context(config,raw_sha,initial_proof=initial) as proof:
        if mode == "preflight":
            result = engine.preflight(config,output,data_dir,device,"fp16",input_config_bytes=raw)
        else:
            engine.train(config,output,data_dir,device,"fp16",resume=resume,input_config_bytes=raw)
    require(source == bound_sources(), "Native-BN bound source changed during execution")
    if mode == "preflight":
        envelope = {"schema":PREFLIGHT_SCHEMA,"status":"complete","source_sha256":source,
            "config_file_sha256":raw_sha,"actual_backend_diagnostic_sha256":DIAGNOSTIC_SHA,
            "engine_result":result,"engine_result_sha256":hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "execution_proof":proof,"interpretation":"Actual official-training-image full pipeline including native-BN proof; no accuracy or promotion evidence."}
        validate_preflight(envelope,config,raw_sha)
        write_json(Path(output)/"nativebn_preflight.json",envelope)
