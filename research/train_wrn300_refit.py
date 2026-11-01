"""Independent scratch50k/zero-validation WRN34-3/300 FP16 refit.

Frozen core and shadow primitives are imported unchanged. This module never
evaluates development/confirmation, loads hidden images, predicts or uploads.
Run real data only through run_wrn300_refit.py and its shared OS-tree guard.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

import runtime
import numpy as np
import torch
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model,count_parameters
from train import DEFAULT_CONFIG,MEAN,STD,augment,code_provenance,save_checkpoint,schedule,tensor_images,training_source
from research.train_shadow import capture_rng,restore_rng,optimizer_step,resolve_precision,validate_partition

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="research_wrn300_full_refit_training_v1"
PREFLIGHT_SCHEMA="research_wrn300_full_refit_preflight_v1"
PARAMETER_COUNT=4166298
STEPS=391
EPOCHS=300
INITIAL_SCALE=65536.0
SIGNATURE={"archive_sha256":"d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
    "split_sha256":"73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e"}
VALIDATION_KEYS=("validation_loss","validation_accuracy","ema_validation_loss","ema_validation_accuracy")
ARTIFACTS=("best.pt","last.pt","config.json","input_config.json","metrics.json","history.json","provenance.json","environment.json")


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_config():
    return {**DEFAULT_CONFIG,"depth":34,"widen_factor":3,"epochs":300,"seed":2026,
        "train_all":True,"final_variant":"model"}


def validate_config(config):
    expected=fixed_config()
    require(config==expected and set(config)==set(expected) and all(type(config[k]) is type(v) for k,v in expected.items()),
        "Only the exact primary2026 scratch50k WRN34-3/300 final-model recipe is registered")


def marker(plan,precision):
    require(precision in ("fp16","fp32"),"Explicit nativeFP16 or CPU-test FP32 required")
    return {"schema":SCHEMA,"method":"scratch_full50k_fixed_final_wrn34_3_300",
        "plan_sha256":plan["plan_sha256"],"parent_delivery_sha256":plan["parent_delivery_sha256"],
        "parent_frozen_manifest_sha256":plan["frozen_manifest_sha256"],"source_sha256":plan["source_sha256"],
        "input_config_sha256":plan["input_config_sha256"],"precision":precision,
        "source_git_commit":plan["source_git_commit"],
        "initial_grad_scaler_scale":INITIAL_SCALE,"grad_scaler_enabled":precision=="fp16",
        "training_examples":50000,"validation_examples":0,"epochs":300,"steps_per_epoch":391,
        "final_variant":"model","training_seed":2026,"split_seed":2026,
        "initialization":"scratch_seeded","parent_weights_loaded":False,
        "independent_development_evaluated":False,"confirmation_evaluated":False,"hidden_images_loaded":False,
        "external_data_loaded":False,"old_refit_helpers_support_this_extension":False}


def load_full_training(data_dir,config):
    """Load official training arrays only; old split IDs are checked, never scored."""
    validate_config(config)
    images,labels,old_train,old_dev=load_training_arrays(data_dir)
    manifest=json.loads((Path(data_dir)/"training_manifest.json").read_text())
    validate_partition(old_train,old_dev,manifest,config)
    source=training_source(data_dir)
    signature={"archive_sha256":source["archive_sha256"],"split_sha256":manifest["split"]["indices_sha256"]}
    require(signature==SIGNATURE and images.shape==(50000,32,32,3) and images.dtype==np.uint8 and
        labels.shape==(50000,) and np.issubdtype(labels.dtype,np.integer) and np.all((labels>=0)&(labels<10)),
        "Full refit needs the exact official50000 RGB images/labels and frozen original split")
    return images,labels,signature,source


def setup(config,device,precision):
    validate_config(config)
    selected,dtype=resolve_precision(precision,device)
    require(selected in ("fp16","fp32"),"Refit training does not silently switch to BF16")
    model=build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}).to(device=device,memory_format=torch.channels_last)
    require(count_parameters(model)==count_parameters(model,True)==PARAMETER_COUNT,"Fixed WRN34-3 parameter count changed")
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"],weight_decay=config["weight_decay"],nesterov=True)
    scaler=torch.amp.GradScaler(device.type,init_scale=INITIAL_SCALE,enabled=selected=="fp16")
    return model,ema,optimizer,scaler,dtype


def step(model,ema,optimizer,scaler,images,targets,config,device,dtype,successful_updates):
    """Original MixUp/Cutout math plus measured success/skip proof, no eval."""
    require(images.shape==(len(targets),32,32,3) and 1<=len(targets)<=128,"One original source minibatch required")
    probe=next(model.parameters());ema_probe=next(ema.parameters())
    before,ema_before=probe.detach().clone(),ema_probe.detach().clone()
    momentum=optimizer.state.get(probe,{}).get("momentum_buffer")
    momentum_before=momentum.detach().clone() if momentum is not None else None
    x=augment(images,config["cutout"])
    mixed=config["mixup_alpha"]>0 and random.random()<config["mixup_probability"]
    lam=float(np.random.beta(config["mixup_alpha"],config["mixup_alpha"])) if mixed else 1.0
    permutation=torch.randperm(len(targets),device=device) if mixed else None
    if mixed:x=x*lam+x[permutation]*(1-lam)
    optimizer.zero_grad(set_to_none=True)
    scale_before=float(scaler.get_scale())
    with torch.autocast(device.type,dtype=dtype,enabled=device.type=="cuda"):
        logits=model(x)
        loss=F.cross_entropy(logits,targets,label_smoothing=config["label_smoothing"])*lam
        if mixed:loss+=F.cross_entropy(logits,targets[permutation],label_smoothing=config["label_smoothing"])*(1-lam)
    updated=optimizer_step(loss,optimizer,scaler)
    if updated:
        gradients=[p.grad for p in model.parameters()]
        require(all(g is not None for g in gradients),"A refit parameter gradient is missing")
        maximum=float(torch.stack(torch._foreach_norm(gradients,float("inf"))).amax())
        require(math.isfinite(maximum) and maximum>0 and not torch.equal(probe.detach(),before),
            "Successful refit step requires finite positive gradients and a changed parameter")
        with torch.no_grad():
            decay=min(config["ema_decay"],(1+successful_updates+1)/(10+successful_updates+1))
            torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
    else:
        after_momentum=optimizer.state.get(probe,{}).get("momentum_buffer")
        require(float(scaler.get_scale())<scale_before and torch.equal(probe.detach(),before) and torch.equal(ema_probe.detach(),ema_before) and
            ((momentum_before is None and after_momentum is None) or (momentum_before is not None and after_momentum is not None and torch.equal(momentum_before,after_momentum))),
            "Overflow must back off scale and preserve online/SGD/EMA probes")
    finite=float(torch.stack(torch._foreach_norm([p.detach() for p in model.parameters()],float("inf"))).amax())
    require(math.isfinite(finite),"Nonfinite refit parameter")
    predicted=logits.detach().argmax(1)
    correct=predicted.eq(targets).sum()*lam
    if mixed:correct+=predicted.eq(targets[permutation]).sum()*(1-lam)
    return {"loss":float(loss.detach()),"correct":float(correct),"updated":updated,"mixed":mixed,
        "lr":float(optimizer.param_groups[0]["lr"]),
        "source_examples":len(targets),"forward_calls":1,"scale_before":scale_before,"scale_after":float(scaler.get_scale()),
        "finite_positive_gradient_verified":updated,"actual_parameter_changed":updated,"skip_probes_preserved":not updated}


def equal_state(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(equal_state(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def overflow_smoke(model,ema,optimizer,scaler,images,targets,config,device,dtype):
    require(scaler.is_enabled(),"Overflow proof requires real enabled GradScaler")
    before=[p.detach().clone() for p in model.parameters()];averages=[p.detach().clone() for p in ema.parameters()]
    previous=copy.deepcopy(optimizer.state_dict())
    handle=next(model.parameters()).register_hook(lambda gradient:torch.full_like(gradient,float("inf")))
    try:report=step(model,ema,optimizer,scaler,images,targets,config,device,dtype,23)
    finally:handle.remove()
    result={"status":"passed","finite_loss":math.isfinite(report["loss"]),"infinite_gradient_injected":True,
        "updated":report["updated"],"scale_before":report["scale_before"],"scale_after":report["scale_after"],
        "all_parameters_unchanged":all(torch.equal(p,old) for p,old in zip(model.parameters(),before)),
        "optimizer_state_unchanged":equal_state(optimizer.state_dict(),previous),
        "ema_parameters_unchanged":all(torch.equal(p,old) for p,old in zip(ema.parameters(),averages)),
        "source_examples":len(targets),"forward_calls":1}
    require(result["finite_loss"] and result["updated"] is False and result["scale_after"]<result["scale_before"] and
        all(result[k] is True for k in ("all_parameters_unchanged","optimizer_state_unchanged","ema_parameters_unchanged")),"Full refit overflow preservation failed")
    return result


def validate_history(history,epochs):
    require(type(epochs) is int and 1<=epochs<=300 and len(history)==epochs,"Complete ordered refit epochs required")
    updates=skipped=0
    for epoch,row in enumerate(history,1):
        integers={"epoch":epoch,"attempted_updates":391,"source_examples":50000,"forward_calls":391,"last_source_batch_size":80}
        require(all(type(row.get(k)) is int and row[k]==v for k,v in integers.items()) and
            type(row.get("successful_updates")) is int and row["successful_updates"]>=1 and
            type(row.get("skipped_updates")) is int and row["skipped_updates"]>=0 and
            row["successful_updates"]+row["skipped_updates"]==391 and
            row.get("selected_variant")=="model" and all(key in row and row[key] is None for key in VALIDATION_KEYS),
            "Refit epoch coverage/updates/fixed variant/no-validation contract differs")
        require(type(row.get("source_order_sha256")) is str and re.fullmatch(r"[0-9a-f]{64}",row["source_order_sha256"]),"Actual epoch source-order fingerprint required")
        require(row.get("finite_positive_gradient_steps")==row["successful_updates"] and
            row.get("actual_parameter_changed_steps")==row["successful_updates"] and row.get("skipped_probe_preserved_steps")==row["skipped_updates"],
            "Refit gradient/update/skip proof counts differ")
        require(all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in
            ("train_loss","train_accuracy","lr","epoch_seconds","elapsed_seconds","grad_scaler_scale")) and
            row["train_accuracy"]<=1 and row["grad_scaler_scale"]>0,"Refit history has invalid finite metrics")
        updates+=row["successful_updates"];skipped+=row["skipped_updates"]
    if epochs==300:require(skipped/117300<=.01,"Final refit overflow skip fraction exceeds1percent")
    return updates,skipped


def validate_state(last,config,precision):
    reference=build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
    expected=reference.state_dict()
    for kind in ("model","ema"):
        state=last.get(kind,{})
        require(set(state)==set(expected) and all(isinstance(state[k],torch.Tensor) and state[k].shape==v.shape and
            state[k].dtype==v.dtype and torch.isfinite(state[k]).all() for k,v in expected.items()),"Refit model/EMA state invalid")
        epoch=last.get("epoch")
        require(type(epoch) is int and 1<=epoch<=300 and all(int(value)==391*epoch for key,value in state.items() if key.endswith("num_batches_tracked")),
            "Every final/resumed BN buffer must count exactly391attempts per actual epoch")
    params=list(reference.parameters());optimizer=last.get("optimizer",{});groups=optimizer.get("param_groups",[])
    require(len(groups)==1 and groups[0].get("params")==list(range(len(params))) and
        groups[0].get("momentum")==.9 and groups[0].get("weight_decay")==.0005 and groups[0].get("nesterov") is True and
        groups[0].get("dampening")==0 and type(groups[0].get("lr")) in (int,float) and math.isfinite(groups[0]["lr"]) and groups[0]["lr"]>0 and
        set(optimizer.get("state",{}))==set(range(len(params))),"Refit exact SGD topology differs")
    expected_lr=schedule(391*last["epoch"]-1,391*300,391,config)
    require(math.isclose(groups[0]["lr"],expected_lr,rel_tol=0,abs_tol=1e-12) and
        last.get("history") and math.isclose(last["history"][-1].get("lr",-1),expected_lr,rel_tol=0,abs_tol=1e-12),
        "Final/resumed SGD LR differs from the original300epoch schedule")
    for index,param in enumerate(params):
        item=optimizer["state"][index];momentum=item.get("momentum_buffer")
        require(set(item)=={"momentum_buffer"} and isinstance(momentum,torch.Tensor) and momentum.shape==param.shape and
            momentum.dtype==param.dtype and torch.isfinite(momentum).all(),"Refit SGD momentum invalid")
    scaler=last.get("grad_scaler")
    if precision=="fp32":require(scaler=={},"CPU-only refit fixture scaler must be disabled")
    else:
        require(isinstance(scaler,dict) and set(scaler)=={"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"} and
            type(scaler["scale"]) in (int,float) and math.isfinite(scaler["scale"]) and scaler["scale"]>0 and
            scaler["growth_factor"]==2.0 and scaler["backoff_factor"]==.5 and scaler["growth_interval"]==2000 and
            type(scaler["_growth_tracker"]) is int and 0<=scaler["_growth_tracker"]<2000,"Exact FP16 scaler state required")
    try:
        random.Random().setstate(last["python_rng"]);np.random.RandomState().set_state(last["numpy_rng"])
        require(last["torch_rng"].dtype==torch.uint8,"Torch RNG bytes required")
        torch.Generator().set_state(last["torch_rng"].cpu())
        require(isinstance(last["cuda_rng"],list) and (precision!="fp16" or len(last["cuda_rng"])==1) and
            all(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0 for v in last["cuda_rng"]),"CUDA RNG state invalid")
    except (KeyError,TypeError,RuntimeError) as error:raise ValueError("Complete refit RNG state required") from error


def validate_resume(last,config,signature,provenance):
    require(last.get("schema")==SCHEMA and last.get("config")==config and last.get("data_signature")==signature and
        last.get("provenance")==provenance,"Refit resume source/config/plan/precision differs")
    epoch=last.get("epoch");require(type(epoch) is int and 1<=epoch<300,"Exact incomplete epoch boundary required")
    updates,skipped=validate_history(last.get("history",[]),epoch)
    require(last.get("updates")==updates and last.get("skipped_updates")==skipped and last.get("attempted_updates")==391*epoch and
        last.get("source_examples_seen")==50000*epoch and last.get("best_accuracy") is None and last.get("best_loss") is None,
        "Refit cumulative resume ledger or no-selection state differs")
    validate_state(last,config,provenance["research_training"]["precision"])


def validate_preflight(report,plan):
    require(report.get("schema")==PREFLIGHT_SCHEMA and report.get("status")=="complete" and report.get("synthetic_only") is False and
        report.get("training_data_loaded") is True and report.get("official_training_only") is True and
        report.get("research_training")==marker(plan,"fp16") and report.get("source_sha256")==plan["source_sha256"] and
        report.get("input_config_sha256")==plan["input_config_sha256"] and report.get("data_signature")==SIGNATURE,
        "Complete actual-image source-bound WRN300 refit preflight required")
    attempts=report.get("attempted_updates")
    require(type(attempts) is int and 23<=attempts<=64,"Native init65536 preflight burn-in capped at64 attempts")
    counts={"training_examples":50000,"validation_examples":0,"parameters":PARAMETER_COUNT,"batch_size":128,
        "warmup_steps":3,"timed_steps":20,"successful_updates":23,"skipped_updates":attempts-23,
        "skipped_probe_preserved_steps":attempts-23,"source_examples_seen":128*attempts,"source_ids_unique_count":128*attempts,"forward_calls":attempts,
        "finite_positive_gradient_steps":23,"actual_parameter_changed_steps":23}
    require(all(type(report.get(k)) is int and report[k]==v for k,v in counts.items()) and
        all(report.get(k) is False for k in ("validation_evaluated","confirmation_evaluated","hidden_images_loaded","external_data_loaded")) and
        report.get("full_training_resident_examples")==50000 and report.get("initial_grad_scaler_scale")==INITIAL_SCALE,
        "Refit preflight full50k residency/update/coverage/precision proof differs")
    smoke=report.get("overflow_smoke",{})
    require(smoke.get("status")=="passed" and smoke.get("updated") is False and smoke.get("source_examples")==128 and smoke.get("forward_calls")==1 and
        all(smoke.get(k) is True for k in ("finite_loss","infinite_gradient_injected","all_parameters_unchanged","optimizer_state_unchanged","ema_parameters_unchanged")) and
        all(type(smoke.get(k)) in (int,float) and math.isfinite(smoke[k]) and smoke[k]>0 for k in ("scale_before","scale_after")) and
        smoke["scale_after"]<smoke["scale_before"],"Independent full optimizer/EMA overflow preservation required")
    durations=report.get("step_seconds",[])
    require(len(durations)==20 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in durations),"20 actual complete pipeline timings required")
    median=statistics.median(durations)
    normal=report.get("normal_attempts",[])
    require(len(normal)==attempts and all(type(row.get("updated")) is bool and type(row.get("mixed")) is bool and row.get("source_examples")==128 and row.get("forward_calls")==1 and
        all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>0 for k in ("elapsed_seconds","scale_before","scale_after")) and
        type(row.get("loss")) in (int,float) and math.isfinite(row["loss"]) and row["loss"]>=0 and
        (row.get("finite_positive_gradient_verified") is True and row.get("actual_parameter_changed") is True if row["updated"] else
         row.get("skip_probes_preserved") is True and row["scale_after"]<row["scale_before"]) for row in normal),"Every natural normal/overflow attempt needs complete measured proof")
    require(sum(row["updated"] for row in normal)==23 and [row["elapsed_seconds"] for row in normal if row["updated"]][3:]==durations and
        normal[0]["scale_before"]==INITIAL_SCALE,"Refit successful timed steps or original init65536 identity differs")
    require(all(type(row.get("lr")) in (int,float) and math.isclose(row["lr"],schedule(index,391*300,391,plan["config"]),rel_tol=0,abs_tol=1e-12)
        for index,row in enumerate(normal)),"Preflight must use the same original attempt-index warmup schedule as formal training")
    require(type(report.get("actual_mixed_batches")) is int and 1<=report["actual_mixed_batches"]==sum(row["mixed"] for row in normal)<=attempts,
        "Actual MixUp execution must be observed in preflight")
    require(math.isclose(report.get("median_step_seconds",-1),median,rel_tol=0,abs_tol=1e-10) and
        math.isclose(report.get("required_training_and_finalization_seconds",-1),1.5*median*391*300+300,rel_tol=0,abs_tol=1e-8),
        "Refit measured median or fixed391x300 guard differs")
    require(isinstance(report.get("source_ids_sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",report["source_ids_sha256"]),"Actual source coverage fingerprint required")
    env=report.get("environment",{})
    require(env.get("device")=="cuda" and env.get("amp_dtype")=="torch.float16" and env.get("gpu") and
        env.get("packages",{}).get("torch") and report.get("precision")=="fp16","Actual explicit native FP16 environment required")
    return report


def preflight(plan,output,data_dir):
    config=plan["config"];device=torch.device("cuda")
    require(not (Path(output)/"native_preflight.json").exists(),"Existing refit preflight immutable")
    runtime.seed_all(2026)
    images,labels,signature,_=load_full_training(data_dir,config)
    x=tensor_images(images,device);y=torch.as_tensor(labels,device=device);del images,labels
    model,ema,optimizer,scaler,dtype=setup(config,device,"fp16")
    model.train();order=torch.randperm(50000,device=device);durations=[];reports=[];successful=0
    torch.cuda.reset_peak_memory_stats(device)
    for index in range(64):
        torch.cuda.synchronize(device);began=time.perf_counter()
        for group in optimizer.param_groups:group["lr"]=schedule(index,391*300,391,config)
        ids=order[index*128:(index+1)*128]
        report=step(model,ema,optimizer,scaler,x[ids],y[ids],config,device,dtype,successful)
        torch.cuda.synchronize(device);elapsed=time.perf_counter()-began
        if report["updated"]:
            successful+=1
            if successful>3:durations.append(elapsed)
        report["elapsed_seconds"]=elapsed
        reports.append(report)
        if successful==23:break
    require(successful==23,"Original init65536 did not produce23successful updates within64attempts")
    smoke=overflow_smoke(model,ema,optimizer,scaler,x[ids],y[ids],config,device,dtype)
    selected=order[:len(reports)*128].detach().cpu().numpy().astype("<i8")
    require(len(np.unique(selected))==len(selected) and selected.min()>=0 and selected.max()<50000,"Preflight must select unique official training indices only")
    median=statistics.median(durations)
    result={"schema":PREFLIGHT_SCHEMA,"status":"complete","synthetic_only":False,"training_data_loaded":True,"official_training_only":True,
        "source_sha256":plan["source_sha256"],"input_config_sha256":plan["input_config_sha256"],"data_signature":signature,
        "research_training":marker(plan,"fp16"),"precision":"fp16","initial_grad_scaler_scale":INITIAL_SCALE,
        "parameters":PARAMETER_COUNT,"batch_size":128,"warmup_steps":3,"timed_steps":20,"attempted_updates":len(reports),
        "successful_updates":23,"skipped_updates":len(reports)-23,"skipped_probe_preserved_steps":len(reports)-23,
        "source_examples_seen":len(selected),"source_ids_unique_count":len(np.unique(selected)),
        "source_ids_sha256":hashlib.sha256(selected.tobytes()).hexdigest(),"forward_calls":len(reports),"normal_attempts":reports,
        "actual_mixed_batches":sum(row["mixed"] for row in reports),
        "finite_positive_gradient_steps":23,"actual_parameter_changed_steps":23,"training_examples":50000,"validation_examples":0,
        "full_training_resident_examples":50000,"validation_evaluated":False,"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
        "step_seconds":durations,"median_step_seconds":median,"p90_step_seconds":float(np.quantile(durations,.9)),
        "required_training_and_finalization_seconds":1.5*median*391*300+300,"overflow_smoke":smoke,
        "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device),
        "environment":{**runtime.environment(),"device":"cuda","amp_dtype":"torch.float16","grad_scaler_enabled":True},
        "formal_initialization":"discard_preflight_then_seed2026_scratch_model_SGD_GradScaler65536",
        "interpretation":"Actual official50k-resident complete augmentation/MixUp/loss/SGD/EMA/proof pipeline; no accuracy or independent-development claim."}
    validate_preflight(result,plan);runtime.write_json(Path(output)/"native_preflight.json",result)
    return result


def train(plan,output,data_dir,*,resume=False):
    config=plan["config"];validate_config(config)
    output=Path(output);device=torch.device("cuda")
    require(resume or not any((output/name).exists() for name in ARTIFACTS if name!="input_config.json"),"Existing refit artifacts immutable; explicit exact resume required")
    runtime.seed_all(2026)
    images,labels,signature,official=load_full_training(data_dir,config)
    x=tensor_images(images,device);y=torch.as_tensor(labels,device=device);del images,labels
    model,ema,optimizer,scaler,dtype=setup(config,device,"fp16")
    provenance={"source_sha256":code_provenance()["source_sha256"],"git_commit":plan["source_git_commit"],"research_training":marker(plan,"fp16")}
    history=[];start_epoch=1;updates=skipped=0;previous_seconds=0.
    if resume:
        last=torch.load(output/"last.pt",map_location=device,weights_only=False)
        validate_resume(last,config,signature,provenance)
        model.load_state_dict(last["model"],strict=True);ema.load_state_dict(last["ema"],strict=True)
        optimizer.load_state_dict(last["optimizer"]);scaler.load_state_dict(last["grad_scaler"])
        history,start_epoch,updates,skipped=last["history"],last["epoch"]+1,last["updates"],last["skipped_updates"]
        previous_seconds=last["elapsed_seconds"];restore_rng(last,device);del last
    require(digest(output/"input_config.json")==plan["input_config_sha256"] and json.loads((output/"input_config.json").read_text())==config,"Exact refit raw config changed")
    runtime.write_json(output/"config.json",config)
    runtime.write_json(output/"environment.json",{**runtime.environment(),"schema":SCHEMA,"device":"cuda","amp_dtype":"torch.float16",
        "training_precision":"fp16","grad_scaler_enabled":True,"initial_grad_scaler_scale":INITIAL_SCALE,"research_training":provenance["research_training"]})
    runtime.write_json(output/"provenance.json",{"schema":SCHEMA,"code":provenance,"official_data":official,"data_signature":signature,
        "train_examples":50000,"validation_examples":0,"parameter_count":PARAMETER_COUNT,"normalization":{"mean":MEAN,"std":STD}})
    started=time.perf_counter()
    for epoch in range(start_epoch,301):
        epoch_started=time.perf_counter();model.train();order=torch.randperm(50000,device=device)
        loss_sum=correct=0.;seen=epoch_updates=epoch_skipped=0
        for batch,start in enumerate(range(0,50000,128)):
            lr=schedule((epoch-1)*391+batch,391*300,391,config)
            for group in optimizer.param_groups:group["lr"]=lr
            ids=order[start:start+128]
            report=step(model,ema,optimizer,scaler,x[ids],y[ids],config,device,dtype,updates)
            updates+=int(report["updated"]);skipped+=int(not report["updated"])
            epoch_updates+=int(report["updated"]);epoch_skipped+=int(not report["updated"])
            loss_sum+=report["loss"]*len(ids);correct+=report["correct"];seen+=len(ids)
        with torch.no_grad():
            for average,online in zip(ema.buffers(),model.buffers()):average.copy_(online)
        elapsed=previous_seconds+time.perf_counter()-started
        order_hash=hashlib.sha256(order.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_accuracy":correct/seen,"lr":lr,"source_order_sha256":order_hash,
            "epoch_seconds":time.perf_counter()-epoch_started,"elapsed_seconds":elapsed,"selected_variant":"model",
            **{key:None for key in VALIDATION_KEYS},"successful_updates":epoch_updates,"skipped_updates":epoch_skipped,
            "attempted_updates":391,"source_examples":seen,"forward_calls":391,"last_source_batch_size":len(ids),
            "grad_scaler_scale":float(scaler.get_scale()),"finite_positive_gradient_steps":epoch_updates,
            "actual_parameter_changed_steps":epoch_updates,"skipped_probe_preserved_steps":epoch_skipped}
        history.append(row);validate_history(history,epoch)
        metrics={"schema":SCHEMA,"status":"complete" if epoch==300 else "running","epochs_completed":epoch,"config":config,
            "parameter_count":PARAMETER_COUNT,"training_examples":50000,"validation_examples":0,
            "best_validation_accuracy":None,"best_validation_loss":None,"elapsed_seconds":elapsed,"research_training":provenance["research_training"],
            "successful_updates":updates,"skipped_updates":skipped,"attempted_updates":391*epoch,"source_examples_seen":50000*epoch,"forward_calls":391*epoch}
        last={"schema":SCHEMA,"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"grad_scaler":scaler.state_dict(),
            "epoch":epoch,"updates":updates,"skipped_updates":skipped,"attempted_updates":391*epoch,"source_examples_seen":50000*epoch,
            "config":config,"model_config":{k:config[k] for k in ("depth","widen_factor","dropout","num_classes")},"parameter_count":PARAMETER_COUNT,
            "data_signature":signature,"provenance":provenance,"history":history,"best_accuracy":None,"best_loss":None,"elapsed_seconds":elapsed,**capture_rng(device)}
        runtime.write_json(output/"history.json",history);runtime.write_json(output/"metrics.json",metrics);save_checkpoint(output/"last.pt",last)
        if epoch==300:
            save_checkpoint(output/"best.pt",{"schema":SCHEMA,"model":model.state_dict(),"model_config":last["model_config"],"config":config,
                "epoch":300,"variant":"model","validation_accuracy":None,"validation_loss":None,"parameter_count":PARAMETER_COUNT,
                "data_signature":signature,"provenance":provenance})
        print(json.dumps({"epoch":epoch,"successful_updates":updates,"skipped_updates":skipped,"elapsed_seconds":elapsed}),flush=True)
