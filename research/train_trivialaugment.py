"""Independent synchronous single-view TA + unchanged WRN34-3/160 parent.

Only a guarded controller may authorize GPU execution. This module supplies
the training math and exact resume state, not a lease/controller or uploader.
"""
from __future__ import annotations

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from runtime import environment, seed_all, write_json
import argparse
import copy
import hashlib
import json
import math
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from research import trivialaugment as ta
from research.train_shadow import capture_rng, evaluate, optimizer_step, resolve_precision, restore_rng, validate_partition
from train import DEFAULT_CONFIG, MEAN, STD, augment as core_augment, code_provenance, save_checkpoint, schedule, sha256, tensor_images, training_source

SCHEMA="research_trivialaugment_training_v1"
PARAMETERS=4166298
BASE_CONFIG="research/configs/wrn34_3.json"
BASE_CONFIG_SHA="2901f0a7d717083c8811aac9cff1910c7a7493eb3c474c260bb9b7f7eb45df02"
TA_SOURCE_SHA="1f81246a4e6f045d5f4beb2198b6d2db9ca3a32662f02829a350062f72d20806"
SOURCE_KEYS=("research/train_trivialaugment.py","research/trivialaugment.py","research/train_shadow.py","research/augmentations.py")


def require(condition,message):
    if not condition: raise ValueError(message)


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fixed_config(seed=2026):
    require(type(seed) is int and seed in (2026,2027),"Only fresh primary2026 or separately authorized repeat2027")
    require(sha256(ROOT/BASE_CONFIG)==BASE_CONFIG_SHA and sha256(Path(ta.__file__))==TA_SOURCE_SHA,"Frozen parent or TA component changed")
    return {**DEFAULT_CONFIG,**json.loads((ROOT/BASE_CONFIG).read_text()),"seed":seed,"ta_space":"fixed_standard","ta_strengths":31}


def validate_config(config):
    require(type(config) is dict and config==fixed_config(config.get("seed"))
            and all(type(config[k]) is type(v) for k,v in fixed_config(config["seed"]).items()),"Exact WRN34-3/160 parent plus fixed TA required")


def input_config(config,raw=None):
    validate_config(config)
    raw=raw if raw is not None else (json.dumps(config,indent=2)+"\n").encode()
    require(isinstance(raw,bytes) and {**DEFAULT_CONFIG,**json.loads(raw)}==config,"Raw input config differs from fixed recipe")
    return raw


def research_provenance(precision,config,raw_sha):
    validate_config(config)
    require(precision in ("bf16","fp32") and isinstance(raw_sha,str) and len(raw_sha)==64
            and all(c in "0123456789abcdef" for c in raw_sha),"Explicit native BF16/CPU FP32 and raw config hash required")
    return {"schema":SCHEMA,"method":"trivialaugment","source_sha256":{key:sha256(ROOT/key) for key in SOURCE_KEYS},
            "precision":precision,"grad_scaler_enabled":False,"ta_space":"fixed_standard","ta_strengths":31,
            "ta_source_sha256":TA_SOURCE_SHA,"ta_rng_algorithm":"PCG64","ta_rng_seed":config["seed"],
            "parent_config_sha256":BASE_CONFIG_SHA,
            "augmentation_order":"TA_raw_RGB32_then_original_crop_flip_normalize_Cutout_then_original_Mixup",
            "views_per_source":1,"training_seed":config["seed"],"split_seed":2026,
            "input_config_sha256":raw_sha,"effective_config_sha256":json_sha(config),
            "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False}


def ta_rng(seed,state=None):
    require(type(seed) is int and seed in (2026,2027),"Fixed explicit TA seed required")
    rng=np.random.Generator(np.random.PCG64(seed))
    if state is not None:
        require(type(state) is dict and set(state)=={"bit_generator","state","has_uint32","uinteger"}
                and state["bit_generator"]=="PCG64" and type(state["state"]) is dict
                and set(state["state"])=={"state","inc"}
                and all(type(state["state"][k]) is int and 0<=state["state"][k]<2**128 for k in ("state","inc"))
                and state["state"]["inc"]%2==1 and type(state["has_uint32"]) is int and state["has_uint32"] in (0,1)
                and type(state["uinteger"]) is int and 0<=state["uinteger"]<2**32,"Complete valid independent PCG64 state required")
        require(state["state"]["inc"]==rng.bit_generator.state["state"]["inc"],"Independent TA RNG stream does not originate from the fixed seed")
        rng.bit_generator.state=copy.deepcopy(state)
    return rng


def prepare_ta_pixels(images,rng):
    """Synchronous CPU/PIL only, before the parent's GPU augmentation RNG."""
    require(isinstance(images,np.ndarray) and images.dtype==np.uint8 and images.ndim==4
            and images.shape[1:]==(32,32,3) and 1<=len(images)<=128,"Original uint8 RGB32 source minibatch required")
    require(isinstance(rng,np.random.Generator) and isinstance(rng.bit_generator,np.random.PCG64),"Independent PCG64 required")
    pixels=[];counts=dict.fromkeys(ta.OPERATIONS,0)
    for image in images:
        output,operation=ta.augment(image,rng,return_trace=True)
        pixels.append(np.asarray(output,dtype=np.uint8));counts[operation.name]+=1
    return np.stack(pixels),counts


def train_step(model,ema,optimizer,scaler,images,y,config,rng,updates,device,dtype):
    """The parent shadow loss/SGD/EMA math, with one independent TA prefix."""
    require(not scaler.is_enabled() and scaler.get_scale()==1.,"TA does not use FP16/scaled updates")
    pixels,counts=prepare_ta_pixels(images,rng)
    x=core_augment(tensor_images(pixels,device),config["cutout"])
    mixed=config["mixup_alpha"]>0 and random.random()<config["mixup_probability"]
    lam=float(np.random.beta(config["mixup_alpha"],config["mixup_alpha"])) if mixed else 1.
    permutation=torch.randperm(len(y),device=device) if mixed else None
    if mixed:x=x*lam+x[permutation]*(1-lam)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type,dtype=dtype,enabled=dtype!=torch.float32):
        logits=model(x)
        loss=F.cross_entropy(logits,y,label_smoothing=config["label_smoothing"])*lam
        if mixed:loss+=F.cross_entropy(logits,y[permutation],label_smoothing=config["label_smoothing"])*(1-lam)
    require(optimizer_step(loss,optimizer,scaler),"Unscaled TA update must succeed without skips")
    with torch.no_grad():
        decay=min(config["ema_decay"],(1+updates+1)/(10+updates+1))
        torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
    return {"loss":loss.detach(),"logits":logits.detach(),"lam":lam,"permutation":permutation,"ta_operation_counts":counts,"updated":True}


def validate_history(history,epochs):
    require(type(epochs) is int and 1<=epochs<=160 and isinstance(history,list) and len(history)==epochs,"Every completed epoch required")
    total=dict.fromkeys(ta.OPERATIONS,0)
    for epoch,row in enumerate(history,1):
        require(type(row.get("epoch")) is int and row["epoch"]==epoch and all(type(row.get(k)) is int and row[k]==v for k,v in
                {"successful_updates":313,"skipped_updates":0,"forward_passes":313,"source_examples_seen":40000,"model_views_seen":40000}.items()),"Complete one-view/SGD source accounting required")
        counts=row.get("ta_operation_counts",{})
        require(type(counts) is dict and set(counts)==set(ta.OPERATIONS)
                and all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())==40000,"Exactly one TA operation per source required")
        for name,value in counts.items():total[name]+=value
        require(type(row.get("grad_scaler_scale")) in (int,float) and row["grad_scaler_scale"]==1.
                and isinstance(row.get("ta_rng_state_sha256"),str) and len(row["ta_rng_state_sha256"])==64
                and all(c in "0123456789abcdef" for c in row["ta_rng_state_sha256"]),"Unscaled state and TA RNG fingerprint required")
        for key in ("train_loss","validation_loss","ema_validation_loss","lr","epoch_seconds","elapsed_seconds"):
            require(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and row[key]>=0,"Finite history value required")
        for key in ("train_accuracy","validation_accuracy","ema_validation_accuracy"):
            require(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and 0<=row[key]<=1,"Finite history accuracy required")
    return total


def best_selection(history):
    best=(-1.,float("-inf"));selected=None
    for row in history:
        online=(row["validation_accuracy"],-row["validation_loss"]);average=(row["ema_validation_accuracy"],-row["ema_validation_loss"])
        variant="ema" if average>=online else "model";metric=average if variant=="ema" else online
        require(row.get("selected_variant")==variant,"Recorded raw/EMA choice differs")
        if metric>best:best=metric;selected=(row["epoch"],variant)
    return selected,best


def validate_saved_state(last,config):
    """Audit actual saved tensors, SGD topology, RNG and independent TA state."""
    with torch.random.fork_rng(devices=[]):reference=build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
    expected=reference.state_dict();updates=last["epoch"]*313
    for kind in ("model","ema"):
        state=last.get(kind,{})
        require(set(state)==set(expected),"Saved model/EMA topology differs")
        for key,value in expected.items():
            actual=state[key]
            require(isinstance(actual,torch.Tensor) and actual.shape==value.shape and actual.dtype==value.dtype and torch.isfinite(actual).all().item(),"Saved model/EMA shape/dtype/finite state differs")
            if key.endswith("num_batches_tracked"):require(actual.item()==updates,"One forward per update BN accounting differs")
    parameters=list(reference.parameters());optimizer=last.get("optimizer",{});groups=optimizer.get("param_groups",[])
    require(len(groups)==1 and groups[0].get("params")==list(range(len(parameters)))
            and groups[0].get("momentum")==.9 and groups[0].get("weight_decay")==.0005 and groups[0].get("nesterov") is True
            and groups[0].get("dampening")==0 and groups[0].get("maximize") is False
            and groups[0].get("lr")==last["history"][-1]["lr"]
            and set(optimizer.get("state",{}))==set(range(len(parameters))),"Actual saved SGD topology/settings differ")
    for index,parameter in enumerate(parameters):
        state=optimizer["state"][index];momentum=state.get("momentum_buffer")
        require(set(state)=={"momentum_buffer"} and isinstance(momentum,torch.Tensor) and momentum.shape==parameter.shape
                and momentum.dtype==torch.float32 and torch.isfinite(momentum).all().item(),"Saved SGD momentum tensor differs")
    require(last.get("grad_scaler")=={} and isinstance(last.get("torch_rng"),torch.Tensor)
            and last["torch_rng"].dtype==torch.uint8 and last["torch_rng"].ndim==1 and last["torch_rng"].numel()>0
            and isinstance(last.get("cuda_rng"),list),"Saved disabled scaler or RNG tensors differ")
    if last["provenance"]["research_training"]["precision"]=="bf16":require(bool(last["cuda_rng"]),"Native CUDA RNG state required")
    for state in last["cuda_rng"]:require(isinstance(state,torch.Tensor) and state.dtype==torch.uint8 and state.ndim==1 and state.numel()>0,"Actual CUDA RNG byte state required")
    require(all(k in last for k in ("python_rng","numpy_rng","ta_rng_state")),"Parent and independent RNG fields required")
    random.Random().setstate(last["python_rng"]);np.random.RandomState(0).set_state(last["numpy_rng"])
    require(type(last["ta_rng_state"]) is dict,"Independent TA RNG state required")
    ta_rng(config["seed"],last["ta_rng_state"])
    require(last.get("ta_rng_state_sha256")==json_sha(last["ta_rng_state"]),"Independent TA RNG state hash differs")


def validate_resume(last,config,signature,provenance,raw):
    require(last.get("schema")==SCHEMA and last.get("formal_training_state") is True and last.get("preflight_state") is False
            and last.get("config")==config and last.get("data_signature")==signature and last.get("provenance")==provenance
            and last.get("input_config_sha256")==hashlib.sha256(raw).hexdigest(),"Resume requires original formal source/config/split/raw input; preflight states are forbidden")
    epoch=last.get("epoch")
    require(type(epoch) is int and 1<=epoch<160 and last.get("updates")==epoch*313 and last.get("skipped_updates")==0
            and last.get("forward_passes")==epoch*313 and last.get("parameter_count")==PARAMETERS
            and last.get("model_config")=={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")},"Resume requires an incomplete exact SGD history/model")
    totals=validate_history(last.get("history"),epoch)
    require(totals==last.get("ta_operation_counts") and last.get("source_examples_seen")==epoch*40000
            and last.get("model_views_seen")==epoch*40000 and last["history"][-1]["ta_rng_state_sha256"]==last.get("ta_rng_state_sha256"),"Resume operation/RNG/source ledger differs")
    _,best=best_selection(last["history"])
    require((last.get("best_accuracy"),-last.get("best_loss",float("inf")))==best,"Resume selection state differs")
    validate_saved_state(last,config)


def train(config,output,data_dir,device,precision="bf16",stop_after_epoch=None,resume=False,input_config_bytes=None):
    validate_config(config);precision,dtype=resolve_precision(precision,device)
    require(precision in ("bf16","fp32"),"Only native BF16 or CPU math checks")
    require(stop_after_epoch is None or type(stop_after_epoch) is int and 1<=stop_after_epoch<=160,"Invalid requested epoch boundary")
    raw=input_config(config,input_config_bytes);raw_sha=hashlib.sha256(raw).hexdigest();output=Path(output);data_dir=Path(data_dir)
    artifact_names=("best.pt","last.pt","history.json","metrics.json","config.json","input_config.json","environment.json","provenance.json")
    if not resume:require(not any((output/name).exists() for name in artifact_names),"Fresh scratch output only; resume must be explicit")
    seed_all(config["seed"])
    images,labels,train_ids,val_ids=load_training_arrays(data_dir)
    manifest=json.loads((data_dir/"training_manifest.json").read_text());validate_partition(train_ids,val_ids,manifest,config)
    source=training_source(data_dir);signature={"archive_sha256":source["archive_sha256"],"split_sha256":manifest["split"]["indices_sha256"]}
    train_pixels=np.ascontiguousarray(images[train_ids]);val_x=tensor_images(images[val_ids],device)
    train_y=torch.as_tensor(labels[train_ids],device=device);val_y=torch.as_tensor(labels[val_ids],device=device);del images,labels
    model_config={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}
    model=build_model(model_config).to(device=device,memory_format=torch.channels_last)
    require(count_parameters(model)==count_parameters(model,True)==PARAMETERS,"Actual parameter count differs")
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=.9,weight_decay=.0005,nesterov=True)
    scaler=torch.amp.GradScaler("cuda",enabled=False);rng=ta_rng(config["seed"])
    provenance=code_provenance();provenance["research_training"]=research_provenance(precision,config,raw_sha)
    history=[];start_epoch=1;updates=0;previous_seconds=0.;best_accuracy=-1.;best_loss=float("inf");counts_total=dict.fromkeys(ta.OPERATIONS,0)
    if resume:
        require((output/"input_config.json").read_bytes()==raw,"Resume raw input bytes changed")
        last=torch.load(output/"last.pt",map_location=device,weights_only=False);validate_resume(last,config,signature,provenance,raw)
        model.load_state_dict(last["model"]);ema.load_state_dict(last["ema"]);optimizer.load_state_dict(last["optimizer"]);scaler.load_state_dict(last["grad_scaler"])
        history=last["history"];start_epoch=last["epoch"]+1;updates=last["updates"];previous_seconds=last["elapsed_seconds"]
        best_accuracy=last["best_accuracy"];best_loss=last["best_loss"];counts_total=last["ta_operation_counts"]
        rng=ta_rng(config["seed"],last["ta_rng_state"]);restore_rng(last,device);del last
    end_epoch=stop_after_epoch or 160;require(start_epoch<=end_epoch,"Requested boundary precedes resume epoch")
    output.mkdir(parents=True,exist_ok=True);(output/"input_config.json").write_bytes(raw);write_json(output/"config.json",config)
    marker=provenance["research_training"]
    write_json(output/"environment.json",{**environment(),"device":device.type,"amp_dtype":str(dtype),"schema":SCHEMA,"trivialaugment_training_precision":precision,"grad_scaler_enabled":False,"research_training":marker})
    write_json(output/"provenance.json",{"schema":SCHEMA,"code":provenance,"official_data":source,"data_signature":signature,
               "train_examples":40000,"validation_examples":5000,"parameter_count":PARAMETERS,"normalization":{"mean":MEAN,"std":STD}})
    print(json.dumps({"event":"trivialaugment_training_started","parameters":PARAMETERS,"training_examples":40000,"validation_examples":5000,"precision":precision,"config":config}),flush=True)
    started=time.perf_counter()
    for epoch in range(start_epoch,end_epoch+1):
        epoch_started=time.perf_counter();model.train();order=torch.randperm(40000,device=device)
        loss_sum=torch.zeros((),device=device);correct=torch.zeros((),device=device);seen=0;counts=dict.fromkeys(ta.OPERATIONS,0)
        for batch,start in enumerate(range(0,40000,128)):
            lr=schedule((epoch-1)*313+batch,160*313,313,config)
            for group in optimizer.param_groups:group["lr"]=lr
            ids=order[start:start+128];y=train_y[ids]
            result=train_step(model,ema,optimizer,scaler,train_pixels[ids.cpu().numpy()],y,config,rng,updates,device,dtype)
            updates+=1
            for name,count in result["ta_operation_counts"].items():counts[name]+=count;counts_total[name]+=count
            with torch.no_grad():
                predicted=result["logits"].argmax(1);lam=result["lam"];permutation=result["permutation"]
                correct+=predicted.eq(y).sum()*lam
                if permutation is not None:correct+=predicted.eq(y[permutation]).sum()*(1-lam)
                loss_sum+=result["loss"]*len(y);seen+=len(y)
        with torch.no_grad():
            for average,online in zip(ema.buffers(),model.buffers()):average.copy_(online)
        regular,average=evaluate(model,val_x,val_y,device,dtype),evaluate(ema,val_x,val_y,device,dtype)
        choice="ema" if (average["accuracy"],-average["loss"])>=(regular["accuracy"],-regular["loss"]) else "model"
        metric=average if choice=="ema" else regular;elapsed=previous_seconds+time.perf_counter()-started;ta_state=copy.deepcopy(rng.bit_generator.state)
        row={"epoch":epoch,"train_loss":loss_sum.item()/seen,"train_accuracy":correct.item()/seen,
             "validation_loss":regular["loss"],"validation_accuracy":regular["accuracy"],"ema_validation_loss":average["loss"],"ema_validation_accuracy":average["accuracy"],
             "selected_variant":choice,"lr":lr,"epoch_seconds":time.perf_counter()-epoch_started,"elapsed_seconds":elapsed,
             "successful_updates":313,"skipped_updates":0,"forward_passes":313,"source_examples_seen":seen,"model_views_seen":seen,
             "grad_scaler_scale":scaler.get_scale(),"ta_operation_counts":counts,"ta_rng_state_sha256":json_sha(ta_state)}
        history.append(row)
        if (metric["accuracy"],-metric["loss"])>(best_accuracy,-best_loss):
            best_accuracy,best_loss=metric["accuracy"],metric["loss"];selected=ema if choice=="ema" else model
            save_checkpoint(output/"best.pt",{"schema":SCHEMA,"model":selected.state_dict(),"model_config":model_config,"config":config,"epoch":epoch,"variant":choice,
                "validation_accuracy":best_accuracy,"validation_loss":best_loss,"parameter_count":PARAMETERS,"data_signature":signature,"provenance":provenance,
                "selected_ta_rng_state_sha256":row["ta_rng_state_sha256"]})
        write_json(output/"history.json",history)
        write_json(output/"metrics.json",{"schema":SCHEMA,"status":"complete" if epoch==160 else "running","epochs_completed":epoch,"best_validation_accuracy":best_accuracy,
            "best_validation_loss":best_loss,"parameter_count":PARAMETERS,"elapsed_seconds":elapsed,"training_examples":40000,"validation_examples":5000,"config":config,
            "research_training":marker,"successful_updates":updates,"skipped_updates":0,"forward_passes":updates,"source_examples_seen":epoch*40000,"model_views_seen":epoch*40000,
            "ta_operation_counts":counts_total,"ta_rng_state_sha256":row["ta_rng_state_sha256"]})
        save_checkpoint(output/"last.pt",{"schema":SCHEMA,"formal_training_state":True,"preflight_state":False,"model":model.state_dict(),"ema":ema.state_dict(),
            "optimizer":optimizer.state_dict(),"grad_scaler":scaler.state_dict(),"epoch":epoch,"updates":updates,"skipped_updates":0,"forward_passes":updates,
            "config":config,"model_config":model_config,"parameter_count":PARAMETERS,"data_signature":signature,"provenance":provenance,"history":history,
            "best_accuracy":best_accuracy,"best_loss":best_loss,"elapsed_seconds":elapsed,"input_config_sha256":raw_sha,"source_examples_seen":epoch*40000,"model_views_seen":epoch*40000,
            "ta_operation_counts":counts_total,"ta_rng_state":ta_state,"ta_rng_state_sha256":row["ta_rng_state_sha256"],**capture_rng(device)})
        print(json.dumps(row),flush=True)
    print(json.dumps({"event":"trivialaugment_training_complete" if end_epoch==160 else "trivialaugment_training_paused","epochs_completed":end_epoch,"successful_updates":updates}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data");parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--precision",choices=("bf16","fp32"),default="bf16");parser.add_argument("--stop-after-epoch",type=int);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();raw=args.config.read_bytes();config={**DEFAULT_CONFIG,**json.loads(raw)}
    train(config,args.output_dir,args.data_dir,torch.device(args.device),args.precision,args.stop_after_epoch,args.resume,raw)


if __name__=="__main__":main()
