"""Independent fixed scratch-KD student training. Host imports are stdlib only.

Only an authenticated controller may call training. There is no CLI launch,
resume, hidden loader, prediction, teacher selection or upload facility.
"""
from __future__ import annotations
import argparse
import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
SCHEMA="research_scratch_kd_training_v1"
PARAMETERS=4289754
TEACHER_PARAMETERS=4166298
BASE_CONFIG="research/configs/wrn16_5.json"
BASE_SHA="d2745cb8d36e7105cf13b0d32f4467024d36e69d9cc316368c45065063c18528"
TEACHER_CHECKPOINT="53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
TEACHER_FREEZE="b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3"
TEACHER_BENCHMARK="e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b"
DATA={"archive_sha256":"d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
      "split_sha256":"73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e"}
SOURCE_KEYS=("research/train_scratch_kd.py","research/scratch_kd.py","research/scratch_kd_lineage.py","research/scratch-kd-training-plan.md",
 "research/configs/wrn16_scratch_kd_control.json","research/configs/wrn16_scratch_kd.json","research/configs/wrn16_scratch_kd_seed2027.json",
 "research/paired_compare_v2.py","research/paired_compare.py","research/benchmark_shadow.py","research/train_shadow.py",
 "research/augmentations.py","research/benchmark.py","research/incumbent_compare.py")


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fixed_config(alpha=.5,seed=2026):
    require(type(alpha) in (int,float) and alpha in (0.,.5) and type(seed) is int and seed in (2026,2027)
        and not (alpha==0 and seed!=2026),"Only exact control26/KD26/KD27 recipes")
    require(sha(ROOT/BASE_CONFIG)==BASE_SHA,"Frozen160 parent config changed")
    tree=ast.parse((ROOT/"train.py").read_text())
    defaults=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="DEFAULT_CONFIG" for x in n.targets))
    return {**defaults,**read(ROOT/BASE_CONFIG),"seed":seed,"kd_temperature":4.,"kd_alpha":float(alpha)}


def validate_config(config):
    expected=fixed_config(config.get("kd_alpha"),config.get("seed"))
    require(config==expected and all(type(config[k]) is type(v) for k,v in expected.items()),"Only parent16/160 plus fixed KD alpha/T and repeat seed supported")


def input_config(config,raw):
    validate_config(config)
    require(isinstance(raw,bytes),"Actual raw input bytes required")
    effective={**fixed_config(config["kd_alpha"],config["seed"]),**json.loads(raw)}
    require(effective==config,"Raw/effective configuration differs")
    # Require every parent/KD raw key, rather than filling omitted treatment keys.
    require(json.loads(raw)=={**read(ROOT/BASE_CONFIG),"seed":config["seed"],"kd_temperature":4.,"kd_alpha":config["kd_alpha"]},"Exact raw parent-plus-KD fields required")
    return raw


def validate_teacher_reference(reference):
    # Root-owned pure lineage verifier is the single canonical8-key contract.
    from research.scratch_kd_lineage import validate_teacher_reference as validate
    return validate(reference)


def research_provenance(config,raw,teacher_reference,precision="bf16"):
    input_config(config,raw);validate_teacher_reference(teacher_reference)
    require(precision in ("bf16","fp32"),"Explicit nativeBF16/CPU math precision required")
    return {"schema":SCHEMA,"method":"scratch_kd","precision":precision,"training_seed":config["seed"],"split_seed":2026,
        "kd_temperature":4.,"kd_alpha":config["kd_alpha"],"teacher_reference_sha256":json_sha(teacher_reference),
        "teacher_checkpoint_sha256":TEACHER_CHECKPOINT,"teacher_frozen_manifest_sha256":TEACHER_FREEZE,"teacher_benchmark_sha256":TEACHER_BENCHMARK,
        "teacher_parameters":TEACHER_PARAMETERS,"student_parameters":PARAMETERS,"teacher_training_seed":2026,"teacher_training_examples":40000,
        "teacher_training_precision":"fp16","teacher_training_attempted_updates":93900,"teacher_training_successful_updates":93864,
        "teacher_training_skipped_updates":36,"teacher_training_git_commit":None,"teacher_forward_precision":precision,
        "source_sha256":{name:sha(ROOT/name) for name in SOURCE_KEYS},"input_config_sha256":hashlib.sha256(raw).hexdigest(),
        "effective_config_sha256":json_sha(config),"grad_scaler_enabled":False,"gradient_clipping":False,"resume_supported":False,
        "teacher_eval":True,"teacher_no_grad":True,"teacher_same_final_mixed_tensor":True,"teacher_TTA":False,
        "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False}


@lru_cache(maxsize=1)
def _deps():
    import runtime  # Must occur only after controller authentication, before Torch.
    import numpy as np
    import torch
    import train as core
    from model import build_model,count_parameters
    from data import load_training_arrays
    from research import scratch_kd as kd
    from research import paired_compare_v2 as paired
    return SimpleNamespace(runtime=runtime,np=np,t=torch,core=core,build_model=build_model,count_parameters=count_parameters,
        load_training_arrays=load_training_arrays,kd=kd,paired=paired)


def load_teacher(reference,device,signature,train_ids,val_ids):
    validate_teacher_reference(reference);m=_deps();t=m.t
    context={"device_type":device.type,"amp_dtype":"torch.bfloat16" if device.type=="cuda" else "torch.float32",
        "torch_version":t.__version__,"training_core_sha256":m.core.code_provenance()["source_sha256"]}
    with m.kd.isolated_teacher_rng():
        checkpoint,document=m.paired.load_experiment(Path(reference["checkpoint_path"]),Path(reference["benchmark_path"]),signature,context,
            m.core.code_provenance()["source_sha256"],train_ids,val_ids)
        require(checkpoint["parameter_count"]==TEACHER_PARAMETERS and checkpoint["epoch"]==298 and checkpoint["variant"]=="model","Typed teacher selected state differs")
        teacher=m.build_model(checkpoint["model_config"]).to(device=device,memory_format=t.channels_last)
        teacher.load_state_dict(checkpoint["model"],strict=True);teacher.eval().requires_grad_(False)
        require(m.count_parameters(teacher)==TEACHER_PARAMETERS and len(teacher.state_dict())==191
            and sum(isinstance(layer,t.nn.BatchNorm2d) for layer in teacher.modules())==31,"Teacher topology differs")
        snapshot=m.kd.teacher_state_snapshot(teacher)
    return teacher,snapshot


def _snapshot(model,ema,optimizer):
    m=_deps()
    return {"model":{k:v.detach().clone() for k,v in model.state_dict().items()},"ema":{k:v.detach().clone() for k,v in ema.state_dict().items()},
        "optimizer":copy.deepcopy(optimizer.state_dict()),"grads":[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
        "rng":m.kd.capture_parent_rng()}


def _rollback(saved,model,ema,optimizer):
    m=_deps();model.load_state_dict(saved["model"]);ema.load_state_dict(saved["ema"]);optimizer.load_state_dict(saved["optimizer"])
    for p,g in zip(model.parameters(),saved["grads"]):p.grad=None if g is None else g.clone()
    m.kd.restore_parent_rng(saved["rng"])


def train_step(model,ema,optimizer,train_x,train_y,order,batch,config,updates,*,epoch=1,teacher=None,teacher_snapshot=None,dtype=None):
    """One original parent update plus fixed KD; all safety/rollback costs included."""
    m=_deps();t=m.t;F=t.nn.functional;device=train_y.device;dtype=dtype or (t.bfloat16 if device.type=="cuda" else t.float32)
    require(type(updates) is int and updates>=0 and type(epoch) is int and epoch>=1,"Incoming update/epoch counters required")
    require(teacher is None or teacher_snapshot is not None,"A provided teacher needs its setup snapshot")
    saved=_snapshot(model,ema,optimizer)
    try:
        lr=m.core.schedule((epoch-1)*313+batch,160*313,313,config)
        for group in optimizer.param_groups:group["lr"]=lr
        ids=order[batch*config["batch_size"]:(batch+1)*config["batch_size"]];y=train_y[ids]
        x=m.core.augment(train_x[ids],config["cutout"])
        mixed=config["mixup_alpha"]>0 and random.random()<config["mixup_probability"]
        lam=float(m.np.random.beta(config["mixup_alpha"],config["mixup_alpha"])) if mixed else 1.
        permutation=t.randperm(len(y),device=device) if mixed else None
        if mixed:x=x*lam+x[permutation]*(1-lam)
        optimizer.zero_grad(set_to_none=True)
        with t.autocast(device.type,dtype=dtype,enabled=dtype!=t.float32):
            logits=model(x)
            # Alpha0 executes the original hard-loss operations/placement exactly.
            if config["kd_alpha"]==0:
                hard=F.cross_entropy(logits,y,label_smoothing=config["label_smoothing"])*lam
                if mixed:hard=hard+F.cross_entropy(logits,y[permutation],label_smoothing=config["label_smoothing"])*(1-lam)
        if config["kd_alpha"]>0:
            require(teacher is not None and teacher_snapshot is not None,"KD requires its immutable fixed teacher")
            with t.autocast(device.type,dtype=dtype,enabled=dtype!=t.float32):teacher_logits=m.kd.frozen_teacher_forward(teacher,x,teacher_snapshot)
            with t.autocast(device.type,enabled=False):
                hard=F.cross_entropy(logits.float(),y,label_smoothing=config["label_smoothing"])*lam
                if mixed:hard=hard+F.cross_entropy(logits.float(),y[permutation],label_smoothing=config["label_smoothing"])*(1-lam)
                components=m.kd.distillation_components(logits,teacher_logits,hard,config["kd_temperature"],config["kd_alpha"])
                loss=components["loss"]
        else:
            loss=m.kd.distillation_loss(None,None,hard,config["kd_temperature"],0.)
            if teacher is not None:m.kd.assert_teacher_immutable(teacher,teacher_snapshot)
        require(bool(t.isfinite(loss).item()),"Nonfinite loss rejected before update")
        loss.backward();grads=[p.grad for p in model.parameters()]
        require(all(g is not None and bool(t.isfinite(g).all().item()) for g in grads),"Nonfinite/missing student gradient rejected")
        norm=t.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt()
        require(bool(t.isfinite(norm).item()) and norm.item()>0,"Finite positive gradient norm required")
        optimizer.step();updates+=1
        with t.no_grad():
            decay=min(config["ema_decay"],(1+updates)/(10+updates))
            t._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
        states=[*model.state_dict().values(),*ema.state_dict().values(),*(v for state in optimizer.state.values() for v in state.values() if t.is_tensor(v))]
        require(all(bool(t.isfinite(v).all().item()) for v in states),"Nonfinite model/EMA/SGD state rejected and rolled back")
        require(all(layer.num_batches_tracked.item()==saved["model"][name+".num_batches_tracked"].item()+1
            for name,layer in model.named_modules() if isinstance(layer,t.nn.BatchNorm2d)),"Student BN must advance exactly once")
        if teacher is not None:m.kd.assert_teacher_immutable(teacher,teacher_snapshot)
        with t.no_grad():
            predicted=logits.detach().argmax(1);correct=predicted.eq(y).sum()*lam
            if mixed:correct+=predicted.eq(y[permutation]).sum()*(1-lam)
            loss_sum=loss.detach()*len(y)
        return {"loss":loss.detach(),"hard_loss":hard.detach(),"soft_loss":components["kd_loss"].detach() if config["kd_alpha"] else None,
            "logits":logits.detach(),"lam":lam,"permutation":permutation,"mixed":mixed,"updated":True,"updates":updates,"lr":lr,
            "student_forward_passes":1,"teacher_forward_passes":int(config["kd_alpha"]>0),"gradient_norm":norm.item(),
            "gradient_tensor_count":len(grads),"state_tensor_count":len(states),"source_examples":len(y),"teacher_inputs":len(y) if config["kd_alpha"] else 0,
            "correct_sum":correct,"loss_sum":loss_sum,"batch_examples":len(y)}
    except BaseException:
        _rollback(saved,model,ema,optimizer)
        if teacher is not None:m.kd.restore_teacher_state(teacher,teacher_snapshot)
        raise


def best_selection(history):
    best=(-1.,float("-inf"));selected=None
    for row in history:
        online=(row["validation_accuracy"],-row["validation_loss"]);average=(row["ema_validation_accuracy"],-row["ema_validation_loss"])
        variant="ema" if average>=online else "model";metric=average if variant=="ema" else online
        require(row["selected_variant"]==variant,"Original model/EMA choice differs")
        if metric>best:best=metric;selected=(row["epoch"],variant)
    return selected,best


def validate_history(history,config):
    require(isinstance(history,list) and len(history)==160,"Complete160 epochs required")
    teacher_passes=313 if config["kd_alpha"] else 0
    for epoch,row in enumerate(history,1):
        require(row.get("epoch")==epoch and type(row["epoch"]) is int and all(type(row.get(k)) is int and row[k]==v for k,v in
            {"successful_updates":313,"skipped_updates":0,"student_forward_passes":313,"teacher_forward_passes":teacher_passes,
             "source_examples_seen":40000,"student_inputs_seen":40000,"teacher_inputs_seen":40000 if config["kd_alpha"] else 0}.items()),"Complete student/teacher update/source ledger required")
        for key in ("train_loss","validation_loss","ema_validation_loss","lr","epoch_seconds","elapsed_seconds"):
            require(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and row[key]>=0,"Finite history scalar required")
        for key in ("train_accuracy","validation_accuracy","ema_validation_accuracy"):
            require(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and 0<=row[key]<=1,"Finite history accuracy required")
    return {"successful_updates":50080,"skipped_updates":0,"student_forward_passes":50080,"teacher_forward_passes":50080 if config["kd_alpha"] else 0,
        "source_examples_seen":6400000,"student_inputs_seen":6400000,"teacher_inputs_seen":6400000 if config["kd_alpha"] else 0}


def validate_saved_state(last,config):
    """Actual last tensors, original SGD and complete parent RNG, no teacher copy."""
    m=_deps();t=m.t
    with m.kd.isolated_teacher_rng():reference=m.build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
    expected=reference.state_dict();updates=160*313
    for kind in ("model","ema"):
        state=last.get(kind,{})
        require(set(state)==set(expected),"Actual last student/EMA topology differs")
        for name,value in expected.items():
            actual=state[name]
            require(isinstance(actual,t.Tensor) and actual.shape==value.shape and actual.dtype==value.dtype and bool(t.isfinite(actual).all().item()),"Last student/EMA tensor shape/dtype/finite state differs")
            if name.endswith("num_batches_tracked"):require(actual.item()==updates,"Last epoch-end BN buffer accounting differs")
    parameters=list(reference.parameters());optimizer=last.get("optimizer",{});groups=optimizer.get("param_groups",[])
    require(len(groups)==1 and groups[0].get("params")==list(range(44)) and groups[0].get("momentum")==.9 and groups[0].get("weight_decay")==.0005
        and groups[0].get("nesterov") is True and groups[0].get("dampening")==0 and groups[0].get("maximize") is False
        and groups[0].get("lr")==last["history"][-1]["lr"] and set(optimizer.get("state",{}))==set(range(44)),"Actual original SGD topology/settings differ")
    for index,parameter in enumerate(parameters):
        state=optimizer["state"][index];value=state.get("momentum_buffer")
        require(set(state)=={"momentum_buffer"} and isinstance(value,t.Tensor) and value.shape==parameter.shape and value.dtype==t.float32
            and bool(t.isfinite(value).all().item()),"Actual saved momentum tensor differs")
    require(isinstance(last.get("torch_rng"),t.Tensor) and last["torch_rng"].dtype==t.uint8 and last["torch_rng"].ndim==1 and last["torch_rng"].numel()>0,
        "Actual saved CPU RNG byte state required")
    if last["provenance"]["research_training"]["precision"]=="bf16":require(isinstance(last.get("cuda_rng"),list) and bool(last["cuda_rng"]),"Formal saved allCUDA RNG byte states required")
    else:require(last.get("cuda_rng") is None or isinstance(last["cuda_rng"],list),"CPU math fixture RNG shape differs")
    for state in last["cuda_rng"] or []:require(isinstance(state,t.Tensor) and state.dtype==t.uint8 and state.ndim==1 and state.numel()>0,"Actual CUDA RNG tensor differs")
    with m.kd.isolated_teacher_rng():
        saved={"python":last["python_rng"],"numpy":last["numpy_rng"],"torch_cpu":last["torch_rng"],"torch_cuda":last["cuda_rng"]}
        m.kd.restore_parent_rng(saved)
        m.kd.assert_parent_rng_equal(saved,m.kd.capture_parent_rng())
    require(not any(key in last for key in ("teacher","teacher_model","teacher_weights","grad_scaler","ta_rng_state")),"Last state must contain only student state plus teacher reference")


def setup(config,data_dir,device,teacher_reference):
    """Unique native setup shared by formal training and actual preflight."""
    validate_config(config);validate_teacher_reference(teacher_reference);data_dir=Path(data_dir)
    m=_deps();t=m.t;device=t.device(device)
    require(device.type=="cuda" and t.cuda.is_available() and t.cuda.is_bf16_supported(including_emulation=False),"Formal KD/control requires real native CUDA BF16")
    m.runtime.seed_all(config["seed"])
    images,labels,train_ids,val_ids=m.load_training_arrays(data_dir)
    manifest=read(data_dir/"training_manifest.json")
    require(len(train_ids)==40000 and len(val_ids)==5000 and not set(train_ids.tolist())&set(val_ids.tolist())
        and manifest["split"]["seed"]==2026 and manifest["split"]["train_indices_sha256"]=="b8472b8b293ad1aacad02abb7fcb7bfe8d37fd8065f280f675dee6ae8f9d2afd","Exact official40k/5000 partition required")
    source=m.core.training_source(data_dir);signature={"archive_sha256":source["archive_sha256"],"split_sha256":manifest["split"]["indices_sha256"]}
    require(signature==DATA,"Official scratch data lineage differs")
    teacher,teacher_snapshot=load_teacher(teacher_reference,device,signature,train_ids,val_ids)
    train_x=m.core.tensor_images(images[train_ids],device);train_y=t.as_tensor(labels[train_ids],device=device)
    val_x=m.core.tensor_images(images[val_ids],device);val_y=t.as_tensor(labels[val_ids],device=device);del images,labels
    model_config={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}
    model=m.build_model(model_config).to(device=device,memory_format=t.channels_last);ema=copy.deepcopy(model).eval().requires_grad_(False)
    require(m.count_parameters(model)==m.count_parameters(model,True)==PARAMETERS,"Single residual student parameter count differs")
    optimizer=t.optim.SGD(model.parameters(),lr=config["lr"],momentum=.9,weight_decay=.0005,nesterov=True)
    return SimpleNamespace(model=model,ema=ema,optimizer=optimizer,train_x=train_x,train_y=train_y,train_ids=train_ids,val_ids=val_ids,val_x=val_x,val_y=val_y,
        teacher=teacher,teacher_snapshot=teacher_snapshot,dtype=t.bfloat16,config=config,teacher_reference=teacher_reference,device=device,
        source=source,signature=signature,model_config=model_config)


def start_epoch(bundle,epoch):
    require(type(epoch) is int and 1<=epoch<=160,"Exact epoch range required")
    bundle.model.train()
    return _deps().t.randperm(len(bundle.train_y),device=bundle.device)


def train(config,output,data_dir,device,*,teacher_reference,input_config_bytes,role=None):
    """Fresh complete160 only. Controller owns all auth/lease/resource admission."""
    validate_config(config);raw=input_config(config,input_config_bytes);validate_teacher_reference(teacher_reference)
    role=role or ("development_control" if config["kd_alpha"]==0 else "primary" if config["seed"]==2026 else "development_repeat")
    require(role==("development_control" if config["kd_alpha"]==0 else "primary" if config["seed"]==2026 else "development_repeat"),"Role differs from exact treatment/seed")
    output=Path(output);data_dir=Path(data_dir)
    artifacts=("best.pt","last.pt","history.json","metrics.json","provenance.json","environment.json","config.json","input_config.json","teacher_reference.json")
    require(not any((output/name).exists() for name in artifacts),"Existing training artifacts refused; resume unsupported")
    bundle=setup(config,data_dir,device,teacher_reference);m=_deps();t=m.t;device=bundle.device
    model,ema,optimizer=bundle.model,bundle.ema,bundle.optimizer
    train_x,train_y,val_x,val_y=bundle.train_x,bundle.train_y,bundle.val_x,bundle.val_y
    teacher,teacher_snapshot=bundle.teacher,bundle.teacher_snapshot;source,signature,model_config=bundle.source,bundle.signature,bundle.model_config
    provenance=m.core.code_provenance();marker=research_provenance(config,raw,teacher_reference);provenance["research_training"]=marker
    output.mkdir(parents=True,exist_ok=True);(output/"input_config.json").write_bytes(raw);m.runtime.write_json(output/"config.json",config)
    m.runtime.write_json(output/"teacher_reference.json",teacher_reference)
    m.runtime.write_json(output/"environment.json",{**m.runtime.environment(),"schema":SCHEMA,"research_training":marker,"role":role})
    m.runtime.write_json(output/"provenance.json",{"schema":SCHEMA,"code":provenance,"official_data":source,"data_signature":signature,
        "train_examples":40000,"validation_examples":5000,"parameter_count":PARAMETERS,"normalization":{"mean":m.core.MEAN,"std":m.core.STD},"role":role})
    history=[];updates=0;best_accuracy=-1.;best_loss=float("inf");started=time.perf_counter()
    for epoch in range(1,161):
        epoch_started=time.perf_counter();order=start_epoch(bundle,epoch)
        loss_sum=t.zeros((),device=device);correct=t.zeros((),device=device);seen=0
        for batch,start in enumerate(range(0,40000,128)):
            result=train_step(model,ema,optimizer,train_x,train_y,order,batch,config,updates,epoch=epoch,teacher=teacher,teacher_snapshot=teacher_snapshot,dtype=t.bfloat16)
            updates=result["updates"]
            with t.no_grad():
                correct+=result["correct_sum"];loss_sum+=result["loss_sum"];seen+=result["batch_examples"]
        with t.no_grad():
            for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
        regular=m.core.evaluate(model,val_x,val_y,device);average=m.core.evaluate(ema,val_x,val_y,device)
        choice="ema" if (average["accuracy"],-average["loss"])>=(regular["accuracy"],-regular["loss"]) else "model"
        metric=average if choice=="ema" else regular;elapsed=time.perf_counter()-started
        row={"epoch":epoch,"train_loss":loss_sum.item()/seen,"train_accuracy":correct.item()/seen,"validation_loss":regular["loss"],"validation_accuracy":regular["accuracy"],
            "ema_validation_loss":average["loss"],"ema_validation_accuracy":average["accuracy"],"selected_variant":choice,"lr":result["lr"],"epoch_seconds":time.perf_counter()-epoch_started,
            "elapsed_seconds":elapsed,"successful_updates":313,"skipped_updates":0,"student_forward_passes":313,"teacher_forward_passes":313 if config["kd_alpha"] else 0,
            "source_examples_seen":seen,"student_inputs_seen":seen,"teacher_inputs_seen":seen if config["kd_alpha"] else 0}
        history.append(row)
        if (metric["accuracy"],-metric["loss"])>(best_accuracy,-best_loss):
            best_accuracy,best_loss=metric["accuracy"],metric["loss"];selected=ema if choice=="ema" else model
            m.core.save_checkpoint(output/"best.pt",{"schema":SCHEMA,"model":selected.state_dict(),"model_config":model_config,"config":config,"epoch":epoch,"variant":choice,
                "validation_accuracy":best_accuracy,"validation_loss":best_loss,"parameter_count":PARAMETERS,"data_signature":signature,"provenance":provenance,"teacher_reference_sha256":json_sha(teacher_reference),"role":role})
        totals={"successful_updates":updates,"skipped_updates":0,"student_forward_passes":updates,"teacher_forward_passes":updates if config["kd_alpha"] else 0,
            "source_examples_seen":epoch*40000,"student_inputs_seen":epoch*40000,"teacher_inputs_seen":epoch*40000 if config["kd_alpha"] else 0}
        m.runtime.write_json(output/"history.json",history)
        m.runtime.write_json(output/"metrics.json",{"schema":SCHEMA,"status":"complete" if epoch==160 else "running","epochs_completed":epoch,"best_validation_accuracy":best_accuracy,
            "best_validation_loss":best_loss,"parameter_count":PARAMETERS,"elapsed_seconds":elapsed,"training_examples":40000,"validation_examples":5000,"config":config,
            "research_training":marker,"teacher_reference_sha256":json_sha(teacher_reference),"role":role,**totals})
        rng=m.kd.capture_parent_rng()
        m.core.save_checkpoint(output/"last.pt",{"schema":SCHEMA,"formal_training_state":True,"preflight_state":False,"model":model.state_dict(),"ema":ema.state_dict(),
            "optimizer":optimizer.state_dict(),"epoch":epoch,"updates":updates,"config":config,"model_config":model_config,"parameter_count":PARAMETERS,"data_signature":signature,
            "provenance":provenance,"history":history,"best_accuracy":best_accuracy,"best_loss":best_loss,"elapsed_seconds":elapsed,"input_config_sha256":hashlib.sha256(raw).hexdigest(),
            "teacher_reference_sha256":json_sha(teacher_reference),"role":role,"python_rng":rng["python"],"numpy_rng":rng["numpy"],"torch_rng":rng["torch_cpu"],"cuda_rng":rng["torch_cuda"],**totals})
        print(json.dumps(row),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv);parser.print_help()


if __name__=="__main__":main()
