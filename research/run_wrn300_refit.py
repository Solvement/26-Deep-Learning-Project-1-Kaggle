"""New guarded full50k WRN300 refit; original core/shadow/refit paths stay frozen.

Explicit prepare, actual-image preflight, training/resume and future prediction.
Bare workers are rejected before Torch. No action is taken without arguments.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
SCHEMA=SUPERVISOR_SCHEMA="research_wrn300_refit_local_supervisor_v1"
PLAN_SCHEMA="research_wrn300_full_refit_plan_v1"
PARENT_DELIVERY_SHA="f2fc4b3f1577b11b7488f3fca713ac3086be572f7042681e83f7bac7c1625fab"
PARENT_FROZEN_SHA="b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3"
PARENT_CHECKPOINT_SHA="53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
BASE_SOURCES={"train.py":"4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py":"79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py":"4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py":"75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
    "research/train_shadow.py":"9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
    "research/augmentations.py":"718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
    "research/process_tree_guard.py":"5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80"}
SOURCE_FILES=(*BASE_SOURCES,"research/train_wrn300_refit.py","research/run_wrn300_refit.py","research/wrn300-refit-plan.md",
    "research/run_local.py","research/watchdog.py","research/profile_augmix.py","research/submit_candidate.py",
    "research/delivery.py","research/delivery_v2.py","research/shadow_repeat_attestation.py","research/repeat_attestation.py",
    "research/benchmark.py","research/benchmark_shadow.py","research/paired_compare.py","research/paired_compare_v2.py","research/campaign.json")
FROZEN_CRLF={
    "research/delivery_v2.py":("bfcdec5bbf0a4e2d67f867ff6761f52a7abf4036b6ba6c0f335ca423d6eb4a3b",
        "88912c25bb1738b9cf1544eabe4c02023eb2c53d231b8de420661b2f9b63d3e5"),
    "research/shadow_repeat_attestation.py":("e5e7c3d0399db7ada0303b639e42dd595293d483a0f73d9c370aaa55a3d0ae6c",
        "58225ce9c3aca232f2a648801bf15da1cbe4a1f14d9262b551f8071711fa8cd0")}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc)


def same_json(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(same_json(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(same_json(a,b) for a,b in zip(actual,expected))
    return actual==expected


def source_snapshot():
    actual={key:sha256(ROOT/key) for key in SOURCE_FILES}
    require(all(actual[key]==value for key,value in BASE_SOURCES.items()),"Frozen original core/shadow/OS guard changed")
    return actual


def committed_source(commit,sources):
    """Keep raw proof hashes; record Git serialization without ignoring content.

Only the two preexisting frozen CRLF files may differ from their Git LF blob,
and both byte fingerprints plus exact CRLF-to-LF equivalence are pinned here.
The recorded Git fingerprint is SHA256 of blob CONTENT, not Git's object ID.
"""
    require(isinstance(commit,str) and re.fullmatch(r"[0-9a-f]{40}",commit),"Exact source Git commit required")
    records={}
    for name,digest in sources.items():
        blob=subprocess.check_output(["git","show",commit+":"+name],cwd=ROOT)
        blob_digest=hashlib.sha256(blob).hexdigest()
        serialization="exact_bytes"
        if blob_digest!=digest:
            require(FROZEN_CRLF.get(name)==(digest,blob_digest),"Unregistered raw/Git source difference")
            raw=(ROOT/name).read_bytes()
            require(hashlib.sha256(raw).hexdigest()==digest and b"\r\n" in raw and raw.replace(b"\r\n",b"\n")==blob,
                "Frozen CRLF serialization has a content difference")
            serialization="frozen_CRLF_to_LF"
        records[name]={"raw_sha256":digest,"git_blob_content_sha256":blob_digest,"serialization":serialization}
    return records


def exclusive(path,document):
    path=Path(path)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(document,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())


def load_parent(path):
    """CPU-only revalidation of the real qualified/scored parent chain."""
    from research.submit_candidate import validate_manifest
    path=Path(path).resolve();require(sha256(path)==PARENT_DELIVERY_SHA,"Only the actual frozen0.878 WRN300 parent is registered")
    summary=validate_manifest(path,ROOT);document=read(path);frozen=read(document["frozen_manifest_path"])
    require(summary.get("delivery_schema")=="research_prediction_delivery_v2" and summary.get("repeat_receipt_schema")=="research_shadow_cloud_repeat_attestation_v1" and
        summary.get("frozen_manifest_sha256")==PARENT_FROZEN_SHA and summary.get("checkpoint_sha256")==PARENT_CHECKPOINT_SHA and
        summary.get("parameter_count")==4166298 and summary.get("policy")=="center80_shift20" and summary.get("training_examples")==40000 and
        frozen.get("full_refit_supported") is False and frozen.get("weight_variant")=="model" and frozen.get("selected_epoch")==298,
        "Qualified original WRN300 parent/freeze/typed repeat differs")
    receipt_path=ROOT/"results/research/submission_receipts/57854294.json";receipt=read(receipt_path)
    require(receipt.get("status")=="complete" and receipt.get("submission_ref")==57854294 and receipt.get("public_score")==.878 and
        receipt.get("private_score") is None and receipt.get("manifest_sha256")==PARENT_DELIVERY_SHA and
        receipt.get("checkpoint_sha256")==PARENT_CHECKPOINT_SHA,"Actual scored WRN300 receipt differs")
    paths={str(path),str(Path(document["frozen_manifest_path"]).resolve()),str(receipt_path.resolve())}
    for key in ("evidence_sha256",):paths.update(frozen.get(key,{}))
    for key in ("repeat_evidence_sha256","test_files_sha256"):paths.update(document.get(key,{}))
    attestation_path=Path(document["repeat"]["receipt_path"]).resolve();attestation=read(attestation_path)
    paths.add(str(attestation_path));paths.add(str(Path(document["csv_path"]).resolve()));paths.add(str(Path(document["test_manifest_path"]).resolve()))
    paths.update(attestation.get("evidence_sha256",{}));paths.update(str((ROOT/name).resolve()) for name in attestation.get("source_sha256",{}))
    return summary,document,frozen,sorted(paths)


def prepare(run,parent_delivery):
    from research.train_wrn300_refit import fixed_config,validate_config,SIGNATURE
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",run or "") and not run.startswith("stability"),"Unique WRN300 refit ID required")
    policy=read(ROOT/"research/campaign.json");admission_budget(policy,now(),0)
    summary,parent,frozen,paths=load_parent(parent_delivery)
    config={**frozen["config"],"train_all":True,"final_variant":"model"};validate_config(config)
    require(config==fixed_config() and frozen["data_signature"]==SIGNATURE,"Parent recipe/data identity differs")
    folder=ROOT/"results/research"/run;require(not folder.exists(),"Use a fresh immutable refit plan directory")
    from train import code_provenance
    source=source_snapshot();source_commit=code_provenance().get("git_commit");git_serialization=committed_source(source_commit,source)
    folder.mkdir(parents=True)
    exclusive(folder/"input_config.json",config)
    plan={"schema":PLAN_SCHEMA,"status":"prepared","run":run,"prepared_at":now().isoformat(),"config":config,
        "input_config_path":str((folder/"input_config.json").resolve()),"input_config_sha256":sha256(folder/"input_config.json"),
        "source_sha256":source,"source_git_commit":source_commit,"source_git_serialization":git_serialization,
        "core_source_sha256":{key:BASE_SOURCES[key] for key in ("train.py","model.py","data.py","runtime.py")},
        "parent_delivery_path":str(Path(parent_delivery).resolve()),"parent_delivery_sha256":PARENT_DELIVERY_SHA,
        "frozen_manifest_path":parent["frozen_manifest_path"],"frozen_manifest_sha256":PARENT_FROZEN_SHA,
        "parent_checkpoint_sha256":PARENT_CHECKPOINT_SHA,"parent_evidence_sha256":{p:sha256(p) for p in paths},
        "policy":"center80_shift20","weight_variant":"model","training_precision":"fp16","initial_grad_scaler_scale":65536.0,
        "parameter_count":4166298,"training_examples":50000,"validation_examples":0,"epochs_completed":300,
        "attempted_updates":117300,"source_examples_seen":15000000,"data_signature":SIGNATURE,"data_dir":frozen["data_dir"],
        "evaluation_context":frozen["evaluation_context"],"initialization":"scratch_seeded","parent_weights_loaded":False,
        "confirmation_evaluated":False,"hidden_images_loaded":False,"independent_development_evaluated":False,
        "old_refit_helpers_support_this_extension":False}
    exclusive(folder/"refit_plan.json",plan)
    return folder/"refit_plan.json"


def validate_plan(path,*,audit_parent=True):
    from research.train_wrn300_refit import validate_config,SIGNATURE
    path=Path(path).resolve();plan=read(path);validate_config(plan.get("config",{}))
    require(plan.get("schema")==PLAN_SCHEMA and plan.get("status")=="prepared" and plan.get("run")==path.parent.name and
        Path(plan.get("input_config_path","")).resolve()==path.parent/"input_config.json" and
        sha256(path.parent/"input_config.json")==plan.get("input_config_sha256") and read(path.parent/"input_config.json")==plan["config"] and
        plan.get("source_sha256")==source_snapshot() and plan.get("core_source_sha256")=={key:BASE_SOURCES[key] for key in ("train.py","model.py","data.py","runtime.py")},
        "Immutable new WRN300 plan/config/source changed")
    fixed={"parent_delivery_sha256":PARENT_DELIVERY_SHA,"frozen_manifest_sha256":PARENT_FROZEN_SHA,"parent_checkpoint_sha256":PARENT_CHECKPOINT_SHA,
        "policy":"center80_shift20","weight_variant":"model","training_precision":"fp16","initial_grad_scaler_scale":65536.0,"parameter_count":4166298,
        "training_examples":50000,"validation_examples":0,"epochs_completed":300,"attempted_updates":117300,"source_examples_seen":15000000,
        "data_signature":SIGNATURE,"initialization":"scratch_seeded","parent_weights_loaded":False,"confirmation_evaluated":False,"hidden_images_loaded":False,
        "independent_development_evaluated":False,"old_refit_helpers_support_this_extension":False}
    require(all(plan.get(k)==v and type(plan[k]) is type(v) for k,v in fixed.items()),"New refit plan protocol/counts differ")
    require(datetime.fromisoformat(plan["prepared_at"]).tzinfo is not None and datetime.fromisoformat(plan["prepared_at"])<=now(),"Aware valid plan time required")
    require(isinstance(plan.get("parent_evidence_sha256"),dict) and plan["parent_evidence_sha256"] and
        all(sha256(p)==digest for p,digest in plan["parent_evidence_sha256"].items()),"Immutable parent evidence changed")
    require(same_json(plan.get("source_git_serialization"),committed_source(plan.get("source_git_commit"),plan["source_sha256"])),
        "Recorded raw/Git source serialization changed")
    if audit_parent:
        _,parent,frozen,paths=load_parent(plan["parent_delivery_path"])
        require(plan["parent_evidence_sha256"]=={p:sha256(p) for p in paths} and plan["frozen_manifest_path"]==parent["frozen_manifest_path"] and
            plan["evaluation_context"]==frozen["evaluation_context"] and plan["data_dir"]==frozen["data_dir"],"Plan differs from actual qualified parent chain")
    return {**plan,"plan_sha256":sha256(path),"plan_path":str(path)}


def admission_budget(policy,current,used,preflight=None):
    require(isinstance(current,datetime) and current.tzinfo is not None and type(used) in (int,float) and math.isfinite(used) and used>=0,
        "Aware time and finite local usage required")
    require(policy.get("concurrent_local_gpu_jobs")==1 and policy.get("local_gpu_seconds_per_utc_day")==21600 and
        policy.get("local_job_timeout_seconds")==10800,"Fixed local concurrency/day/job limits changed")
    stop=datetime.fromisoformat(policy["stops_at"]);require(stop.tzinfo is not None,"Aware hard stop required")
    require(stop.astimezone(timezone.utc)==datetime(2026,11,5,5,tzinfo=timezone.utc),"Fixed five-day campaign hard stop changed")
    budget=min(10800,21600-used,(stop-current).total_seconds());require(budget>=1,"No WRN300 refit local budget remains")
    if preflight is not None:
        median=preflight.get("median_step_seconds")
        require(type(median) in (int,float) and math.isfinite(median) and median>0 and 1.5*median*391*300+300<=budget,
            "Measured actual WRN30050k pipeline cannot fit the fixed guard in remaining budget")
    return budget


def validate_worker(folder,manifest_sha):
    """Actual live OS containment and source/input identity, before Torch."""
    require(folder.parent.resolve()==(ROOT/"results/research").resolve() and sha256(folder/"refit_inputs.json")==manifest_sha,
        "Controlled WRN300 refit worker manifest required")
    receipt=read(folder/"supervisor.json");process,parent=psutil.Process(),psutil.Process(os.getppid())
    started=datetime.fromisoformat(receipt.get("started_at",""));maximum={"preflight":180,"train":10800,"predict":300}.get(receipt.get("mode"),0)
    require(receipt.get("schema")==SCHEMA and receipt.get("status") in ("training","benchmarking") and
        receipt.get("input_manifest_sha256")==manifest_sha and started.tzinfo is not None and
        type(receipt.get("reserved_seconds")) in (int,float) and math.isfinite(receipt["reserved_seconds"]) and 0<receipt["reserved_seconds"]<=maximum and
        type(receipt.get("deadline_unix")) in (int,float) and math.isfinite(receipt["deadline_unix"]) and
        time.time()<receipt["deadline_unix"]<=started.timestamp()+receipt["reserved_seconds"]+.1 and started.timestamp()<=time.time()+.1 and
        receipt.get("pid")==parent.pid and abs(receipt.get("process_created_at",0)-parent.create_time())<.01 and
        receipt.get("child_pid")==process.pid and abs(receipt.get("child_created_at",0)-process.create_time())<.01,"Exact bounded live WRN300 supervisor required")
    lease=read(ROOT/"results/research/local_lease.json")
    require(lease.get("pid")==parent.pid and abs(lease.get("process_created_at",0)-parent.create_time())<.01 and
        lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False,"Canonical live refit lease required")
    matches=[]
    for path in folder.glob("process_tree_*"):
        try:spec,ready=read(path/"spec.json"),read(path/"ready.json")
        except (OSError,ValueError):continue
        if ready.get("pid")==process.pid and abs(ready.get("created_at",0)-process.create_time())<.01:matches.append((path.resolve(),spec,ready))
    require(len(matches)==1,"Exactly one live refit OS guard required")
    path,spec,ready=matches[0];nonce=path.name.removeprefix("process_tree_");guard=(ROOT/"research/process_tree_guard.py").resolve()
    require(re.fullmatch(r"[0-9a-f]{32}",nonce) and spec.get("schema")==ready.get("schema")=="owned_python_process_tree_v1" and
        spec.get("nonce")==ready.get("nonce")==nonce and spec.get("host_pid")==parent.pid and abs(spec.get("host_created_at",0)-parent.create_time())<.01 and
        Path(spec.get("cwd","")).resolve()==ROOT.resolve() and Path(spec.get("script","")).resolve()==Path(__file__).resolve() and
        spec.get("script_sha256")==ready.get("script_sha256")==sha256(__file__) and
        spec.get("guard_source_sha256")==sha256(guard)==BASE_SOURCES["research/process_tree_guard.py"] and
        spec.get("arguments")==["--_worker",str(folder),"--_manifest-sha",manifest_sha] and
        ready.get("method")==("windows_job_v1" if os.name=="nt" else "posix_session_v1") and
        type(spec.get("deadline_unix")) in (int,float) and time.time()<spec["deadline_unix"]<=receipt["deadline_unix"]+.1 and
        spec.get("job_name")==("Local\\SaberAugMix_"+nonce if os.name=="nt" else None),"Refit guard identity/source/deadline differs")
    command=process.cmdline()
    require(len(command)==4 and Path(command[0]).resolve()==Path(sys.executable).resolve() and Path(command[1]).resolve()==guard and
        command[2]=="--bootstrap" and Path(command[3]).resolve()==path/"spec.json","Actual guarded refit bootstrap required")
    from research.profile_augmix import verify_os_boundary
    verify_os_boundary(spec,ready,process)
    manifest=read(folder/"refit_inputs.json")
    require(manifest.get("mode")==receipt.get("mode") and manifest.get("source_sha256")==receipt.get("source_sha256")==source_snapshot() and
        type(manifest.get("input_sha256")) is dict and bool(manifest["input_sha256"]) and manifest["input_sha256"]==receipt.get("input_sha256") and
        all(sha256(p)==digest for p,digest in manifest["input_sha256"].items()) and
        manifest.get("plan_sha256")==receipt.get("plan_sha256")==sha256(manifest["plan_path"]),"Refit bound sources/inputs/plan changed")
    return manifest


def worker(folder,manifest_sha):
    manifest=validate_worker(folder,manifest_sha)
    import runtime
    import torch
    from research import train_wrn300_refit as engine
    require(torch.cuda.is_available(),"Actual CUDA required")
    plan=validate_plan(manifest["plan_path"],audit_parent=False)
    if manifest["mode"]=="preflight":engine.preflight(plan,folder,plan["data_dir"])
    elif manifest["mode"]=="train":engine.train(plan,folder,plan["data_dir"],resume=manifest.get("resume",False))
    else:
        from research.wrn300_refit_delivery import predict
        predict(Path(manifest["plan_path"]),folder,device="cuda",guard_dir=folder)


def validate_records(plan,checkpoint,last,metrics,history,provenance,environment):
    """Deep typed final-state audit, with zero development evaluation."""
    import torch
    from research import train_wrn300_refit as engine
    config=plan["config"];expected=engine.marker(plan,"fp16")
    require(all(doc.get("schema")==engine.SCHEMA for doc in (checkpoint,last,metrics,provenance,environment)),"Independent WRN300 refit artifact schemas required")
    require(all(same_json(mark,expected) for mark in (checkpoint.get("provenance",{}).get("research_training"),last.get("provenance",{}).get("research_training"),
        metrics.get("research_training"),provenance.get("code",{}).get("research_training"),environment.get("research_training"))),
        "Refit explicit source/plan/precision markers differ")
    model_config={k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}
    require(checkpoint.get("config")==last.get("config")==metrics.get("config")==config and
        checkpoint.get("model_config")==last.get("model_config")==model_config and
        checkpoint.get("data_signature")==last.get("data_signature")==provenance.get("data_signature")==engine.SIGNATURE and
        checkpoint.get("parameter_count")==last.get("parameter_count")==metrics.get("parameter_count")==provenance.get("parameter_count")==4166298,
        "Refit actual model/config/official signature/parameters differ")
    require(all(type(value) is int for value in (metrics.get("training_examples"),metrics.get("validation_examples"),provenance.get("train_examples"),
        provenance.get("validation_examples"),metrics.get("attempted_updates"),last.get("attempted_updates"),metrics.get("source_examples_seen"),last.get("source_examples_seen"))),
        "Refit coverage/update counts must be integers, never booleans")
    updates,skipped=engine.validate_history(history,300)
    require(metrics.get("status")=="complete" and metrics.get("epochs_completed")==last.get("epoch")==checkpoint.get("epoch")==300 and
        checkpoint.get("variant")=="model" and last.get("history")==history and
        metrics.get("training_examples")==provenance.get("train_examples")==50000 and metrics.get("validation_examples")==provenance.get("validation_examples")==0 and
        all(key in checkpoint and checkpoint[key] is None for key in ("validation_accuracy","validation_loss")) and
        all(key in metrics and metrics[key] is None for key in ("best_validation_accuracy","best_validation_loss")) and
        last.get("best_accuracy") is None and last.get("best_loss") is None,"Full50k refit must be fixed final300/model with no dev selection")
    require(type(metrics.get("successful_updates")) is int and metrics["successful_updates"]==last.get("updates")==updates and
        type(metrics.get("skipped_updates")) is int and metrics["skipped_updates"]==last.get("skipped_updates")==skipped and
        metrics.get("attempted_updates")==last.get("attempted_updates")==117300 and metrics.get("forward_calls")==117300 and
        metrics.get("source_examples_seen")==last.get("source_examples_seen")==15000000,"Full50k refit cumulative update/source counts differ")
    require(checkpoint["provenance"].get("source_sha256")==last["provenance"].get("source_sha256")==provenance["code"].get("source_sha256")==plan["core_source_sha256"],"Refit frozen core source differs")
    require(all(doc.get("git_commit")==plan["source_git_commit"] for doc in (checkpoint["provenance"],last["provenance"],provenance["code"])),
        "Refit recorded source commit differs from its immutable plan")
    engine.validate_state(last,config,"fp16")
    require(last["grad_scaler"]["scale"]==history[-1]["grad_scaler_scale"],"Final refit scaler differs from final recorded epoch")
    require(engine.equal_state(checkpoint["model"],last["model"]),"Fixed final weights differ from final online state")
    require(environment.get("training_precision")=="fp16" and environment.get("amp_dtype")=="torch.float16" and
        environment.get("grad_scaler_enabled") is True and environment.get("initial_grad_scaler_scale")==65536.0 and environment.get("device")=="cuda",
        "Refit actual training precision differs; BF16 prediction is a separate context")


def receipt_timing(receipt,plan,*,successful=True):
    require(receipt.get("schema")==SCHEMA and receipt.get("backend")=="local" and receipt.get("source_sha256")==plan["source_sha256"] and
        receipt.get("plan_sha256")==plan["plan_sha256"],"Refit attempt schema/source/plan differs")
    started,ended=[datetime.fromisoformat(receipt[key]) for key in ("started_at","finished_at")]
    stop=datetime(2026,11,5,5,tzinfo=timezone.utc)
    maximum={"preflight":180,"train":10800,"predict":300}.get(receipt.get("mode"),0)
    require(maximum>0 and started.tzinfo is not None and ended.tzinfo is not None and datetime.fromisoformat(plan["prepared_at"])<=started<=ended<=stop and
        type(receipt.get("reserved_seconds")) in (int,float) and math.isfinite(receipt["reserved_seconds"]) and 0<=receipt["reserved_seconds"]<=maximum and
        type(receipt.get("elapsed_seconds")) in (int,float) and math.isfinite(receipt["elapsed_seconds"]) and
        0<=receipt["elapsed_seconds"]<=receipt["reserved_seconds"]+1 and abs((ended-started).total_seconds()-receipt["elapsed_seconds"])<=5,
        "Refit attempt timestamps/budget/hardstop differ")
    if successful:
        require(receipt.get("status")=="complete" and receipt["reserved_seconds"]>0 and type(receipt.get("deadline_unix")) in (int,float) and
            math.isfinite(receipt["deadline_unix"]) and ended.timestamp()<=receipt["deadline_unix"]+1,"Completed refit exceeded its bounded deadline")


def guard_evidence(folder,receipt,*,successful=True):
    """Verify durable proof for every finished attempt; OS boundary is mandatory."""
    paths=[]
    proofs=receipt.get("process_tree_guard_results",[])
    if receipt.get("child_pid") is None and not successful:
        require(receipt.get("process_tree_cleanup_verified") is not False and not proofs,"Unknown pre-child attempt cannot be ignored")
        return paths
    candidates=[Path(folder)/"refit_inputs.json",*Path(folder).glob("refit_inputs_attempt_*.json")]
    matching=[p for p in candidates if p.is_file() and sha256(p)==receipt.get("input_manifest_sha256")]
    require(matching,"Original attempt input manifest bytes were not preserved")
    inputs=read(matching[0])
    require(inputs.get("mode")==receipt.get("mode") and inputs.get("plan_sha256")==receipt.get("plan_sha256") and
        inputs.get("source_sha256")==receipt.get("source_sha256") and inputs.get("input_sha256")==receipt.get("input_sha256") and
        all(sha256(p)==h for p,h in inputs.get("input_sha256",{}).items()),"Original attempt input manifest binding differs")
    paths.extend(str(p.resolve()) for p in matching)
    require(receipt.get("process_tree_cleanup_verified") is True and isinstance(proofs,list) and proofs,"Refit OS-tree cleanup not verified")
    for proof in proofs:
        nonce=proof.get("nonce")
        require(isinstance(nonce,str) and re.fullmatch(r"[0-9a-f]{32}",nonce),"Refit tree proof nonce invalid")
        directory=Path(folder)/("process_tree_"+nonce);spec,host=read(directory/"spec.json"),read(directory/"host_result.json")
        require(host==proof and host.get("cleanup_verified") is True and spec.get("nonce")==nonce and
            spec.get("host_pid")==receipt.get("pid") and abs(spec.get("host_created_at",0)-receipt.get("process_created_at",0))<.01 and
            spec.get("script_sha256")==sha256(__file__) and spec.get("guard_source_sha256")==BASE_SOURCES["research/process_tree_guard.py"] and
            Path(spec.get("script","")).resolve()==Path(__file__).resolve() and Path(spec.get("cwd","")).resolve()==ROOT.resolve(),
            "Refit actual guard owner/source/cleanup differs")
        require(Path(host.get("guard_directory","")).resolve()==directory.resolve() and
            spec.get("arguments")==["--_worker",str(Path(folder).resolve()),"--_manifest-sha",receipt.get("input_manifest_sha256")] and
            type(spec.get("deadline_unix")) in (int,float) and math.isfinite(spec["deadline_unix"]) and
            spec["deadline_unix"]<=receipt.get("deadline_unix",0)+.1 and spec["deadline_unix"]>datetime.fromisoformat(receipt["started_at"]).timestamp() and
            spec.get("job_name")==("Local\\SaberAugMix_"+nonce if host.get("method")=="windows_job_v1" else None),"Refit guard command/directory/deadline differs")
        names=["spec.json","host_result.json"]
        if successful:
            ready,done=read(directory/"ready.json"),read(directory/"completion.json")
            require(ready.get("schema")==done.get("schema")==host.get("schema")=="owned_python_process_tree_v1" and
                ready.get("method")==done.get("method")==host.get("method") and host.get("method") in ("windows_job_v1","posix_session_v1") and
                ready.get("nonce")==done.get("nonce")==nonce and ready.get("pid")==done.get("pid")==host.get("child_pid")==receipt.get("child_pid") and
                abs(ready.get("created_at",0)-receipt.get("child_created_at",0))<.01 and abs(done.get("created_at",0)-receipt.get("child_created_at",0))<.01 and
                ready.get("script_sha256")==done.get("script_sha256")==spec["script_sha256"] and host.get("completion_authenticated") is True and
                all(type(value) is int and value==0 for value in (host.get("child_returncode"),host.get("completion_exit_code"),done.get("exit_code"))) and host.get("timed_out") is False,
                "Successful refit needs authenticated exact child completion")
            names.extend(("ready.json","completion.json"))
        for name in ("ready.json","completion.json","watchdog_ready.json"):
            if (directory/name).exists() and name not in names:names.append(name)
        paths.extend(str((directory/name).resolve()) for name in names)
    return paths


def validate_refit(plan_path,*,allow_current_host=False):
    import runtime
    import torch
    from research import train_wrn300_refit as engine
    plan_path=Path(plan_path).resolve();plan=validate_plan(plan_path);folder=plan_path.parent
    require(not (folder/"benchmark.json").exists(),"Full50k refit cannot claim overlapping dev benchmark")
    receipt=read(folder/"supervisor.json")
    require(receipt.get("schema")==SCHEMA and receipt.get("mode")=="train" and receipt.get("status")=="complete" and
        receipt.get("plan_sha256")==plan["plan_sha256"] and receipt.get("source_sha256")==plan["source_sha256"] and receipt.get("submitted") is False,
        "Completed source-bound WRN300 refit supervisor required")
    require(receipt.get("trainer")=="wrn300_full_refit" and receipt.get("precision")=="fp16" and receipt.get("config_sha256")==plan["input_config_sha256"] and
        receipt.get("initialization")=="scratch_seeded" and receipt.get("parent_weights_loaded") is False and receipt.get("confirmation_evaluated") is False and
        receipt.get("hidden_images_loaded") is False and type(receipt.get("training_examples")) is int and receipt["training_examples"]==50000 and
        type(receipt.get("validation_examples")) is int and receipt["validation_examples"]==0,"Refit actual supervisor precision/counts/initialization differs")
    receipt_timing(receipt,plan)
    from research.run_local import assert_stopped
    if allow_current_host:
        require(receipt.get("pid")==os.getpid() and abs(receipt.get("process_created_at",0)-psutil.Process().create_time())<.01,"Only the exact current owning host may finalize")
        assert_stopped({**receipt,"pid":None})
    else:assert_stopped(receipt)
    proof_paths=guard_evidence(folder,receipt)
    checkpoint,last=[torch.load(folder/name,map_location="cpu",weights_only=False) for name in ("best.pt","last.pt")]
    metrics,history,provenance,environment=[read(folder/name) for name in ("metrics.json","history.json","provenance.json","environment.json")]
    validate_records(plan,checkpoint,last,metrics,history,provenance,environment)
    hashes={name:sha256(folder/name) for name in engine.ARTIFACTS}
    require(receipt.get("artifact_sha256")==hashes and receipt.get("checkpoint_sha256")==hashes["best.pt"] and read(folder/"config.json")==plan["config"],"Refit final artifact hashes/config differ")
    preflight_path=Path(receipt["preflight_path"]);pf,pref_binding=checked_preflight(preflight_path,plan)
    require(receipt.get("preflight_sha256")==sha256(preflight_path),"Refit admission preflight changed")
    inputs_path=folder/"refit_inputs.json";inputs=read(inputs_path)
    binding=inputs.get("input_sha256",{})
    require(inputs.get("mode")=="train" and inputs.get("plan_sha256")==plan["plan_sha256"] and Path(inputs.get("plan_path","")).resolve()==plan_path and
        inputs.get("source_sha256")==plan["source_sha256"] and binding==receipt.get("input_sha256") and
        sha256(inputs_path)==receipt.get("input_manifest_sha256") and all(sha256(p)==h for p,h in binding.items()) and
        all(binding.get(p)==h for p,h in {**plan["parent_evidence_sha256"],**pref_binding,str(plan_path):plan["plan_sha256"],
            str(Path(plan["input_config_path"]).resolve()):plan["input_config_sha256"]}.items()),"Refit final input/parent/preflight binding differs")
    require(all(environment.get(k)==pf["environment"].get(k) for k in ("gpu","cuda_runtime","cudnn")) and
        environment.get("packages",{}).get("torch")==pf["environment"].get("packages",{}).get("torch"),"Refit actual training context differs from measured preflight")
    evidence={str(plan_path),str(folder/"supervisor.json"),str(folder/"refit_inputs.json"),str(preflight_path),
        str(preflight_path.parent/"supervisor.json"),*pref_binding,*binding,*plan["parent_evidence_sha256"],*proof_paths,
        *(str((folder/name).resolve()) for name in engine.ARTIFACTS),*(str((ROOT/name).resolve()) for name in plan["source_sha256"])}
    for path in folder.glob("supervisor_attempt_*.json"):
        attempt=read(path);require(attempt.get("source_sha256")==plan["source_sha256"] and attempt.get("plan_sha256")==plan["plan_sha256"],"Refit previous attempt identity changed")
        assert_stopped(attempt)
        receipt_timing(attempt,plan,successful=False)
        evidence.add(str(path.resolve()))
        evidence.update(guard_evidence(folder,attempt,successful=False))
    evidence.update(str(p.resolve()) for p in folder.glob("supervisor_raw_*.json"))
    return {"run":plan["run"],"plan_path":str(plan_path),"plan_sha256":plan["plan_sha256"],"config":plan["config"],
        "parent_delivery_path":plan["parent_delivery_path"],"parent_delivery_sha256":PARENT_DELIVERY_SHA,
        "parent_manifest_path":plan["parent_delivery_path"],"parent_manifest_sha256":PARENT_DELIVERY_SHA,
        "frozen_manifest_path":plan["frozen_manifest_path"],"frozen_manifest_sha256":PARENT_FROZEN_SHA,"parent_checkpoint_sha256":PARENT_CHECKPOINT_SHA,
        "checkpoint_path":str((folder/"best.pt").resolve()),"checkpoint_sha256":hashes["best.pt"],"parameter_count":4166298,
        "policy":"center80_shift20","weight_variant":"model","training_examples":50000,"validation_examples":0,"epochs_completed":300,
        "training_precision":"fp16","initial_grad_scaler_scale":65536.0,"attempted_updates":117300,"source_examples_seen":15000000,
        "successful_updates":metrics["successful_updates"],"skipped_updates":metrics["skipped_updates"],"data_signature":plan["data_signature"],
        "source_sha256":plan["source_sha256"],"core_source_sha256":plan["core_source_sha256"],"evaluation_context":plan["evaluation_context"],
        "source_git_commit":plan["source_git_commit"],
        "source_git_serialization":plan["source_git_serialization"],
        "evidence_paths":sorted(evidence),"initialization":"scratch_seeded","parent_weights_loaded":False,"independent_development_evaluated":False}


def checked_preflight(path,plan):
    from research import train_wrn300_refit as engine
    from research.run_local import assert_stopped
    path=Path(path).resolve();receipt_path=path.parent/"supervisor.json";receipt=read(receipt_path)
    require(path.name=="native_preflight.json" and receipt.get("schema")==SCHEMA and receipt.get("mode")=="preflight" and
        receipt.get("status")=="complete" and receipt.get("source_sha256")==plan["source_sha256"] and
        receipt.get("plan_sha256")==plan["plan_sha256"] and receipt.get("artifact_sha256")==sha256(path),"Completed same-plan native refit preflight required")
    receipt_timing(receipt,plan)
    inputs=read(path.parent/"refit_inputs.json")
    require(inputs.get("mode")=="preflight" and inputs.get("plan_sha256")==plan["plan_sha256"] and inputs.get("source_sha256")==plan["source_sha256"] and
        inputs.get("input_sha256")==receipt.get("input_sha256") and sha256(path.parent/"refit_inputs.json")==receipt.get("input_manifest_sha256") and
        all(sha256(p)==h for p,h in inputs.get("input_sha256",{}).items()),"Preflight actual supervised inputs changed")
    assert_stopped(receipt);proofs=guard_evidence(path.parent,receipt)
    report=engine.validate_preflight(read(path),plan)
    return report,{str(p):sha256(p) for p in (path,receipt_path,path.parent/"refit_inputs.json",*(Path(p) for p in proofs))}


def parent(args):
    import runtime
    from research import train_wrn300_refit as engine
    from research.run_local import acquire_lease,used_seconds,assert_stopped,close_interrupted_attempt
    from research.process_tree_guard import run_guarded,may_release_lease
    plan=validate_plan(args.plan);mode=args.mode
    require(mode in ("preflight","train","predict") and (mode=="train")==bool(args.preflight) and
        (not args.resume or mode=="train"),"Only training takes same-plan preflight/exact resume")
    if mode=="train":
        require(args.run in (None,plan["run"]),"Formal refit uses its existing prepared plan directory")
        folder=Path(args.plan).resolve().parent
    else:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",args.run or "") and not args.run.startswith("stability"),"A fresh distinct engineering/prediction run required")
        folder=ROOT/"results/research"/args.run;require(not folder.exists(),"Use a fresh engineering/prediction directory")
    policy=read(ROOT/"research/campaign.json");admission_budget(policy,now(),0)
    manifest_path=Path(plan["data_dir"])/"training_manifest.json";official=read(manifest_path)
    paths=[Path(args.plan).resolve(),Path(plan["input_config_path"]),manifest_path,
        *(Path(plan["data_dir"])/name for name in official["files_sha256"]),*(ROOT/key for key in plan["source_sha256"]),
        *(Path(p) for p in plan["parent_evidence_sha256"])]
    preflight=None;extra={}
    if args.preflight:preflight,extra=checked_preflight(args.preflight,plan)
    if mode=="predict":
        info=validate_refit(args.plan);paths.extend(Path(p) for p in info["evidence_paths"])
        paths.extend(Path(plan["data_dir"])/name for name in ("test.csv","sample_submission.csv","official_test.npz","official_test_source_manifest.json"))
        paths.extend(ROOT/name for name in ("research/wrn300_refit_delivery.py","research/submit_wrn300_refit.py"))
    binding={str(p.resolve()):sha256(p) for p in paths};binding.update(extra)
    receipt_path=folder/"supervisor.json";previous=None
    if args.resume:
        previous=read(receipt_path)
        require(previous.get("schema")==SCHEMA and previous.get("mode")=="train" and previous.get("status")!="complete" and
            previous.get("plan_sha256")==plan["plan_sha256"] and previous.get("source_sha256")==plan["source_sha256"] and
            previous.get("input_sha256")==binding and previous.get("preflight_sha256")==sha256(args.preflight) and
            (folder/"last.pt").is_file() and not (folder/"best.pt").exists() and may_release_lease(previous),"Exact source/input/stopped incomplete refit resume required")
        assert_stopped(previous)
        if previous.get("child_pid") is not None:guard_evidence(folder,previous,successful=False)
    elif mode=="train":
        require(not receipt_path.exists() and set(p.name for p in folder.iterdir())=={"refit_plan.json","input_config.json"},"Prepared refit folder must not contain an old attempt")
    lease=ROOT/"results/research/local_lease.json";require(not lease.exists(),"Refit requires explicitly reconciled free local lease")
    acquire_lease(lease);began=time.perf_counter();owns=False;receipt={}
    try:
        day=now().replace(hour=0,minute=0,second=0,microsecond=0);excluded=receipt_path if args.resume else None
        budget=admission_budget(policy,now(),used_seconds(ROOT/"results/research",day,day+timedelta(days=1),exclude=excluded),preflight)
        if mode!="train":budget=min(budget,180 if mode=="preflight" else 300)
        if preflight:
            actual,expected=runtime.environment(),preflight["environment"]
            require(all(actual.get(k)==expected.get(k) for k in ("gpu","cuda_runtime","cudnn")) and
                actual["packages"]["torch"]==expected["packages"]["torch"],"Actual refit hardware/runtime differs from measured preflight")
        if args.resume:
            old_inputs=(folder/"refit_inputs.json").read_bytes()
            require(previous.get("input_manifest_sha256") is None or hashlib.sha256(old_inputs).hexdigest()==previous["input_manifest_sha256"],
                "Original resume input manifest changed")
            archived=close_interrupted_attempt(receipt_path);owns=True
            number=archived.stem.removeprefix("supervisor_attempt_")
            with (folder/("refit_inputs_attempt_"+number+".json")).open("xb") as stream:
                stream.write(old_inputs);stream.flush();os.fsync(stream.fileno())
            receipt={"schema":SCHEMA,"backend":"local","run":plan["run"],"mode":"train","status":"starting","started_at":now().isoformat(),
                "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"reserved_seconds":0,
                "previous_attempt":archived.name,"source_sha256":plan["source_sha256"],"plan_sha256":plan["plan_sha256"],"input_sha256":binding}
            budget=admission_budget(policy,now(),used_seconds(ROOT/"results/research",day,day+timedelta(days=1),exclude=receipt_path),preflight)
        elif mode!="train":folder.mkdir(parents=True);owns=True
        else:owns=True
        receipt={"schema":SCHEMA,"backend":"local","run":folder.name,"mode":mode,"trainer":"wrn300_full_refit","precision":"fp16" if mode!="predict" else "parent_frozen_inference",
            "status":"training" if mode=="train" else "benchmarking","started_at":now().isoformat(),"pid":os.getpid(),"process_created_at":psutil.Process().create_time(),
            "reserved_seconds":budget,"deadline_unix":time.time()+budget,"source_sha256":plan["source_sha256"],"input_sha256":binding,
            "utc_budget_day":day.date().isoformat(),
            "plan_path":str(Path(args.plan).resolve()),"plan_sha256":plan["plan_sha256"],"config_sha256":plan["input_config_sha256"],
            "initialization":"scratch_seeded","parent_weights_loaded":False,"training_examples":50000,"validation_examples":0,"resumed":args.resume,
            "submitted":False,"confirmation_evaluated":False,"hidden_images_loaded":mode=="predict"}
        if args.resume:receipt.update(previous_attempt=archived.name,resumed_checkpoint_sha256=sha256(folder/"last.pt"))
        if args.preflight:receipt.update(preflight_path=str(args.preflight.resolve()),preflight_sha256=sha256(args.preflight))
        manifest={"mode":mode,"plan_path":str(Path(args.plan).resolve()),"plan_sha256":plan["plan_sha256"],"source_sha256":plan["source_sha256"],
            "input_sha256":binding,"resume":args.resume}
        runtime.write_json(folder/"refit_inputs.json",manifest);receipt["input_manifest_sha256"]=sha256(folder/"refit_inputs.json")
        runtime.write_json(receipt_path,receipt)
        remaining=budget-(time.perf_counter()-began)-(300 if mode=="train" else 0);require(remaining>0,"No bounded refit runtime remains")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder),"--_manifest-sha",receipt["input_manifest_sha256"]],
            folder/(mode+"_"+str(time.time_ns())+".log"),remaining,receipt_path,receipt,cwd=ROOT)
        require(source_snapshot()==plan["source_sha256"] and all(sha256(p)==h for p,h in binding.items()),"Refit source/input changed during execution")
        if mode=="preflight":
            engine.validate_preflight(read(folder/"native_preflight.json"),plan)
            receipt.update(artifact="native_preflight.json",artifact_sha256=sha256(folder/"native_preflight.json"))
        elif mode=="train":
            import torch
            best,last=[torch.load(folder/name,map_location="cpu",weights_only=False) for name in ("best.pt","last.pt")]
            validate_records(plan,best,last,read(folder/"metrics.json"),read(folder/"history.json"),read(folder/"provenance.json"),read(folder/"environment.json"))
            receipt.update(artifact="best.pt",checkpoint_sha256=sha256(folder/"best.pt"),artifact_sha256={name:sha256(folder/name) for name in engine.ARTIFACTS})
        else:receipt.update(artifact="delivery.json",artifact_sha256=sha256(folder/"delivery.json"))
        receipt.update(status="complete",finished_at=now().isoformat(),elapsed_seconds=time.perf_counter()-began)
        runtime.write_json(receipt_path,receipt)
        if mode=="predict":
            try:
                from research.wrn300_refit_delivery import validate_refit_delivery
                validate_refit_delivery(folder/"delivery.json")
            except BaseException:
                exclusive(folder/"validation_failed_receipt.json",receipt)
                raise
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__);raise
    finally:
        if owns and receipt:
            receipt.update(elapsed_seconds=time.perf_counter()-began,finished_at=now().isoformat())
            if not may_release_lease(receipt):receipt["status"]="cleanup_unverified"
            runtime.write_json(receipt_path,receipt)
        if lease.exists():
            owner=read(lease)
            if owner.get("pid")==os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time())<.01:
                if may_release_lease(receipt):lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(receipt_path));runtime.write_json(lease,owner)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true");parser.add_argument("--run");parser.add_argument("--parent-delivery",type=Path)
    parser.add_argument("--plan",type=Path);parser.add_argument("--mode",choices=("preflight","train","predict"));parser.add_argument("--preflight",type=Path)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS);parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker:
        require(not any((args.prepare,args.run,args.parent_delivery,args.plan,args.mode,args.preflight,args.resume)) and args._manifest_sha,"Exact guarded refit worker arguments required")
        worker(args._worker.resolve(),args._manifest_sha)
    elif args.prepare:
        require(args.run and args.parent_delivery and not any((args.plan,args.mode,args.preflight,args.resume,args._manifest_sha)),"Prepare needs fresh run and parent delivery only")
        print(json.dumps({"plan_path":str(prepare(args.run,args.parent_delivery))}))
    elif args.mode:
        require(args.plan is not None and args.parent_delivery is None and args._manifest_sha is None,"Explicit refit plan required")
        parent(args)
    else:
        require(not any((args.run,args.parent_delivery,args.plan,args.preflight,args.resume,args._manifest_sha)),"Explicit prepare or bounded mode required")
        parser.print_help()


if __name__=="__main__":main()
