"""Primary-only TA prediction under its own bounded guard; offline audit is stdlib.

No training, repeat prediction, refit, upload or old-schema dispatch is provided.
Only the authenticated worker imports runtime/Torch and reloads actual tensors.
"""
from __future__ import annotations
import argparse
import csv
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
from research import preflight_core_control as control
from research import profile_augmix as guard_checks
from research import incumbent_compare as aggregate
from research import trivialaugment_paired_v2 as paired
from research import run_trivialaugment_repeat as repeat
from scripts.submit_verified import COMPETITION,SAMPLE_SHA,validate_csv

SCHEMA="research_trivialaugment_prediction_supervisor_v1"
INPUT_SCHEMA="research_trivialaugment_prediction_inputs_v1"
DELIVERY_SCHEMA="research_trivialaugment_prediction_delivery_v1"
AUDIT_SCHEMA="research_trivialaugment_guarded_tensor_audit_v1"
PLAN="research/trivialaugment-prediction-plan.md"
FROZEN_PATH=ROOT/"results/research/ta_primary_freeze_20261103_v2.json"
FROZEN_SHA="88682c6f719ccc4791adaaa26086bd0fa3e0778de526d463da89da1721099e9a"
POLICY="center80_shift20"
TEST_CSV_SHA="d0db3026e3a7bc397d6f59b7ff6825960cc32a5e2aef57df63c9140b2ad31e5d"
PINS={"research/run_trivialaugment_repeat.py":"81b824218a71f8c2cac3acda5a9c711ea1a75d357b917c9e97092d2f223fe489",
      "research/trivialaugment_paired_v2.py":"c839ceec890b73a02c4197e07ea01f6f5ef5203ed9dcbbee98a57ace566d43ad",
      "research/process_tree_guard.py":"5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80",
      "scripts/submit_verified.py":"50007df3dc233c60fe3eb11c4fb0698d708091d1b77b24aa845470265e4ad027"}
ARTIFACTS=("best.pt","last.pt","config.json","input_config.json","metrics.json","history.json","provenance.json","environment.json")
require,sha,read,write=control.require,control.sha256,control.read,control.write


def utc_now():return datetime.now(timezone.utc)


def stamp(value):
    parsed=datetime.fromisoformat(value);require(parsed.tzinfo is not None,"Aware artifact timestamp required");return parsed


def snapshots(paths):return {str(Path(p).resolve()):sha(p) for p in paths}


def exact(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(exact(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(exact(a,b) for a,b in zip(actual,expected))
    return actual==expected


def source_snapshot():
    require(all(sha(ROOT/n)==v for n,v in PINS.items()),"Frozen TA delivery dependencies changed")
    require(sha(FROZEN_PATH)==FROZEN_SHA,"Use the actual pre-outcome frozen primary26 manifest")
    sources=repeat.source_snapshot();require(sources==read(FROZEN_PATH)["planned_repeat_source_sha256"],"Frozen50 repeat source closure changed")
    return {**sources,**PINS,PLAN:sha(ROOT/PLAN),"research/trivialaugment_delivery.py":sha(__file__)}


def inactive(receipt):
    from research.process_tree_guard import matching_process
    for pid_key,time_key in (("pid","process_created_at"),("child_pid","child_created_at")):
        pid,created=receipt.get(pid_key),receipt.get(time_key)
        require(type(pid) is int and pid>0 and aggregate.number(created) and created>0
                and matching_process(pid,created) is None,"Matching completed-job process is active or unidentified")


def benchmark_evidence(folder,document,data_dir):
    """Exact closure emitted by the frozen TA typed benchmark; no tensor load."""
    checkpoint=folder/"best.pt";inc=paired.INCUMBENT_CHECKPOINT
    paths=[checkpoint,*(folder/n for n in ARTIFACTS if n!="best.pt"),ROOT/repeat.BASELINE,aggregate.INCUMBENT_PATH,inc,
       ROOT/"research/incumbent-promotion-plan.md",*(inc.parent/n for n in ("metrics.json","history.json","provenance.json","environment.json")),
       aggregate.INCUMBENT_PATH.with_name("benchmark.core.json"),*(ROOT/n for n in
        ("research/benchmark_trivialaugment.py","research/paired_compare_v2.py","research/paired_compare.py","research/benchmark_shadow.py",
         "research/benchmark.py","research/incumbent_compare.py","research/configs/wrn34_3.json")),
       *(data_dir/n for n in ("training_manifest.json","official_train_images.npy","official_train_labels.npy","official_train_ids.npy","official_split_indices.npz")),
       *(ROOT/n for n in ("research/train_trivialaugment.py","research/trivialaugment.py","research/train_shadow.py","research/augmentations.py",*aggregate.CORE))]
    expected=snapshots(paths)
    require(document.get("evidence_sha256")==expected
       and document.get("trivialaugment_benchmark_source_sha256")==repeat.PINS["research/benchmark_trivialaugment.py"]
       and document.get("checkpoint_sha256")==sha(checkpoint)
       and document.get("training_metrics_sha256")==sha(folder/"metrics.json")
       and document.get("training_history_sha256")==sha(folder/"history.json"),"Actual full typed benchmark/selected/last/history/source closure differs")
    for policy in aggregate.POLICIES:
        expected_gate=paired.point_gate(document["policies"][policy],read(ROOT/repeat.BASELINE)["policies"]["flip"])
        require(exact(document.get("promotion",{}).get(policy),expected_gate),"Recorded original policy gate differs")
    return paths


def completed_repeat(frozen_path=FROZEN_PATH):
    """Deep stdlib receipt/source/benchmark audit; actual tensor audit is later."""
    frozen_path=Path(frozen_path).resolve();require(frozen_path==FROZEN_PATH.resolve() and sha(frozen_path)==FROZEN_SHA,"Only the actual frozen TA primary is supported")
    sources=repeat.source_snapshot();frozen=repeat.validate_freeze(frozen_path,sources)
    folder=Path(frozen["repeat_run_path"]).resolve();manifest=read(folder/"inputs.json");receipt=read(folder/"supervisor.json")
    result=read(folder/"repeat_result.json");document=read(folder/"benchmark.json");digest=sha(folder/"inputs.json")
    preflight_path=Path(manifest["preflight_path"]).resolve()
    preflight,pref_paths=repeat.validate_preflight_bundle(preflight_path,sources,frozen_path)
    binding,identity=repeat.binding_for(Path(frozen["data_dir"]),sources,frozen_path,preflight_path)
    expected_manifest={"schema":repeat.INPUT_SCHEMA,"run":folder.name,"recipe":repeat.RECIPE,"seed":2027,"mode":"formal","role":"development_repeat",
       "data_dir":frozen["data_dir"],"effective_config":repeat.configuration(),"effective_config_sha256":control.json_sha(repeat.configuration()),
       "config_sha256":repeat.CONFIGS[2027][1],"source_sha256":sources,"input_sha256":binding,"data_identity":identity,
       "frozen_manifest_path":str(frozen_path),"frozen_manifest_sha256":FROZEN_SHA,"preflight_path":str(preflight_path)}
    require(all(exact(manifest.get(k),v) for k,v in expected_manifest.items()) and re.fullmatch(r"[0-9a-f]{40}",manifest.get("source_git_commit","")),
            "Actual predeclared fresh27 input/source/config binding differs")
    fixed={"schema":repeat.SCHEMA,"run":folder.name,"backend":"local","mode":"formal","role":"development_repeat","status":"complete",
       "source_sha256":sources,"input_sha256":binding,"input_manifest_sha256":digest,"confirmation_evaluated":False,
       "formal_training_admitted":True,"precision":"bf16","seed":2027,"config_sha256":repeat.CONFIGS[2027][1],
       "preflight_path":str(preflight_path),"frozen_manifest_path":str(frozen_path),"frozen_manifest_sha256":FROZEN_SHA,
       "artifact":"repeat_result.json","artifact_sha256":sha(folder/"repeat_result.json"),"process_tree_cleanup_verified":True}
    require(all(exact(receipt.get(k),v) for k,v in fixed.items()) and receipt.get("cleanup_requires_reconciliation") is not True,
            "Complete actual native TA repeat receipt required")
    start,end=stamp(receipt["started_at"]),stamp(receipt["finished_at"]);duration=(end-start).total_seconds()
    pref_receipt=read(preflight_path.parent/"supervisor.json")
    require(stamp(frozen["frozen_at"])<stamp(pref_receipt["started_at"])<=stamp(pref_receipt["finished_at"])<start<=end<=utc_now()
       and end<=stamp(read(ROOT/"research/campaign.json")["stops_at"])
       and aggregate.number(receipt.get("reserved_seconds"),maximum=10800) and repeat.guard_seconds(preflight["median_step_seconds"])<=receipt["reserved_seconds"]
       and aggregate.number(receipt.get("elapsed_seconds"),maximum=10800) and 0<receipt["elapsed_seconds"]<=receipt["reserved_seconds"]
       and 0<duration<=receipt["reserved_seconds"]+.1 and abs(duration-receipt["elapsed_seconds"])<=1
       and aggregate.number(receipt.get("deadline_unix")) and end.timestamp()<=receipt["deadline_unix"]+.1
       and receipt["deadline_unix"]<=start.timestamp()+receipt["reserved_seconds"]+.1,"Repeat freeze/admission/chronology/accounting differs")
    guard_paths=repeat.completed_guard_paths(folder,receipt,digest);inactive(receipt)
    comparison=repeat.compare_repeat_benchmark(document,frozen);screen=repeat.scalar_screen(document,comparison)
    require(document.get("point_and_case_gates_passed") is True and screen["scalar_screen_passed"] is True
       and read(folder/"incumbent_comparison.json")==comparison,"Repeat fixed weighted mean/per-case gates or comparison failed")
    expected_result={"schema":repeat.REPEAT_SCHEMA,"status":"complete","role":"development_repeat","seed":2027,"recipe":repeat.RECIPE,
       "formal_training_admitted":True,"training_precision":"bf16","config":repeat.configuration(),"config_sha256":repeat.CONFIGS[2027][1],
       "source_sha256":sources,"source_git_commit":manifest["source_git_commit"],"input_manifest_sha256":digest,"data_identity":identity,
       "frozen_manifest_path":str(frozen_path),"frozen_manifest_sha256":FROZEN_SHA,"frozen_primary_checkpoint_sha256":paired.PRIMARY_CHECKPOINT_SHA,
       "policy":POLICY,"preflight_path":str(preflight_path),"preflight_sha256":sha(preflight_path),"checkpoint_sha256":sha(folder/"best.pt"),
       "benchmark_sha256":sha(folder/"benchmark.json"),"incumbent_comparison_sha256":sha(folder/"incumbent_comparison.json"),
       "epochs_completed":160,"successful_updates":50080,"skipped_updates":0,"source_examples_seen":6400000,"model_views_seen":6400000,
       "ta_operation_counts":read(folder/"metrics.json")["ta_operation_counts"],"scalar_screen":screen,"independent_repeat":True,
       "repeat_weights_hidden_use_authorized":False,"primary_replacement_authorized":False,"hidden_images_loaded":False,
       "confirmation_evaluated":False,"submission_authorized":False}
    require(exact(result,expected_result) and sha(folder/"best.pt")!=paired.PRIMARY_CHECKPOINT_SHA,"Actual complete validation-only repeat result differs")
    paths=benchmark_evidence(folder,document,Path(frozen["data_dir"]))
    paths += [folder/n for n in ("inputs.json","supervisor.json","repeat_result.json","benchmark.json","incumbent_comparison.json")]
    paths += [*pref_paths,*guard_paths,*map(Path,binding),*(ROOT/n for n in sources)]
    require(all(sha(p)==v for p,v in binding.items()),"Immutable repeat inputs changed")
    return frozen,document,snapshots(paths)


def official_test_paths(data_dir):
    """Fingerprint only the audited unlabeled export; never decode here."""
    data_dir=Path(data_dir)
    paths=[data_dir/n for n in ("sample_submission.csv","test.csv","official_test.npz","official_test_source_manifest.json","kaggle_test_manifest.json")]
    require(all(p.is_file() for p in paths) and sha(paths[0])==SAMPLE_SHA and sha(paths[1])==TEST_CSV_SHA,"Audited official test/sample inputs required")
    source=read(paths[3]);manifest=read(paths[4]);archive=sha(paths[2])
    require(source.get("competition")==COMPETITION and source.get("source")=="official_kaggle_competition"
       and source.get("no_external_data") is True and source.get("hidden_labels_loaded") is False and source.get("test_labels_in_archive") is False
       and source.get("test_csv_sha256")==TEST_CSV_SHA and source.get("sample_submission_csv_sha256")==SAMPLE_SHA and source.get("archive_sha256")==archive,
       "Official unlabeled test source identity differs")
    require(manifest.get("competition")==COMPETITION and type(manifest.get("count")) is int and manifest["count"]==2000
       and manifest.get("shape")==[2000,32,32,3] and manifest.get("dtype")=="uint8" and manifest.get("layout")=="NHWC"
       and manifest.get("order")=="sample_submission.csv ID order" and manifest.get("labels_loaded") is False
       and manifest.get("test_csv_sha256")==TEST_CSV_SHA and manifest.get("sample_submission_sha256")==SAMPLE_SHA
       and manifest.get("source_provenance",{}).get("archive_sha256")==archive
       and manifest.get("source_provenance",{}).get("source_manifest_sha256")==sha(paths[3]),"Canonical official test manifest differs")
    return paths


def qualified_inputs(frozen_path=FROZEN_PATH):
    sources=source_snapshot();frozen,repeat_document,evidence=completed_repeat(frozen_path)
    primary_folder=Path(frozen["checkpoint_path"]).parent;primary_document=read(frozen["benchmark_path"])
    require(sha(frozen["checkpoint_path"])==paired.PRIMARY_CHECKPOINT_SHA and frozen.get("selected_epoch")==159
       and frozen.get("weight_variant")=="ema" and frozen.get("parameter_count")==4166298 and frozen.get("policy")==POLICY,
       "Only the actual frozen primary26 epoch159EMA may enter hidden prediction")
    primary_paths=benchmark_evidence(primary_folder,primary_document,Path(frozen["data_dir"]))
    paths=[FROZEN_PATH,*map(Path,frozen["evidence_sha256"]),*primary_paths,Path(frozen["benchmark_path"]),*(ROOT/n for n in sources)]
    binding={**evidence,**snapshots(paths)}
    return frozen,{"primary":primary_document,"repeat":repeat_document},binding,sources


def binding_for(frozen_path=FROZEN_PATH):
    frozen,documents,evidence,sources=qualified_inputs(frozen_path)
    tests=official_test_paths(Path(frozen["data_dir"]))
    return frozen,documents,{**evidence,**snapshots(tests)},sources


def tensor_audit_expected(frozen,documents):
    folders={"primary":Path(frozen["checkpoint_path"]).parent,"repeat":Path(frozen["repeat_run_path"])}
    return {"schema":AUDIT_SCHEMA,"status":"complete","performed_in_authenticated_prediction_worker":True,
       "validator_source_sha256":sha(ROOT/"research/benchmark_trivialaugment.py"),"worker_source_sha256":sha(__file__),
       "frozen_manifest_sha256":FROZEN_SHA,"confirmation_evaluated":False,"hidden_images_loaded_during_tensor_audit":False,
       "artifacts":{role:{"checkpoint_sha256":sha(folder/"best.pt"),"last_sha256":sha(folder/"last.pt"),
          "training_integrity_sha256":control.json_sha(documents[role]["research_training_integrity"]),
          "actual_selected_last_optimizer_parent_and_TA_RNG_reloaded":True} for role,folder in folders.items()}}


def prediction_inputs(run,frozen,documents,binding,sources):
    return {"schema":INPUT_SCHEMA,"run":run,"mode":"predict","frozen_manifest_path":str(FROZEN_PATH.resolve()),
       "frozen_manifest_sha256":FROZEN_SHA,"source_sha256":sources,"input_sha256":binding,"checkpoint_sha256":paired.PRIMARY_CHECKPOINT_SHA,
       "repeat_checkpoint_sha256":documents["repeat"]["checkpoint_sha256"],"policy":POLICY,"data_signature":frozen["data_signature"],
       "evaluation_context":frozen["evaluation_context"]}


def validate_worker(folder,manifest_sha):
    folder=Path(folder).resolve()
    require(folder.parent==(ROOT/"results/research").resolve() and aggregate.is_sha(manifest_sha)
       and sha(folder/"inputs.json")==manifest_sha,"Controlled new TA prediction manifest required")
    manifest,receipt=read(folder/"inputs.json"),read(folder/"supervisor.json")
    validate_owner(folder,manifest_sha,manifest,receipt,psutil.Process(),psutil.Process(os.getppid()))
    frozen,documents,binding,sources=binding_for(Path(manifest["frozen_manifest_path"]))
    fixed=prediction_inputs(folder.name,frozen,documents,binding,sources)
    require(exact(manifest,fixed) and receipt.get("source_sha256")==sources and receipt.get("input_sha256")==binding
       and receipt.get("input_manifest_sha256")==manifest_sha and receipt.get("confirmation_evaluated") is False
       and receipt.get("submitted") is False,"Actual primary/repeat/source/data/test prediction binding differs")
    return manifest,frozen,documents


def audit_tensors(frozen,documents,train_ids,val_ids):
    """Actual CPU tensor reload inside the already authenticated CUDA worker."""
    import runtime
    import torch
    from research import benchmark_trivialaugment as typed
    from research import benchmark as core
    loaded_primary=None
    for role,folder in (("primary",Path(frozen["checkpoint_path"]).parent),("repeat",Path(frozen["repeat_run_path"]))):
        checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=False)
        last=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        metrics,history,provenance,env=[read(folder/n) for n in ("metrics.json","history.json","provenance.json","environment.json")]
        integrity=typed.validate_training(checkpoint,metrics,history,provenance,env,last,(folder/"input_config.json").read_bytes());del last
        require(exact(integrity,documents[role]["research_training_integrity"])
           and checkpoint["config"]==read(folder/"config.json")==documents[role]["config"],"Actual TA selected/last/SGD/RNG integrity differs")
        core.validate_checkpoint(checkpoint,train_ids,val_ids,frozen["data_signature"],metrics,aggregate.CORE,history)
        require(checkpoint["config"]["seed"]==({"primary":2026,"repeat":2027}[role]),"Actual tensor role/seed differs")
        if role=="primary":loaded_primary=checkpoint
        else:del checkpoint
    return loaded_primary,tensor_audit_expected(frozen,documents)


def worker(folder,manifest_sha):
    manifest,frozen,documents=validate_worker(folder,manifest_sha)  # Every ownership/binding check precedes runtime/Torch.
    import runtime
    import torch
    from data import load_training_arrays,load_kaggle_test
    from model import build_model,count_parameters
    from research import benchmark as core
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),"Actual native CUDA BF16 prediction required")
    runtime.seed_all(2026);device=torch.device("cuda");env=runtime.environment()
    paired.validate_environment(env,read(Path(frozen["checkpoint_path"]).parent/"environment.json"))
    require(torch.__version__==aggregate.CONTEXT["torch_version"] and str(core.benchmark_dtype(device))=="torch.bfloat16",
            "Actual prediction context differs from matched benchmark")
    _,_,train_ids,val_ids=load_training_arrays(Path(frozen["data_dir"]))
    checkpoint,audit=audit_tensors(frozen,documents,train_ids,val_ids)
    # No hidden image may be decoded before both actual typed tensor audits succeed.
    require(all(sha(p)==v for p,v in manifest["input_sha256"].items()),"Input changed during typed tensor audit")
    write(Path(folder)/"tensor_audit.json",audit)
    model=build_model(checkpoint["model_config"])
    require(count_parameters(model)==count_parameters(model,True)==4166298,"Actual single student parameter budget differs")
    model.load_state_dict(checkpoint["model"],strict=True);model=model.to(device=device,memory_format=torch.channels_last).eval()
    ids,images=load_kaggle_test(Path(frozen["data_dir"]))
    require(len(ids)==len(images)==2000 and len(set(ids))==2000,"Exactly2000 official unique hidden IDs required")
    predictions=[]
    with torch.inference_mode():
        for start in range(0,2000,128):
            batch=torch.as_tensor(images[start:start+128],device=device)
            with torch.autocast(device.type,dtype=core.benchmark_dtype(device)):
                logits=core.infer(model,batch,POLICY)
            require(torch.isfinite(logits).all().item(),"Nonfinite hidden inference logits")
            predictions.extend(logits.argmax(1).cpu().tolist())
    require(all(torch.equal(t.cpu(),checkpoint["model"][n]) for n,t in model.state_dict().items()),"Frozen weights/BN changed during inference")
    require(source_snapshot()==manifest["source_sha256"] and all(sha(p)==v for p,v in manifest["input_sha256"].items()),"Source/input changed during prediction")
    csv_path=Path(folder)/"submission.csv"
    with csv_path.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["id","label"]);writer.writerows(zip(ids,predictions))
    digest=validate_csv(csv_path,Path(frozen["data_dir"])/"sample_submission.csv")
    document={"schema":DELIVERY_SCHEMA,"status":"complete","created_at":utc_now().isoformat(),
       "run":Path(frozen["checkpoint_path"]).parent.name,"competition":COMPETITION,
       "frozen_manifest_path":str(FROZEN_PATH.resolve()),"frozen_manifest_sha256":FROZEN_SHA,
       "checkpoint_path":frozen["checkpoint_path"],"checkpoint_sha256":paired.PRIMARY_CHECKPOINT_SHA,
       "checkpoint_epoch":159,"weight_variant":"ema","parameter_count":4166298,"policy":POLICY,"seed":2026,
       "training_examples":40000,"validation_examples":5000,"repeat_run_path":frozen["repeat_run_path"],
       "repeat_checkpoint_sha256":documents["repeat"]["checkpoint_sha256"],"repeat_receipt_schema":repeat.SCHEMA,
       "repeat_receipt_sha256":sha(Path(frozen["repeat_run_path"])/"supervisor.json"),"repeat_validator_source_sha256":PINS["research/run_trivialaugment_repeat.py"],
       "prediction_input_manifest_sha256":manifest_sha,"source_sha256":manifest["source_sha256"],"input_sha256":manifest["input_sha256"],
       "tensor_audit_sha256":sha(Path(folder)/"tensor_audit.json"),"actual_environment":env,"evaluation_context":frozen["evaluation_context"],
       "csv_path":str(csv_path.resolve()),"submission_sha256":digest,"rows":2000,"columns":["id","label"],"ids_match_official_order":True,
       "generator_source_sha256":sha(__file__),"confirmation_evaluated":False,"hidden_test_labels_loaded":False,
       "repeat_weights_used_for_hidden_prediction":False,"external_data_loaded":False,"bn_recalibrated":False}
    require(not (Path(folder)/"delivery.json").exists(),"Existing prediction manifest is immutable");write(Path(folder)/"delivery.json",document)



def validate_owner(folder, manifest_sha, manifest, receipt, process, parent):
    """Authenticate this script, never an old helper's script identity."""
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "predicting"
            and receipt.get("mode") == "predict" and receipt.get("backend") == "local"
            and receipt.get("run") == manifest.get("run") == folder.name
            and receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None
            and type(receipt.get("reserved_seconds")) is int and receipt["reserved_seconds"] == 180
            and aggregate.number(receipt.get("deadline_unix"))
            and started.timestamp() <= time.time()+.1 < receipt["deadline_unix"] <= started.timestamp()+180.1
            and receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at",0)-parent.create_time()) < .01
            and receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at",0)-process.create_time()) < .01,
            "Exact live bounded incumbent supervisor required")
    lease = read(ROOT/"results/research/local_lease.json")
    require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at",0)-parent.create_time()) < .01
            and control.lease_can_release(lease, {}), "Canonical local lease is foreign or unreconciled")
    matches = []
    for directory in folder.glob("process_tree_*"):
        try:
            spec, ready = read(directory/"spec.json"), read(directory/"ready.json")
        except (OSError, ValueError):
            continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at",0)-process.create_time()) < .01:
            matches.append((directory.resolve(), spec, ready))
    require(len(matches) == 1, "Exactly one actual incumbent OS guard required")
    directory, spec, ready = matches[0]
    nonce, guard = directory.name.removeprefix("process_tree_"), ROOT/"research/process_tree_guard.py"
    require(re.fullmatch(r"[0-9a-f]{32}",nonce) and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1"
            and spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid
            and abs(spec.get("host_created_at",0)-parent.create_time()) < .01
            and Path(spec.get("cwd","")).resolve() == ROOT.resolve()
            and Path(spec.get("script","")).resolve() == Path(__file__).resolve()
            and spec.get("script_sha256") == ready.get("script_sha256") == sha(__file__)
            and spec.get("guard_source_sha256") == sha(guard) == PINS["research/process_tree_guard.py"]
            and spec.get("arguments") == ["--_worker",str(folder),"--_manifest-sha",manifest_sha]
            and ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1")
            and aggregate.number(spec.get("deadline_unix")) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1
            and spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None),
            "New script/nonce/argv/source/deadline guard contract differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve()
            and Path(command[1]).resolve() == guard.resolve() and command[2] == "--bootstrap"
            and Path(command[3]).resolve() == directory/"spec.json", "Actual guarded bootstrap command required")
    guard_checks.verify_os_boundary(spec, ready, process)



def validate_completed_guard(folder,receipt,spec,ready,completion,proof,manifest_sha):
    """Offline own identity/duration/cleanup proof; never kill a reused PID."""
    start,end=stamp(receipt["started_at"]),stamp(receipt["finished_at"])
    duration=(end-start).total_seconds();elapsed=receipt.get("elapsed_seconds")
    nonce=spec.get("nonce","");method="windows_job_v1" if os.name=="nt" else "posix_session_v1"
    require(receipt.get("backend")=="local" and receipt.get("mode")=="predict"
       and type(receipt.get("reserved_seconds")) is int and receipt["reserved_seconds"]==180
       and all(type(receipt.get(k)) is int and receipt[k]>0 for k in ("pid","child_pid"))
       and all(aggregate.number(receipt.get(k)) and receipt[k]>0 for k in ("process_created_at","child_created_at"))
       and math.isfinite(duration) and 0<duration<=180.1 and aggregate.number(elapsed,maximum=180.1)
       and elapsed>0 and abs(duration-elapsed)<=1 and end<=datetime.now(timezone.utc)
       and aggregate.number(receipt.get("deadline_unix")) and end.timestamp()<=receipt["deadline_unix"]+.1
       and receipt["deadline_unix"]<=start.timestamp()+180.1
       and aggregate.number(spec.get("deadline_unix")) and start.timestamp()<spec["deadline_unix"]<=receipt["deadline_unix"]+.1,
       "Complete bounded positive wall/monotonic duration and180s reservation required")
    require(spec.get("schema")==ready.get("schema")==completion.get("schema")==proof.get("schema")=="owned_python_process_tree_v1"
       and re.fullmatch(r"[0-9a-f]{32}",nonce) and nonce==ready.get("nonce")==completion.get("nonce")==proof.get("nonce")
       and Path(proof.get("guard_directory","")).resolve()==Path(folder)/("process_tree_"+nonce)
       and spec.get("host_pid")==receipt.get("pid") and spec.get("host_created_at")==receipt.get("process_created_at")
       and spec.get("script")==str(Path(__file__).resolve()) and spec.get("script_sha256")==ready.get("script_sha256")==completion.get("script_sha256")==sha(__file__)
       and spec.get("guard_source_sha256")==proof.get("guard_source_sha256")==PINS["research/process_tree_guard.py"]
       and spec.get("arguments")==["--_worker",str(folder),"--_manifest-sha",manifest_sha]
       and Path(spec.get("cwd","")).resolve()==ROOT.resolve() and ready.get("method")==completion.get("method")==proof.get("method")==method
       and spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name=="nt" else None)
       and type(completion.get("exit_code")) is int and completion["exit_code"]==0 and ready.get("pid")==completion.get("pid")==receipt.get("child_pid")
       and ready.get("created_at")==completion.get("created_at")==receipt.get("child_created_at")
       and proof.get("child_pid")==receipt.get("child_pid") and proof.get("child_created_at")==receipt.get("child_created_at")
       and receipt.get("process_tree_cleanup_verified") is True and proof.get("cleanup_verified") is True
       and proof.get("timed_out") is False and proof.get("completion_authenticated") is True
       and type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"]==0
       and type(proof.get("child_returncode")) is int and proof["child_returncode"]==0,
       "Own script/argv/source/nonce/method/PID-time/cleanup proof differs")
    host=Path(proof["guard_directory"])/"host_result.json"
    require(host.is_file() and read(host)==proof,"Actual host cleanup result must equal the unique supervisor proof")
    from research.process_tree_guard import matching_process
    for pid,created in ((receipt.get("pid"),receipt.get("process_created_at")),
                        (ready.get("pid"),ready.get("created_at")),
                        (ready.get("watchdog_pid"),ready.get("watchdog_created_at"))):
        if pid is not None:
            require(type(pid) is int and pid>0 and aggregate.number(created)
                    and matching_process(pid,created) is None,"Matching completed-pair process is still alive or unidentified")


def parent(run,frozen_path=FROZEN_PATH):
    require(isinstance(run,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",run) and not run.startswith("stability"),"Unique fresh prediction run required")
    frozen,documents,binding,sources=binding_for(frozen_path)
    policy,account=read(ROOT/"research/campaign.json"),control.accounting()
    require(stamp(policy["stops_at"])==datetime(2026,11,5,5,tzinfo=timezone.utc),"Original campaign hard stop required")
    base=ROOT/"results/research";folder=base/run;lease=base/"local_lease.json"
    require(not folder.exists() and not lease.exists(),"Existing prediction artifacts or lease require reconciliation")
    def budget():
        current=account["now"]();day=current.replace(hour=0,minute=0,second=0,microsecond=0)
        return control.bounded_seconds(policy,current,account["used_seconds"](base,day,day+timedelta(days=1)))
    budget()
    from research.process_tree_guard import run_guarded,may_release_lease
    control.claim_lease(lease);began,owns,receipt=time.perf_counter(),False,{}
    try:
        seconds=budget();folder.mkdir(parents=True,exist_ok=False);owns=True;start=account["now"]()
        manifest=prediction_inputs(run,frozen,documents,binding,sources)
        write(folder/"inputs.json",manifest);digest=sha(folder/"inputs.json")
        receipt={"schema":SCHEMA,"run":run,"mode":"predict","backend":"local","status":"predicting",
           "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":start.isoformat(),
           "reserved_seconds":int(seconds),"deadline_unix":start.timestamp()+seconds,"source_sha256":sources,
           "input_sha256":binding,"input_manifest_sha256":digest,"confirmation_evaluated":False,"submitted":False}
        write(folder/"supervisor.json",receipt)
        remaining=receipt["deadline_unix"]-time.time()-2;require(remaining>0,"No owned prediction runtime remains")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",digest],
                    folder/"prediction.log",remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        document=read(folder/"delivery.json")
        require(document.get("schema")==DELIVERY_SCHEMA and document.get("status")=="complete"
           and document.get("prediction_input_manifest_sha256")==digest and document.get("checkpoint_sha256")==paired.PRIMARY_CHECKPOINT_SHA
           and document.get("submission_sha256")==validate_csv(folder/"submission.csv",Path(frozen["data_dir"])/"sample_submission.csv")
           and exact(read(folder/"tensor_audit.json"),tensor_audit_expected(frozen,documents))
           and source_snapshot()==sources and all(sha(p)==v for p,v in binding.items()) and time.time()<=receipt["deadline_unix"],
           "Complete bounded unchanged primary-only prediction artifacts required")
        receipt.update(status="complete",artifact="delivery.json",artifact_sha256=sha(folder/"delivery.json"))
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__);raise
    finally:
        if owns:
            receipt.update(finished_at=account["now"]().isoformat(),elapsed_seconds=time.perf_counter()-began)
            if not may_release_lease(receipt):receipt["status"]="cleanup_unverified"
            write(folder/"supervisor.json",receipt)
        if lease.exists():
            owner=read(lease)
            if owner.get("pid")==os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time())<.01:
                if control.lease_can_release(owner,receipt):lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"));write(lease,owner)
    return receipt


def prediction_guard_paths(folder,receipt,digest):
    proofs=receipt.get("process_tree_guard_results",[])
    require(isinstance(proofs,list) and len(proofs)==1,"Exactly one actual closed prediction guard required")
    proof=proofs[0];directory=Path(proof.get("guard_directory","")).resolve()
    require(directory.parent==folder and re.fullmatch(r"process_tree_[0-9a-f]{32}",directory.name),"Controlled prediction guard directory required")
    paths=[directory/n for n in ("spec.json","ready.json","completion.json","host_result.json")]
    require(all(p.is_file() for p in paths),"Complete actual native guard files required")
    spec,ready,completion,host=map(read,paths)
    require(exact(host,proof),"Actual prediction host cleanup differs from supervisor proof")
    validate_completed_guard(folder,receipt,spec,ready,completion,proof,digest)
    return paths


def validate_delivery(manifest_path,require_prediction=True):
    """No Torch import: verify completed owned tensor audit/lineage/CSV evidence."""
    require(require_prediction is True,"Only complete actual TA prediction delivery is supported")
    path=Path(manifest_path).resolve();folder=path.parent
    require(path.name=="delivery.json" and folder.parent==(ROOT/"results/research").resolve(),"Controlled completed TA delivery manifest required")
    document=read(path)
    require(document.get("schema")==DELIVERY_SCHEMA and document.get("status")=="complete","Independent completed TA delivery schema required")
    frozen,documents,binding,sources=binding_for(Path(document["frozen_manifest_path"]))
    manifest=read(folder/"inputs.json");receipt=read(folder/"supervisor.json");digest=sha(folder/"inputs.json")
    require(exact(manifest,prediction_inputs(folder.name,frozen,documents,binding,sources)),"Actual prediction input manifest differs")
    fixed_receipt={"schema":SCHEMA,"run":folder.name,"mode":"predict","backend":"local","status":"complete",
       "source_sha256":sources,"input_sha256":binding,"input_manifest_sha256":digest,
       "confirmation_evaluated":False,"submitted":False,"process_tree_cleanup_verified":True,
       "artifact":"delivery.json","artifact_sha256":sha(path)}
    require(all(exact(receipt.get(k),v) for k,v in fixed_receipt.items()) and receipt.get("cleanup_requires_reconciliation") is not True,
            "Actual completed source-bound prediction supervisor required")
    prediction_guard_paths(folder,receipt,digest)
    started,ended=stamp(receipt["started_at"]),stamp(receipt["finished_at"])
    require(stamp(read(Path(frozen["repeat_run_path"])/"supervisor.json")["finished_at"])<=started<=stamp(document["created_at"])<=ended
       and ended<=stamp(read(ROOT/"research/campaign.json")["stops_at"]),"Prediction must follow qualifying repeat and finish before hard stop")
    expected={"schema":DELIVERY_SCHEMA,"status":"complete","run":Path(frozen["checkpoint_path"]).parent.name,"competition":COMPETITION,
       "frozen_manifest_path":str(FROZEN_PATH.resolve()),"frozen_manifest_sha256":FROZEN_SHA,
       "checkpoint_path":frozen["checkpoint_path"],"checkpoint_sha256":paired.PRIMARY_CHECKPOINT_SHA,"checkpoint_epoch":159,"weight_variant":"ema",
       "parameter_count":4166298,"policy":POLICY,"seed":2026,"training_examples":40000,"validation_examples":5000,
       "repeat_run_path":frozen["repeat_run_path"],"repeat_checkpoint_sha256":documents["repeat"]["checkpoint_sha256"],
       "repeat_receipt_schema":repeat.SCHEMA,"repeat_receipt_sha256":sha(Path(frozen["repeat_run_path"])/"supervisor.json"),
       "repeat_validator_source_sha256":PINS["research/run_trivialaugment_repeat.py"],"prediction_input_manifest_sha256":digest,
       "source_sha256":sources,"input_sha256":binding,"tensor_audit_sha256":sha(folder/"tensor_audit.json"),"evaluation_context":frozen["evaluation_context"],
       "csv_path":str((folder/"submission.csv").resolve()),"rows":2000,"columns":["id","label"],"ids_match_official_order":True,
       "generator_source_sha256":sha(__file__),"confirmation_evaluated":False,"hidden_test_labels_loaded":False,
       "repeat_weights_used_for_hidden_prediction":False,"external_data_loaded":False,"bn_recalibrated":False}
    require(set(document)==set(expected)|{"created_at","actual_environment","submission_sha256"}
       and all(exact(document.get(k),v) for k,v in expected.items()),"Actual frozen primary/typed repeat/source/CSV metadata differs")
    paired.validate_environment(document["actual_environment"],read(Path(frozen["checkpoint_path"]).parent/"environment.json"))
    require(exact(read(folder/"tensor_audit.json"),tensor_audit_expected(frozen,documents)),"Actual guarded selected/last/SGD/RNG tensor-audit evidence differs")
    csv_digest=validate_csv(folder/"submission.csv",Path(frozen["data_dir"])/"sample_submission.csv")
    require(document.get("submission_sha256")==csv_digest and all(sha(p)==v for p,v in binding.items()),"Official CSV or immutable qualified input changed")
    return {"run":expected["run"],"competition":COMPETITION,"checkpoint_sha256":paired.PRIMARY_CHECKPOINT_SHA,
       "parameter_count":4166298,"policy":POLICY,"seed":2026,"training_examples":40000,"validation_examples":5000,
       "csv_path":expected["csv_path"],"submission_sha256":csv_digest,"manifest_sha256":sha(path),"frozen_manifest_sha256":FROZEN_SHA,
       "repeat_receipt_schema":repeat.SCHEMA,"repeat_receipt_sha256":expected["repeat_receipt_sha256"],
       "repeat_validator_source_sha256":PINS["research/run_trivialaugment_repeat.py"]}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=="--_worker":
        require(len(argv)==4 and argv[2]=="--_manifest-sha","Exact hidden worker arguments required");return worker(Path(argv[1]),argv[3])
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--predict",action="store_true");actions.add_argument("--validate",type=Path)
    parser.add_argument("--run");parser.add_argument("--frozen",type=Path,default=FROZEN_PATH)
    args=parser.parse_args(argv)
    if args.predict:
        require(args.run is not None,"Prediction requires a unique owned run");return parent(args.run,args.frozen.resolve())
    if args.validate:
        require(args.run is None,"Offline validation does not start a prediction");result=validate_delivery(args.validate)
        print(json.dumps(result,sort_keys=True));return result
    require(args.run is None,"No prediction action supplied");parser.print_help()


if __name__=="__main__":main()
