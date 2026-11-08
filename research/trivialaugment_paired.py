"""Typed TA primary pairing and pre-repeat freeze; no predictions or upload.

The supervisor and early OS authentication explicitly bind this new script.
Only generic frozen accounting/OS-guard/statistical primitives are reused.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timedelta, timezone
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
from research import paired_incumbent as supervision
from research import preflight_core_control as control
from research import profile_augmix as guard_checks
from research import incumbent_compare as aggregate
from research import run_trivialaugment_primary as primary

SCHEMA="research_trivialaugment_pair_supervisor_v1"
INPUT_SCHEMA="research_trivialaugment_pair_inputs_v1"
RESULT_SCHEMA="research_trivialaugment_dual_image_bootstrap_v1"
FREEZE_SCHEMA="research_trivialaugment_primary_freeze_v1"
PLAN="research/trivialaugment_delivery_plan.md"
POLICY,SEED,REPLICATES="center80_shift20",20261031,10000
INCUMBENT_CHECKPOINT=supervision.INCUMBENT_CHECKPOINT
BASELINE_CHECKPOINT=ROOT/"results/regularized/best.pt"
BASELINE_CHECKPOINT_SHA="502979bc03301aa2f0273a7414f86799302b908a3abada622a88e4578c4b734f"
PRIMARY_CHECKPOINT_SHA="dcfe213533481aa6c90d2fdd9ac853dc291c124098c9fe0ab8866337ab3f0fba"
PRIMARY_BENCHMARK_SHA="47cfe0ebedbd3f01be162648e8362c71df7607dd5389929d67da14342e79b205"
PINS={**supervision.PINS,**primary.PINS,
      "research/paired_incumbent.py":"a9d9edc1eed8bc2b3fd2c057954a83c4d958d5d01076be0d3c4a67b61a82fe50",
      "research/run_trivialaugment_primary.py":"2463387debe4bbb9850e0e605c335d4df0a3b87279b8b1d08e993bc07bb10f43"}
require,sha,read,write=control.require,control.sha256,control.read,control.write


def source_snapshot():
    require(all(sha(ROOT/name)==digest for name,digest in PINS.items()),"Frozen TA/pair/guard dependencies changed")
    return {**primary.source_snapshot(),**PINS,PLAN:sha(ROOT/PLAN),
            "research/trivialaugment_paired.py":sha(__file__)}


def point_gate(a,b):
    clean=a["clean_accuracy"]-b["clean_accuracy"];stress=a["stress_accuracy"]-b["stress_accuracy"]
    return {"passed":((clean>=.001-1e-12 and stress>=-.002-1e-12) or
                      (clean>=-.002-1e-12 and stress>=.005-1e-12)),
            "clean_delta":clean,"stress_delta":stress,
            "thresholds":{"clean_improvement":.001,"stress_noninferiority":-.002,
                          "stress_improvement":.005,"clean_noninferiority":-.002}}


def validate_screening(candidate,incumbent,baseline):
    summary=aggregate.compare_documents(candidate,incumbent)
    a=candidate["policies"][POLICY]
    old=point_gate(a,baseline["policies"]["flip"])
    mean=point_gate(a,incumbent["policies"][POLICY])
    deltas={c:summary["cases"][c]["accuracy_delta"] for c in aggregate.CASES[1:]}
    prospective={"policy":POLICY,"incumbent_checkpoint_sha256":aggregate.INCUMBENT_CHECKPOINT_SHA,
       "incumbent_benchmark_sha256":aggregate.INCUMBENT_BENCHMARK_SHA,"mean_gate":mean,
       "clean_nll_delta":summary["same_policy_deltas"]["clean_nll"],"case_accuracy_deltas":deltas,
       "noise_accuracy_delta":deltas["noise_004"],"maximum_allowed_case_accuracy_drop":.01,
       "per_case_noninferiority_passed":all(d>=-.01-1e-12 for d in deltas.values()),
       "failed_cases":[c for c,d in deltas.items() if d<-.01-1e-12],
       "passed":mean["passed"] and all(d>=-.01-1e-12 for d in deltas.values()),
       "prospective_plan_sha256":primary.PINS["research/incumbent-promotion-plan.md"],
       "automatic_delivery_authorization":False}
    marker=candidate.get("research_training_integrity",{}).get("training",{})
    require(marker.get("schema")=="research_trivialaugment_training_v1" and marker.get("training_seed")==2026
            and marker.get("precision")=="bf16" and marker.get("source_sha256")==
            {name:PINS[name] for name in ("research/train_trivialaugment.py","research/trivialaugment.py",
                                         "research/train_shadow.py","research/augmentations.py")},"Actual independent TA primary marker required")
    require(candidate.get("fixed_primary_policy")==POLICY and candidate.get("promotion",{}).get(POLICY)==old
            and candidate.get("prospective_incumbent")==prospective and old["passed"] and prospective["passed"]
            and candidate.get("point_and_case_gates_passed") is True,"All predeclared weighted gates must pass before pairing")
    return summary


def binding_for(data_dir,candidate_checkpoint,candidate_benchmark,sources):
    paths,identity=control.official_inputs(data_dir)
    candidate,checksum=aggregate.read_document(candidate_benchmark)
    incumbent,inc_sha=aggregate.read_document(aggregate.INCUMBENT_PATH)
    baseline_path=ROOT/primary.BASELINE;baseline=read(baseline_path)
    require(checksum==PRIMARY_BENCHMARK_SHA and sha(candidate_checkpoint)==PRIMARY_CHECKPOINT_SHA
            and candidate["checkpoint_sha256"]==PRIMARY_CHECKPOINT_SHA
            and inc_sha==aggregate.INCUMBENT_BENCHMARK_SHA and sha(baseline_path)==primary.BASELINE_SHA
            and sha(INCUMBENT_CHECKPOINT)==aggregate.INCUMBENT_CHECKPOINT_SHA
            and sha(BASELINE_CHECKPOINT)==BASELINE_CHECKPOINT_SHA,"Exact completed primary and trusted comparators required")
    summary=validate_screening(candidate,incumbent,baseline)
    expected=[candidate_checkpoint,*(candidate_checkpoint.parent/name for name in
        ("last.pt","metrics.json","history.json","provenance.json","environment.json","config.json","input_config.json")),
        baseline_path,aggregate.INCUMBENT_PATH,INCUMBENT_CHECKPOINT,ROOT/"research/incumbent-promotion-plan.md",
        *(INCUMBENT_CHECKPOINT.parent/name for name in ("metrics.json","history.json","provenance.json","environment.json")),
        aggregate.INCUMBENT_PATH.with_name("benchmark.core.json"),
        *(ROOT/name for name in ("research/benchmark_trivialaugment.py","research/paired_compare_v2.py",
           "research/paired_compare.py","research/benchmark_shadow.py","research/benchmark.py",
           "research/incumbent_compare.py","research/configs/wrn34_3.json")),*paths,
        *(ROOT/name for name in ("research/train_trivialaugment.py","research/trivialaugment.py",
            "research/train_shadow.py","research/augmentations.py",*aggregate.CORE))]
    expected_map={str(path.resolve()):sha(path) for path in expected}
    require(candidate.get("evidence_sha256")==expected_map
            and candidate.get("trivialaugment_benchmark_source_sha256")==PINS["research/benchmark_trivialaugment.py"],
            "Full original TA benchmark evidence closure differs")
    for checkpoint,document in ((candidate_checkpoint,candidate),(INCUMBENT_CHECKPOINT,incumbent),(BASELINE_CHECKPOINT,baseline)):
        require(sha(checkpoint)==document["checkpoint_sha256"],"Actual selected checkpoint hash differs")
        for filename,key in (("metrics.json","training_metrics_sha256"),("history.json","training_history_sha256")):
            path=checkpoint.parent/filename;require(sha(path)==document[key],"Actual training metrics/history differ");paths.append(path)
        paths += [checkpoint.parent/name for name in ("config.json","provenance.json","environment.json")]
    paths += [*expected,candidate_benchmark,BASELINE_CHECKPOINT,*(ROOT/name for name in sources)]
    return {str(p.resolve()):sha(p) for p in paths},identity,summary


def validate_worker(folder,manifest_sha):
    folder=Path(folder).resolve()
    require(folder.parent==(ROOT/"results/research").resolve() and aggregate.is_sha(manifest_sha)
            and sha(folder/"inputs.json")==manifest_sha,"Controlled TA pair input manifest required")
    manifest,receipt=read(folder/"inputs.json"),read(folder/"supervisor.json")
    validate_owner(folder,manifest_sha,manifest,receipt,psutil.Process(),psutil.Process(os.getppid()))
    sources=source_snapshot()
    binding,identity,summary=binding_for(Path(manifest["data_dir"]),Path(manifest["candidate_checkpoint"]),Path(manifest["candidate_benchmark"]),sources)
    require(manifest.get("schema")==INPUT_SCHEMA and manifest.get("source_sha256")==receipt.get("source_sha256")==sources
            and manifest.get("input_sha256")==receipt.get("input_sha256")==binding and manifest.get("data_identity")==identity
            and manifest.get("comparison")==summary and receipt.get("confirmation_evaluated") is False
            and receipt.get("submitted") is False and re.fullmatch(r"[0-9a-f]{40}",manifest.get("source_git_commit","")),
            "TA actual source/input/config/context binding differs")
    return manifest


def validate_bootstrap(value,clean_delta,stress_delta,candidate=None,reference=None):
    require(value.get("original_images")==5000 and value.get("replicates")==10000 and value.get("seed")==SEED
            and value.get("sampling_unit")=="original_image_with_all_five_stress_views" and value.get("confidence_level")==.95,
            "Complete original-image10000 cluster bootstrap required")
    for name,n,delta in (("clean",5000,clean_delta),("stress",25000,stress_delta)):
        row=value.get(name,{});counts=[row.get(k) for k in ("candidate_only_correct","baseline_only_correct","both_correct","both_wrong")]
        ci=row.get("percentile_ci95",[])
        require(all(type(v) is int and 0<=v<=n for v in counts) and sum(counts)==n
                and aggregate.number(row.get("delta_accuracy"),minimum=-1,maximum=1)
                and math.isclose(row["delta_accuracy"],delta,rel_tol=0,abs_tol=1e-12)
                and math.isclose((counts[0]-counts[1])/n,delta,rel_tol=0,abs_tol=1e-12)
                and isinstance(ci,list) and len(ci)==2 and all(aggregate.number(v,minimum=-1,maximum=1) for v in ci)
                and ci[0]<=ci[1],"Paired counts/delta/finite interval differ")
        if candidate is not None:
            require(math.isclose((counts[0]+counts[2])/n,candidate[name+"_accuracy"],rel_tol=0,abs_tol=1e-12)
                and math.isclose((counts[1]+counts[2])/n,reference[name+"_accuracy"],rel_tol=0,abs_tol=1e-12),
                "Paired absolute marginal accuracies differ from full benchmark")
        wins,losses=row.get("original_image_net_wins"),row.get("original_image_net_losses")
        require(type(wins) is int and type(losses) is int and 0<=wins<=5000 and 0<=losses<=5000 and wins+losses<=5000
                and (name!="clean" or (wins==counts[0] and losses==counts[1])),"Original-image discordance totals differ")


def validate_result(result,manifest,manifest_sha):
    fixed={"schema":RESULT_SCHEMA,"status":"complete","candidate_policy":POLICY,"incumbent_policy":POLICY,
       "baseline_policy":"flip","candidate_checkpoint_sha256":PRIMARY_CHECKPOINT_SHA,
       "candidate_benchmark_sha256":PRIMARY_BENCHMARK_SHA,"incumbent_benchmark_sha256":aggregate.INCUMBENT_BENCHMARK_SHA,
       "baseline_benchmark_sha256":primary.BASELINE_SHA,
       "incumbent_checkpoint_sha256":aggregate.INCUMBENT_CHECKPOINT_SHA,"baseline_checkpoint_sha256":BASELINE_CHECKPOINT_SHA,
       "data_signature":aggregate.DATA,"evaluation_context":aggregate.CONTEXT,"shape":[5000,6],
       "source_sha256":manifest["source_sha256"],"input_sha256":manifest["input_sha256"],"input_manifest_sha256":manifest_sha,
       "comparison":manifest["comparison"],"correctness_dtype":"uint8_binary","original_ids_aligned":True,
       "raw_flags_persisted":False,"automatic_promotion":False,"delivery_authorized":False,
       "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,"actual_TA_full_state_audited":True}
    require(all(type(result.get(k)) is type(v) and result[k]==v for k,v in fixed.items()),"Complete independent TA dual-pair evidence required")
    for name in ("original_id_order_sha256","development_index_order_sha256","candidate_correctness_sha256",
                 "incumbent_correctness_sha256","baseline_correctness_sha256"):
        require(aggregate.is_sha(result.get(name)),"Actual ordered original-ID/flag fingerprints required")
    delta=manifest["comparison"]["same_policy_deltas"]
    a=read(manifest["candidate_benchmark"])["policies"][POLICY];b=read(ROOT/primary.BASELINE)["policies"]["flip"]
    inc=read(aggregate.INCUMBENT_PATH)["policies"][POLICY]
    validate_bootstrap(result.get("incumbent_bootstrap",{}),delta["clean_accuracy"],delta["mean_stress_accuracy"],a,inc)
    validate_bootstrap(result.get("legacy_bootstrap",{}),a["clean_accuracy"]-b["clean_accuracy"],a["stress_accuracy"]-b["stress_accuracy"],a,b)


def worker(folder,manifest_sha):
    manifest=validate_worker(folder,manifest_sha)  # Own OS identity and every input before Torch.
    import numpy as np
    import torch
    from data import load_training_arrays
    from model import build_model,count_parameters
    from runtime import seed_all
    from train import code_provenance,tensor_images
    from research import paired_compare_v2 as paired
    from research import benchmark_trivialaugment as typed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.__version__==aggregate.CONTEXT["torch_version"] and code_provenance()["source_sha256"]==aggregate.CORE,
            "Actual pinned native BF16 CUDA context required")
    seed_all(2026);device=torch.device("cuda")
    require(str(paired.suite.benchmark_dtype(device))=="torch.bfloat16","Actual evaluation dtype differs")
    images,labels,train_ids,val_ids=load_training_arrays(Path(manifest["data_dir"]))
    official=np.load(Path(manifest["data_dir"])/"official_train_ids.npy",allow_pickle=False)
    require(val_ids.shape==(5000,) and len(set(val_ids.tolist()))==5000 and len(train_ids)==40000
            and not set(train_ids.tolist())&set(val_ids.tolist()),"Independent unique official40k/5000 required")
    ids=[str(v) for v in official[val_ids]];require(len(set(ids))==5000,"Original official IDs must be unique")
    x,y=tensor_images(images[val_ids],device),torch.as_tensor(labels[val_ids],device=device)
    vectors=[]
    for path,evidence,policy,is_ta in ((Path(manifest["candidate_checkpoint"]),Path(manifest["candidate_benchmark"]),POLICY,True),
          (INCUMBENT_CHECKPOINT,aggregate.INCUMBENT_PATH,POLICY,False),(BASELINE_CHECKPOINT,ROOT/primary.BASELINE,"flip",False)):
        if is_ta:
            checkpoint=torch.load(path,map_location="cpu",weights_only=False);last=torch.load(path.parent/"last.pt",map_location="cpu",weights_only=False)
            metrics,history,provenance,environment=[read(path.parent/name) for name in ("metrics.json","history.json","provenance.json","environment.json")]
            integrity=typed.validate_training(checkpoint,metrics,history,provenance,environment,last,(path.parent/"input_config.json").read_bytes());del last
            document=read(evidence)
            require(integrity==document["research_training_integrity"] and checkpoint["config"]==read(path.parent/"config.json"),"Actual selected/last TA tensor/RNG/SGD integrity differs")
            paired.validate_document(document,checkpoint,{"checkpoint_sha256":sha(path),"training_metrics_sha256":sha(path.parent/"metrics.json"),
                "training_history_sha256":sha(path.parent/"history.json")},aggregate.DATA,aggregate.CONTEXT,aggregate.CORE,PINS["research/benchmark.py"])
            paired.suite.validate_checkpoint(checkpoint,train_ids,val_ids,aggregate.DATA,metrics,aggregate.CORE,history)
        else:
            checkpoint,document=paired.load_experiment(path,evidence,aggregate.DATA,aggregate.CONTEXT,aggregate.CORE,train_ids,val_ids)
        model=build_model(checkpoint["model_config"])
        require(count_parameters(model)==count_parameters(model,True)==checkpoint["parameter_count"]<=5000000,"Actual single-model parameter budget differs")
        model.load_state_dict(checkpoint["model"],strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),"Selected tensor state is nonfinite")
        model=model.to(device=device,memory_format=torch.channels_last).eval()
        vector=paired.collect_correctness(model,x,y,policy,128,device,document["policies"][policy])
        require(vector.shape==(5000,6) and np.isin(vector,[0,1]).all(),"Full5000x6 flags required")
        vectors.append(vector);del model,checkpoint
    result={"schema":RESULT_SCHEMA,"status":"complete","candidate_policy":POLICY,"incumbent_policy":POLICY,"baseline_policy":"flip",
       "candidate_checkpoint_sha256":PRIMARY_CHECKPOINT_SHA,"incumbent_checkpoint_sha256":aggregate.INCUMBENT_CHECKPOINT_SHA,
       "candidate_benchmark_sha256":PRIMARY_BENCHMARK_SHA,"incumbent_benchmark_sha256":aggregate.INCUMBENT_BENCHMARK_SHA,
       "baseline_benchmark_sha256":primary.BASELINE_SHA,
       "baseline_checkpoint_sha256":BASELINE_CHECKPOINT_SHA,"data_signature":aggregate.DATA,"evaluation_context":aggregate.CONTEXT,
       "source_sha256":manifest["source_sha256"],"input_sha256":manifest["input_sha256"],"input_manifest_sha256":manifest_sha,
       "comparison":manifest["comparison"],"shape":[5000,6],"correctness_dtype":"uint8_binary","original_ids_aligned":True,
       "original_id_order_sha256":control.json_sha(ids),"development_index_order_sha256":hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest(),
       **{name+"_correctness_sha256":hashlib.sha256(v.astype(np.uint8).tobytes()).hexdigest() for name,v in zip(("candidate","incumbent","baseline"),vectors)},
       "incumbent_bootstrap":paired.paired_bootstrap(vectors[0],vectors[1],seed=SEED,replicates=REPLICATES),
       "legacy_bootstrap":paired.paired_bootstrap(vectors[0],vectors[2],seed=SEED,replicates=REPLICATES),
       "raw_flags_persisted":False,"automatic_promotion":False,"delivery_authorized":False,"actual_TA_full_state_audited":True,
       "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False}
    require(source_snapshot()==manifest["source_sha256"] and all(sha(p)==v for p,v in manifest["input_sha256"].items()),"Evidence changed during paired inference/bootstrap")
    validate_result(result,manifest,manifest_sha);output=Path(folder)/"paired_incumbent.json"
    require(not output.exists(),"Existing TA pairing cannot be overwritten");write(output,result)


def validate_owner(folder, manifest_sha, manifest, receipt, process, parent):
    """Authenticate this script, never an old helper's script identity."""
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "benchmarking"
            and receipt.get("mode") == "paired-incumbent" and receipt.get("backend") == "local"
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


def parent(run, candidate_checkpoint, candidate_benchmark, data_dir):
    require(isinstance(run,str) and re.fullmatch(r"[A-Za-z0-9_]+",run) and not run.startswith("stability"), "Unique diagnostic run ID required")
    sources = source_snapshot()
    binding, identity, summary = binding_for(data_dir,candidate_checkpoint,candidate_benchmark,sources)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}",commit), "Actual source revision required")
    policy, account = read(ROOT/"research/campaign.json"), control.accounting()
    require(datetime.fromisoformat(policy["stops_at"]).astimezone(timezone.utc) == datetime(2026,11,5,5,tzinfo=timezone.utc), "Fixed campaign hard stop differs")
    base = ROOT/"results/research"; folder = base/run; lease = base/"local_lease.json"
    require(not folder.exists() and not lease.exists(), "Existing artifacts/lease need explicit reconciliation")
    def budget():
        current = account["now"](); day = current.replace(hour=0,minute=0,second=0,microsecond=0)
        return control.bounded_seconds(policy,current,account["used_seconds"](base,day,day+timedelta(days=1)))
    budget()
    from research.process_tree_guard import run_guarded, may_release_lease
    control.claim_lease(lease)
    began, owns, receipt = time.perf_counter(), False, {}
    try:
        seconds = budget(); folder.mkdir(parents=True,exist_ok=False); owns = True; started = account["now"]()
        manifest = {"schema":INPUT_SCHEMA,"run":run,"candidate_checkpoint":str(candidate_checkpoint),
                    "candidate_benchmark":str(candidate_benchmark),"data_dir":str(data_dir),"source_git_commit":commit,
                    "source_sha256":sources,"input_sha256":binding,"data_identity":identity,"comparison":summary}
        write(folder/"inputs.json",manifest); manifest_sha = sha(folder/"inputs.json")
        receipt = {"schema":SCHEMA,"run":run,"mode":"paired-incumbent","backend":"local","status":"benchmarking",
                   "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":started.isoformat(),
                   "reserved_seconds":int(seconds),"deadline_unix":started.timestamp()+seconds,
                   "source_sha256":sources,"input_sha256":binding,"input_manifest_sha256":manifest_sha,
                   "confirmation_evaluated":False,"submitted":False}
        write(folder/"supervisor.json",receipt)
        remaining = receipt["deadline_unix"]-time.time()-2
        require(remaining > 0, "Diagnostic setup exhausted reservation")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",manifest_sha],
                    folder/"paired_incumbent.log",remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        result = read(folder/"paired_incumbent.json"); validate_result(result,manifest,manifest_sha)
        require(source_snapshot() == sources and all(sha(path) == checksum for path,checksum in binding.items())
                and time.time() <= receipt["deadline_unix"], "Complete bounded unchanged paired evidence required")
        receipt.update(status="complete",artifact="paired_incumbent.json",artifact_sha256=sha(folder/"paired_incumbent.json"))
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__); raise
    finally:
        if owns:
            receipt.update(finished_at=account["now"]().isoformat(),elapsed_seconds=time.perf_counter()-began)
            if not may_release_lease(receipt): receipt["status"] = "cleanup_unverified"
            write(folder/"supervisor.json",receipt)
        if lease.exists():
            owner = read(lease)
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time()) < .01:
                if control.lease_can_release(owner,receipt): lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"))
                    write(lease,owner)
    return receipt


def stamp(value):
    parsed=datetime.fromisoformat(value);require(parsed.tzinfo is not None,"Timezone-aware freeze timestamps required");return parsed


def validate_completed_guard(folder,receipt,spec,ready,completion,proof,manifest_sha):
    """Offline own identity/duration/cleanup proof; never kill a reused PID."""
    start,end=stamp(receipt["started_at"]),stamp(receipt["finished_at"])
    duration=(end-start).total_seconds();elapsed=receipt.get("elapsed_seconds")
    nonce=spec.get("nonce","");method="windows_job_v1" if os.name=="nt" else "posix_session_v1"
    require(receipt.get("backend")=="local" and receipt.get("mode")=="paired-incumbent"
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
       and receipt.get("process_tree_cleanup_verified") is True and proof.get("cleanup_verified") is True
       and proof.get("timed_out") is False and proof.get("completion_authenticated") is True
       and type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"]==0
       and type(proof.get("child_returncode")) is int and proof["child_returncode"]==0,
       "Own script/argv/source/nonce/method/PID-time/cleanup proof differs")
    from research.process_tree_guard import matching_process
    for pid,created in ((receipt.get("pid"),receipt.get("process_created_at")),
                        (ready.get("pid"),ready.get("created_at")),
                        (ready.get("watchdog_pid"),ready.get("watchdog_created_at"))):
        if pid is not None:
            require(type(pid) is int and pid>0 and aggregate.number(created)
                    and matching_process(pid,created) is None,"Matching completed-pair process is still alive or unidentified")


def completed_pair(folder):
    folder=Path(folder).resolve();manifest=read(folder/"inputs.json");receipt=read(folder/"supervisor.json")
    result=read(folder/"paired_incumbent.json");manifest_sha=sha(folder/"inputs.json")
    sources=source_snapshot();binding,identity,summary=binding_for(Path(manifest["data_dir"]),Path(manifest["candidate_checkpoint"]),Path(manifest["candidate_benchmark"]),sources)
    require(folder.parent==(ROOT/"results/research").resolve() and receipt.get("schema")==SCHEMA and receipt.get("status")=="complete"
       and receipt.get("mode")=="paired-incumbent" and receipt.get("backend")=="local"
       and type(receipt.get("reserved_seconds")) is int and receipt["reserved_seconds"]==180
       and receipt.get("confirmation_evaluated") is False and receipt.get("submitted") is False
       and receipt.get("run")==manifest.get("run")==folder.name and receipt.get("input_manifest_sha256")==manifest_sha
       and manifest.get("schema")==INPUT_SCHEMA and manifest.get("source_sha256")==receipt.get("source_sha256")==sources
       and manifest.get("input_sha256")==receipt.get("input_sha256")==binding and manifest.get("comparison")==summary
       and manifest.get("data_identity")==identity and receipt.get("artifact")=="paired_incumbent.json"
       and receipt.get("artifact_sha256")==sha(folder/"paired_incumbent.json") and receipt.get("process_tree_cleanup_verified") is True,
       "Actual completed, source-bound cleaned TA pair receipt required")
    proofs=receipt.get("process_tree_guard_results",[])
    require(len(proofs)==1 and proofs[0].get("cleanup_verified") is True and proofs[0].get("timed_out") is False
       and proofs[0].get("completion_authenticated") is True and proofs[0].get("completion_exit_code")==0
       and proofs[0].get("child_returncode")==0 and proofs[0].get("guard_source_sha256")==PINS["research/process_tree_guard.py"]
       and aggregate.number(receipt.get("elapsed_seconds"),maximum=180)
       and stamp(receipt["started_at"])<=stamp(receipt["finished_at"])<=datetime.now(timezone.utc)
       and aggregate.number(receipt.get("deadline_unix"))
       and stamp(receipt["finished_at"]).timestamp()<=receipt["deadline_unix"]+.1
       and receipt["deadline_unix"]<=stamp(receipt["started_at"]).timestamp()+180.1,
       "Actual authenticated bounded OS cleanup proof required")
    validate_result(result,manifest,manifest_sha)
    paths=[folder/name for name in ("inputs.json","supervisor.json","paired_incumbent.json")]
    directory=Path(proofs[0]["guard_directory"]).resolve()
    require(directory.parent==folder and directory.name.startswith("process_tree_"),"Actual TA guard evidence directory required")
    for name in ("spec.json","ready.json","completion.json"):
        require((directory/name).is_file(),"Actual OS authentication evidence missing");paths.append(directory/name)
    spec,ready,completion=[read(directory/name) for name in ("spec.json","ready.json","completion.json")]
    validate_completed_guard(folder,receipt,spec,ready,completion,proofs[0],manifest_sha)
    return manifest,result,{**binding,**{str(p.resolve()):sha(p) for p in paths}}


def repeat_contract(decision,config):
    path=ROOT/primary.CONFIGS[2027][0]
    require(decision.get("planned_repeat_seed")==2027 and decision.get("planned_repeat_config_path")==str(path.resolve())
       and decision.get("planned_repeat_config_sha256")==sha(path)==primary.CONFIGS[2027][1]
       and decision.get("planned_repeat_effective_config")==primary.configuration(2027)
       and decision.get("planned_repeat_effective_config_sha256")==control.json_sha(primary.configuration(2027))
       and primary.configuration(2027)=={**config,"seed":2027},"Exact predeclared scratch27 raw/effective configuration required")
    mapping=decision.get("planned_repeat_source_sha256",{})
    required={"research/run_trivialaugment_repeat.py","research/trivialaugment-repeat-plan.md",*source_snapshot()}
    require(isinstance(mapping,dict) and required<=set(mapping) and all(isinstance(n,str) and not Path(n).is_absolute()
       and ".." not in Path(n).parts and aggregate.is_sha(v) and sha(ROOT/n)==v for n,v in mapping.items())
       and all(mapping.get(n)==v for n,v in source_snapshot().items()),"Explicit real final repeat/learning/guard source closure required")
    run=Path(decision.get("repeat_run_path","")).resolve()
    require(run.parent==(ROOT/"results/research").resolve() and re.fullmatch(r"[A-Za-z0-9_]+",run.name)
       and not run.name.startswith("stability"),"Unique controlled future repeat folder required")
    return run,mapping


def freeze(paired_dir,decision_path,output):
    require(not Path(output).exists(),"Existing primary freeze cannot be overwritten")
    manifest,result,evidence=completed_pair(paired_dir);decision=read(decision_path);now=datetime.now(timezone.utc)
    benchmark=read(manifest["candidate_benchmark"]);config=benchmark["config"]
    run,sources=repeat_contract(decision,config)
    require(not run.exists() and stamp(decision.get("decided_at",""))<=now
       and now<datetime(2026,11,5,5,tzinfo=timezone.utc),"Freeze must precede every fresh27 attempt and hard stop")
    for key,value in (("checkpoint_sha256",PRIMARY_CHECKPOINT_SHA),("benchmark_sha256",PRIMARY_BENCHMARK_SHA),
          ("paired_sha256",sha(Path(paired_dir)/"paired_incumbent.json")),("policy",POLICY)):
        require(decision.get(key)==value,"Explicit primary decision identity differs")
    # Preserve legacy delivery's robustness-driven confidence requirement only.
    old=benchmark["promotion"][POLICY]
    if old["clean_delta"]<.001-1e-12:
        require(result["legacy_bootstrap"]["stress"]["percentile_ci95"][0]>0,"Legacy robustness-only paired confidence requirement failed")
    extra=[Path(decision_path),ROOT/primary.CONFIGS[2027][0],*(ROOT/name for name in sources)]
    value={"schema":FREEZE_SCHEMA,"status":"frozen","frozen_at":now.isoformat(),"primary_decision":decision,
       "primary_decision_path":str(Path(decision_path).resolve()),"primary_decision_sha256":sha(decision_path),
       "checkpoint_path":manifest["candidate_checkpoint"],"checkpoint_sha256":PRIMARY_CHECKPOINT_SHA,
       "benchmark_path":manifest["candidate_benchmark"],"benchmark_sha256":PRIMARY_BENCHMARK_SHA,
       "paired_path":str((Path(paired_dir)/"paired_incumbent.json").resolve()),"paired_sha256":sha(Path(paired_dir)/"paired_incumbent.json"),
       "data_dir":manifest["data_dir"],"policy":POLICY,"config":config,"effective_config_sha256":control.json_sha(config),
       "model_config":benchmark["model_config"],"parameter_count":benchmark["parameter_count"],
       "selected_epoch":benchmark["checkpoint_epoch"],"weight_variant":benchmark["checkpoint_variant"],
       "data_signature":aggregate.DATA,"evaluation_context":aggregate.CONTEXT,
       **{k:decision[k] for k in ("planned_repeat_seed","planned_repeat_config_path","planned_repeat_config_sha256",
          "planned_repeat_effective_config","planned_repeat_effective_config_sha256","planned_repeat_source_sha256","repeat_run_path")},
       "evidence_sha256":{**evidence,**{str(p.resolve()):sha(p) for p in extra}},
       "confirmation_evaluated":False,"repeat_prediction_authorized":False,"delivery_authorized":False,"full_refit_supported":False}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2,allow_nan=False)
    return value


def validate_frozen(path):
    value=read(path)
    require(value.get("schema")==FREEZE_SCHEMA and value.get("status")=="frozen" and value.get("policy")==POLICY
       and value.get("checkpoint_sha256")==PRIMARY_CHECKPOINT_SHA and value.get("benchmark_sha256")==PRIMARY_BENCHMARK_SHA
       and value.get("confirmation_evaluated") is False and value.get("repeat_prediction_authorized") is False
       and value.get("delivery_authorized") is False and value.get("full_refit_supported") is False,
       "Independent primary-only TA freeze required")
    manifest,result,evidence=completed_pair(Path(value["paired_path"]).parent)
    decision=value["primary_decision"];repeat_contract(decision,value["config"])
    benchmark=read(value["benchmark_path"])
    extra=[Path(value["primary_decision_path"]),ROOT/primary.CONFIGS[2027][0],*(ROOT/name for name in value["planned_repeat_source_sha256"])]
    expected_evidence={**evidence,**{str(p.resolve()):sha(p) for p in extra}}
    require(stamp(decision["decided_at"])<=stamp(value["frozen_at"])<=datetime.now(timezone.utc)
       and value["paired_sha256"]==sha(value["paired_path"]) and value["checkpoint_path"]==manifest["candidate_checkpoint"]
       and value["benchmark_path"]==manifest["candidate_benchmark"] and value["config"]==read(value["benchmark_path"])["config"]
       and value.get("data_dir")==manifest["data_dir"] and value.get("model_config")==benchmark["model_config"]
       and value.get("parameter_count")==benchmark["parameter_count"] and value.get("selected_epoch")==benchmark["checkpoint_epoch"]
       and value.get("weight_variant")==benchmark["checkpoint_variant"] and value.get("data_signature")==aggregate.DATA
       and value.get("evaluation_context")==aggregate.CONTEXT
       and read(value["primary_decision_path"])==decision and sha(value["primary_decision_path"])==value["primary_decision_sha256"]
       and decision.get("checkpoint_sha256")==PRIMARY_CHECKPOINT_SHA and decision.get("benchmark_sha256")==PRIMARY_BENCHMARK_SHA
       and decision.get("paired_sha256")==value["paired_sha256"] and decision.get("policy")==POLICY
       and value["effective_config_sha256"]==control.json_sha(value["config"])
       and all(value.get(k)==decision.get(k) for k in ("planned_repeat_seed","planned_repeat_config_path","planned_repeat_config_sha256",
          "planned_repeat_effective_config","planned_repeat_effective_config_sha256","planned_repeat_source_sha256","repeat_run_path"))
       and value.get("evidence_sha256")==expected_evidence
       and all(sha(p)==v for p,v in value["evidence_sha256"].items()),"Frozen TA primary/decision/source evidence changed")
    return value


def main(argv=None):
    # Exact hidden worker argv is authenticated before loading any numerical runtime.
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=="--_worker":
        require(len(argv)==4 and argv[2]=="--_manifest-sha","Exact worker argv required");return worker(Path(argv[1]),argv[3])
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest="mode")
    p=subs.add_parser("pair");p.add_argument("--run",required=True)
    for name in ("candidate-checkpoint","candidate-benchmark"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    f=subs.add_parser("freeze")
    for name in ("paired-dir","decision","output"):f.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.mode=="pair":return parent(args.run,args.candidate_checkpoint.resolve(),args.candidate_benchmark.resolve(),args.data_dir.resolve())
    if args.mode=="freeze":return freeze(args.paired_dir.resolve(),args.decision.resolve(),args.output.resolve())
    parser.print_help()


if __name__=="__main__":main()
