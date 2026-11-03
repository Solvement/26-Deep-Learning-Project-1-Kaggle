"""Bounded official-image WRN16-5/300 native-BF16 engineering preflight.

No formal training, accuracy evaluation, saved weights, hidden inputs or upload.
Host imports only stdlib/psutil. Own authenticated OS boundary precedes Torch.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timedelta,timezone
import hashlib
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
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research import preflight_core_control as control
from research import incumbent_compare as aggregate
SCHEMA="research_wrn16_300_preflight_supervisor_v1"
INPUT_SCHEMA="research_wrn16_300_preflight_inputs_v1"
REPORT_SCHEMA="research_wrn16_300_actual_image_preflight_v1"
NAME="wrn16_5_300"
HARD_STOP=datetime(2026,11,5,5,tzinfo=timezone.utc)
PLAN="research/wrn16-5-300-plan.md"
CONFIGS={2026: ('research/configs/wrn16_5_300.json', 'a11bf0c5443b105b2bf25bbb7c61c94c0cd0ee635eba1daaf2e6605367c75d93'), 2027: ('research/configs/wrn16_5_300_seed2027.json', '614f6c2391b962d40cad959adf55fdadbb052b4fb6fcdcd0d54cc5705f5798af')}
RECIPES={NAME:{"configs":CONFIGS,"parameters":4289754,"bn_modules":13,"epochs":300,"future_trainer":"frozen"}}
PINS={**control.PINS,
 "research/preflight_core_control.py":"85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761",
 "research/configs/wrn16_5.json":"d2745cb8d36e7105cf13b0d32f4467024d36e69d9cc316368c45065063c18528",
 "research/configs/wrn16_5_seed2027.json":"2b4d0e98b8357d930994957857522579d04c3b4af6af6c0bbbebfc2372ed49a7",
 "research/incumbent-promotion-plan.md":"10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7",
 "research/incumbent_compare.py":"277f7803d855a90a5bd92e582e2beda85cb18557402d9a4332428d2b70a49d0d",
 "research/paired_compare_v2.py":"7218d65bd0f33ae6e3563763730d6ce161a9e90131cdfb21c9feb10ad5bb5199",
 "research/paired_compare.py":"034c850037b71ef151ad1f3f168afb3c475f170ad2a436d70e406e329fba5e65",
 "research/benchmark.py":"d30acccd66a18f8de198824a57b565cb8e16cd7044fa1cec6608007c80d37f16"}
COMPARISONS={
 "parent160":("results/research/wrn16_5_s2026/benchmark.json","c5d457e6e249052851b042c5f7fa19267cb44e62613ecb8b28ff55b6238cb211","426a986e34a8179e94fe3065eaf8e0db444dab7bbc52f236f35dec75b055aacb"),
 "incumbent300":("results/research/wrn34_3_300_s2026_local_v3/benchmark.json","e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b","53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc")}
require,sha256,read,write,json_sha=control.require,control.sha256,control.read,control.write,control.json_sha
DATA_IDENTITY=control.DATA_IDENTITY
ARRAYS=control.ARRAYS
accounting,claim_lease,lease_can_release,official_inputs=(control.accounting,control.claim_lease,control.lease_can_release,control.official_inputs)
core_step=control.core_step


def source_snapshot(root=ROOT):
    actual={name:sha256(root/name) for name in PINS}
    require(actual==PINS,"Frozen WRN16-300 dependencies changed")
    for name,digest in CONFIGS.values():
        require(sha256(root/name)==digest,"Exact raw300 configs changed")
        actual[name]=digest
    for name in (PLAN,"research/preflight_wrn16_300.py","research/campaign.json"):
        actual[name]=sha256(root/name)
    return actual


def configuration(seed,root=ROOT,recipe=NAME):
    require(recipe==NAME and type(seed) is int and seed in CONFIGS,"Only exact new300 primary/repeat configs supported")
    for name,digest in CONFIGS.values():require(sha256(root/name)==digest,"Exact raw300 configs required")
    parent_path=root/"research/configs/wrn16_5.json"
    require(sha256(parent_path)==PINS["research/configs/wrn16_5.json"],"Frozen160 parent config changed")
    primary,repeat=[read(root/CONFIGS[s][0]) for s in (2026,2027)]
    require(primary=={**read(parent_path),"epochs":300} and repeat=={**primary,"seed":2027},"Only schedule length160 to300 and repeat seed may differ")
    tree=ast.parse((root/"train.py").read_text())
    defaults=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="DEFAULT_CONFIG" for t in n.targets))
    return {**defaults,**(primary if seed==2026 else repeat)}


def comparison_inputs(root=ROOT):
    paths=[];documents=[]
    for name,(relative,digest,checkpoint) in COMPARISONS.items():
        path=root/relative
        require(sha256(path)==digest,"Frozen parent/incumbent raw benchmark changed")
        document=read(path);aggregate.validate_document(document)
        require(document["checkpoint_sha256"]==checkpoint,"Frozen comparison checkpoint identity differs")
        paths.append(path);documents.append(document)
    aggregate.compare_documents(documents[0],documents[1])
    return paths,{name:values[1] for name,values in COMPARISONS.items()}


def repeat_plan(seed,frozen_path,sources):
    if seed==2026:
        require(frozen_path is None,"Primary engineering cannot borrow repeat freeze")
        return [],None
    require(seed==2027 and frozen_path is not None,"Repeat27 requires its already frozen primary plan")
    path=Path(frozen_path).resolve();frozen=read(path);decision=frozen.get("primary_decision",{})
    require(frozen.get("schema")=="research_candidate_freeze_v2" and frozen.get("status")=="frozen"
        and frozen.get("config")==configuration(2026) and frozen.get("parameter_count")==4289754
        and frozen.get("policy")=="center80_shift20" and frozen.get("confirmation_evaluated") is False
        and frozen.get("repeat_prediction_authorized") is False and frozen.get("full_refit_supported") is False
        and decision.get("seed")==2026 and decision.get("planned_repeat_seed")==2027
        and decision.get("planned_repeat_config_sha256")==CONFIGS[2027][1]
        and decision.get("planned_repeat_source_sha256")==sources,"Exact qualified primary/future27 config/source freeze required")
    now=datetime.now(timezone.utc);decided=datetime.fromisoformat(decision.get("decided_at",""));stamp=datetime.fromisoformat(frozen.get("frozen_at",""))
    require(decided.tzinfo is not None and stamp.tzinfo is not None and decided<=stamp<now,"Primary/source selection must precede repeat preflight")
    evidence=frozen.get("evidence_sha256",{})
    require(isinstance(evidence,dict) and evidence,"Frozen actual evidence closure required")
    for name,digest in evidence.items():require(re.fullmatch(r"[0-9a-f]{64}",digest or "") and sha256(Path(name))==digest,"Frozen source/primary evidence changed")
    mandatory={}
    for key in ("checkpoint_path","benchmark_path","paired_path","primary_decision_path","baseline_path"):
        require(isinstance(frozen.get(key),str) and frozen[key],"Complete actual freeze paths required")
        target=Path(frozen[key]).resolve();require(str(target) in evidence and evidence[str(target)]==sha256(target),"Every primary/pair/decision path must be in frozen evidence")
        mandatory[key]=target
    require(read(mandatory["primary_decision_path"])==decision and sha256(mandatory["checkpoint_path"])==decision.get("checkpoint_sha256")
        and sha256(mandatory["paired_path"])==decision.get("paired_sha256"),"Actual frozen primary/pair/decision identities differ")
    primary=Path(frozen["benchmark_path"]);document=read(primary)
    aggregate.validate_document(document)
    require(document["config"]==configuration(2026) and document["checkpoint_sha256"]==decision.get("checkpoint_sha256")
        and sha256(primary)==decision.get("benchmark_sha256") and document.get("promotion",{}).get("center80_shift20",{}).get("passed") is True,"Frozen primary full benchmark/gate differs")
    require(all(frozen.get(key)==document.get(other) for key,other in (("model_config","model_config"),("selected_epoch","checkpoint_epoch"),
        ("weight_variant","checkpoint_variant"),("data_signature","data_signature"),("evaluation_context","evaluation_context"))),"Frozen constructor/context/selected state differs")
    pair=read(mandatory["paired_path"]);baseline=read(mandatory["baseline_path"])
    require(sha256(mandatory["baseline_path"])=="eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
        and baseline.get("checkpoint_sha256")=="502979bc03301aa2f0273a7414f86799302b908a3abada622a88e4578c4b734f","Actual trusted original baseline required")
    expected_pair={"schema":"exploratory_paired_image_bootstrap_v2","status":"complete","candidate_policy":"center80_shift20","baseline_policy":"flip",
        "candidate_checkpoint_sha256":decision["checkpoint_sha256"],"candidate_benchmark_sha256":decision["benchmark_sha256"],
        "baseline_checkpoint_sha256":baseline["checkpoint_sha256"],"baseline_benchmark_sha256":sha256(mandatory["baseline_path"]),
        "data_signature":document["data_signature"],"evaluation_context":document["evaluation_context"],
        "source_sha256":sha256(ROOT/"research/paired_compare_v2.py"),"paired_helper_source_sha256":sha256(ROOT/"research/paired_compare.py"),
        "benchmark_source_sha256":PINS["research/benchmark.py"],"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,"automatic_promotion":False}
    require(all(type(pair.get(k)) is type(v) and pair[k]==v for k,v in expected_pair.items()),"Actual primary legacy paired-v2 binding differs")
    boot=pair.get("bootstrap",{})
    require(all(type(boot.get(k)) is type(v) and boot[k]==v for k,v in {"original_images":5000,"replicates":10000,"seed":20261031,
        "confidence_level":.95,"sampling_unit":"original_image_with_all_five_stress_views"}.items()),"Complete original-image cluster bootstrap required")
    for metric,total in (("clean",5000),("stress",25000)):
        row=boot.get(metric,{})
        counts=[row.get(k) for k in ("candidate_only_correct","baseline_only_correct","both_correct","both_wrong")]
        a=document["policies"]["center80_shift20"][metric+"_accuracy"];b=baseline["policies"]["flip"][metric+"_accuracy"]
        require(all(type(v) is int and v>=0 for v in counts) and sum(counts)==total and
            all(type(x) in (int,float) and math.isfinite(x) and abs(x-y)<=1e-12 for x,y in ((row.get("delta_accuracy"),a-b),((counts[0]-counts[1])/total,a-b),
            ((counts[0]+counts[2])/total,a),((counts[1]+counts[2])/total,b))),"Actual paired correctness marginals differ")
        ci=row.get("percentile_ci95")
        require(isinstance(ci,list) and len(ci)==2 and all(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1 for v in ci) and ci[0]<=ci[1],"Full conditional interval required")
    selected=document["policies"]["center80_shift20"];reference=baseline["policies"]["flip"]
    clean_path=selected["clean_accuracy"]-reference["clean_accuracy"]>=.001-1e-12 and selected["stress_accuracy"]-reference["stress_accuracy"]>=-.002-1e-12
    require(clean_path or boot["stress"]["percentile_ci95"][0]>0,"Original robustness-driven delivery paired condition failed")
    inc=read(ROOT/COMPARISONS["incumbent300"][0]);comparison=aggregate.compare_documents(document,inc)
    delta=comparison["same_policy_deltas"];clean,stress=delta["clean_accuracy"],delta["mean_stress_accuracy"]
    require(((clean>=.001-1e-12 and stress>=-.002-1e-12) or (stress>=.005-1e-12 and clean>=-.002-1e-12))
        and all(value["accuracy_delta"]>=-.01-1e-12 for key,value in comparison["cases"].items() if key!="clean"),"Failed primary cannot start a repeat preflight")
    return [path,*map(Path,evidence)],sha256(path)


def formal_guard_seconds(report,policy,current,used):
    """Admission arithmetic only; never launches or changes an old receipt."""
    require(type(used) in (int,float) and math.isfinite(used) and used>=0,"Actual finite day use required")
    require(policy.get("concurrent_local_gpu_jobs")==1 and policy.get("local_gpu_seconds_per_utc_day")==21600
        and policy.get("local_job_timeout_seconds")==10800 and current.tzinfo is not None,"Fixed campaign resource contract required")
    stop=datetime.fromisoformat(policy["stops_at"])
    require(stop.tzinfo is not None and stop==HARD_STOP,"Original campaign hard stop changed")
    median=report.get("median_step_seconds")
    require(type(median) in (int,float) and math.isfinite(median) and median>0,"Actual positive step median required")
    guard=1.5*median*313*300+300
    require(report.get("conservative_pipeline_seconds")==guard and guard<=min(10800,21600-used,(stop-current).total_seconds()),"Actual300 guard exceeds job/day/hard-stop budget")
    return guard


def bounded_seconds(policy,current,used):
    require(datetime.fromisoformat(policy["stops_at"])==HARD_STOP,"Original campaign hard stop changed")
    return control.bounded_seconds(policy,current,used)


def day_usage(account,base,current,exclude=None,started=None,elapsed=None):
    """Use one captured current time and prorate this live job across UTC days."""
    require(current.tzinfo is not None,"Aware current time required")
    day=current.astimezone(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    used=account["used_seconds"](base,day,day+timedelta(days=1),exclude=exclude)
    if started is not None:
        require(started.tzinfo is not None and current>started and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,"Actual positive live span/elapsed required")
        overlap=max(0.,(min(current,day+timedelta(days=1))-max(started,day)).total_seconds())
        used+=elapsed*overlap/(current-started).total_seconds()
    return used

def validate_worker(folder, manifest_sha):
    """Authenticate the real OS boundary before importing runtime/Torch/data."""
    folder = Path(folder).resolve()
    require(folder.parent == (ROOT/"results/research").resolve() and re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
            sha256(folder/"inputs.json") == manifest_sha, "Controlled immutable preflight inputs required")
    manifest, receipt = read(folder/"inputs.json"), read(folder/"supervisor.json")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "benchmarking" and
            receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
            type(receipt.get("reserved_seconds")) is int and receipt["reserved_seconds"] == 180 and
            type(receipt.get("deadline_unix")) in (int, float) and math.isfinite(receipt["deadline_unix"]) and
            started.timestamp() <= time.time()+.1 and time.time() < receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
            receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
            receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at", 0)-process.create_time()) < .01,
            "Exact live bounded preflight supervisor required")
    lease = read(ROOT/"results/research/local_lease.json")
    require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at", 0)-parent.create_time()) < .01 and
            lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False and
            lease.get("status") not in ("unknown", "failed", "cleanup_unverified"), "Canonical live local lease required")
    matches = []
    for directory in folder.glob("process_tree_*"):
        try:
            spec, ready = read(directory/"spec.json"), read(directory/"ready.json")
        except (OSError, ValueError):
            continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at", 0)-process.create_time()) < .01:
            matches.append((directory, spec, ready))
    require(len(matches) == 1, "Exactly one actual preflight OS guard required")
    directory, spec, ready = matches[0]
    nonce = directory.name.removeprefix("process_tree_")
    guard = (ROOT/"research/process_tree_guard.py").resolve()
    require(re.fullmatch(r"[0-9a-f]{32}", nonce) and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
            spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid and
            abs(spec.get("host_created_at", 0)-parent.create_time()) < .01 and Path(spec.get("cwd", "")).resolve() == ROOT.resolve() and
            Path(spec.get("script", "")).resolve() == Path(__file__).resolve() and
            spec.get("script_sha256") == ready.get("script_sha256") == sha256(__file__) and
            spec.get("guard_source_sha256") == sha256(guard) == PINS["research/process_tree_guard.py"] and
            spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha] and
            ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
            type(spec.get("deadline_unix")) in (int, float) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
            spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "Actual guard identity/source/deadline differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and Path(command[1]).resolve() == guard and
            command[2] == "--bootstrap" and Path(command[3]).resolve() == directory/"spec.json", "Actual guarded bootstrap required")
    from research.profile_augmix import verify_os_boundary
    verify_os_boundary(spec, ready, process)
    sources = source_snapshot()
    recipe = manifest.get("recipe")
    config = configuration(manifest.get("seed"), recipe=recipe)
    expected_config = RECIPES[recipe]["configs"][manifest["seed"]][1]
    paths, identity = official_inputs(Path(manifest["data_dir"]))
    comparisons, comparison_hashes = comparison_inputs()
    frozen_paths, frozen_sha = repeat_plan(manifest["seed"], manifest.get("frozen_manifest_path"), sources)
    expected_binding = {str(path.resolve()):sha256(path) for path in [*paths, *comparisons, *frozen_paths, ROOT/RECIPES[recipe]["configs"][manifest["seed"]][0], *(ROOT/name for name in sources)]}
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == receipt.get("run") == folder.name and
            recipe == NAME and receipt.get("recipe") == recipe and
            manifest.get("comparison_sha256") == comparison_hashes and
            manifest.get("frozen_manifest_sha256") == frozen_sha and
            receipt.get("mode") == "preflight" and receipt.get("backend") == "local" and receipt.get("precision") == "bf16" and
            receipt.get("confirmation_evaluated") is False and receipt.get("submitted") is False and
            isinstance(manifest.get("source_git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", manifest["source_git_commit"]) and
            manifest.get("source_sha256") == receipt.get("source_sha256") == sources and
            manifest.get("config_sha256") == receipt.get("config_sha256") == expected_config and manifest.get("effective_config") == config and
            manifest.get("effective_config_sha256") == json_sha(config) and manifest.get("data_identity") == identity and
            manifest.get("input_sha256") == receipt.get("input_sha256") == expected_binding,
            "Exact source/config/official-input preflight binding differs")
    return manifest

def worker(folder, manifest_sha):
    setup_began = time.perf_counter()
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # DLL isolation precedes Torch; never imported by the host.
    import copy
    import numpy as np
    import torch
    import train as core
    from data import load_training_arrays
    from model import build_model, count_parameters
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False), "Actual native CUDA BF16 required; emulation/FP16/CPU are forbidden")
    config = manifest["effective_config"]
    recipe = RECIPES[manifest["recipe"]]
    runtime.seed_all(config["seed"])
    images, labels, train_ids, _ = load_training_arrays(Path(manifest["data_dir"]))
    require(len(train_ids) == 40000, "Only the original 40k training pool is allowed")
    require(hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest() == DATA_IDENTITY["train_indices_sha256"], "Actual training row identities differ")
    device = torch.device("cuda")
    train_x, train_y = core.tensor_images(images[train_ids], device), torch.as_tensor(labels[train_ids], device=device)
    model = build_model({k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}).to(device=device, memory_format=torch.channels_last)
    require(count_parameters(model) == count_parameters(model, True) == recipe["parameters"], "Actual total/trainable parameter count differs")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False) if recipe["future_trainer"] == "shadow" else None
    model.train()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    require(len(bns) == recipe["bn_modules"] and all(m.num_batches_tracked.item() == 0 for m in bns), "Original scratch BN topology/state required")
    order = torch.randperm(len(train_y), device=device)
    selected = train_ids[order[:23*128].cpu().numpy()]
    original_ids = np.load(Path(manifest["data_dir"])/"official_train_ids.npy", allow_pickle=False)[selected]
    require(len(np.unique(selected)) == 2944, "Preflight must use 2944 unique official training rows")
    rows, durations = [], []
    parameters, averages = list(model.parameters()), list(ema.parameters())
    require(len(parameters) == len(averages) == 44, "Actual complete WRN16 gradient tensor count differs")
    require(all(p.dtype == torch.float32 for p in [*parameters, *averages]), "Original FP32 model/EMA master parameters required")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_began
    for batch in range(23):
        torch.cuda.synchronize()
        began = time.perf_counter()
        before, before_ema = parameters[0].detach().clone(), averages[0].detach().clone()
        counters = [m.num_batches_tracked.item() for m in bns]
        row = core_step(core, torch, model, ema, optimizer, train_x, train_y, order, batch, config, batch, recipe["future_trainer"], scaler)
        state_values = [*model.state_dict().values(), *ema.state_dict().values(),
                        *(value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))]
        require(len(state_values) == 210, "All83model+83EMA+44momentum state tensors required")
        finite = bool(torch.stack([torch.isfinite(p).all() for p in state_values]).all().item())
        changed = (parameters[0]-before).abs().max().item()
        ema_changed = (averages[0]-before_ema).abs().max().item()
        require(finite and changed > 0 and ema_changed > 0 and
                all(m.num_batches_tracked.item() == old+1 for m, old in zip(bns, counters)), "Finite real SGD/EMA updates and exactly one BN forward required")
        row.update(gradient_tensor_count=len(parameters), state_tensor_count=len(state_values), parameter_probe_max_delta=changed, ema_probe_max_delta=ema_changed, all_parameters_finite=True, bn_single_forward=True)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-began)
        rows.append(row)
    env = runtime.environment()
    report = {"schema":REPORT_SCHEMA, "status":"complete", "engineering_only":True, "synthetic_data":False,
        "accuracy_evaluated":False, "formal_training_admitted":False, "hidden_images_loaded":False, "confirmation_evaluated":False,
        "training_precision":"bf16", "native_bf16_without_emulation":True, "grad_scaler_enabled":False,
        "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
        "recipe":manifest["recipe"], "future_trainer":recipe["future_trainer"], "epochs":config["epochs"],
        "planned_formal_attempted_updates":313*config["epochs"], "disabled_scaler_scale":scaler.get_scale() if scaler is not None else None,
        "seed":config["seed"], "parameter_count":recipe["parameters"], "trainable_parameter_count":recipe["parameters"],
        "config_sha256":manifest["config_sha256"], "effective_config_sha256":manifest["effective_config_sha256"],
        "source_sha256":manifest["source_sha256"], "input_manifest_sha256":manifest_sha,
        "source_git_commit":manifest["source_git_commit"], "data_identity":manifest["data_identity"],
        "environment":env, "environment_sha256":json_sha(env), "training_pool_examples":40000, "evaluation_examples":0,
        "batch_size":128, "warmup_updates":3, "timed_updates":20, "successful_updates":23, "skipped_updates":0,
        "source_examples_seen":2944, "unique_source_examples":2944,
        "ordered_training_indices_sha256":hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "ordered_official_ids_sha256":hashlib.sha256(np.ascontiguousarray(original_ids).tobytes()).hexdigest(),
        "gradient_tensor_count":44, "state_tensor_count":210, "comparison_sha256":manifest["comparison_sha256"],
        "frozen_manifest_sha256":manifest["frozen_manifest_sha256"], "bn_modules":recipe["bn_modules"], "bn_forward_counts":[int(m.num_batches_tracked.item()) for m in bns],
        "ema_buffer_rule":"Original epoch-end-only copy; no epoch boundary occurs in this preflight.", "ema_buffer_copies":0,
        "steps":rows, "step_seconds":durations, "median_step_seconds":statistics.median(durations[3:]),
        "setup_seconds":setup_seconds, "full_step_seconds":sum(durations),
        "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(), "preflight_state_discarded":True}
    report["conservative_pipeline_seconds"] = 1.5*report["median_step_seconds"]*313*config["epochs"]+300
    report["timing_scope"] = "Actual uint8 gather/augment/MixUp/transfer-resident model/loss/backward/SGD/EMA/metrics and all finite/update/BN proofs; 3 warmup plus 20 timed full batches. Initial official input staging is charged by supervisor wall time."
    validate_report(report, manifest, manifest_sha)
    write(folder/"wrn16_300_preflight.json", report)

def validate_report(report, manifest, manifest_sha):
    require(manifest.get("recipe") == NAME, "Fixed named control required")
    recipe = RECIPES[manifest["recipe"]]
    config = configuration(manifest.get("seed"), recipe=manifest["recipe"])
    require(manifest.get("effective_config") == config and manifest.get("effective_config_sha256") == json_sha(config) and
            manifest.get("config_sha256") == recipe["configs"][manifest["seed"]][1] and manifest.get("data_identity") == DATA_IDENTITY,
            "Predeclared raw/effective configuration and official data identity required")
    expected = {"schema":REPORT_SCHEMA, "status":"complete", "engineering_only":True, "synthetic_data":False,
        "accuracy_evaluated":False, "formal_training_admitted":False, "hidden_images_loaded":False, "confirmation_evaluated":False,
        "training_precision":"bf16", "native_bf16_without_emulation":True, "grad_scaler_enabled":False,
        "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
        "recipe":manifest["recipe"], "future_trainer":recipe["future_trainer"], "epochs":recipe["epochs"],
        "planned_formal_attempted_updates":313*recipe["epochs"], "disabled_scaler_scale":1. if recipe["future_trainer"] == "shadow" else None,
        "seed":manifest["seed"], "parameter_count":recipe["parameters"], "trainable_parameter_count":recipe["parameters"],
        "training_pool_examples":40000, "evaluation_examples":0, "batch_size":128, "warmup_updates":3, "timed_updates":20,
        "successful_updates":23, "skipped_updates":0, "source_examples_seen":2944, "unique_source_examples":2944,
        "gradient_tensor_count":44, "state_tensor_count":210, "comparison_sha256":manifest["comparison_sha256"],
        "frozen_manifest_sha256":manifest["frozen_manifest_sha256"], "bn_modules":recipe["bn_modules"], "bn_forward_counts":[23]*recipe["bn_modules"], "ema_buffer_copies":0, "preflight_state_discarded":True,
        "config_sha256":manifest["config_sha256"], "effective_config_sha256":manifest["effective_config_sha256"],
        "source_sha256":manifest["source_sha256"], "input_manifest_sha256":manifest_sha,
        "source_git_commit":manifest["source_git_commit"], "data_identity":manifest["data_identity"]}
    require(all(type(report.get(k)) is type(v) and report[k] == v for k,v in expected.items()), "Complete typed actual-image BF16 evidence required")
    rows, durations = report.get("steps", []), report.get("step_seconds", [])
    require(len(rows) == len(durations) == 23 and all(type(s) in (int,float) and math.isfinite(s) and s > 0 for s in durations), "23 positive complete full-step timings required")
    for index, row in enumerate(rows, 1):
        require(type(row.get("updates")) is int and row["updates"] == index and row.get("gradient_tensor_count") == 44 and type(row.get("gradient_tensor_count")) is int and
                row.get("state_tensor_count") == 210 and type(row.get("state_tensor_count")) is int and row.get("all_parameters_finite") is True and row.get("bn_single_forward") is True and
                type(row.get("mixed")) is bool and row.get("logits_dtype") == "torch.bfloat16" and
                all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k] > 0 for k in
                    ("loss", "gradient_norm", "lr", "parameter_probe_max_delta", "ema_probe_max_delta")), "Finite positive actual update proof required")
        require(row["lr"] == manifest["effective_config"]["lr"]*index/(5*313), "Original initial-epoch warmup schedule required")
    median = statistics.median(durations[3:])
    require(all(type(report.get(key)) in (int,float) and math.isfinite(report[key]) and report[key]>0 for key in
                ("median_step_seconds","conservative_pipeline_seconds","full_step_seconds")) and
            report.get("median_step_seconds") == median and report.get("conservative_pipeline_seconds") == 1.5*median*313*recipe["epochs"]+300,
            "Fixed actual full-pipeline median/guard differs")
    require(type(report.get("setup_seconds")) in (int,float) and math.isfinite(report["setup_seconds"]) and report["setup_seconds"] > 0 and
            report.get("full_step_seconds") == sum(durations), "Complete setup/full-step wall accounting required")
    env = report.get("environment", {})
    fixed_env={"device":"cuda","gpu":"NVIDIA GeForce RTX 5060 Laptop GPU","amp_dtype":"torch.bfloat16","python_version":"3.14.3",
               "cuda_runtime":"13.0","cudnn":92000,"cudnn_benchmark":True,"deterministic_algorithms":False}
    require(all(type(env.get(k)) is type(v) and env[k]==v for k,v in fixed_env.items())
            and isinstance(env.get("python"),str) and Path(env["python"]).is_absolute() and isinstance(env.get("platform"),str) and bool(env["platform"])
            and type(env.get("gpu_memory_GiB")) in (int,float) and math.isfinite(env["gpu_memory_GiB"]) and env["gpu_memory_GiB"]>0
            and type(env.get("process_local_cudnn_path_entries_removed")) is int and env["process_local_cudnn_path_entries_removed"]>=0
            and isinstance(env.get("packages"),dict) and env["packages"].get("torch")=="2.12.0+cu130"
            and set(env["packages"])=={"torch","torchvision","numpy","Pillow","matplotlib","nbformat","reportlab"}
            and all(value is None or isinstance(value,str) and bool(value) for value in env["packages"].values()) and
            report.get("environment_sha256") == json_sha(env) and type(report.get("peak_gpu_memory_bytes")) is int and report["peak_gpu_memory_bytes"] > 0,
            "Actual native CUDA environment and peak allocation required")
    for key in ("ordered_training_indices_sha256", "ordered_official_ids_sha256"):
        require(isinstance(report.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", report[key]), "Private source identity hashes required")
    json.dumps(report, allow_nan=False)

def parent(args):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}", args.run or "") and not args.run.startswith("stability"), "Fresh controlled run ID required")
    sources, config = source_snapshot(), configuration(args.seed, recipe=args.recipe)
    configs = RECIPES[args.recipe]["configs"]
    policy = read(ROOT/"research/campaign.json")
    account = accounting()
    data_dir = args.data_dir.resolve()
    paths, identity = official_inputs(data_dir)
    comparisons, comparison_hashes = comparison_inputs()
    frozen_paths, frozen_sha = repeat_plan(args.seed, args.frozen_manifest, sources)
    binding = {str(path.resolve()):sha256(path) for path in [*paths, *comparisons, *frozen_paths, ROOT/configs[args.seed][0], *(ROOT/name for name in sources)]}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Actual source commit required")
    base, folder = ROOT/"results/research", ROOT/"results/research"/args.run
    lease = base/"local_lease.json"
    require(not folder.exists() and not lease.exists(), "Existing artifacts/lease require explicit reconciliation; no restart/resume here")
    current=account["now"]()
    bounded_seconds(policy,current,day_usage(account,base,current))
    from research.process_tree_guard import run_guarded, may_release_lease
    claim_lease(lease)
    began, owns, receipt = time.perf_counter(), False, {}
    try:
        current=account["now"]()
        seconds=bounded_seconds(policy,current,day_usage(account,base,current))
        folder.mkdir(parents=True, exist_ok=False)
        owns = True
        started = current
        manifest = {"schema":INPUT_SCHEMA, "recipe":args.recipe, "seed":args.seed, "run":args.run, "data_dir":str(data_dir),
            "source_sha256":sources, "config_sha256":configs[args.seed][1], "effective_config":config,
            "effective_config_sha256":json_sha(config), "source_git_commit":commit, "data_identity":identity, "input_sha256":binding,
            "comparison_sha256":comparison_hashes,"frozen_manifest_path":str(args.frozen_manifest.resolve()) if args.frozen_manifest else None,
            "frozen_manifest_sha256":frozen_sha}
        write(folder/"inputs.json", manifest)
        manifest_sha = sha256(folder/"inputs.json")
        receipt = {"schema":SCHEMA, "backend":"local", "run":args.run, "status":"benchmarking", "mode":"preflight",
            "trainer":"wrn16_300_engineering", "recipe":args.recipe, "precision":"bf16", "pid":os.getpid(), "process_created_at":psutil.Process().create_time(),
            "started_at":started.isoformat(), "reserved_seconds":seconds, "deadline_unix":started.timestamp()+seconds,
            "source_sha256":sources, "config_sha256":configs[args.seed][1], "input_sha256":binding, "input_manifest_sha256":manifest_sha,
            "confirmation_evaluated":False, "submitted":False}
        write(folder/"supervisor.json", receipt)
        remaining = receipt["deadline_unix"]-time.time()-2
        require(remaining > 0, "Preflight preparation exhausted its reservation")
        run_guarded([sys.executable, str(Path(__file__).resolve()), "--_worker", str(folder.resolve()), "--_manifest-sha", manifest_sha],
                    folder/"preflight.log", remaining, folder/"supervisor.json", receipt, cwd=ROOT)
        result = read(folder/"wrn16_300_preflight.json")
        validate_report(result, manifest, manifest_sha)
        require(source_snapshot() == sources and all(sha256(path) == digest for path,digest in binding.items()) and time.time() <= receipt["deadline_unix"], "Bounded unchanged complete preflight required")
        try:
            current=account["now"]()
            actual_used=day_usage(account,base,current,exclude=folder/"supervisor.json",started=started,elapsed=time.perf_counter()-began)
            formal_guard_seconds(result,policy,current,actual_used)
            receipt["formal_timing_guard_passed"]=True
        except ValueError:
            receipt["formal_timing_guard_passed"]=False
        receipt["formal_training_admitted"]=False
        receipt.update(status="complete", artifact="wrn16_300_preflight.json", artifact_sha256=sha256(folder/"wrn16_300_preflight.json"))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns:
            receipt.update(elapsed_seconds=time.perf_counter()-began, finished_at=account["now"]().isoformat())
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            write(folder/"supervisor.json", receipt)
        if lease.exists():
            owner = read(lease)
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at", 0)-psutil.Process().create_time()) < .01:
                if lease_can_release(owner, receipt):
                    lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False, supervisor=str(folder/"supervisor.json"))
                    write(lease, owner)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--seed",type=int,choices=(2026,2027))
    parser.add_argument("--frozen-manifest",type=Path)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker is not None:
        require(args.run is None and args.seed is None and args.frozen_manifest is None and args._manifest_sha is not None,"Only own authenticated worker signature allowed")
        worker(args._worker,args._manifest_sha)
    elif args.run is None and args.seed is None and args.frozen_manifest is None and args._manifest_sha is None:
        parser.print_help()
    else:
        require(args.run is not None and args.seed is not None and args._manifest_sha is None,"Explicit fresh run/seed required")
        require((args.seed==2026 and args.frozen_manifest is None) or (args.seed==2027 and args.frozen_manifest is not None),"Repeat27 requires its actual prior primary freeze")
        args.recipe=NAME
        parent(args)


if __name__=="__main__":main()

