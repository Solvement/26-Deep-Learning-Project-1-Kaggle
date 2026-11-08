"""Fresh WRN34-3/160 TrivialAugment BF16 preflight or primary under one OS guard.

No arguments show help. No resume, repeat, hidden prediction, upload or cloud.
"""
from __future__ import annotations

import argparse
import ast
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
from research import preflight_batched_core_control as b
from research import profile_augmix as p

SCHEMA = "research_trivialaugment_primary_supervisor_v1"
INPUT_SCHEMA = "research_trivialaugment_primary_inputs_v1"
RESULT_SCHEMA = "research_trivialaugment_actual_preflight_v1"
PRIMARY_SCHEMA = "research_trivialaugment_primary_result_v1"
RECIPE = "wrn34_3_trivialaugment"
PLAN = "research/trivialaugment-primary-plan.md"
PLAN_SHA = "72e2f204509b952d8c0cf4b498ddd61fa3e719980b83eac9d3959635fdec7fd7"
CONFIGS = {
    2026: ("research/configs/wrn34_3_trivialaugment.json", "8447aaf8356f692d317e5a13d37ab2faf76f5c6f20700055e263d277ad58f764"),
    2027: ("research/configs/wrn34_3_trivialaugment_seed2027.json", "9686ddea85d0f56ba1f58ff8668056c146a78b675f12f0624799f2fd0b2f4981"),
}
PINS = {
    'research/train_trivialaugment.py': '0d4e1458b8a0bc8c30805371f43dfff42a690398d07cfca405e07bfe8e5e0f0c',
    'research/benchmark_trivialaugment.py': '0aed38f1d6957b22cf4d59294ee34dc85109b8da58ca091d7dadee25d22e548d',
    'research/trivialaugment.py': '1f81246a4e6f045d5f4beb2198b6d2db9ca3a32662f02829a350062f72d20806',
    'research/trivialaugment-training-plan.md': '248f87da5c7d69781cd53aeb1024a8dc745736b7c40df464f64ce1dfe7e413ea',
    'research/trivialaugment-implementation-plan.md': '3db70106b6d3b9b4492b3333badfdf729ef0e06297ab27b5cc4317dcec6906fe',
    'research/licenses/TrivialAugment-MIT.txt': 'b97576d1345e463979111dd49bffea6cd5bbe010b7919fbfb514f3478e6c5835',
    'research/configs/wrn34_3.json': '2901f0a7d717083c8811aac9cff1910c7a7493eb3c474c260bb9b7f7eb45df02',
    'research/paired_compare_v2.py': '7218d65bd0f33ae6e3563763730d6ce161a9e90131cdfb21c9feb10ad5bb5199',
    'research/paired_compare.py': '034c850037b71ef151ad1f3f168afb3c475f170ad2a436d70e406e329fba5e65',
    'research/benchmark_shadow.py': 'aab840a944125f45ef070099e70270103c62a5acabf081ead7f26cfdb6f7d6c9',
    "research/preflight_batched_core_control.py": "595ed01921af12b9822a620bec54c2c4ac83c99822c1b0cdba603ae7e3b79096",
    "research/batched_finite_proof.py": "c7bb0c161fab8b3a0361c1179f44d59d991428dd9d7bc8485f28d7cd61ee59ff",
    "research/incumbent_compare.py": "277f7803d855a90a5bd92e582e2beda85cb18557402d9a4332428d2b70a49d0d",
    "research/incumbent-comparison-plan.md": "97269ed9f6144393084cde6709f6c8fd0c8cc316d8f3e920741462ebebde1f7d",
    "research/incumbent-promotion-plan.md": "10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7",
    "research/benchmark.py": "d30acccd66a18f8de198824a57b565cb8e16cd7044fa1cec6608007c80d37f16",
}
BASELINE = "results/research/baseline_v3_verified.json"
BASELINE_SHA = "eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
INCUMBENT = "results/research/wrn34_3_300_s2026_local_v3/benchmark.json"
INCUMBENT_SHA = "e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b"
TIMING_SCOPE = "Uninstrumented actual40k CPU uint8 gather, GPU index-to-CPU, all PIL TA/packing/H2D, original augment/Mixup/train_step/98gradient checks/SGD/EMA/metrics and complete state/gradient/norm/31BN/probe/RNG/count proofs and allocations;3warmup+20timed updates."


OPERATIONS = ("identity", "autocontrast", "equalize", "rotate", "solarize", "color", "posterize",
              "contrast", "brightness", "sharpness", "shear_x", "shear_y", "translate_x", "translate_y")
INCUMBENT_CHECKPOINT = "results/research/colab_wrn34_3_300_fp16_s2026_v3/candidate/best.pt"
INCUMBENT_CHECKPOINT_SHA = "53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"





def configuration(seed=2026):
    c.require(type(seed) is int and seed in CONFIGS, "Only the frozen primary and planned repeat configs exist")
    for name,digest in CONFIGS.values():
        c.require(c.sha256(ROOT/name) == digest, "Exact new TrivialAugment raw config required")
    primary,repeat = [c.read(ROOT/CONFIGS[s][0]) for s in (2026,2027)]
    base = c.read(ROOT/"research/configs/wrn34_3.json")
    c.require(primary == {**base,"ta_space":"fixed_standard","ta_strengths":31} and repeat == {**primary,"seed":2027},
              "Only fixed-standard TA fields and the declared repeat seed may differ")
    tree = ast.parse((ROOT/"train.py").read_text())
    defaults = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=="DEFAULT_CONFIG" for t in n.targets))
    return {**defaults, **(primary if seed==2026 else repeat)}


def source_snapshot():
    pins = {**PINS, PLAN:PLAN_SHA, **dict(CONFIGS.values())}
    c.require(all(c.sha256(ROOT/name) == digest for name,digest in pins.items()), "Frozen method/config/decision sources changed")
    configuration()
    return {**c.source_snapshot(), **pins, "research/run_trivialaugment_primary.py":c.sha256(__file__)}


def binding_for(data_dir, sources, preflight_path=None):
    paths,identity = c.official_inputs(data_dir)
    for name,digest in ((BASELINE,BASELINE_SHA),(INCUMBENT,INCUMBENT_SHA)):
        c.require(c.sha256(ROOT/name) == digest, "Pinned complete local baseline/incumbent required")
        paths.append(ROOT/name)
    checkpoint=ROOT/INCUMBENT_CHECKPOINT
    c.require(c.sha256(checkpoint)==INCUMBENT_CHECKPOINT_SHA, "Exact actual incumbent checkpoint required by typed TA benchmark")
    paths += [checkpoint, *(checkpoint.parent/name for name in ("metrics.json","history.json","provenance.json","environment.json")),
              (ROOT/INCUMBENT).with_name("benchmark.core.json"), *(ROOT/name for name in sources)]
    if preflight_path is not None:
        _,evidence=validate_preflight_bundle(Path(preflight_path),sources)
        paths += evidence
    return {str(path.resolve()):c.sha256(path) for path in paths},identity


def guard_seconds(median):
    c.require(type(median) in (int,float) and math.isfinite(median) and median > 0, "Finite positive actual median required")
    return 1.5*median*313*160+300


def admission_budget(policy,current,used,mode,preflight=None):
    c.require(mode in ("preflight","formal") and isinstance(current,datetime) and current.tzinfo is not None and
              type(used) in (int,float) and math.isfinite(used) and used >= 0, "Valid mode/time/accounting required")
    stop=datetime.fromisoformat(policy["stops_at"])
    c.require(stop.tzinfo is not None and stop==datetime.fromisoformat("2026-11-05T05:00:00+00:00") and current<stop,
              "The original hard stop applies to both preflight and formal")
    if mode=="preflight":
        c.require(preflight is None, "Preflight is a fresh measurement, not a resumed report")
        return c.bounded_seconds(policy,current,used)
    c.require(policy.get("concurrent_local_gpu_jobs")==1 and policy.get("local_job_timeout_seconds")==10800 and
              policy.get("local_gpu_seconds_per_utc_day")==21600, "Original concurrency/day/job/hard-stop limits required")
    budget=min(10800,21600-used,(stop-current).total_seconds())
    c.require(preflight is not None and budget >= guard_seconds(preflight.get("median_step_seconds")),
              "Actual TrivialAugment preflight cannot fit original 1.5 safety margin plus300 finalization")
    return budget


def scalar_screen(benchmark,comparison):
    old=benchmark["promotion"]["center80_shift20"]["passed"]
    counts={}
    for name,row in comparison["cases"].items():
        values=[]
        for key in ("candidate_accuracy","incumbent_accuracy"):
            c.require(type(row[key]) in (int,float) and math.isfinite(row[key]) and 0<=row[key]<=1,
                      "Finite real case accuracy required")
            value=row[key]*5000
            c.require(abs(value-round(value))<=1e-8, "Complete5000 case accuracy must encode an integer correct count")
            values.append(round(value))
        counts[name]=values[0]-values[1]
    c.require(set(counts)=={"clean","noise_004","boxblur3","contrast065","brightness080","resize24"}, "All six matched cases required")
    clean=counts["clean"];stress=sum(v for k,v in counts.items() if k!="clean")
    mean_gate=(clean>=5 and stress>=-50) or (stress>=125 and clean>=-10)
    regressions=[name for name,value in counts.items() if name!="clean" and value < -50]
    passed=old is True and mean_gate and not regressions
    return dict(schema="trivialaugment_prospective_scalar_screen_v1",policy="center80_shift20",
        original_baseline_weighted_gate_passed=old,incumbent_mean_gate_passed=bool(mean_gate),
        stress_case_floor=-.01,stress_case_floor_violations=regressions,scalar_screen_passed=bool(passed),
        correct_count_deltas=counts,stress_correct_count_delta=stress,
        noise_accuracy_delta=comparison["cases"]["noise_004"]["accuracy_delta"],
        next_action="root_review_before_required_pairing" if passed else "stop_no_pair_repeat_or_submit",
        promotion_authorized=False,submission_authorized=False)


def validate_worker(folder, manifest_sha):
    """Authenticate this entry's actual lease/OS boundary before Torch imports."""
    folder = Path(folder).resolve()
    c.require(folder.parent == (ROOT/"results/research").resolve() and
              re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
              c.sha256(folder/"inputs.json") == manifest_sha, "Controlled TrivialAugment manifest required")
    manifest, receipt = c.read(folder/"inputs.json"), c.read(folder/"supervisor.json")
    mode=manifest.get("mode")
    c.require(mode in ("preflight","formal") and manifest.get("role")==({"preflight":"engineering_preflight","formal":"exploratory_primary"}[mode]), "Own new primary/preflight role required")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    c.require(receipt.get("schema") == SCHEMA and receipt.get("status") == {"preflight":"benchmarking","formal":"training"}[mode] and
              receipt.get("backend") == "local" and receipt.get("mode") == mode and receipt.get("role") == manifest["role"] and
              receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
              type(receipt.get("reserved_seconds")) in (int,float) and math.isfinite(receipt["reserved_seconds"]) and 0 < receipt["reserved_seconds"] <= (180 if mode=="preflight" else 10800) and (mode!="preflight" or receipt["reserved_seconds"]==180) and
              type(receipt.get("deadline_unix")) in (int, float) and math.isfinite(receipt["deadline_unix"]) and
              started.timestamp() <= time.time()+.1 and time.time() < receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
              receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
              receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at", 0)-process.create_time()) < .01,
              "Exact live bounded TrivialAugment supervisor required")
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
    c.require(len(matches) == 1, "Exactly one actual TrivialAugment OS guard required")
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
              spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "TrivialAugment guard identity/source differs")
    command = process.cmdline()
    c.require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
              Path(command[1]).resolve() == guard.resolve() and command[2] == "--bootstrap" and
              Path(command[3]).resolve() == directory/"spec.json", "Actual TrivialAugment bootstrap required")
    p.verify_os_boundary(spec, ready, process)
    sources = source_snapshot()
    config = configuration()
    binding, identity = binding_for(Path(manifest["data_dir"]), sources, manifest.get("preflight_path"))
    c.require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("recipe") == RECIPE and
              manifest.get("seed") == 2026 and type(manifest.get("seed")) is int and
              manifest.get("run") == receipt.get("run") == folder.name and
              manifest.get("effective_config") == config and manifest.get("effective_config_sha256") == c.json_sha(config) and
              manifest.get("config_sha256") == CONFIGS[2026][1] and
              manifest.get("source_sha256") == receipt.get("source_sha256") == sources and
              manifest.get("data_identity") == identity and manifest.get("input_sha256") == receipt.get("input_sha256") == binding and
              receipt.get("confirmation_evaluated") is False and receipt.get("formal_training_admitted") is (mode=="formal"),
              "Exact fixed source/config/official input binding differs")
    c.require(isinstance(manifest.get("source_git_commit"),str) and re.fullmatch(r"[0-9a-f]{40}",manifest["source_git_commit"]), "Actual source commit required")
    c.require((mode=="formal") is (manifest.get("preflight_path") is not None), "Formal requires its own actual preflight; preflight cannot reuse state")
    if mode=="formal":
        report,_=validate_preflight_bundle(Path(manifest["preflight_path"]),sources)
        c.require(guard_seconds(report["median_step_seconds"])<=receipt["reserved_seconds"], "Actual admitted full-job guard required before Torch")
    return manifest


def validate_preflight_bundle(path,sources):
    path=Path(path).resolve();folder=path.parent
    c.require(path.name=="trivialaugment_preflight.json" and folder.parent==(ROOT/"results/research").resolve(),
              "Only an owned complete TrivialAugment preflight is admissible")
    inputs=folder/"inputs.json";receipt_path=folder/"supervisor.json"
    manifest,receipt,report=c.read(inputs),c.read(receipt_path),c.read(path)
    digest=c.sha256(inputs)
    validate_result(report,{**manifest,"manifest_sha256":digest})
    expected_binding,identity=binding_for(Path(manifest["data_dir"]),sources)
    c.require(manifest.get("schema")==INPUT_SCHEMA and manifest.get("mode")=="preflight" and
              manifest.get("role")=="engineering_preflight" and manifest.get("preflight_path") is None and
              manifest.get("source_sha256")==sources and manifest.get("effective_config")==configuration() and
              manifest.get("seed")==2026 and type(manifest.get("seed")) is int and manifest.get("run")==folder.name and
              receipt.get("schema")==SCHEMA and receipt.get("mode")=="preflight" and receipt.get("role")=="engineering_preflight" and
              receipt.get("status")=="complete" and receipt.get("process_tree_cleanup_verified") is True and
              receipt.get("cleanup_requires_reconciliation") is not True and receipt.get("run")==folder.name and
              receipt.get("seed")==2026 and type(receipt.get("seed")) is int and receipt.get("precision")=="bf16" and
              receipt.get("config_sha256")==CONFIGS[2026][1] and receipt.get("artifact")=="trivialaugment_preflight.json" and
              receipt.get("source_sha256")==sources and receipt.get("input_manifest_sha256")==digest and
              receipt.get("input_sha256")==manifest.get("input_sha256")==expected_binding and manifest.get("data_identity")==identity and
              type(receipt.get("reserved_seconds")) is int and receipt.get("reserved_seconds")==180 and
              receipt.get("artifact_sha256")==c.sha256(path) and receipt.get("formal_training_admitted") is False and
              all(c.sha256(p)==h for p,h in manifest["input_sha256"].items()) and not list(folder.glob("*.pt")),
              "Immutable complete cleaned actual preflight/source/data/state-discard proof required")
    started,finished=map(datetime.fromisoformat,(receipt["started_at"],receipt["finished_at"]))
    c.require(started.tzinfo is not None and finished.tzinfo is not None and 0 < (finished-started).total_seconds() <=180.1 and
              type(receipt.get("elapsed_seconds")) in (int,float) and math.isfinite(receipt["elapsed_seconds"]) and
              0 < receipt["elapsed_seconds"] <=180.1 and
              report["setup_seconds"]+report["full_step_seconds"] <= receipt["elapsed_seconds"]+.1,
              "Completed bounded180s preflight accounting must cover all setup/steps")
    for key,stamp in (("pid","process_created_at"),("child_pid","child_created_at")):
        try:process=psutil.Process(receipt[key])
        except psutil.NoSuchProcess:continue
        c.require(abs(process.create_time()-receipt[stamp]) >= .01, "Preflight owner/child still active")
    return report,[path,inputs,receipt_path]


def parent(run,data_dir,mode,preflight_path=None):
    c.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",run or "") and not run.startswith("stability") and
              mode in ("preflight","formal") and (mode=="formal") is (preflight_path is not None), "Fresh explicit primary/preflight mode required")
    sources,config=source_snapshot(),configuration()
    preflight_path=Path(preflight_path).resolve() if preflight_path is not None else None
    preflight=validate_preflight_bundle(preflight_path,sources)[0] if preflight_path is not None else None
    binding,identity=binding_for(data_dir,sources,preflight_path)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    c.require(re.fullmatch(r"[0-9a-f]{40}",commit), "Actual fixed source revision required")
    base=ROOT/"results/research";folder=base/run;lease=base/"local_lease.json"
    c.require(not folder.exists() and not lease.exists(), "Fresh-only artifacts and lease; unknown allocations need reconciliation")
    account,policy=c.accounting(),c.read(ROOT/"research/campaign.json")
    def budget():
        now=account["now"]();day=now.replace(hour=0,minute=0,second=0,microsecond=0)
        return admission_budget(policy,now,account["used_seconds"](base,day,day+timedelta(days=1)),mode,preflight)
    budget()
    from research.process_tree_guard import run_guarded,may_release_lease
    c.claim_lease(lease)
    began,owns,receipt=time.perf_counter(),False,{}
    try:
        seconds=budget();folder.mkdir(parents=True,exist_ok=False);owns=True;started=account["now"]()
        role="engineering_preflight" if mode=="preflight" else "exploratory_primary"
        manifest=dict(schema=INPUT_SCHEMA,run=run,recipe=RECIPE,seed=2026,mode=mode,role=role,data_dir=str(data_dir),
            effective_config=config,effective_config_sha256=c.json_sha(config),config_sha256=CONFIGS[2026][1],
            source_sha256=sources,input_sha256=binding,data_identity=identity,source_git_commit=commit,
            preflight_path=str(preflight_path) if preflight_path is not None else None)
        c.write(folder/"inputs.json",manifest);digest=c.sha256(folder/"inputs.json")
        receipt=dict(schema=SCHEMA,run=run,backend="local",mode=mode,role=role,status="benchmarking" if mode=="preflight" else "training",
            pid=os.getpid(),process_created_at=psutil.Process().create_time(),started_at=started.isoformat(),
            reserved_seconds=seconds,deadline_unix=started.timestamp()+seconds,source_sha256=sources,
            input_sha256=binding,input_manifest_sha256=digest,confirmation_evaluated=False,formal_training_admitted=(mode=="formal"),
            precision="bf16",seed=2026,config_sha256=CONFIGS[2026][1],preflight_path=manifest["preflight_path"])
        c.write(folder/"supervisor.json",receipt)
        remaining=receipt["deadline_unix"]-time.time()-2;c.require(remaining>0, "No owned runtime remains")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",digest],
            folder/"worker.log",remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        artifact="trivialaugment_preflight.json" if mode=="preflight" else "primary_result.json"
        result=c.read(folder/artifact)
        if mode=="preflight":validate_result(result,{**manifest,"manifest_sha256":digest})
        else:c.require(result.get("schema")==PRIMARY_SCHEMA and result.get("status")=="complete" and
                       result.get("input_manifest_sha256")==digest and result.get("source_sha256")==sources and
                       result.get("checkpoint_sha256")==c.sha256(folder/"best.pt") and
                       result.get("benchmark_sha256")==c.sha256(folder/"benchmark.json"), "Complete own primary result required")
        c.require(source_snapshot()==sources and all(c.sha256(path)==sha for path,sha in binding.items()) and
                  time.time()<=receipt["deadline_unix"], "Complete bounded immutable source/input evidence required")
        receipt.update(status="complete",artifact=artifact,artifact_sha256=c.sha256(folder/artifact))
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__);raise
    finally:
        if owns:
            receipt.update(finished_at=account["now"]().isoformat(),elapsed_seconds=time.perf_counter()-began)
            if not may_release_lease(receipt):receipt["status"]="cleanup_unverified"
            c.write(folder/"supervisor.json",receipt)
        if lease.exists():
            owner=c.read(lease)
            if owner.get("pid")==os.getpid() and abs(owner.get("process_created_at",0)-psutil.Process().create_time())<.01:
                if c.lease_can_release(owner,receipt):lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"))
                    c.write(lease,owner)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run");parser.add_argument("--mode",choices=("preflight","formal"))
    parser.add_argument("--preflight",type=Path);parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS);parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker is not None:
        c.require(args.run is args.mode is args.preflight is None and args._manifest_sha is not None, "Only authenticated worker arguments allowed")
        worker(args._worker,args._manifest_sha)
    elif args.run is not None:
        c.require(args.mode is not None and args._manifest_sha is None, "Explicit mode required")
        parent(args.run,args.data_dir.resolve(),args.mode,args.preflight)
    else:
        c.require(args.mode is args.preflight is args._manifest_sha is None, "No action without an explicit fresh run")
        parser.print_help()


def validate_operation_counts(counts,total):
    c.require(type(counts) is dict and set(counts)==set(OPERATIONS) and
              all(type(value) is int and value>=0 for value in counts.values()) and sum(counts.values())==total,
              "Complete14-operation nonnegative one-view TA accounting required")


def execute_step(r,torch,model,ema,optimizer,train_pixels,train_y,order,index,config,scaler,
                 parameters,averages,bns,template,proof,state,rng):
    before,before_ema=parameters[0].detach().clone(),averages[0].detach().clone()
    counters=proof.snapshot_bn_counters([m.num_batches_tracked for m in bns])
    rng_before=r.json_sha(rng.bit_generator.state)
    lr=r.schedule(index,160*313,313,config)
    for group in optimizer.param_groups:group["lr"]=lr
    ids=order[index*config["batch_size"]:(index+1)*config["batch_size"]]
    y=train_y[ids]
    # This gather, synchronous PIL/packing and H2D all execute inside the worker timer.
    images=train_pixels[ids.cpu().numpy()]
    device=y.device;dtype=torch.bfloat16 if device.type=="cuda" else torch.float32
    result=r.train_step(model,ema,optimizer,scaler,images,y,config,rng,state["updates"],device,dtype)
    c.require(result["updated"] is True, "Every unscaled actual TA update must succeed")
    state["updates"]+=1
    counts=result["ta_operation_counts"];validate_operation_counts(counts,len(y))
    for name,value in counts.items():state["ta_operation_counts"][name]+=value
    with torch.no_grad():
        predicted=result["logits"].argmax(1);lam=result["lam"];permutation=result["permutation"]
        state["correct"]+=predicted.eq(y).sum()*lam
        if permutation is not None:state["correct"]+=predicted.eq(y[permutation]).sum()*(1-lam)
        state["loss_sum"]+=result["loss"]*len(y);state["seen"]+=len(y)
    grads=[parameter.grad for parameter in parameters]
    c.require(all(g is not None for g in grads), "Every original parameter needs its actual gradient")
    norm=torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt().item()
    values=b.full_state_values(model,ema,optimizer,template,torch)
    finite=bool(proof.all_finite([*values,*grads]).item())
    changed=(parameters[0]-before).abs().max().item();ema_changed=(averages[0]-before_ema).abs().max().item()
    bn=proof.prove_bn_increment(counters,[m.num_batches_tracked for m in bns])
    loss=result["loss"].item();rng_after=r.json_sha(rng.bit_generator.state)
    c.require(not scaler.is_enabled() and scaler.get_scale()==1. and state["updates"]==index+1 and finite and
              math.isfinite(norm) and norm>0 and math.isfinite(loss) and loss>0 and changed>0 and ema_changed>0 and
              bn["all_incremented"] and result["logits"].dtype==dtype and rng_before!=rng_after,
              "Real native TA/SGD/EMA/finite/norm/BN/independent-RNG proofs required")
    return dict(loss=loss,gradient_norm=norm,lr=lr,updates=state["updates"],mixed=permutation is not None,logits_dtype=str(dtype),
        parameter_probe_max_delta=changed,ema_probe_max_delta=ema_changed,all_parameters_finite=True,bn_single_forward=True,
        bn_counters_before=bn["before"],bn_counters_after=bn["after"],state_tensor_count=len(values),ta_operation_counts=counts,
        ta_rng_before_sha256=rng_before,ta_rng_after_sha256=rng_after)


def formal_worker(folder,manifest_sha):
    manifest=validate_worker(folder,manifest_sha)
    import runtime
    import torch
    from research import train_trivialaugment as trainer
    from research import benchmark_trivialaugment as benchmark
    from research import incumbent_compare
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False), "Native CUDA BF16 required")
    preflight,_=validate_preflight_bundle(Path(manifest["preflight_path"]),manifest["source_sha256"])
    runtime.seed_all(2026)
    env=runtime.environment();ignored={"process_local_cudnn_path_entries_removed"}
    c.require({k:v for k,v in env.items() if k not in ignored}==
              {k:v for k,v in preflight["environment"].items() if k not in ignored}, "Actual native/Pillow/package context differs from admission")
    config=manifest["effective_config"]
    trainer.train(config,folder,Path(manifest["data_dir"]),torch.device("cuda"),precision="bf16",resume=False,
                  input_config_bytes=(ROOT/CONFIGS[2026][0]).read_bytes())
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/"research/benchmark_trivialaugment.py"),"--checkpoint",str(Path(folder)/"best.pt"),
            "--output",str(Path(folder)/"benchmark.json"),"--baseline",str(ROOT/BASELINE),
            "--data-dir",manifest["data_dir"],"--device","cuda"]
        benchmark.main()
    finally:sys.argv=previous
    evidence,metrics=c.read(Path(folder)/"benchmark.json"),c.read(Path(folder)/"metrics.json")
    c.require(evidence.get("schema")=="official_development_stress_v3" and evidence.get("status")=="complete" and
              evidence.get("examples")==5000 and evidence.get("training_examples")==40000 and evidence.get("config")==config and
              evidence.get("parameter_count")==4166298 and evidence.get("checkpoint_sha256")==c.sha256(Path(folder)/"best.pt") and
              metrics.get("schema")==trainer.SCHEMA and metrics.get("status")=="complete" and metrics.get("epochs_completed")==160 and
              metrics.get("successful_updates")==metrics.get("forward_passes")==50080 and metrics.get("skipped_updates")==0 and
              metrics.get("source_examples_seen")==metrics.get("model_views_seen")==6400000 and
              evidence.get("research_training_integrity",{}).get("training",{}).get("schema")==trainer.SCHEMA,
              "Complete actual typed TA160/one-view/full native benchmark required")
    validate_operation_counts(metrics.get("ta_operation_counts"),6400000)
    comparison=incumbent_compare.run(Path(folder)/"benchmark.json",Path(folder)/"incumbent_comparison.json")
    screen=scalar_screen(evidence,comparison)
    # The typed benchmark's gates also remain mandatory; an aggregate helper cannot override them.
    if evidence.get("point_and_case_gates_passed") is not True:
        screen.update(scalar_screen_passed=False,next_action="stop_no_pair_repeat_or_submit")
    result=dict(schema=PRIMARY_SCHEMA,status="complete",role="exploratory_primary",seed=2026,recipe=RECIPE,formal_training_admitted=True,
        training_precision="bf16",config=config,config_sha256=manifest["config_sha256"],source_sha256=manifest["source_sha256"],
        source_git_commit=manifest["source_git_commit"],input_manifest_sha256=manifest_sha,data_identity=manifest["data_identity"],
        preflight_path=manifest["preflight_path"],preflight_sha256=c.sha256(manifest["preflight_path"]),
        checkpoint_sha256=evidence["checkpoint_sha256"],benchmark_sha256=c.sha256(Path(folder)/"benchmark.json"),
        incumbent_comparison_sha256=c.sha256(Path(folder)/"incumbent_comparison.json"),epochs_completed=160,successful_updates=50080,
        skipped_updates=0,source_examples_seen=6400000,model_views_seen=6400000,ta_operation_counts=metrics["ta_operation_counts"],
        scalar_screen=screen,paired=False,repeat=False,hidden_images_loaded=False,confirmation_evaluated=False,submission_authorized=False)
    c.write(Path(folder)/"primary_result.json",result)


def worker(folder,manifest_sha):
    manifest=validate_worker(folder,manifest_sha)
    if manifest["mode"]=="preflight":preflight_worker(folder,manifest_sha)
    else:formal_worker(folder,manifest_sha)



def validate_result(report, manifest):
    exact = {"schema": RESULT_SCHEMA, "status":"complete", "engineering_only":True,
             "formal_training_admitted":False, "accuracy_evaluated":False,
             "confirmation_evaluated":False, "hidden_images_loaded":False, "external_data_loaded":False,
             "recipe":RECIPE, "seed":2026, "training_precision":"bf16", "native_bf16_without_emulation":True,
             "grad_scaler_enabled":False, "disabled_scaler_scale":1.,
             "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
             "parameter_count":4166298, "trainable_parameter_count":4166298, "bn_modules":31,
             "epochs":160, "planned_formal_attempted_updates":50080, "training_pool_examples":40000,
             "evaluation_examples":0, "batch_size":128, "warmup_updates":3, "timed_updates":20,
             "successful_updates":23, "skipped_updates":0, "source_examples_seen":2944,
             "unique_source_examples":2944, "bn_forward_counts":[23]*31, "ema_buffer_copies":0,
             "state_tensor_count":480, "state_float32_count":418, "state_int64_count":62,
             "preflight_state_discarded":True, "ta_rng_algorithm":"PCG64", "ta_rng_seed":2026, "model_views_seen":2944, "views_per_source":1,
             "source_sha256":manifest["source_sha256"], "config_sha256":manifest["config_sha256"],
             "effective_config_sha256":manifest["effective_config_sha256"],
             "input_manifest_sha256":manifest["manifest_sha256"], "source_git_commit":manifest["source_git_commit"],
             "data_identity":manifest["data_identity"], "timing_scope":TIMING_SCOPE}
    extra = {"steps", "step_seconds", "median_step_seconds", "full_step_seconds", "setup_seconds",
             "conservative_pipeline_seconds", "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes",
             "environment", "environment_sha256", "inventory_sha256",
             "ordered_training_indices_sha256", "ordered_official_ids_sha256", "ta_operation_counts", "ta_rng_start_sha256", "ta_rng_end_sha256"}
    c.require(set(report) == set(exact)|extra and
              all(type(report.get(k)) is type(v) and report[k] == v for k, v in exact.items()),
              "Exact engineering-only source-bound preflight report required")
    c.require(manifest["config_sha256"] == CONFIGS[2026][1] and
              manifest["data_identity"] == c.DATA_IDENTITY and
              manifest["effective_config_sha256"] == c.json_sha(configuration()),
              "Exact unchanged TrivialAugment160 configuration and official split required")
    rows, durations = report["steps"], report["step_seconds"]
    c.require(len(rows) == len(durations) == 23 and
              all(type(s) in (int,float) and math.isfinite(s) and s > 0 for s in durations),
              "23 timed complete successful updates required")
    for index, row in enumerate(rows):
        fields = {"loss", "gradient_norm", "lr", "mixed", "updates", "logits_dtype",
                  "parameter_probe_max_delta", "ema_probe_max_delta", "all_parameters_finite",
                  "bn_single_forward", "bn_counters_before", "bn_counters_after", "state_tensor_count", "ta_operation_counts", "ta_rng_before_sha256", "ta_rng_after_sha256"}
        c.require(set(row) == fields and type(row["updates"]) is int and row["updates"] == index+1 and
                  row["all_parameters_finite"] is True and row["bn_single_forward"] is True and
                  row["bn_counters_before"] == [index]*31 and row["bn_counters_after"] == [index+1]*31 and
                  all(type(v) is int for v in row["bn_counters_before"]+row["bn_counters_after"]) and
                  type(row["state_tensor_count"]) is int and row["state_tensor_count"] == 480 and
                  type(row["mixed"]) is bool and row["logits_dtype"] == "torch.bfloat16" and
                  all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] > 0 for k in
                      ("loss", "gradient_norm", "lr", "parameter_probe_max_delta", "ema_probe_max_delta")) and
                  row["lr"] == .1*(index+1)/(5*313), "Complete original gradient/update/probe/BN proofs required")
    validate_operation_counts(report["ta_operation_counts"],2944)
    totals=dict.fromkeys(OPERATIONS,0)
    previous=report["ta_rng_start_sha256"]
    for row in rows:
        validate_operation_counts(row["ta_operation_counts"],128)
        c.require(row["ta_rng_before_sha256"]==previous and row["ta_rng_after_sha256"]!=previous,
                  "Contiguous independently advancing TA RNG fingerprints required")
        for key in ("ta_rng_before_sha256","ta_rng_after_sha256"):
            c.require(isinstance(row[key],str) and re.fullmatch(r"[0-9a-f]{64}",row[key]), "Actual independent TA RNG hash required")
        previous=row["ta_rng_after_sha256"]
        for name,value in row["ta_operation_counts"].items():totals[name]+=value
    c.require(totals==report["ta_operation_counts"] and previous==report["ta_rng_end_sha256"], "All23 TA operation totals/RNG endpoint required")
    for key in ("ta_rng_start_sha256","ta_rng_end_sha256"):
        c.require(isinstance(report[key],str) and re.fullmatch(r"[0-9a-f]{64}",report[key]), "Actual initial/final independent RNG fingerprints required")
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
    c.json_sha(report)


def preflight_worker(folder, manifest_sha):
    setup_began = time.perf_counter()
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # Authentication and real OS containment precede Torch.
    import copy
    import numpy as np
    import torch
    import train as core
    from research import train_trivialaugment as trainer
    from data import load_training_arrays
    from model import build_model, count_parameters
    from research import batched_finite_proof as proof
    c.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
              "Actual native CUDA BF16 required; emulation/FP16/CPU forbidden")
    config = manifest["effective_config"]
    trainer.validate_config(config)
    runtime.seed_all(2026)
    images, labels, train_ids, _ = load_training_arrays(Path(manifest["data_dir"]))
    c.require(len(train_ids) == 40000 and hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest() ==
              c.DATA_IDENTITY["train_indices_sha256"], "Actual complete official40k pool required")
    device = torch.device("cuda")
    train_pixels=np.ascontiguousarray(images[train_ids]);train_y=torch.as_tensor(labels[train_ids],device=device)
    c.require(train_pixels.dtype==np.uint8 and train_pixels.shape==(40000,32,32,3), "Complete original raw CPU uint8 pool required")
    del images,labels
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
    template = b.inventory_template(model,ema,torch)
    floats = sum(row[0] == "torch.float32" for row in template.values())
    integers = sum(row[0] == "torch.int64" for row in template.values())
    c.require(len(template) == 480 and floats == 418 and integers == 62, "Full original state inventory required")
    order = torch.randperm(len(train_y),device=device)
    selected = train_ids[order[:23*128].cpu().numpy()]
    original_ids = np.load(Path(manifest["data_dir"])/"official_train_ids.npy",allow_pickle=False)[selected]
    c.require(len(np.unique(selected)) == 2944, "2944 unique actual sources required")
    rows, durations = [], []
    rng=trainer.ta_rng(2026);rng_start=trainer.json_sha(rng.bit_generator.state)
    state=dict(updates=0,correct=torch.zeros((),device=device),loss_sum=torch.zeros((),device=device),seen=0,
               ta_operation_counts=dict.fromkeys(OPERATIONS,0))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_began
    for index in range(23):
        torch.cuda.synchronize()
        began = time.perf_counter()
        row = execute_step(trainer,torch,model,ema,optimizer,train_pixels,train_y,order,index,
                           config,scaler,parameters,averages,bns,template,proof,state,rng)
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
        epochs=160,planned_formal_attempted_updates=50080,training_pool_examples=len(train_ids),evaluation_examples=0,
        batch_size=128,warmup_updates=3,timed_updates=20,successful_updates=rows[-1]["updates"],skipped_updates=0,
        source_examples_seen=2944,unique_source_examples=len(np.unique(selected)),bn_forward_counts=rows[-1]["bn_counters_after"],
        ema_buffer_copies=0,state_tensor_count=len(template),state_float32_count=floats,state_int64_count=integers,
        preflight_state_discarded=True,ta_rng_algorithm="PCG64",ta_rng_seed=2026,model_views_seen=state["seen"],views_per_source=1,
        ta_operation_counts=state["ta_operation_counts"],ta_rng_start_sha256=rng_start,ta_rng_end_sha256=trainer.json_sha(rng.bit_generator.state),source_sha256=manifest["source_sha256"],
        config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"],
        input_manifest_sha256=manifest_sha,source_git_commit=manifest["source_git_commit"],data_identity=manifest["data_identity"],
        timing_scope=TIMING_SCOPE,steps=rows,step_seconds=durations,median_step_seconds=statistics.median(durations[3:]),
        full_step_seconds=sum(durations),setup_seconds=setup_seconds,conservative_pipeline_seconds=guard_seconds(statistics.median(durations[3:])),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        environment=env,environment_sha256=c.json_sha(env),inventory_sha256=c.json_sha(template),
        ordered_training_indices_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        ordered_official_ids_sha256=hashlib.sha256(np.ascontiguousarray(original_ids).tobytes()).hexdigest())
    validate_result(report,{**manifest,"manifest_sha256":manifest_sha})
    c.write(Path(folder)/"trivialaugment_preflight.json",report)


if __name__=="__main__":main()
