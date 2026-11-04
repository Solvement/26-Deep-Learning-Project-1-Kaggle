"""Guarded local native-training-BN AugMix preflight, train or complete v3.

No arguments show help. Formal primary2026 requires this protocol's actual-image
preflight and measured 1.5x wall-time guard. No predictions or uploads exist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.profile_augmix import verify_os_boundary
from research.augmix_nativebn import (CONFIG_SHA, DIAGNOSTIC_SHA, bound_sources, require, sha256)

SCHEMA = "research_augmix_nativebn_local_supervisor_v1"
CONFIG = "research/configs/wrn22_augmix_jsd.json"


def admission_budget(policy, current, used, preflight=None):
    require(isinstance(current,datetime) and current.tzinfo is not None and
        type(used) in (int,float) and math.isfinite(used) and used >= 0,"Aware time and finite local usage required")
    require(policy.get("concurrent_local_gpu_jobs") == 1 and policy.get("local_gpu_seconds_per_utc_day") == 21600 and
        policy.get("local_job_timeout_seconds") == 10800,"Fixed local concurrency/day/job limits changed")
    deadline = datetime.fromisoformat(policy["stops_at"])
    require(deadline.tzinfo is not None,"Aware hard stop required")
    budget = min(10800,21600-used,(deadline-current).total_seconds())
    require(budget >= 1,"No native-BN local budget remains")
    if preflight is not None:
        median = preflight.get("median_step_seconds")
        require(type(median) in (int,float) and math.isfinite(median) and median > 0,"Actual full-pipeline median required")
        require(1.5*median*313*160+300 <= budget,"Native-BN actual pipeline cannot fit fixed training/finalization guard")
    return budget


def validate_worker(folder, manifest_sha):
    """Reject a direct worker before importing Torch, data or the engine."""
    require(folder.parent.resolve() == (ROOT/"results/research").resolve() and
        sha256(folder/"nativebn_inputs.json") == manifest_sha,"Controlled source-bound native-BN worker required")
    receipt = json.loads((folder/"supervisor.json").read_text())
    process,parent = psutil.Process(),psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at",""))
    mode = receipt.get("mode")
    maximum = {"preflight":180,"train":10800,"benchmark":300}.get(mode,0)
    require(receipt.get("schema") == SCHEMA and receipt.get("status") in ("training","benchmarking") and
        receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
        type(receipt.get("reserved_seconds")) in (int,float) and math.isfinite(receipt["reserved_seconds"]) and
        0 < receipt["reserved_seconds"] <= maximum and type(receipt.get("deadline_unix")) in (int,float) and
        math.isfinite(receipt["deadline_unix"]) and time.time() < receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
        started.timestamp() <= time.time()+.1 and receipt.get("pid") == parent.pid and
        abs(receipt.get("process_created_at",0)-parent.create_time()) < .01 and receipt.get("child_pid") == process.pid and
        abs(receipt.get("child_created_at",0)-process.create_time()) < .01,"Exact bounded live native-BN supervisor required")
    lease = json.loads((ROOT/"results/research/local_lease.json").read_text())
    require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at",0)-parent.create_time()) < .01 and
        lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False,
        "Canonical native-BN lease is missing, foreign or unknown")
    matches = []
    for path in folder.glob("process_tree_*"):
        try: spec,ready = [json.loads((path/name).read_text()) for name in ("spec.json","ready.json")]
        except (OSError,ValueError): continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at",0)-process.create_time()) < .01:
            matches.append((path.resolve(),spec,ready))
    require(len(matches) == 1,"Exactly one live native-BN OS guard required")
    path,spec,ready = matches[0]
    nonce = path.name.removeprefix("process_tree_")
    guard = (ROOT/"research/process_tree_guard.py").resolve()
    from research.profile_augmix import PINNED_SOURCES
    require(re.fullmatch(r"[0-9a-f]{32}",nonce) and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
        spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid and
        abs(spec.get("host_created_at",0)-parent.create_time()) < .01 and Path(spec.get("cwd","")).resolve() == ROOT.resolve() and
        Path(spec.get("script","")).resolve() == Path(__file__).resolve() and
        spec.get("script_sha256") == ready.get("script_sha256") == sha256(__file__) and
        spec.get("guard_source_sha256") == sha256(guard) == PINNED_SOURCES["research/process_tree_guard.py"] and
        spec.get("arguments") == ["--_worker",str(folder),"--_manifest-sha",manifest_sha] and
        ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
        type(spec.get("deadline_unix")) in (int,float) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
        spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None),"Native-BN guard nonce/source/deadline differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and
        Path(command[1]).resolve() == guard and command[2] == "--bootstrap" and Path(command[3]).resolve() == path/"spec.json",
        "Native-BN worker must use the exact actual OS bootstrap")
    verify_os_boundary(spec,ready,process)
    manifest = json.loads((folder/"nativebn_inputs.json").read_text())
    require(manifest.get("source_sha256") == receipt.get("source_sha256") == bound_sources() and
        manifest.get("config_sha256") == CONFIG_SHA and manifest.get("diagnostic_sha256") == DIAGNOSTIC_SHA and
        manifest.get("mode") in (("train","benchmark") if mode == "train" else (mode,)) and
        type(manifest.get("input_sha256")) is dict and bool(manifest["input_sha256"]) and
        manifest["input_sha256"] == receipt.get("input_sha256") and
        manifest["input_sha256"].get(str((ROOT/CONFIG).resolve())) == CONFIG_SHA and
        all(sha256(path) == digest for path,digest in manifest["input_sha256"].items()),
        "Native-BN source/config/input binding changed")
    return manifest


def worker(folder, manifest_sha):
    manifest = validate_worker(folder,manifest_sha)
    import runtime  # Process-local DLL isolation must precede Torch.
    import torch
    from research import augmix_nativebn as protocol
    from research.train_augmix import fixed_config
    require(torch.cuda.is_available(),"Actual native CUDA required")
    raw = (ROOT/CONFIG).read_bytes()
    require(sha256(ROOT/CONFIG) == CONFIG_SHA,"Exact native-BN raw config required")
    mode = manifest["mode"]
    if mode == "benchmark":
        from research.benchmark_augmix_nativebn import benchmark
        benchmark(Path(manifest["checkpoint"]),folder/"benchmark.json",Path(manifest["baseline"]),Path(manifest["data_dir"]),"cuda")
    else:
        protocol.run_engine(mode,fixed_config(),folder,Path(manifest["data_dir"]),torch.device("cuda"),raw,resume=manifest.get("resume",False))


def checked_preflight(path, sources):
    from research import augmix_nativebn as protocol
    from research.train_augmix import fixed_config
    from research.benchmark_augmix import validate_pool_document
    from research.run_local import assert_stopped
    path = Path(path).resolve()
    receipt_path = path.parent/"supervisor.json"
    receipt = json.loads(receipt_path.read_text())
    require(path.name == "nativebn_preflight.json" and receipt.get("schema") == SCHEMA and receipt.get("mode") == "preflight" and
        receipt.get("status") == "complete" and receipt.get("source_sha256") == sources and
        receipt.get("artifact_sha256") == sha256(path) and receipt.get("process_tree_cleanup_verified") is True,
        "Completed source-bound native-BN preflight supervisor required")
    assert_stopped(receipt)
    envelope = json.loads(path.read_text())
    report = protocol.validate_preflight(envelope,fixed_config(),CONFIG_SHA)
    original,pool = path.parent/"engineering_benchmark.json",path.parent/"worker_pool_preflight.json"
    require(json.loads(original.read_text()) == envelope["engine_result"] and sha256(pool) == report["worker_pool_receipt_sha256"],
        "Native-BN original actual pipeline/pool artifact differs")
    validate_pool_document(json.loads(pool.read_text()))
    return report,{str(p):sha256(p) for p in (path,receipt_path,original,pool)}


def parent(args):
    from runtime import environment,write_json
    from research.run_local import now,used_seconds,acquire_lease,assert_stopped,close_interrupted_attempt
    from research.process_tree_guard import run_guarded,may_release_lease
    from research import augmix_nativebn as protocol
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",args.run or "") and not args.run.startswith("stability"),"Fresh native-BN run ID required")
    require(not args.resume or args.mode == "train","Resume only exact native-BN training")
    require((args.mode == "train") == (args.preflight is not None) and (args.mode == "benchmark") == (args.checkpoint is not None),
        "Train requires native-BN preflight; only benchmark accepts checkpoint")
    sources = bound_sources()
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    admission_budget(policy,now(),0)
    data_dir = args.data_dir.resolve()
    baseline = (ROOT/policy["baseline_benchmark"]).resolve()
    official = data_dir/"training_manifest.json"
    document = json.loads(official.read_text())
    paths = [(ROOT/CONFIG).resolve(),baseline,official,*(data_dir/name for name in document["files_sha256"])]
    require(sha256(ROOT/CONFIG) == CONFIG_SHA and all(path.is_file() for path in paths),"Trusted baseline/official native-BN inputs missing")
    binding = {str(path.resolve()):sha256(path) for path in paths}
    preflight = None
    if args.preflight:
        preflight,extra = checked_preflight(args.preflight,sources);binding.update(extra)
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        candidates = [checkpoint,*(checkpoint.parent/name for name in ("last.pt","metrics.json","history.json","config.json","input_config.json","provenance.json","environment.json"))]
        require(all(path.is_file() for path in candidates),"Complete native-BN candidate evidence required")
        binding.update({str(path):sha256(path) for path in candidates})
    base,folder = ROOT/"results/research",ROOT/"results/research"/args.run
    previous = None
    if args.resume:
        previous = json.loads((folder/"supervisor.json").read_text())
        require(previous.get("schema") == SCHEMA and previous.get("mode") == "train" and previous.get("status") != "complete" and
            previous.get("source_sha256") == sources and previous.get("input_sha256") == binding and
            previous.get("config_sha256") == CONFIG_SHA and (folder/"last.pt").is_file() and not (folder/"benchmark.json").exists() and
            previous.get("process_tree_cleanup_verified") is True,"Exact stopped source-bound native-BN resume required")
        assert_stopped(previous)
    else: require(not folder.exists(),"Existing native-BN artifacts are immutable")
    lease = base/"local_lease.json"
    require(not lease.exists(),"Native-BN requires an explicitly reconciled free local lease")
    acquire_lease(lease)
    began,owns,receipt = time.perf_counter(),False,{}
    try:
        day = now().replace(hour=0,minute=0,second=0,microsecond=0)
        excluded = folder/"supervisor.json" if args.resume else None
        budget = admission_budget(policy,now(),used_seconds(base,day,day+timedelta(days=1),exclude=excluded),preflight)
        if args.mode != "train": budget = min(budget,180 if args.mode == "preflight" else 300)
        if preflight:
            actual,expected = environment(),preflight["environment"]
            require(all(actual.get(k) == expected.get(k) for k in ("gpu","cuda_runtime","cudnn")) and
                actual["packages"]["torch"] == expected["packages"]["torch"],"Actual native-BN environment differs from measured preflight")
        if args.resume:
            archived = close_interrupted_attempt(folder/"supervisor.json")
            owns = True
            receipt = {"schema":SCHEMA,"mode":"train","backend":"local","run":args.run,"status":"starting",
                "started_at":now().isoformat(),"pid":os.getpid(),"process_created_at":psutil.Process().create_time(),
                "reserved_seconds":0,"previous_attempt":archived.name,"source_sha256":sources,"input_sha256":binding,"config_sha256":CONFIG_SHA}
            budget = admission_budget(policy,now(),used_seconds(base,day,day+timedelta(days=1),exclude=excluded),preflight)
        else: folder.mkdir(parents=True,exist_ok=False);owns = True
        receipt = {"schema":SCHEMA,"backend":"local","run":args.run,"mode":args.mode,
            "status":"training" if args.mode == "train" else "benchmarking","trainer":"augmix_nativebn","precision":"fp16",
            "pid":os.getpid(),"process_created_at":psutil.Process().create_time(),"started_at":now().isoformat(),
            "reserved_seconds":budget,"deadline_unix":time.time()+budget,"source_sha256":sources,"input_sha256":binding,
            "config_sha256":CONFIG_SHA,"resumed":args.resume,"submitted":False,"confirmation_evaluated":False}
        if args.resume: receipt.update(previous_attempt=archived.name,resumed_checkpoint_sha256=sha256(folder/"last.pt"))
        manifest = {"mode":args.mode,"source_sha256":sources,"config_sha256":CONFIG_SHA,"diagnostic_sha256":DIAGNOSTIC_SHA,
            "input_sha256":binding,"data_dir":str(data_dir),"baseline":str(baseline),"checkpoint":str(args.checkpoint.resolve()) if args.checkpoint else None,"resume":args.resume}
        def execute(mode):
            manifest["mode"] = mode
            if mode == "benchmark" and args.mode == "train": manifest["checkpoint"] = str(folder/"best.pt")
            write_json(folder/"nativebn_inputs.json",manifest)
            receipt["input_manifest_sha256"] = sha256(folder/"nativebn_inputs.json")
            write_json(folder/"supervisor.json",receipt)
            remaining = budget-(time.perf_counter()-began)
            require(remaining > 0,"No bounded native-BN execution time remains")
            run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder),"--_manifest-sha",receipt["input_manifest_sha256"]],
                folder/(mode+"_"+str(time.time_ns())+".log"),remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        execute(args.mode)
        if args.mode == "train":
            require(all(sha256(path) == digest for path,digest in binding.items()) and sources == bound_sources(),"Native-BN source/input changed during training")
            receipt["status"] = "benchmarking";execute("benchmark")
        artifact = folder/("nativebn_preflight.json" if args.mode == "preflight" else "benchmark.json")
        result = json.loads(artifact.read_text())
        require(result.get("status") == "complete" and sources == bound_sources() and
            all(sha256(path) == digest for path,digest in binding.items()),"Complete unchanged native-BN evidence required")
        if args.mode == "preflight": protocol.validate_preflight(result,__import__("research.train_augmix",fromlist=["fixed_config"]).fixed_config(),CONFIG_SHA)
        else: require(result.get("research_training_integrity",{}).get("schema") == "research_augmix_nativebn_integrity_v1","Deep typed native-BN benchmark required")
        receipt.update(status="complete",artifact=artifact.name,artifact_sha256=sha256(artifact))
        if args.mode != "preflight": receipt["benchmark_passed"] = result["promotion"]["center80_shift20"]["passed"]
    except BaseException as error:
        receipt.update(status="failed",error_type=type(error).__name__);raise
    finally:
        if owns and receipt:
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
    parser.add_argument("--mode",choices=("preflight","train","benchmark"))
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        require(args.run is None and args.mode is None and args.preflight is None and args.checkpoint is None and not args.resume and args._manifest_sha,
            "Exact native-BN guarded worker arguments required")
        worker(args._worker.resolve(),args._manifest_sha)
    elif args.run:
        require(args.mode is not None and args._manifest_sha is None,"Explicit native-BN mode required")
        parent(args)
    else:
        require(args.mode is None and args._manifest_sha is None and not args.resume and args.preflight is None and args.checkpoint is None,"Explicit native-BN run required")
        parser.print_help()


if __name__ == "__main__": main()
