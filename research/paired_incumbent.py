"""One guarded matched-incumbent development pair; no delivery authorization.

Parent and authentication use stdlib/psutil only. Torch is imported only by an
authenticated worker inside its own OS process boundary and canonical lease.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import preflight_core_control as control
from research import profile_augmix as guard_checks
from research import incumbent_compare as aggregate

SCHEMA = "matched_incumbent_pair_supervisor_v1"
INPUT_SCHEMA = "matched_incumbent_pair_inputs_v1"
RESULT_SCHEMA = "matched_incumbent_image_bootstrap_v1"
PLAN = "research/paired-incumbent-plan.md"
POLICY, SEED, REPLICATES = "center80_shift20", 20261031, 10000
INCUMBENT_CHECKPOINT = ROOT / "results/research/colab_wrn34_3_300_fp16_s2026_v3/candidate/best.pt"
PINS = {
    **aggregate.CORE,
    "research/benchmark.py": aggregate.BENCHMARK_SOURCE_SHA,
    "research/paired_compare.py": "034c850037b71ef151ad1f3f168afb3c475f170ad2a436d70e406e329fba5e65",
    "research/paired_compare_v2.py": "7218d65bd0f33ae6e3563763730d6ce161a9e90131cdfb21c9feb10ad5bb5199",
    "research/benchmark_shadow.py": "aab840a944125f45ef070099e70270103c62a5acabf081ead7f26cfdb6f7d6c9",
    "research/train_shadow.py": "9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
    "research/augmentations.py": "718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
    "research/benchmark_sam.py": "ec7d93ff8aa05e658f0fb44170d43a76b1f8d0a711fe7320b82a9717a83a5376",
    "research/train_sam.py": "6c172b127ee0749e6973903c192156bcc2f0bb88c8c6c22b74f8a0d1e089d8f1",
    "research/sam.py": "682171d56134e82e14cd538bee67b7eec8dffa63c3c788b77e4636e5a09ad01c",
    "research/benchmark_resizemix.py": "64508b91668b84627aecdce16f644c1512b94c9a6628470e5e4b2b8702590c40",
    "research/train_resizemix.py": "6fc49c6b76af09a753cb784b367c39e1be4ddb47b918e5cee8c2d3dfff298ae8",
    "research/resizemix.py": "042d909df03063abba4cf4e8cb9401b2f48c247734741780d97c9c0b30cbced1",
    "research/preflight_core_control.py": "85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761",
    "research/profile_augmix.py": "82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66",
    "research/process_tree_guard.py": "5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80",
    "research/run_local.py": "6c42cfd08aaad6b39c1acd23ed788044a68c31b047a677a0218308a1dd15a069",
    "research/incumbent_compare.py": "277f7803d855a90a5bd92e582e2beda85cb18557402d9a4332428d2b70a49d0d",
}
require, sha, read, write = control.require, control.sha256, control.read, control.write


def source_snapshot():
    actual = {name: sha(ROOT/name) for name in PINS}
    require(actual == PINS, "Frozen paired/incumbent/guard dependencies differ")
    return {**actual, PLAN: sha(ROOT/PLAN), "research/paired_incumbent.py": sha(__file__),
            "research/campaign.json": sha(ROOT/"research/campaign.json")}


def binding_for(data_dir, candidate_checkpoint, candidate_benchmark, sources):
    """Hash actual source/cache/checkpoint/history/config before Torch/lease."""
    data_paths, identity = control.official_inputs(data_dir)
    a, _ = aggregate.read_document(candidate_benchmark)
    b, incumbent_sha = aggregate.read_document(aggregate.INCUMBENT_PATH)
    summary = aggregate.compare_documents(a, b)
    require(incumbent_sha == aggregate.INCUMBENT_BENCHMARK_SHA,
            "Actual incumbent local benchmark bytes differ")
    paths = [*data_paths, candidate_checkpoint, candidate_benchmark, INCUMBENT_CHECKPOINT,
             aggregate.INCUMBENT_PATH, *(ROOT/name for name in sources)]
    for checkpoint, benchmark, document in ((candidate_checkpoint, candidate_benchmark, a),
                                           (INCUMBENT_CHECKPOINT, aggregate.INCUMBENT_PATH, b)):
        require(checkpoint.is_file() and sha(checkpoint) == document["checkpoint_sha256"],
                "Actual checkpoint bytes differ from the complete benchmark")
        for name, key in (("metrics.json", "training_metrics_sha256"), ("history.json", "training_history_sha256")):
            path = checkpoint.parent/name
            require(sha(path) == document[key], "Actual metrics/history bytes differ")
            paths.append(path)
        config_path = checkpoint.parent/"config.json"
        require(read(config_path) == document["config"], "Actual effective training configuration differs")
        paths.append(config_path)
        integrity = document.get("research_training_integrity")
        if integrity is not None:
            require(integrity["training"].get("schema") in ("research_shadow_training_v1",
                    "research_sam_training_v1", "research_resizemix_training_v1"),
                    "No existing typed validator supports this independent trainer")
            paths += [checkpoint.parent/name for name in ("provenance.json", "environment.json")]
            core = benchmark.with_name(benchmark.stem+".core.json")
            require(sha(core) == integrity.get("core_benchmark_sha256") and read(core) ==
                    {k:v for k,v in document.items() if k != "research_training_integrity"},
                    "Actual method/core benchmark binding differs")
            paths.append(core)
            for name, checksum in integrity["training"]["source_sha256"].items():
                require(name in sources and sources[name] == checksum,
                        "Unsupported or changed typed method source")
    binding = {str(path.resolve()): sha(path) for path in paths}
    return binding, identity, summary


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


def validate_worker(folder, manifest_sha):
    folder = Path(folder).resolve()
    require(folder.parent == (ROOT/"results/research").resolve() and re.fullmatch(r"[0-9a-f]{64}",manifest_sha or "")
            and sha(folder/"inputs.json") == manifest_sha, "Controlled new paired input manifest required")
    manifest, receipt = read(folder/"inputs.json"), read(folder/"supervisor.json")
    validate_owner(folder, manifest_sha, manifest, receipt, psutil.Process(), psutil.Process(os.getppid()))
    sources = source_snapshot()
    binding, identity, summary = binding_for(Path(manifest["data_dir"]), Path(manifest["candidate_checkpoint"]),
                                            Path(manifest["candidate_benchmark"]), sources)
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("source_sha256") == receipt.get("source_sha256") == sources
            and manifest.get("input_sha256") == receipt.get("input_sha256") == binding
            and manifest.get("data_identity") == identity and manifest.get("comparison") == summary
            and receipt.get("confirmation_evaluated") is False and receipt.get("submitted") is False,
            "Actual source/config/checkpoint/data/benchmark input binding differs")
    require(isinstance(manifest.get("source_git_commit"),str) and re.fullmatch(r"[0-9a-f]{40}",manifest["source_git_commit"]),
            "Actual source revision required")
    return manifest


def validate_result(result, manifest, manifest_sha):
    summary = manifest["comparison"]
    fixed = {"schema": RESULT_SCHEMA, "status": "complete", "candidate_policy": POLICY, "incumbent_policy": POLICY,
             "data_signature": aggregate.DATA, "evaluation_context": aggregate.CONTEXT,
             "candidate_checkpoint_sha256": summary["candidate"]["checkpoint_sha256"],
             "incumbent_checkpoint_sha256": aggregate.INCUMBENT_CHECKPOINT_SHA,
             "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
             "input_manifest_sha256": manifest_sha, "comparison": summary,
             "shape": [5000,6], "correctness_dtype": "uint8_binary", "raw_flags_persisted": False,
             "original_ids_aligned": True, "automatic_promotion": False, "delivery_authorized": False,
             "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}
    require(all(type(result.get(k)) is type(v) and result[k] == v for k,v in fixed.items()), "Complete typed incumbent pair required")
    for key in ("original_id_order_sha256", "development_index_order_sha256", "candidate_correctness_sha256", "incumbent_correctness_sha256"):
        require(aggregate.is_sha(result.get(key)), "Private in-memory row/flag hashes required")
    bootstrap = result.get("bootstrap", {})
    require(bootstrap.get("original_images") == 5000 and bootstrap.get("replicates") == 10000
            and bootstrap.get("seed") == SEED and bootstrap.get("sampling_unit") == "original_image_with_all_five_stress_views"
            and bootstrap.get("confidence_level") == .95, "Fixed original-image 10000-cluster bootstrap required")
    for kind, denominator, delta in (("clean",5000,summary["same_policy_deltas"]["clean_accuracy"]),
                                     ("stress",25000,summary["same_policy_deltas"]["mean_stress_accuracy"])):
        row = bootstrap.get(kind,{})
        counts = [row.get(k) for k in ("candidate_only_correct","baseline_only_correct","both_correct","both_wrong")]
        require(all(type(v) is int and 0 <= v <= denominator for v in counts) and sum(counts) == denominator
                and type(row.get("delta_accuracy")) in (int,float) and math.isfinite(row["delta_accuracy"])
                and math.isclose(row["delta_accuracy"],delta,rel_tol=0,abs_tol=1e-12)
                and math.isclose((counts[0]-counts[1])/denominator,delta,rel_tol=0,abs_tol=1e-12)
                and isinstance(row.get("percentile_ci95"),list) and len(row["percentile_ci95"]) == 2
                and all(aggregate.number(v,minimum=-1,maximum=1) for v in row["percentile_ci95"])
                and row["percentile_ci95"][0] <= row["percentile_ci95"][1], "Inconsistent paired aggregate/counts/interval")


def worker(folder, manifest_sha):
    manifest = validate_worker(folder, manifest_sha)  # All ownership/source proofs precede Torch.
    import numpy as np
    from research import paired_compare_v2 as paired
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from train import code_provenance, tensor_images
    from runtime import seed_all
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.__version__ == aggregate.CONTEXT["torch_version"], "Actual native local BF16 CUDA required")
    device = torch.device("cuda")
    require(str(paired.suite.benchmark_dtype(device)) == "torch.bfloat16" and code_provenance()["source_sha256"] == aggregate.CORE,
            "Actual evaluation precision/core differs")
    seed_all(2026)
    images, labels, train_ids, val_ids = load_training_arrays(Path(manifest["data_dir"]))
    official_ids = np.load(Path(manifest["data_dir"])/"official_train_ids.npy", allow_pickle=False)
    require(val_ids.shape == (5000,) and len(set(val_ids.tolist())) == 5000 and not set(train_ids.tolist()) & set(val_ids.tolist()),
            "Unique independent original development rows required")
    ids = [str(i) for i in official_ids[val_ids]]
    require(len(set(ids)) == 5000, "Unique official original IDs required")
    vectors, docs = [], []
    dev_images, dev_labels = tensor_images(images[val_ids],device), torch.as_tensor(labels[val_ids],device=device)
    for path, evidence in ((Path(manifest["candidate_checkpoint"]),Path(manifest["candidate_benchmark"])),
                           (INCUMBENT_CHECKPOINT,aggregate.INCUMBENT_PATH)):
        checkpoint, document = paired.load_experiment(path,evidence,aggregate.DATA,aggregate.CONTEXT,aggregate.CORE,train_ids,val_ids)
        model = build_model(checkpoint["model_config"])
        require(count_parameters(model) == checkpoint["parameter_count"] <= 5_000_000
                and count_parameters(model,trainable_only=True) <= 5_000_000, "Actual total/trainable parameter count differs")
        model.load_state_dict(checkpoint["model"],strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()), "Actual model tensor state is non-finite")
        model = model.to(device=device,memory_format=torch.channels_last).eval()
        vector = paired.collect_correctness(model,dev_images,dev_labels,POLICY,128,device,document["policies"][POLICY])
        require(vector.shape == (5000,6) and np.isin(vector,[0,1]).all(), "Complete six-case flags required")
        vectors.append(vector); docs.append(document)
        del model, checkpoint
    require([doc["checkpoint_sha256"] for doc in docs] == [manifest["comparison"]["candidate"]["checkpoint_sha256"],aggregate.INCUMBENT_CHECKPOINT_SHA],
            "Actual selected checkpoint identities differ")
    bootstrap = paired.paired_bootstrap(*vectors,seed=SEED,replicates=REPLICATES)
    require(source_snapshot() == manifest["source_sha256"] and all(sha(path) == checksum for path,checksum in manifest["input_sha256"].items()),
            "Source/evidence changed during paired inference/bootstrap")
    result = {"schema": RESULT_SCHEMA,"status":"complete","candidate_policy":POLICY,"incumbent_policy":POLICY,
              "data_signature":aggregate.DATA,"evaluation_context":aggregate.CONTEXT,
              "candidate_checkpoint_sha256":docs[0]["checkpoint_sha256"],"incumbent_checkpoint_sha256":docs[1]["checkpoint_sha256"],
              "source_sha256":manifest["source_sha256"],"input_sha256":manifest["input_sha256"],"input_manifest_sha256":manifest_sha,
              "comparison":manifest["comparison"],"shape":[5000,6],"correctness_dtype":"uint8_binary","raw_flags_persisted":False,
              "original_id_order_sha256":control.json_sha(ids),
              "development_index_order_sha256":hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest(),
              "candidate_correctness_sha256":hashlib.sha256(vectors[0].astype(np.uint8).tobytes()).hexdigest(),
              "incumbent_correctness_sha256":hashlib.sha256(vectors[1].astype(np.uint8).tobytes()).hexdigest(),
              "original_ids_aligned":True,"bootstrap":bootstrap,"automatic_promotion":False,"delivery_authorized":False,
              "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
              "interpretation":"Both weighted models were rerun on the same ordered5000 original dev IDs with all six cases.10000 image-cluster draws keep five stress views together. Conditional reused-development evidence, not independent confirmation, seed variance, hidden accuracy or new delivery authorization."}
    validate_result(result,manifest,manifest_sha)
    output = Path(folder)/"paired_incumbent.json"
    require(not output.exists(), "Existing paired evidence cannot be overwritten")
    write(output,result)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--candidate-checkpoint",type=Path)
    parser.add_argument("--candidate-benchmark",type=Path)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        require(args.run is None and args.candidate_checkpoint is None and args.candidate_benchmark is None
                and args._manifest_sha is not None, "Only exact worker arguments allowed")
        worker(args._worker,args._manifest_sha)
    elif args.run:
        require(args._manifest_sha is None and args.candidate_checkpoint is not None and args.candidate_benchmark is not None,
                "Run requires explicit candidate checkpoint and complete benchmark")
        parent(args.run,args.candidate_checkpoint.resolve(),args.candidate_benchmark.resolve(),args.data_dir.resolve())
    else:
        require(args._manifest_sha is None and args.candidate_checkpoint is None and args.candidate_benchmark is None, "Incomplete diagnostic arguments")
        parser.print_help()


if __name__ == "__main__":
    main()
