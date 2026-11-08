"""Prepare or supervise only the registered AugMix+JSD primary on one Colab T4.

Preparation is offline. Launch records one durable allocation intent, uses the
shared cloud lease and never retries an unknown allocation or remote outcome.
No existing method router, runtime, source file or delivery helper is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import uuid
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import run_colab as common

ColabCLI, sha256, write_json, now = common.ColabCLI, common.sha256, common.write_json, common.now
private_directory, wsl_path = common.private_directory, common.wsl_path
acquire_lease, parse_usage = common.acquire_lease, common.parse_usage
validate_resource_usage, stop_and_verify = common.validate_resource_usage, common.stop_and_verify
CORE, CACHE, BASELINE_SHA, CHUNK_BYTES = common.CORE, common.CACHE, common.BASELINE_SHA, common.CHUNK_BYTES
FROZEN_COMMON_SHA = "836c6546c9ba82cb9f14432b2a285c5da23e0e0bc5ddd571ea4946fa062eae4d"
TRAIN_MANIFEST_SHA = "88c25b1069b4d14936d021c4466adace6acc76d6c98009df4b6012631e2cb68a"
CACHE_SHA = {
    "official_train_images.npy": "4a5241847d2fdc2d50a9ad6e386b47f26d121db6cb2e43b24e72190724314e48",
    "official_train_labels.npy": "9dfee6f275bac0f14e63de8d1091cd1f4487a16d30c6d8726f61d1b8f999c745",
    "official_train_ids.npy": "453bc47d56258ca79cf3b32d15f48aa07ffdf4bd0e4438cb4d4e1140aaa9d07f",
    "official_split_indices.npz": "2fe25e1395cae002e91880ff3ab36404fc63ece8238c5577d40d9613075e9679"}
SIGNATURE = {"archive_sha256": "d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
             "split_sha256": "73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e"}
CACHE_LAYOUTS = (("official_train_images.npy", (50000, 32, 32, 3), "uint8"),
                 ("official_train_labels.npy", (50000,), "int64"),
                 ("official_train_ids.npy", (50000,), "<U5"))
METHOD_SOURCES = ("research/train_augmix.py", "research/augmix.py", "research/train_shadow.py", "research/augmentations.py")
SOURCES = (*CORE, "research/train_shadow.py", "research/augmentations.py", "research/benchmark.py",
           "research/augmix.py", "research/train_augmix.py", "research/benchmark_augmix.py")
HOST_SOURCES = ("research/run_colab_augmix.py", "research/run_colab.py", "research/colab_cli.py",
                "research/campaign.json", "research/augmix-plan.md")
PACKAGE_MANIFEST = "augmix_bundle_manifest.json"
SCHEMA = "private_colab_augmix_input_v1"
REMOTE_SCHEMA = "research_augmix_colab_remote_v1"
POOL_OUTPUTS = ("preflight/worker_pool_preflight.json", *(f"candidate/worker_pool_epoch{e:04d}.json" for e in range(1, 161)))
COMPLETE_OUTPUTS = (*POOL_OUTPUTS, "preflight/engineering_benchmark.json", "preflight/preflight.log",
    "baseline/benchmark.json", "baseline/benchmark.log", "candidate/best.pt", "candidate/last.pt",
    "candidate/metrics.json", "candidate/history.json", "candidate/config.json", "candidate/input_config.json", "candidate/environment.json",
    "candidate/provenance.json", "candidate/training.log", "candidate/benchmark.log", "candidate/benchmark.json",
    "candidate/benchmark.core.json", "remote_receipt.json")
OUTPUTS = (*COMPLETE_OUTPUTS, "preflight_failure.json", "controller.log")
MAX_INPUT_BYTES, MAX_OUTPUT_BYTES = 192*1024*1024, 128*1024*1024
PREDECESSOR_RUN = "colab_wrn34_3_300_fp16_s2027_v3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(raw):
    from train import DEFAULT_CONFIG
    from research.train_augmix import fixed_config, validate_config as validate_training_config
    config = {**DEFAULT_CONFIG, **raw}
    expected = fixed_config()
    validate_training_config(config)
    require(all(expected[k] == v for k,v in {"seed": 2026, "epochs": 160, "batch_size": 128,
        "depth": 22, "widen_factor": 4, "cutout": 0, "mixup_alpha": 0., "mixup_probability": 0.,
        "augmix_jsd_weight": 12., "augmix_width": 3, "augmix_depth": -1, "augmix_severity": 3}.items()),
        "Registered AugMix primary semantics changed")
    require(config == expected and all(type(config[k]) is type(v) for k, v in expected.items()),
        "Only the fixed scratch seed2026 WRN22-4/160/B128 AugMix+12JSD recipe is supported")
    return config


def safe_zip(archive, allowed, maximum):
    names = archive.namelist()
    require(len(set(names)) == len(names) and set(names) == set(allowed) and
        sum(item.file_size for item in archive.infolist()) <= maximum and all(
        not n.endswith("/") and "\\" not in n and ":" not in n and not PurePosixPath(n).is_absolute() and
        ".." not in PurePosixPath(n).parts and (archive.getinfo(n).external_attr >> 16) & 0o170000 != 0o120000
        for n in names), "ZIP exact member allowlist, size or safe-path check failed")


def stream_sha(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def npy_header(stream, expected_shape, expected_dtype, file_size):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "Only standard non-object NPY headers are allowed")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    require(shape == expected_shape and not fortran and dtype == np.dtype(expected_dtype) and not dtype.hasobject and
        stream.tell()+math.prod(shape)*dtype.itemsize == file_size, "Official cache NPY shape/dtype/length differs")


def validate_data(root):
    data = Path(root)/"data"
    require(sha256(data/"training_manifest.json") == TRAIN_MANIFEST_SHA, "Frozen official data manifest changed")
    manifest = json.loads((data/"training_manifest.json").read_text())
    require(manifest.get("competition") == "cifar-10-res-net-architecture-challenge-dl" and
        manifest.get("source_type") == "official_kaggle_competition_training" and
        manifest.get("external_training_data_loaded") is False and manifest.get("official_test_labels_loaded") is False and
        manifest.get("files_sha256") == CACHE_SHA and manifest.get("split", {}).get("seed") == 2026 and
        manifest["split"].get("train_count") == 40000 and manifest["split"].get("validation_count") == 5000 and
        manifest["split"].get("indices_sha256") == SIGNATURE["split_sha256"] and
        manifest.get("source_provenance", {}).get("official_train_npz_sha256") == SIGNATURE["archive_sha256"],
        "Only the exact previously verified official 40k/5k training cache is allowed")
    for name, digest in CACHE_SHA.items():
        require(sha256(data/name) == digest, "Official cache file fingerprint changed")
    for name, shape, dtype in CACHE_LAYOUTS:
        path = data/name
        with path.open("rb") as stream:
            npy_header(stream, shape, dtype, path.stat().st_size)
    with zipfile.ZipFile(data/"official_split_indices.npz") as archive:
        members = {name+".npy" for name in ("train", "validation", "confirmation", "seed")}
        safe_zip(archive, members, 500000)
        for name, shape in (("train", (40000,)), ("validation", (5000,)), ("confirmation", (5000,)), ("seed", ())):
            with archive.open(name+".npy") as stream:
                npy_header(stream, shape, "int64", archive.getinfo(name+".npy").file_size)
    return manifest


def entries(root, config_path):
    root = Path(root)
    result = {name: root/name for name in SOURCES}
    result.update({"data/"+name: root/"data"/name for name in CACHE})
    result.update({"baseline/"+name: root/"results/regularized"/name for name in ("best.pt", "metrics.json", "history.json")})
    result["research/configs/candidate.json"] = Path(config_path)
    return result


def prepare(root, config_path, folder, run):
    root, config_path, folder = Path(root).resolve(), Path(config_path).resolve(), Path(folder).resolve()
    common.validate_run(run)
    require(not (folder/"input_manifest.json").exists(), "Prepared AugMix evidence is immutable")
    raw = json.loads(config_path.read_bytes())
    config = validate_config(raw)
    require(sha256(root/"research/run_colab.py") == FROZEN_COMMON_SHA, "Frozen common supervisor changed")
    for name, digest in CORE.items():
        require(sha256(root/name) == digest, "Frozen primary core changed")
    require(sha256(root/"results/regularized/best.pt") == BASELINE_SHA, "Trusted original baseline differs")
    validate_data(root)
    files = entries(root, config_path)
    fingerprints = {name: sha256(path) for name, path in files.items()}
    host = {name: sha256(root/name) for name in HOST_SOURCES}
    package = {"schema": "augmix_colab_bundle_v1", "trainer": "augmix_jsd", "run": run, "files_sha256": fingerprints,
        "host_source_sha256": host, "config": config, "raw_config_sha256": fingerprints["research/configs/candidate.json"],
        "data_signature": SIGNATURE, "hidden_images_packaged": False, "external_data_packaged": False, "confirmation_evaluated": False}
    write_json(folder/PACKAGE_MANIFEST, package)
    files[PACKAGE_MANIFEST] = folder/PACKAGE_MANIFEST
    fingerprints = {name: sha256(path) for name, path in files.items()}
    bundle = folder/"input.private.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in files.items():
            archive.write(path, name)
    chunks = []
    with bundle.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            path = folder/f"input_{len(chunks):03d}.private.part"
            with path.open("xb") as target:
                target.write(block)
            chunks.append({"name": path.name, "bytes": len(block), "sha256": sha256(path)})
    manifest = {**package, "schema": SCHEMA, "raw_config": raw, "config_source_path": str(config_path),
        "files_sha256": fingerprints, "bundle_sha256": sha256(bundle), "chunks": chunks,
        "split_note": "Complete frozen partition file required; confirmation indices are never selected or evaluated."}
    write_json(folder/"input_manifest.json", manifest)
    validate_package(root, folder)
    return manifest


def validate_package(root, folder):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    m = json.loads((folder/"input_manifest.json").read_text())
    require(m.get("schema") == SCHEMA and m.get("trainer") == "augmix_jsd" and m.get("data_signature") == SIGNATURE and
        m.get("hidden_images_packaged") is False and m.get("external_data_packaged") is False and
        m.get("confirmation_evaluated") is False, "Unsupported AugMix input manifest")
    common.validate_run(m["run"])
    for name, digest in CORE.items():
        require(sha256(root/name) == digest, "Frozen primary core changed")
    require(sha256(root/"results/regularized/best.pt") == BASELINE_SHA, "Trusted original baseline differs")
    validate_data(root)
    require(validate_config(m["raw_config"]) == m["config"], "Raw/effective AugMix configuration differs")
    files = entries(root, Path(m["config_source_path"]))
    files[PACKAGE_MANIFEST] = folder/PACKAGE_MANIFEST
    require(m["files_sha256"] == {name: sha256(path) for name, path in files.items()} and
        m["host_source_sha256"] == {name: sha256(root/name) for name in HOST_SOURCES} and
        sha256(root/"research/run_colab.py") == FROZEN_COMMON_SHA and
        m["raw_config_sha256"] == m["files_sha256"]["research/configs/candidate.json"], "Prepared source/config fingerprints differ")
    package = json.loads((folder/PACKAGE_MANIFEST).read_text())
    expected_package = {k: m[k] for k in ("trainer", "run", "host_source_sha256", "config", "raw_config_sha256", "data_signature",
        "hidden_images_packaged", "external_data_packaged", "confirmation_evaluated")}
    expected_package.update(schema="augmix_colab_bundle_v1", files_sha256={k:v for k,v in m["files_sha256"].items() if k != PACKAGE_MANIFEST})
    require(package == expected_package, "Inner package manifest differs")
    bundle = folder/"input.private.zip"
    require(bundle.stat().st_size <= MAX_INPUT_BYTES and sha256(bundle) == m["bundle_sha256"], "Prepared input ZIP changed")
    with zipfile.ZipFile(bundle) as archive:
        safe_zip(archive, files, MAX_INPUT_BYTES)
        for name, digest in m["files_sha256"].items():
            with archive.open(name) as stream:
                require(stream_sha(stream) == digest, "Packaged member fingerprint differs")
    combined = hashlib.sha256()
    require(bool(m.get("chunks")), "Input chunks are missing")
    for index, part in enumerate(m["chunks"]):
        require(part.get("name") == f"input_{index:03d}.private.part" and type(part.get("bytes")) is int and
            0 < part["bytes"] <= CHUNK_BYTES, "Input chunk sequence/size differs")
        path = folder/part["name"]
        require(path.stat().st_size == part["bytes"] and sha256(path) == part["sha256"], "Input chunk changed")
        with path.open("rb") as stream:
            while block := stream.read(1024*1024):
                combined.update(block)
    require(combined.hexdigest() == m["bundle_sha256"], "Input chunks do not reconstruct the exact bundle")
    return m


def validate_preflight_report(report, manifest, remaining):
    expected = {"schema": "research_augmix_preflight_v1", "status": "complete", "synthetic_only": False,
        "official_training_only": True, "precision": "fp16", "grad_scaler_enabled": True, "parameters": 4298970,
        "initial_grad_scaler_scale": 256.,
        "config": manifest["config"], "input_config_sha256": manifest["raw_config_sha256"], "batch_size": 128,
        "model_batch_size": 384, "warmup_steps": 3, "timed_steps": 20, "attempted_updates": 23,
        "successful_updates": 23, "skipped_updates": 0, "parameter_changed_steps": 23, "forward_calls": 23,
        "single_forward_per_step": True, "source_examples_seen": 2944, "model_views_seen": 8832,
        "num_workers": 2, "prefetch_factor": 2, "multiprocessing_context": "spawn", "worker_cuda_initialized": False,
        "persistent_workers": False, "all_workers_stopped": True,
        "training_examples": 40000, "validation_examples": 5000, "validation_evaluated": False,
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
        "source_ids_are_training_only": True, "source_ids_unique_count": 2944, "data_signature": SIGNATURE,
        "model_config": {"depth": 22, "widen_factor": 4, "dropout": 0., "num_classes": 10},
        "source_sha256": {name: manifest["files_sha256"][name] for name in (*CORE, *METHOD_SOURCES)}}
    require(all(report.get(k) == v for k, v in expected.items()), "Real AugMix+JSD preflight source/count/pipeline evidence differs")
    ids = report.get("preflight_source_ids_sha256", "")
    require(isinstance(ids, str) and bool(re.fullmatch(r"[0-9a-f]{64}", ids)), "Preflight training-source row fingerprint is missing")
    smoke = report.get("overflow_smoke", {})
    required = {"status": "passed", "finite_loss": True, "infinite_gradient_injected": True, "updated": False,
        "all_parameters_unchanged": True, "optimizer_state_unchanged": True, "ema_parameters_unchanged": True,
        "source_examples": 128, "model_views": 384, "forward_calls": 1}
    require(all(smoke.get(k) == v for k, v in required.items()) and
        all(type(smoke.get(k)) in (int, float) and math.isfinite(smoke[k]) and smoke[k] > 0 for k in ("scale_before", "scale_after")) and
        smoke["scale_before"] == 256. and smoke["scale_after"] == 128.,
        "Finite-loss overflow did not prove the registered init256 native-FP16 skip")
    environment = report.get("environment", {})
    require(environment.get("gpu") == "Tesla T4" and environment.get("device") in ("cuda", "cuda:0") and
        environment.get("amp_dtype") == "torch.float16", "AugMix preflight must use the actual T4/native-FP16 profile")
    median, p90 = report.get("median_step_seconds"), report.get("p90_step_seconds")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in
        (median, p90, report.get("images_per_second"), report.get("peak_gpu_memory_bytes"), remaining)) and
        p90 >= median, "AugMix timings/memory/remaining deadline are invalid")
    required_seconds = 1.5*median*313*160+300
    require(math.isclose(report.get("required_training_and_benchmark_seconds", -1), required_seconds, rel_tol=1e-9),
        "AugMix preflight changed the predeclared 1.5x runtime formula")
    if required_seconds >= remaining-180:
        raise RuntimeError("AugMix real pipeline does not fit the fixed budget; batch/views/JSD must not be changed")
    return {"required_training_and_benchmark_seconds": required_seconds, "runtime_safety_multiplier": 1.5,
        "pipeline_overhead_allowance_seconds": 300, "transfer_cleanup_reserve_seconds": 180,
        "remaining_seconds_at_admission": remaining, "runtime_budget_fit": True}


def validate_pool_record(pool):
    require(pool.get("schema") == "research_augmix_worker_pool_v1" and pool.get("status") == "closed" and
        pool.get("num_workers") == 2 and pool.get("prefetch_factor") == 2 and pool.get("context") == "spawn" and
        pool.get("worker_cuda_initialized") is False and pool.get("all_workers_stopped") is True and
        type(pool.get("owner_pid")) is int and pool["owner_pid"] > 0 and
        type(pool.get("owner_created_at")) in (float, int) and math.isfinite(pool["owner_created_at"]) and pool["owner_created_at"] > 0 and
        isinstance(pool.get("workers"), list) and len(pool["workers"]) == 2, "AugMix CPU pool identity/closure proof differs")
    require(all(type(v.get("pid")) is int and v["pid"] > 0 and type(v.get("created_at")) in (int, float) and
        math.isfinite(v["created_at"]) and v["created_at"] > 0 for v in pool["workers"]) and
        len({pool["owner_pid"], *(v["pid"] for v in pool["workers"])}) == 3 and
        all(isinstance(pool.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", pool[k]) for k in ("source_ids_sha256", "sample_seeds_sha256")),
        "AugMix worker PID/create or source/RNG fingerprints are invalid")


def guard_identity(manifest, session, deadline):
    body = common.remote_guard(deadline)
    return {"schema": "augmix_runtime_guard_v1", "session": session, "deadline_unix": deadline,
        "input_bundle_sha256": manifest["bundle_sha256"], "guard_body_sha256": hashlib.sha256(body.encode()).hexdigest()}


def remote_guard(manifest, session, deadline):
    identity = guard_identity(manifest, session, deadline)
    return common.remote_guard(deadline)+"\n_augmix_guard_identity = "+repr(identity)+"\nprint('AUGMIX_GUARD_JSON='+json.dumps(_augmix_guard_identity))\n"


def remote_program(manifest, session, deadline, manifest_sha256):
    constants = {"CORE": CORE, "METHOD_SOURCES": METHOD_SOURCES, "SIGNATURE": SIGNATURE,
        "CHUNK_BYTES": CHUNK_BYTES, "MAX_INPUT_BYTES": MAX_INPUT_BYTES, "PACKAGE_MANIFEST": PACKAGE_MANIFEST,
        "COMPLETE_OUTPUTS": COMPLETE_OUTPUTS, "OUTPUTS": OUTPUTS, "REMOTE_SCHEMA": REMOTE_SCHEMA}
    program = "import hashlib,json,math,os,re,subprocess,sys,time,traceback,zipfile\nfrom pathlib import Path,PurePosixPath\n"
    program += "\n".join(name+"="+repr(value) for name, value in constants.items())+"\n"
    program += "\n".join(inspect.getsource(fn) for fn in (require, safe_zip, stream_sha, validate_preflight_report, validate_pool_record))
    program += "\nM="+repr(manifest)+"\nD="+repr(deadline)+"\nEXPECTED_GUARD="+repr(guard_identity(manifest, session, deadline))
    program += "\nMANIFEST_SHA="+repr(manifest_sha256)+"\nSESSION="+repr(session)+"\n"
    return program+'''
P=Path('/content'); R=P/SESSION
require(globals().get('_augmix_guard_identity')==EXPECTED_GUARD, 'The exact self-release guard must remain armed before any work')
R.mkdir(exist_ok=False)
receipt={'schema':REMOTE_SCHEMA,'status':'running','trainer':'augmix_jsd','run':M['run'],
    'input_bundle_sha256':M['bundle_sha256'],'input_manifest_sha256':MANIFEST_SHA,
    'host_source_sha256':M['host_source_sha256'],'source_files_sha256':M['files_sha256'],
    'guard_identity':EXPECTED_GUARD,'confirmation_evaluated':False,'hidden_images_loaded':False,'external_data_loaded':False,
    'stage_history':[]}
def stage(name,args,log):
    remaining=D-time.time()-180
    if remaining<1:raise TimeoutError('No bounded AugMix stage time remains')
    receipt['stage_history'].append({'stage':name,'started_unix':time.time()})
    with (R/log).open('w') as output:
        subprocess.run([sys.executable,*args],cwd=R,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    receipt['stage_history'][-1].update(status='complete',finished_unix=time.time())
try:
    bundle=R/'input.zip'
    with bundle.open('xb') as assembled:
        for i,part in enumerate(M['chunks']):
            require(part['name']==f'input_{i:03d}.private.part','Input chunk sequence changed')
            raw=(P/part['name']).read_bytes()
            require(len(raw)==part['bytes'] and 0<len(raw)<=CHUNK_BYTES and hashlib.sha256(raw).hexdigest()==part['sha256'],'Input chunk changed')
            assembled.write(raw)
    with bundle.open('rb') as stream:require(stream_sha(stream)==M['bundle_sha256'],'Input bundle changed')
    with zipfile.ZipFile(bundle) as archive:
        safe_zip(archive,M['files_sha256'],MAX_INPUT_BYTES)
        for name,digest in M['files_sha256'].items():
            with archive.open(name) as stream:require(stream_sha(stream)==digest,'Input member changed')
        archive.extractall(R)
    package=json.loads((R/PACKAGE_MANIFEST).read_text())
    require(package['files_sha256']=={k:v for k,v in M['files_sha256'].items() if k!=PACKAGE_MANIFEST},'Inner source manifest changed')
    require(package['config']==M['config'] and package['host_source_sha256']==M['host_source_sha256'] and package['data_signature']==SIGNATURE,'Inner admission manifest differs')
    os.chdir(R);sys.path.insert(0,str(R))
    from research.train_augmix import validate_config,validate_preflight
    validate_config(M['config'])
    (R/'preflight').mkdir()
    base=['research/train_augmix.py','--config','research/configs/candidate.json','--data-dir','data','--device','cuda','--precision','fp16']
    stage('preflight',base+['--output-dir','preflight','--preflight'],'preflight/preflight.log')
    report=json.loads((R/'preflight/engineering_benchmark.json').read_text())
    pool_path=R/'preflight/worker_pool_preflight.json'
    require(hashlib.sha256(pool_path.read_bytes()).hexdigest()==report.get('worker_pool_receipt_sha256'), 'Preflight CPU worker receipt changed')
    validate_pool_record(json.loads(pool_path.read_text()))
    expected_source={name:M['files_sha256'][name] for name in (*CORE,*METHOD_SOURCES)}
    validate_preflight(report,M['config'],M['raw_config_sha256'],expected_source_sha256=expected_source,require_t4=True)
    receipt['preflight_admission']=validate_preflight_report(report,M,D-time.time())
    receipt['preflight_admitted_unix']=time.time()
    receipt['preflight_report_sha256']=hashlib.sha256((R/'preflight/engineering_benchmark.json').read_bytes()).hexdigest()
    stage('baseline',['research/benchmark.py','--checkpoint','baseline/best.pt','--output','baseline/benchmark.json','--data-dir','data','--device','cuda'],'baseline/benchmark.log')
    # A separate fresh process discards all engineering model/optimizer/EMA/RNG state.
    (R/'candidate').mkdir()
    stage('training',base+['--output-dir','candidate'],'candidate/training.log')
    metrics=json.loads((R/'candidate/metrics.json').read_text())
    require(metrics.get('schema')=='research_augmix_training_v1' and metrics.get('status')=='complete' and
        metrics.get('epochs_completed')==160 and metrics.get('attempted_updates')==50080 and
        metrics.get('successful_updates',0)+metrics.get('skipped_updates',0)==50080 and
        metrics.get('source_examples_seen')==6400000 and metrics.get('model_views_seen')==19200000 and
        metrics.get('forward_calls')==50080,'Incomplete AugMix source-batch training; views are not optimizer updates')
    stage('benchmark',['research/benchmark_augmix.py','--checkpoint','candidate/best.pt','--output','candidate/benchmark.json',
        '--baseline','baseline/benchmark.json','--data-dir','data','--device','cuda'],'candidate/benchmark.log')
    evidence=json.loads((R/'candidate/benchmark.json').read_text())
    require(evidence.get('status')=='complete' and evidence.get('examples')==5000 and
        set(evidence.get('policies',{}))=={'plain','flip','center80_shift20'},'Incomplete AugMix v3 benchmark')
    receipt.update(status='complete',benchmark_passed=evidence['promotion']['center80_shift20']['passed'])
except BaseException as error:
    receipt.update(status='failed',error_type=type(error).__name__)
    (R/'controller.log').write_text(traceback.format_exc())
    if not any(x['stage']=='training' for x in receipt['stage_history']):
        (R/'preflight_failure.json').write_text(json.dumps({'schema':'augmix_admission_failure_v1','status':'failed',
            'error_type':type(error).__name__,'formal_training_started':False,'batch_size':128,'model_batch_size':384,
            'fallback_attempted':False,'confirmation_evaluated':False}))
finally:
    receipt['finished_unix']=time.time()
    (R/'remote_receipt.json').write_text(json.dumps(receipt,indent=2))
    members=sorted(n for n in OUTPUTS if (R/n).is_file())
    if receipt['status']=='complete':require(set(members)==set(COMPLETE_OUTPUTS),'Complete output member set differs')
    artifact=R/'artifacts.zip'
    with zipfile.ZipFile(artifact,'x',compression=zipfile.ZIP_STORED) as archive:
        for name in members:archive.write(R/name,name)
    receipt['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt['artifact_members']=members
    receipt['artifact_chunks']=[]
    with artifact.open('rb') as stream:
        while raw:=stream.read(CHUNK_BYTES):
            name=f'augmix_artifacts_{len(receipt["artifact_chunks"]):03d}.part'
            (P/name).write_bytes(raw)
            receipt['artifact_chunks'].append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    print('AUGMIX_RESULT_JSON='+json.dumps(receipt))
'''


def extract_artifacts(archive, folder, remote):
    require(remote.get("schema") == REMOTE_SCHEMA and remote.get("trainer") == "augmix_jsd" and
        remote.get("status") in {"complete", "failed"}, "Unsupported AugMix remote result")
    require(sha256(archive) == remote.get("artifact_sha256"), "AugMix output archive hash differs")
    members = remote.get("artifact_members", [])
    require(isinstance(members, list) and len(set(members)) == len(members) and
        "remote_receipt.json" in members and set(members) <= set(OUTPUTS) and
        (remote["status"] != "complete" or set(members) == set(COMPLETE_OUTPUTS)), "AugMix result member declaration differs")
    with zipfile.ZipFile(archive) as source:
        safe_zip(source, members, MAX_OUTPUT_BYTES)
        inner = json.loads(source.read("remote_receipt.json"))
        require(inner == {k:v for k,v in remote.items() if k not in {"artifact_sha256", "artifact_chunks", "artifact_members"}},
            "Original remote receipt differs from the hashed output archive")
        if remote["status"] == "complete":
            for name in POOL_OUTPUTS:
                pool = json.loads(source.read(name))
                validate_pool_record(pool)
        require(all(not (folder/name).exists() for name in members), "Existing output evidence must never be overwritten")
        source.extractall(folder)


def validate_downloaded(folder, manifest, runtime):
    """CPU-only re-audit of downloaded method records before declaring completion."""
    import torch
    from research import benchmark
    from research.benchmark_augmix import validate_augmix
    candidate = folder/"candidate"
    read = lambda path: json.loads(path.read_text())
    report = read(folder/"preflight/engineering_benchmark.json")
    validate_preflight_report(report, manifest, 10800.)
    require(report.get("worker_pool_receipt_sha256") == sha256(folder/"preflight/worker_pool_preflight.json"),
        "Downloaded real-preflight CPU pool fingerprint differs")
    require(report.get("environment", {}).get("packages", {}).get("torch") == runtime.get("torch_version"),
        "Preflight Torch differs from the native runtime guard")
    checkpoint = torch.load(candidate/"best.pt", map_location="cpu", weights_only=False)
    last = torch.load(candidate/"last.pt", map_location="cpu", weights_only=False)
    metrics, history, provenance, environment = [read(candidate/name) for name in
        ("metrics.json", "history.json", "provenance.json", "environment.json")]
    raw = (candidate/"input_config.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest["raw_config_sha256"] and
        read(candidate/"config.json") == checkpoint.get("config") == manifest["config"], "Downloaded AugMix raw/effective recipe differs")
    integrity = validate_augmix(checkpoint, metrics, history, provenance, environment, raw, last)
    evidence = read(candidate/"benchmark.json")
    recorded = evidence.get("research_training_integrity", {})
    require(all(recorded.get(k) == v for k,v in integrity.items()) and
        recorded.get("validator_source_sha256") == manifest["files_sha256"]["research/benchmark_augmix.py"] and
        recorded.get("core_benchmark_sha256") == sha256(candidate/"benchmark.core.json") and
        recorded.get("provenance_document_sha256") == sha256(candidate/"provenance.json") and
        recorded.get("environment_document_sha256") == sha256(candidate/"environment.json") and
        evidence.get("fixed_primary_policy") == "center80_shift20" and
        read(candidate/"benchmark.core.json") == {k:v for k,v in evidence.items() if k not in {"research_training_integrity", "fixed_primary_policy"}},
        "Downloaded typed AugMix benchmark integrity differs")
    require(evidence.get("schema") == benchmark.SCHEMA and evidence.get("status") == "complete" and
        evidence.get("examples") == 5000 and evidence.get("training_examples") == 40000 and
        evidence.get("parameter_count") == 4298970 and evidence.get("checkpoint_sha256") == sha256(candidate/"best.pt") and
        evidence.get("data_signature") == SIGNATURE and evidence.get("config") == manifest["config"] and
        evidence.get("training_metrics_sha256") == sha256(candidate/"metrics.json") and
        evidence.get("training_history_sha256") == sha256(candidate/"history.json") and
        set(evidence.get("policies", {})) == set(benchmark.POLICIES) and
        all(evidence.get(k) is False for k in ("confirmation_evaluated", "hidden_images_loaded", "external_data_loaded")),
        "Downloaded AugMix v3 benchmark is incomplete or mismatched")
    context = {"device_type": "cuda", "amp_dtype": "torch.float16", "torch_version": runtime["torch_version"],
        "training_core_sha256": CORE}
    require(evidence.get("evaluation_context") == context, "AugMix evaluation precision/runtime differs")
    baseline_path = folder/"baseline/benchmark.json"
    baseline = read(baseline_path)
    benchmark.validate_baseline(baseline, evidence["policies"], SIGNATURE, 128, context,
        manifest["files_sha256"]["research/benchmark.py"])
    require(evidence.get("baseline_sha256") == sha256(baseline_path), "Same-runtime original baseline fingerprint differs")
    gates = {policy: benchmark.promotion_gate(report, baseline["policies"]["flip"]) for policy, report in evidence["policies"].items()}
    require(evidence.get("promotion") == gates, "Downloaded AugMix gate metadata differs")
    return gates["center80_shift20"]["passed"]


def cleanup_once(cli, folder, receipt):
    if receipt.get("allocation_returned_success") is not True:
        return {"runtime_stop_verified": False, "cleanup_outcome": "allocation_identity_unknown_no_stop"}
    try:
        acquire_lease(folder/"cleanup_intent.json", {"session": receipt["session"], "started_at": now().isoformat(),
            "pid": os.getpid(), "process_created_at": psutil.Process().create_time()})
    except FileExistsError:
        return {"runtime_stop_verified": False, "cleanup_outcome": "already_claimed_no_second_stop"}
    result = stop_and_verify(cli, receipt["session"], allocation_known=True)
    if (result.get("usage_after") or {}).get("rate_compute_units_per_hour") != 0:
        result["runtime_stop_verified"] = False
    return result


def bounded_timeout(deadline, maximum, reserve):
    remaining = deadline-time.time()-reserve
    if remaining < 1:
        raise TimeoutError("No bounded transport time remains before the fixed deadline")
    return min(maximum, remaining)


def validate_predecessor(root):
    """This next experiment cannot bypass the current WRN300 cleanup audit."""
    path = Path(root)/"results/research"/PREDECESSOR_RUN/"supervisor.json"
    receipt = json.loads(path.read_text())
    require(receipt.get("run") == "wrn34_3_300_fp16_s2027_v3" and receipt.get("finished_at") and
        receipt.get("runtime_stop_verified") is True and receipt.get("allocation_returned_success") is True and
        receipt.get("allocation_identity_verified") is True and receipt.get("stop_reported_termination") is True and
        receipt.get("stop_exit_code") == 0 and receipt.get("usage_after", {}).get("active_assignments") == 0 and
        receipt.get("usage_after", {}).get("rate_compute_units_per_hour") == 0,
        "The current WRN300 runtime must have known completed cleanup before AugMix allocation")
    finished = datetime.fromisoformat(receipt["finished_at"])
    require(finished.tzinfo is not None and finished <= now(), "Predecessor cleanup time is invalid")
    require(type(receipt.get("pid")) is int and receipt["pid"] > 0 and type(receipt.get("process_created_at")) in (int,float) and
        math.isfinite(receipt["process_created_at"]), "Predecessor process identity is incomplete")
    try:
        owner = psutil.Process(receipt["pid"])
        require(abs(owner.create_time()-receipt["process_created_at"]) >= .01, "Previous exact controller is still alive")
    except psutil.NoSuchProcess:
        pass
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def watchdog(folder, parent, created, distro):
    folder = Path(folder)
    while True:
        receipt = json.loads((folder/"supervisor.json").read_text())
        if receipt.get("finished_at"):
            return
        try:
            alive = abs(psutil.Process(parent).create_time()-created) < .01
        except psutil.NoSuchProcess:
            alive = False
        if not alive or time.time() >= receipt["deadline_unix"]:
            result = cleanup_once(ColabCLI(folder/"private", distro), folder, receipt)
            write_json(folder/"watchdog_cleanup.json", result)
            return
        time.sleep(2)


def launch(folder, distro="Ubuntu"):
    folder = Path(folder).resolve()
    manifest = validate_package(ROOT, folder)
    predecessor = validate_predecessor(ROOT)
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    require(policy.get("colab_additional_purchases_authorized") is False, "This entrypoint never purchases compute")
    deadline = min(datetime.fromisoformat(policy["stops_at"]).timestamp(), time.time()+10800)
    if deadline-time.time() < 600:
        raise RuntimeError("Insufficient campaign window remains")
    receipt_path = folder/"supervisor.json"
    if receipt_path.exists():
        raise FileExistsError("Allocation intent already exists; unknown outcomes are never retried")
    session = "saber-augmix-"+manifest["run"]+"-"+uuid.uuid4().hex[:8]
    lease_path = ROOT/"results/research/cloud_lease.json"
    receipt = {"schema": "research_augmix_cloud_supervisor_v1", "trainer": "augmix_jsd", "backend": "colab",
        "run": manifest["run"], "session": session, "status": "preparing", "pid": os.getpid(),
        "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat(), "deadline_unix": deadline,
        "assignment_attempts": 0, "allocation_returned_success": False, "remote_guard_verified": False, "precision": "fp16",
        "resource_policy": "existing_Colab_entitlement_only_no_purchases", "no_new_purchases": True,
        "confirmation_evaluated": False, "hidden_images_loaded": False, "submitted": False,
        "input_bundle_sha256": manifest["bundle_sha256"], "input_manifest_sha256": sha256(folder/"input_manifest.json"),
        "resource_predecessor_cleanup": predecessor,
        "host_source_sha256": manifest["host_source_sha256"], "source_files_sha256": manifest["files_sha256"]}
    acquire_lease(lease_path, receipt)
    guard, cli = None, None
    try:
        private_directory(folder/"private")
        cli = ColabCLI(folder/"private", distro)
        before = parse_usage(cli.require(["usage"], 30))
        validate_resource_usage(before, expected_assignments=0)
        require(before["rate_compute_units_per_hour"] == 0, "Previous runtime usage requires reconciliation")
        receipt.update(status="allocating", assignment_attempts=1, usage_before=before)
        write_json(receipt_path, receipt)  # Persist before the only allocation call.
        guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(folder),
            "--parent", str(os.getpid()), "--created", str(receipt["process_created_at"]), "--distro", distro],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        receipt.update(watchdog_pid=guard.pid, watchdog_created_at=psutil.Process(guard.pid).create_time())
        write_json(receipt_path, receipt)
        allocation = cli.require(["new", "--gpu", "T4", "--session", session], 150)
        require("[colab] Session READY." in allocation, "Allocation identity was not persisted; retain intent and lease")
        receipt["allocation_returned_success"] = True
        write_json(receipt_path, receipt)
        identity = guard_identity(manifest, session, deadline)
        guard_path = folder/"remote_guard.py"
        guard_path.write_text(remote_guard(manifest, session, deadline), encoding="utf-8")
        receipt.update(guard_identity=identity, remote_guard_source_sha256=sha256(guard_path))
        write_json(receipt_path, receipt)
        output = cli.require(["exec", "--session", session, "--file", wsl_path(guard_path), "--timeout", "60"], 90)
        typed = re.findall(r"AUGMIX_GUARD_JSON=(\{[^\n]+\})", output)
        native = re.findall(r"SABER_GUARD_JSON=(\{[^\n]+\})", output)
        require(bool(typed) and bool(native) and json.loads(typed[-1]) == identity, "Exact AugMix guard identity/hash not verified")
        hardware = json.loads(native[-1])
        require(hardware.get("status") == "armed" and hardware.get("deadline_unix") == deadline and
            hardware.get("device") == "Tesla T4" and hardware.get("amp_dtype") == "float16", "Native T4 self-release guard differs")
        receipt.update(remote_guard_verified=True, runtime_preflight=hardware)
        usage = parse_usage(cli.require(["usage"], 30))
        validate_resource_usage(usage, expected_assignments=1)
        receipt.update(status="uploading", usage_after_assignment=usage)
        write_json(receipt_path, receipt)
        for part in manifest["chunks"]:
            local = folder/part["name"]
            require(sha256(local) == part["sha256"], "Input chunk changed before upload")
            cli.require(["upload", "--session", session, wsl_path(local), "content/"+part["name"]], bounded_timeout(deadline, 180, 180))
        program = folder/"remote_program.py"
        program.write_text(remote_program(manifest, session, deadline, receipt["input_manifest_sha256"]), encoding="utf-8")
        receipt.update(status="training_and_benchmarking", remote_program_sha256=sha256(program))
        write_json(receipt_path, receipt)
        remaining = int(deadline-time.time()-150)
        if remaining < 120:
            raise TimeoutError("No supervised AugMix execution window remains")
        output = cli.require(["exec", "--session", session, "--file", wsl_path(program), "--timeout", str(remaining)], remaining+30)
        matches = re.findall(r"AUGMIX_RESULT_JSON=(\{[^\n]+\})", output)
        require(bool(matches), "Remote execution outcome is unknown; do not retry")
        remote = json.loads(matches[-1])
        require(remote.get("input_bundle_sha256") == manifest["bundle_sha256"] and
            remote.get("input_manifest_sha256") == receipt["input_manifest_sha256"] and
            remote.get("host_source_sha256") == manifest["host_source_sha256"] and
            remote.get("source_files_sha256") == manifest["files_sha256"] and remote.get("guard_identity") == identity,
            "Remote source/input/guard provenance differs")
        receipt["remote_status"] = remote["status"]
        artifact = folder/"artifacts.private.zip"
        require(bool(remote.get("artifact_chunks")), "Remote artifact chunks are missing")
        with artifact.open("xb") as stream:
            for index, part in enumerate(remote["artifact_chunks"]):
                require(part.get("name") == f"augmix_artifacts_{index:03d}.part" and type(part.get("bytes")) is int and
                    0 < part["bytes"] <= CHUNK_BYTES, "Output chunk sequence/size differs")
                local = folder/part["name"]
                require(not local.exists(), "Existing download evidence must not be replaced")
                cli.require(["download", "--session", session, "content/"+part["name"], wsl_path(local)], bounded_timeout(deadline, 90, 5))
                require(local.stat().st_size == part["bytes"] and sha256(local) == part["sha256"], "Downloaded artifact chunk changed")
                with local.open("rb") as source:
                    while block := source.read(1024*1024):
                        stream.write(block)
        extract_artifacts(artifact, folder, remote)
        validate_package(ROOT, folder)
        qualified = False
        if remote["status"] == "complete":
            qualified = validate_downloaded(folder, manifest, receipt["runtime_preflight"])
            require(remote.get("benchmark_passed") is qualified, "Remote gate differs from downloaded evidence")
        receipt.update(status="complete" if remote["status"] == "complete" else "remote_failed",
            artifact_sha256=remote["artifact_sha256"], benchmark_passed=qualified)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if receipt["assignment_attempts"]:
            try:
                receipt.update(cleanup_once(cli, folder, receipt))
            except Exception as error:
                receipt.update(runtime_stop_verified=False, cleanup_error_type=type(error).__name__)
        else:
            receipt["runtime_stop_verified"] = True
        receipt["finished_at"] = now().isoformat()
        if receipt["status"] == "complete" and not receipt["runtime_stop_verified"]:
            receipt["status"] = "cleanup_unverified"
        write_json(receipt_path, receipt)
        if receipt["runtime_stop_verified"] and lease_path.exists():
            owner = json.loads(lease_path.read_text())
            if all(owner.get(k) == receipt[k] for k in ("session", "pid", "process_created_at")):
                lease_path.unlink()
        if guard and guard.poll() is None:
            guard.terminate(); guard.wait(timeout=10)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--watchdog", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--created", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog, args.parent, args.created, args.distro)
        return
    if not args.run or not args.config:
        parser.error("--run and --config are required")
    common.validate_run(args.run)
    folder = ROOT/"results/research"/("colab_"+args.run)
    if folder.exists():
        manifest = validate_package(ROOT, folder)
        if not args.launch or sha256(args.config) != manifest["raw_config_sha256"]:
            raise FileExistsError("Prepared AugMix directory exists; inspect it and launch only once")
    else:
        private_directory(folder)
        try:
            prepare(ROOT, args.config, folder, args.run)
        except BaseException:
            write_json(folder/"preparation_failed.json", {"status": "failed", "at": now().isoformat()})
            raise
    if args.launch:
        launch(folder, args.distro)
    print(json.dumps({"run": args.run, "prepared": True, "launch_requested": args.launch}))


if __name__ == "__main__":
    main()
