"""Explicit local scratch-KD control26/primary26 preflight or formal job.

No arguments show help. No auto-formal, repeat27, resume, prediction or upload.
The host is pure stdlib/psutil; own OSJob authentication precedes all numerics.
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
import signal
import statistics
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research import preflight_core_control as c
from research import profile_augmix as p
from research import train_scratch_kd as engine
from research import scratch_kd_lineage as lineage
SCHEMA="research_scratch_kd_local_supervisor_v1"
INPUT_SCHEMA="research_scratch_kd_local_inputs_v1"
PREF_SCHEMA="research_scratch_kd_native_preflight_v1"
FORMAL_SCHEMA="research_scratch_kd_local_result_v1"
PLAN="research/scratch-kd-controller-plan.md"
HARD_STOP=datetime(2026,11,5,5,tzinfo=timezone.utc)
ROLES={"control26":{"alpha":0.,"engine_role":"development_control","config":"research/configs/wrn16_scratch_kd_control.json"},
       "primary26":{"alpha":.5,"engine_role":"primary","config":"research/configs/wrn16_scratch_kd.json"}}
PINS={'data.py': '4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b', 'model.py': '79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848', 'research/augmentations.py': '718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873', 'research/benchmark.py': 'd30acccd66a18f8de198824a57b565cb8e16cd7044fa1cec6608007c80d37f16', 'research/benchmark_scratch_kd.py': 'edc6c58fcc9b46e6bc4b375a820a5775f5242506874791fb230d738a806f250b', 'research/benchmark_shadow.py': 'aab840a944125f45ef070099e70270103c62a5acabf081ead7f26cfdb6f7d6c9', 'research/configs/wrn16_scratch_kd.json': 'f86296652b78cfd7f1ab53fe517d3331acbc780c5b26ee1e5f5d861cb461fcf5', 'research/configs/wrn16_scratch_kd_control.json': '33b8954368c9eeeb42a17a43c06d6deb71677acd71f32b4e99adb55efb17f109', 'research/configs/wrn16_scratch_kd_seed2027.json': '80659bf6f441cd12d7d9dad79b97f9098a6b4b93e9d6de3f0d57feec2144ebef', 'research/configs/wrn34_3.json': '2901f0a7d717083c8811aac9cff1910c7a7493eb3c474c260bb9b7f7eb45df02', 'research/incumbent-comparison-plan.md': '97269ed9f6144393084cde6709f6c8fd0c8cc316d8f3e920741462ebebde1f7d', 'research/incumbent-promotion-plan.md': '10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7', 'research/incumbent_compare.py': '277f7803d855a90a5bd92e582e2beda85cb18557402d9a4332428d2b70a49d0d', 'research/paired_compare.py': '034c850037b71ef151ad1f3f168afb3c475f170ad2a436d70e406e329fba5e65', 'research/paired_compare_v2.py': '7218d65bd0f33ae6e3563763730d6ce161a9e90131cdfb21c9feb10ad5bb5199', 'research/preflight_core_control.py': '85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761', 'research/process_tree_guard.py': '5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80', 'research/profile_augmix.py': '82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66', 'research/run_local.py': '6c42cfd08aaad6b39c1acd23ed788044a68c31b047a677a0218308a1dd15a069', 'research/scratch-distillation-followup.md': 'b6c916bd778e1042cfcd6a0f0bacb85ec66fc014ef282ab6204be4a4727b3d8f', 'research/scratch-distillation-plan.md': 'a2e9aec6f6a00422958505d37a3cea21f5b48e6ae3f6dddec906910adf196c1b', 'research/scratch-kd-training-plan.md': '10ceaffa600d7598d5183698b790d7db80aba9a3289e8248bcbe770ae1de98e0', 'research/scratch_kd.py': '7bb624fd4fa2e6f3989b03e36c659ff69488b4945799c353ac329004e626c81b', 'research/scratch_kd_lineage.py': 'ed430a87186c72352bef4f470cc0d5a5d435f32d2dcef071918688012dfe2e0e', 'research/train_scratch_kd.py': 'bd2b3c9c0f97ad9aa3d499a2033b3bba6c7b8c6805018dc9a17ef6aec55507a9', 'research/train_shadow.py': '9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061', 'research/wrn16-5-plan.md': '1d7485c64f4911defad68ecc0b5d3b14ac4e35c65f3dfc68e07734404c558559', 'research/wrn34-3-300-bf16-plan.md': '2678357b601d7cfcbb2eb6efc9920633920601b66f6551cba3a6b79c1384fd2e', 'runtime.py': '75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4', 'train.py': '4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be'}
BASELINE="results/research/baseline_v3_verified.json"
BASELINE_SHA="eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d"
PREF_ARTIFACT="scratch_kd_preflight.json"
FORMAL_ARTIFACT="scratch_kd_result.json"
require,read,write,sha=c.require,c.read,c.write,c.sha256
TIMING_SCOPE="Actual shared setup/official40k residency and H2D charged to supervisor; whole shared step includes original GPU gather/augment/Mixup, student44grad/210state/13BN, rollback model+EMA+SGD+priorgradient+allRNG snapshots, directSGD/EMA/metrics, teacher191state/31BN immutable finite/no-grad/allRNG proofs and positive-alpha teacherforward;3warmup+20timed steps."


def source_snapshot():
    actual={name:sha(ROOT/name) for name in PINS}
    require(actual==PINS,"Frozen KD learning/reference/guard/accounting dependencies changed")
    return {**actual,PLAN:sha(ROOT/PLAN),"research/run_scratch_kd.py":sha(__file__),
        "tests/test_scratch_kd_controller.py":sha(ROOT/"tests/test_scratch_kd_controller.py"),"research/campaign.json":sha(ROOT/"research/campaign.json")}


def configuration(role):
    require(role in ROLES,"Only fixed control26 or primary26 supported")
    value=ROLES[role];raw=(ROOT/value["config"]).read_bytes();config=engine.fixed_config(value["alpha"],2026)
    engine.input_config(config,raw)
    require(sha(ROOT/value["config"])==PINS[value["config"]],"Exact raw treatment config changed")
    return config


def stamp(value):
    require(isinstance(value,str),"Aware evidence timestamp required")
    result=datetime.fromisoformat(value);require(result.tzinfo is not None,"Aware evidence timestamp required");return result


def finite(value,positive=False):return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def guard_seconds(median):
    require(finite(median,True),"Finite positive measured median required")
    return 1.5*median*313*160+300


def budget(policy,current,used,mode,preflight=None):
    require(current.tzinfo is not None and finite(used) and mode in ("preflight","formal"),"Valid captured time/accounting/mode required")
    require(stamp(policy["stops_at"])==HARD_STOP and current<HARD_STOP and policy.get("concurrent_local_gpu_jobs")==1
        and policy.get("local_gpu_seconds_per_utc_day")==21600 and policy.get("local_job_timeout_seconds")==10800,"Original budget/hard stop contract changed")
    available=min(10800,21600-used,(HARD_STOP-current).total_seconds())
    if mode=="preflight":require(preflight is None and available>=180,"Full180s fresh engineering reservation required");return 180
    require(preflight is not None and available>=guard_seconds(preflight["median_step_seconds"]),"Own actual preflight exceeds job/day/hard-stop budget")
    return available


def current_usage(account,base,current):
    day=current.astimezone(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    return account["used_seconds"](base,day,day+timedelta(days=1))


def base_binding(data_dir,sources):
    paths,identity=c.official_inputs(Path(data_dir));reference=lineage.build_teacher_reference(data_dir=Path(data_dir))
    require(sha(ROOT/BASELINE)==BASELINE_SHA,"Exact trusted native original baseline required")
    paths+=[ROOT/BASELINE,*(Path(name) for name in reference["evidence_sha256"]),*(ROOT/name for name in sources)]
    return {str(path.resolve()):sha(path) for path in paths},identity,reference


def env_context(env):
    # DLL PATH-filter removal count is process-local housekeeping, not a runtime.
    return {key:value for key,value in env.items() if key!="process_local_cudnn_path_entries_removed"}


def validate_environment(env):
    expected={"device":"cuda","gpu":"NVIDIA GeForce RTX 5060 Laptop GPU","python_version":"3.14.3","cuda_runtime":"13.0",
        "cudnn":92000,"amp_dtype":"torch.bfloat16","cudnn_benchmark":True,"deterministic_algorithms":False}
    require(isinstance(env,dict) and all(type(env.get(k)) is type(v) and env[k]==v for k,v in expected.items())
        and isinstance(env.get("python"),str) and Path(env["python"]).is_absolute() and isinstance(env.get("platform"),str) and bool(env["platform"])
        and finite(env.get("gpu_memory_GiB"),True) and type(env.get("process_local_cudnn_path_entries_removed")) is int and env["process_local_cudnn_path_entries_removed"]>=0
        and isinstance(env.get("packages"),dict) and env["packages"].get("torch")=="2.12.0+cu130"
        and set(env["packages"])=={"torch","torchvision","numpy","Pillow","matplotlib","nbformat","reportlab"}
        and all(value is None or isinstance(value,str) and bool(value) for value in env["packages"].values()),"Exact actual native local runtime required")
    return env_context(env)


def inactive(pid,created):
    require(type(pid) is int and pid>0 and finite(created,True),"Original process PID/creation identity required")
    try:process=psutil.Process(pid)
    except psutil.NoSuchProcess:return True
    return abs(process.create_time()-created)>=.01

def validate_worker(folder, manifest_sha):
    """Authenticate the real OS boundary before importing runtime/Torch/data."""
    folder = Path(folder).resolve()
    require(folder.parent == (ROOT/"results/research").resolve() and re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
            sha(folder/"inputs.json") == manifest_sha, "Controlled immutable preflight inputs required")
    manifest, receipt = read(folder/"inputs.json"), read(folder/"supervisor.json")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == ("benchmarking" if manifest.get("mode")=="preflight" else "training") and
            receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
            finite(receipt.get("reserved_seconds"),True) and receipt["reserved_seconds"] <= (180 if manifest.get("mode")=="preflight" else 10800) and (manifest.get("mode")!="preflight" or receipt["reserved_seconds"]==180) and
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
            spec.get("script_sha256") == ready.get("script_sha256") == sha(__file__) and
            spec.get("guard_source_sha256") == sha(guard) == PINS["research/process_tree_guard.py"] and
            spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha] and
            ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
            type(spec.get("deadline_unix")) in (int, float) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
            spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "Actual guard identity/source/deadline differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and Path(command[1]).resolve() == guard and
            command[2] == "--bootstrap" and Path(command[3]).resolve() == directory/"spec.json", "Actual guarded bootstrap required")
    from research.profile_augmix import verify_os_boundary
    verify_os_boundary(spec, ready, process)
    return validate_live_binding(folder, manifest_sha, manifest, receipt)


def validate_live_binding(folder,manifest_sha,manifest,receipt):
    sources = source_snapshot()
    mode,role=manifest.get("mode"),manifest.get("role")
    require(mode in ("preflight","formal") and role in ROLES,"Own fixed mode/role required")
    expected,identity,reference=stage_binding(Path(manifest["data_dir"]),sources,mode,role,manifest.get("control_preflight"),manifest.get("kd_preflight"),manifest.get("control_run"))
    require(manifest.get("schema")==INPUT_SCHEMA and manifest.get("run")==receipt.get("run")==folder.name and manifest.get("seed")==2026
        and type(manifest["seed"]) is int and receipt.get("mode")==mode and receipt.get("role")==role
        and receipt.get("backend")=="local" and receipt.get("seed")==2026 and type(receipt.get("seed")) is int and receipt.get("precision")=="bf16"
        and receipt.get("config_sha256")==manifest.get("config_sha256") and receipt.get("formal_training_admitted") is (mode=="formal")
        and receipt.get("engine_role")==ROLES[role]["engine_role"] and manifest.get("effective_config")==configuration(role)
        and manifest.get("effective_config_sha256")==c.json_sha(configuration(role)) and manifest.get("config_sha256")==PINS[ROLES[role]["config"]]
        and manifest.get("teacher_reference")==reference and manifest.get("source_sha256")==receipt.get("source_sha256")==sources
        and manifest.get("data_identity")==identity and manifest.get("input_sha256")==receipt.get("input_sha256")==expected
        and receipt.get("confirmation_evaluated") is False and receipt.get("hidden_images_loaded") is False
        and isinstance(manifest.get("source_git_commit"),str) and re.fullmatch(r"[0-9a-f]{40}",manifest["source_git_commit"]),"Actual role/config/source/input/reference stage binding differs")
    if mode=="formal":
        reports=cohort(manifest["control_preflight"],manifest["kd_preflight"],sources)
        require(guard_seconds(reports[role]["median_step_seconds"])<=receipt["reserved_seconds"],"Own role actual guard exceeds formal reservation")
    return manifest


def completed_guard_paths(folder,receipt,manifest_sha):
    """Audit the actual closed native boundary, not just its supervisor flag."""
    from research.process_tree_guard import _valid_child_record
    records=receipt.get("process_tree_guard_results")
    require(type(records) is list and len(records)==1, "Exactly one actual closed preflight OS guard required")
    proof=records[0]
    require(type(proof) is dict and proof.get("schema")=="owned_python_process_tree_v1" and
              proof.get("cleanup_verified") is True and proof.get("timed_out") is False and
              proof.get("completion_authenticated") is True and type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"]==0 and
              proof.get("method") in ("windows_job_v1","posix_session_v1") and
              type(proof.get("child_pid")) is int and proof["child_pid"]==receipt.get("child_pid") and
              type(proof.get("child_created_at")) in (int,float) and math.isfinite(proof["child_created_at"]) and
              abs(proof["child_created_at"]-receipt.get("child_created_at",0))<.01 and
              type(proof.get("child_returncode")) is int and
              proof["child_returncode"] in ((0,) if proof["method"]=="windows_job_v1" else (0,-signal.SIGKILL)) and
              proof.get("guard_source_sha256")==PINS["research/process_tree_guard.py"],
              "Authenticated successful native cleanup/completion evidence required")
    directory=Path(proof.get("guard_directory","")).resolve()
    nonce=directory.name.removeprefix("process_tree_")
    require(directory.parent==Path(folder).resolve() and directory.name=="process_tree_"+nonce and
              re.fullmatch(r"[0-9a-f]{32}",nonce) and proof.get("nonce")==nonce,
              "Actual controlled unique preflight guard directory required")
    paths=[directory/name for name in ("spec.json","ready.json","completion.json","host_result.json")]
    spec,ready,completion,host=map(read,paths)
    require(host==proof and spec.get("schema")=="owned_python_process_tree_v1" and spec.get("nonce")==nonce and
              spec.get("host_pid")==receipt.get("pid") and type(spec.get("host_created_at")) in (int,float) and
              abs(spec["host_created_at"]-receipt.get("process_created_at",0))<.01 and
              Path(spec.get("cwd","")).resolve()==ROOT.resolve() and Path(spec.get("script","")).resolve()==Path(__file__).resolve() and
              spec.get("script_sha256")==sha(__file__) and
              spec.get("guard_source_sha256")==PINS["research/process_tree_guard.py"] and
              spec.get("arguments")==["--_worker",str(Path(folder).resolve()),"--_manifest-sha",manifest_sha] and
              type(spec.get("deadline_unix")) in (int,float) and math.isfinite(spec["deadline_unix"]) and
              stamp(receipt["started_at"]).timestamp()<spec["deadline_unix"]<=receipt.get("deadline_unix",0)+.1 and
              _valid_child_record(ready,spec,proof["child_pid"],proof["child_created_at"]) and
              _valid_child_record(completion,spec,proof["child_pid"],proof["child_created_at"]) and
              ready.get("method")==completion.get("method")==proof["method"] and
              type(completion.get("exit_code")) is int and completion["exit_code"]==0 and
              spec.get("job_name")==("Local\\SaberAugMix_"+nonce if proof["method"]=="windows_job_v1" else None),
              "Actual nonce/source/argv/identity/exit-zero native records differ")
    if proof["method"]=="posix_session_v1":
        watchdog=directory/"watchdog_ready.json";watch=read(watchdog)
        require(_valid_child_record(watch,spec,ready.get("watchdog_pid"),ready.get("watchdog_created_at")) and
                  watch.get("leader_pid")==proof["child_pid"] and
                  abs(watch.get("leader_created_at",0)-proof["child_created_at"])<.01,
                  "Actual authenticated POSIX watchdog proof required")
        paths.append(watchdog)
    return paths


def owned_folder(value):
    folder=Path(value).resolve()
    require(folder.parent==(ROOT/"results/research").resolve() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",folder.name)
        and not folder.name.startswith("stability"),"Controlled unique local folder required")
    return folder


def validate_preflight_report(report,manifest,digest):
    role=manifest["role"];config=configuration(role);teacher_pass=23 if role=="primary26" else 0
    fixed={"schema":PREF_SCHEMA,"status":"complete","mode":"preflight","role":role,"seed":2026,"epochs":160,
        "engineering_only":True,"accuracy_evaluated":False,"formal_training_admitted":False,"confirmation_evaluated":False,
        "hidden_images_loaded":False,"external_data_loaded":False,"native_bf16_without_emulation":True,"training_precision":"bf16",
        "parameter_count":4289754,"trainable_parameter_count":4289754,"teacher_parameters":4166298,
        "teacher_state_tensors":191,"teacher_bn_modules":31,"gradient_tensor_count":44,"state_tensor_count":210,"bn_modules":13,
        "master_parameter_dtype":"torch.float32","ema_parameter_dtype":"torch.float32","grad_scaler_enabled":False,"gradient_clipping":False,
        "batch_size":128,"training_pool_examples":40000,"evaluation_examples":0,"warmup_updates":3,"timed_updates":20,
        "successful_updates":23,"skipped_updates":0,"source_examples_seen":2944,"unique_source_examples":2944,
        "student_forward_passes":23,"teacher_forward_passes":teacher_pass,"student_inputs_seen":2944,"teacher_inputs_seen":2944 if teacher_pass else 0,
        "planned_formal_attempted_updates":50080,"bn_forward_counts":[23]*13,"ema_buffer_copies":0,"preflight_state_discarded":True,
        "source_sha256":manifest["source_sha256"],"config_sha256":manifest["config_sha256"],"effective_config_sha256":c.json_sha(config),
        "input_manifest_sha256":digest,"teacher_reference_sha256":engine.json_sha(manifest["teacher_reference"]),"data_identity":manifest["data_identity"],"timing_scope":TIMING_SCOPE}
    require(all(type(report.get(k)) is type(v) and report[k]==v for k,v in fixed.items()),"Complete own actual native scratch-KD role proof required")
    rows,durations=report.get("steps"),report.get("step_seconds")
    require(isinstance(rows,list) and isinstance(durations,list) and len(rows)==len(durations)==23 and all(finite(v,True) for v in durations),"All23 full-step timings required")
    for index,row in enumerate(rows,1):
        expected={"updates":index,"gradient_tensor_count":44,"state_tensor_count":210,"source_examples":128,
            "student_forward_passes":1,"teacher_forward_passes":int(role=="primary26"),"teacher_inputs":128 if role=="primary26" else 0,
            "bn_counters_after":[index]*13,"teacher_immutable_verified":True,"teacher_eval_no_grad":True,"rollback_snapshots_in_timer":True}
        expected["logits_dtype"]="torch.bfloat16"
        require(all(type(row.get(k)) is type(v) and row[k]==v for k,v in expected.items()) and type(row.get("mixed")) is bool and
            all(finite(row.get(k),True) for k in ("loss","hard_loss","gradient_norm","parameter_probe_max_delta","ema_probe_max_delta","lr")),"Complete finite true student/teacher update proof required")
        require(row["lr"]==.1*index/(5*313) and (finite(row.get("soft_loss")) if role=="primary26" else row.get("soft_loss") is None),"Original warmup/KD role diagnostics differ")
    median=statistics.median(durations[3:]);validate_environment(report.get("environment"))
    require(report.get("environment_sha256")==engine.json_sha(report["environment"]) and finite(report.get("median_step_seconds"),True) and report["median_step_seconds"]==median
        and finite(report.get("conservative_pipeline_seconds"),True) and report["conservative_pipeline_seconds"]==guard_seconds(median)
        and finite(report.get("full_step_seconds"),True) and report["full_step_seconds"]==sum(durations)
        and finite(report.get("setup_seconds"),True) and type(report.get("peak_gpu_memory_bytes")) is int and report["peak_gpu_memory_bytes"]>0,"Native environment and exact whole-pipeline accounting required")
    for key in ("ordered_training_indices_sha256","ordered_official_ids_sha256","initial_student_state_sha256"):
        require(isinstance(report.get(key),str) and re.fullmatch(r"[0-9a-f]{64}",report[key]),"Actual private source-order hash required")


def completed_envelope(folder,sources,mode,role):
    folder=owned_folder(folder);manifest=read(folder/"inputs.json");receipt=read(folder/"supervisor.json");digest=sha(folder/"inputs.json")
    name=PREF_ARTIFACT if mode=="preflight" else FORMAL_ARTIFACT;artifact=read(folder/name)
    require(manifest.get("schema")==INPUT_SCHEMA and manifest.get("mode")==mode and manifest.get("role")==role and manifest.get("run")==folder.name
        and receipt.get("schema")==SCHEMA and receipt.get("mode")==mode and receipt.get("role")==role and receipt.get("engine_role")==ROLES[role]["engine_role"]
        and receipt.get("backend")=="local" and receipt.get("status")=="complete" and receipt.get("process_tree_cleanup_verified") is True
        and receipt.get("cleanup_requires_reconciliation") is not True and receipt.get("run")==folder.name and receipt.get("input_manifest_sha256")==digest
        and receipt.get("artifact")==name and receipt.get("artifact_sha256")==sha(folder/name) and receipt.get("source_sha256")==manifest.get("source_sha256")==sources
        and manifest.get("effective_config")==configuration(role) and manifest.get("config_sha256")==PINS[ROLES[role]["config"]]
        and manifest.get("effective_config_sha256")==c.json_sha(configuration(role)) and type(manifest.get("seed")) is int and manifest["seed"]==2026
        and receipt.get("precision")=="bf16" and type(receipt.get("seed")) is int and receipt["seed"]==2026 and receipt.get("config_sha256")==manifest["config_sha256"]
        and receipt.get("confirmation_evaluated") is False and receipt.get("hidden_images_loaded") is False
        and receipt.get("formal_training_admitted") is (mode=="formal"),"Own actual complete immutable role receipt required")
    started,finished=stamp(receipt["started_at"]),stamp(receipt["finished_at"]);duration=(finished-started).total_seconds();reserved=receipt.get("reserved_seconds")
    require(finite(reserved,True) and reserved<=(180 if mode=="preflight" else 10800) and (mode!="preflight" or reserved==180)
        and 0<duration<=reserved+.1 and finished.timestamp()<=time.time()+.1 and finite(receipt.get("elapsed_seconds"),True)
        and receipt["elapsed_seconds"]<=reserved+.1 and abs(receipt["elapsed_seconds"]-duration)<=5
        and finite(receipt.get("deadline_unix"),True) and started.timestamp()<receipt["deadline_unix"]<=started.timestamp()+reserved+.1,
        "Closed own reservation/duration/deadline accounting differs")
    require(inactive(receipt.get("pid"),receipt.get("process_created_at")) and inactive(receipt.get("child_pid"),receipt.get("child_created_at")),"Previous owner/child identities remain active")
    require(isinstance(manifest.get("input_sha256"),dict) and bool(manifest["input_sha256"]) and receipt.get("input_sha256")==manifest["input_sha256"]
        and all(sha(path)==value for path,value in manifest["input_sha256"].items()),"Completed source/input bytes changed")
    paths=[folder/"inputs.json",folder/"supervisor.json",folder/name,*map(Path,manifest["input_sha256"]),*completed_guard_paths(folder,receipt,digest)]
    return artifact,manifest,receipt,paths


def validate_preflight_bundle(folder,sources,role):
    report,manifest,receipt,paths=completed_envelope(folder,sources,"preflight",role)
    require(manifest.get("control_preflight") is None and manifest.get("kd_preflight") is None and manifest.get("control_run") is None,
        "Engineering is a fresh fixed role, not a reused formal bundle")
    expected,identity,reference=base_binding(Path(manifest["data_dir"]),sources)
    require(manifest["input_sha256"]==expected and manifest.get("data_identity")==identity and manifest.get("teacher_reference")==reference,
        "Exact base-stage official inputs/source/teacher required")
    validate_preflight_report(report,manifest,sha(Path(folder)/"inputs.json"))
    require(not list(Path(folder).glob("*.pt")),"All engineering state must be discarded")
    require(guard_seconds(report["median_step_seconds"])<=10800,"Numerically valid but timing-rejected preflight cannot admit formal")
    return report,paths


def cohort(control_preflight,kd_preflight,sources):
    require(control_preflight is not None and kd_preflight is not None and Path(control_preflight).resolve()!=Path(kd_preflight).resolve(),"Two independent actual role preflights required")
    reports={};paths=[]
    for role,path in (("control26",control_preflight),("primary26",kd_preflight)):
        report,extra=validate_preflight_bundle(owned_folder(path),sources,role);reports[role]=report;paths+=extra
    require(env_context(reports["control26"]["environment"])==env_context(reports["primary26"]["environment"]),"Both fixed roles require one actual native runtime cohort")
    require(all(reports["control26"][key]==reports["primary26"][key] for key in
        ("initial_student_state_sha256","ordered_training_indices_sha256","ordered_official_ids_sha256")),"Control/KD actual initial student and input order must match")
    return reports


def validate_control_bundle(folder,sources):
    result,manifest,receipt,paths=completed_envelope(folder,sources,"formal","control26")
    require(manifest.get("control_run") is None,"A control cannot borrow a different control")
    expected,identity,reference=stage_binding(Path(manifest["data_dir"]),sources,"formal","control26",manifest.get("control_preflight"),manifest.get("kd_preflight"),None)
    require(expected==manifest["input_sha256"] and manifest.get("teacher_reference")==reference and manifest.get("data_identity")==identity,"Actual complete control-stage inputs differ")
    folder=Path(folder).resolve();benchmark=read(folder/"benchmark.json");integrity=benchmark.get("research_training_integrity",{})
    require(result.get("schema")==FORMAL_SCHEMA and result.get("status")=="complete" and result.get("role")=="control26"
        and result.get("engine_role")=="development_control" and result.get("source_sha256")==sources and result.get("input_manifest_sha256")==sha(folder/"inputs.json")
        and result.get("checkpoint_sha256")==sha(folder/"best.pt") and result.get("benchmark_sha256")==sha(folder/"benchmark.json")
        and result.get("teacher_reference_sha256")==engine.json_sha(reference) and result.get("epochs_completed")==160
        and result.get("config")==configuration("control26") and result.get("config_sha256")==PINS[ROLES["control26"]["config"]]
        and result.get("confirmation_evaluated") is False and result.get("hidden_images_loaded") is False
        and result.get("prediction_authorized") is False and result.get("submission_authorized") is False,
        "Actual own newly completed alpha0 control result required")
    require(benchmark.get("schema")=="official_development_stress_v3" and benchmark.get("status")=="complete" and benchmark.get("examples")==5000
        and benchmark.get("training_examples")==40000 and benchmark.get("parameter_count")==4289754 and benchmark.get("config")==configuration("control26")
        and benchmark.get("checkpoint_sha256")==sha(folder/"best.pt") and benchmark.get("scratch_kd_benchmark_source_sha256")==PINS["research/benchmark_scratch_kd.py"]
        and benchmark.get("training_metrics_sha256")==sha(folder/"metrics.json") and benchmark.get("training_history_sha256")==sha(folder/"history.json")
        and benchmark.get("benchmark_source_sha256")==PINS["research/benchmark.py"]
        and integrity.get("schema")=="research_scratch_kd_integrity_v1" and integrity.get("role")=="development_control"
        and integrity.get("actual_selected_last_optimizer_parent_RNG_tensors_audited") is True and integrity.get("actual_teacher_lineage_audited") is True
        and integrity.get("teacher_weights_embedded") is False,"Actual typed full control audit must precede KD formal")
    marker=engine.research_provenance(configuration("control26"),(folder/"input_config.json").read_bytes(),reference)
    require(integrity.get("training")==marker and all(type(integrity.get(k)) is int and integrity[k]==v for k,v in
        {"successful_updates":50080,"skipped_updates":0,"student_forward_passes":50080,"teacher_forward_passes":0,"source_examples_seen":6400000,"student_inputs_seen":6400000,"teacher_inputs_seen":0}.items()),"Full control update/source/teacher ledger differs")
    core=read(ROOT/BASELINE)
    require(benchmark.get("evaluation_context")==core["evaluation_context"] and benchmark.get("data_signature")==core["data_signature"]
        and benchmark.get("bn_recalibrated") is False and set(benchmark.get("policies",{}))=={"plain","flip","center80_shift20"},"Full same-context control benchmark required")
    for policy,row in benchmark["policies"].items():
        require(set(row.get("cases",{}))==set(core["policies"][policy]["cases"]),"All6 control cases required")
        for case,value in row["cases"].items():
            require(type(value.get("examples")) is int and value["examples"]==5000 and type(value.get("accuracy")) in (int,float) and math.isfinite(value["accuracy"]) and 0<=value["accuracy"]<=1
                and abs(value["accuracy"]*5000-round(value["accuracy"]*5000))<=1e-8 and finite(value.get("loss"))
                and value.get("input_pixel_sha256")==core["policies"][policy]["cases"][case]["input_pixel_sha256"],"Actual complete control pixels/counts/finite metrics required")
        require(row.get("clean_accuracy")==row["cases"]["clean"]["accuracy"] and row.get("clean_loss")==row["cases"]["clean"]["loss"]
            and finite(row.get("stress_accuracy")) and math.isclose(row["stress_accuracy"],sum(v["accuracy"] for k,v in row["cases"].items() if k!="clean")/5,rel_tol=0,abs_tol=1e-12),"Actual control aggregate differs from18case evidence")
    require(isinstance(benchmark.get("evidence_sha256"),dict) and bool(benchmark["evidence_sha256"])
        and all(sha(path)==value for path,value in benchmark["evidence_sha256"].items()),"Actual typed control artifact closure changed")
    paths+=[folder/name for name in ("best.pt","last.pt","benchmark.json","metrics.json","history.json","provenance.json","environment.json","input_config.json","config.json","teacher_reference.json")]
    paths+=map(Path,benchmark["evidence_sha256"])
    return result,paths


def stage_binding(data_dir,sources,mode,role,control_preflight=None,kd_preflight=None,control_run=None):
    binding,identity,reference=base_binding(data_dir,sources)
    if mode=="preflight":
        require(control_preflight is None and kd_preflight is None and control_run is None,"Preflight has base-stage inputs only")
    else:
        require(mode=="formal","Known mode required");cohort(control_preflight,kd_preflight,sources)
        extra=[]
        for r,path in (("control26",control_preflight),("primary26",kd_preflight)):
            _,paths=validate_preflight_bundle(Path(path),sources,r);extra+=paths
        if role=="primary26":
            require(control_run is not None,"KD formal requires its actual completed control")
            _,paths=validate_control_bundle(owned_folder(control_run),sources);extra+=paths
        else:require(role=="control26" and control_run is None,"Control formal does not consume future control evidence")
        binding.update({str(Path(path).resolve()):sha(path) for path in extra})
    return binding,identity,reference


def preflight_worker(folder,digest):
    setup_began=time.perf_counter();manifest=validate_worker(folder,digest)
    import runtime  # Own live OS identity/source binding was verified first.
    import numpy as np
    import torch
    config=manifest["effective_config"];bundle=engine.setup(config,Path(manifest["data_dir"]),"cuda",manifest["teacher_reference"])
    require(bundle.train_x.dtype==torch.uint8 and tuple(bundle.train_x.shape)==(40000,32,32,3)
        and bundle.train_y.numel()==40000 and len(bundle.val_ids)==5000 and len(bundle.train_ids)==40000
        and hashlib.sha256(bundle.train_ids.astype("<i8").tobytes()).hexdigest()==c.DATA_IDENTITY["train_indices_sha256"],"Actual complete original40k residency and source identities required")
    parameters,averages=list(bundle.model.parameters()),list(bundle.ema.parameters())
    bns=[layer for layer in bundle.model.modules() if isinstance(layer,torch.nn.BatchNorm2d)]
    require(len(parameters)==len(averages)==44 and len(bns)==13 and all(layer.num_batches_tracked.item()==0 for layer in bns)
        and all(value.dtype==torch.float32 for value in [*parameters,*averages]),"Actual scratch student/EMA/BN topology differs")
    require(len(bundle.teacher.state_dict())==191 and sum(isinstance(layer,torch.nn.BatchNorm2d) for layer in bundle.teacher.modules())==31,"Actual full teacher191/31 proof scope required")
    require(bundle.dtype==torch.bfloat16 and type(bundle.optimizer) is torch.optim.SGD and not hasattr(bundle,"grad_scaler"),"Native directSGD without scaler required")
    initial=hashlib.sha256()
    for name,value in bundle.model.state_dict().items():
        initial.update(name.encode());initial.update(str((tuple(value.shape),str(value.dtype))).encode());initial.update(value.detach().cpu().contiguous().numpy().tobytes())
    initial_hash=initial.hexdigest()
    order=engine.start_epoch(bundle,1);selected=bundle.train_ids[order[:2944].cpu().numpy()]
    official_ids=np.load(Path(manifest["data_dir"])/"official_train_ids.npy",allow_pickle=False)[selected]
    require(len(np.unique(selected))==2944,"Unique actual training sources required")
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup_began
    rows=[];durations=[]
    for index in range(23):
        torch.cuda.synchronize();began=time.perf_counter()
        before=parameters[0].detach().clone();before_ema=averages[0].detach().clone()
        result=engine.train_step(bundle.model,bundle.ema,bundle.optimizer,bundle.train_x,bundle.train_y,order,index,config,index,
            epoch=1,teacher=bundle.teacher,teacher_snapshot=bundle.teacher_snapshot,dtype=bundle.dtype)
        counters=[int(layer.num_batches_tracked.item()) for layer in bns]
        changed=(parameters[0]-before).abs().max().item();ema_changed=(averages[0]-before_ema).abs().max().item()
        row={key:result[key] for key in ("updates","gradient_tensor_count","state_tensor_count","source_examples","student_forward_passes","teacher_forward_passes","teacher_inputs","mixed","gradient_norm","lr")}
        row.update(loss=result["loss"].item(),hard_loss=result["hard_loss"].item(),soft_loss=None if result["soft_loss"] is None else result["soft_loss"].item(),
            parameter_probe_max_delta=changed,ema_probe_max_delta=ema_changed,bn_counters_after=counters,teacher_immutable_verified=True,
            teacher_eval_no_grad=True,rollback_snapshots_in_timer=True,logits_dtype=str(result["logits"].dtype))
        require(result["logits"].dtype==torch.bfloat16,"Actual student forward dtype differs")
        require(changed>0 and ema_changed>0 and counters==[index+1]*13,"Actual parameter/EMA changes and single-forward studentBN required")
        torch.cuda.synchronize();durations.append(time.perf_counter()-began);rows.append(row)
    env=runtime.environment();role=manifest["role"];teacher_pass=23 if role=="primary26" else 0
    report={"schema":PREF_SCHEMA,"status":"complete","mode":"preflight","role":role,"seed":2026,"epochs":160,"engineering_only":True,
        "accuracy_evaluated":False,"formal_training_admitted":False,"confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
        "native_bf16_without_emulation":True,"training_precision":"bf16","parameter_count":4289754,"trainable_parameter_count":4289754,
        "teacher_parameters":4166298,"teacher_state_tensors":191,"teacher_bn_modules":31,"gradient_tensor_count":44,"state_tensor_count":210,"bn_modules":13,
        "master_parameter_dtype":"torch.float32","ema_parameter_dtype":"torch.float32","grad_scaler_enabled":False,"gradient_clipping":False,
        "batch_size":128,"training_pool_examples":40000,"evaluation_examples":0,"warmup_updates":3,"timed_updates":20,"successful_updates":23,"skipped_updates":0,
        "source_examples_seen":2944,"unique_source_examples":2944,"student_forward_passes":23,"teacher_forward_passes":teacher_pass,"student_inputs_seen":2944,
        "teacher_inputs_seen":2944 if teacher_pass else 0,"planned_formal_attempted_updates":50080,"bn_forward_counts":[23]*13,"ema_buffer_copies":0,
        "preflight_state_discarded":True,"source_sha256":manifest["source_sha256"],"config_sha256":manifest["config_sha256"],"effective_config_sha256":c.json_sha(config),
        "input_manifest_sha256":digest,"teacher_reference_sha256":engine.json_sha(manifest["teacher_reference"]),"data_identity":manifest["data_identity"],
        "timing_scope":TIMING_SCOPE,"steps":rows,"step_seconds":durations,"median_step_seconds":statistics.median(durations[3:]),"full_step_seconds":sum(durations),
        "setup_seconds":setup_seconds,"conservative_pipeline_seconds":guard_seconds(statistics.median(durations[3:])),"environment":env,"environment_sha256":engine.json_sha(env),
        "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"ordered_training_indices_sha256":hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "ordered_official_ids_sha256":hashlib.sha256(np.ascontiguousarray(official_ids).tobytes()).hexdigest(),"initial_student_state_sha256":initial_hash}
    validate_preflight_report(report,manifest,digest);write(Path(folder)/PREF_ARTIFACT,report)


def formal_worker(folder,digest):
    manifest=validate_worker(folder,digest)
    import runtime
    from research import benchmark_scratch_kd as benchmark
    runtime.seed_all(2026);env=runtime.environment();validate_environment(env)
    reports=cohort(manifest["control_preflight"],manifest["kd_preflight"],manifest["source_sha256"])
    require(env_context(env)==env_context(reports[manifest["role"]]["environment"]),"Actual formal runtime differs from its role's measured native cohort")
    config=manifest["effective_config"];role=manifest["role"];folder=Path(folder)
    engine.train(config,folder,Path(manifest["data_dir"]),"cuda",teacher_reference=manifest["teacher_reference"],
        input_config_bytes=(ROOT/ROLES[role]["config"]).read_bytes(),role=ROLES[role]["engine_role"])
    control=Path(manifest["control_run"]) if role=="primary26" else None
    evidence=benchmark.run_benchmark(folder/"best.pt",folder/"benchmark.json",ROOT/BASELINE,Path(manifest["data_dir"]),
        control_benchmark=control/"benchmark.json" if control else None,control_checkpoint=control/"best.pt" if control else None)
    result={"schema":FORMAL_SCHEMA,"status":"complete","role":role,"engine_role":ROLES[role]["engine_role"],"seed":2026,"epochs_completed":160,
        "config":config,"config_sha256":manifest["config_sha256"],"source_sha256":manifest["source_sha256"],"input_manifest_sha256":digest,
        "checkpoint_sha256":sha(folder/"best.pt"),"benchmark_sha256":sha(folder/"benchmark.json"),"teacher_reference_sha256":engine.json_sha(manifest["teacher_reference"]),
        "successful_updates":50080,"skipped_updates":0,"source_examples_seen":6400000,"student_forward_passes":50080,"teacher_forward_passes":50080 if role=="primary26" else 0,
        "student_inputs_seen":6400000,"teacher_inputs_seen":6400000 if role=="primary26" else 0,"point_and_case_gates_passed":evidence["point_and_case_gates_passed"],
        "confirmation_evaluated":False,"hidden_images_loaded":False,"prediction_authorized":False,"submission_authorized":False,
        "control_run":manifest["control_run"],"control_is_not_a_submission_candidate":True}
    write(folder/FORMAL_ARTIFACT,result)


def worker(folder,digest):
    manifest=validate_worker(folder,digest)
    if manifest["mode"]=="preflight":preflight_worker(folder,digest)
    else:formal_worker(folder,digest)


def parent(args):
    require(args.role in ROLES and args.mode in ("preflight","formal") and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}",args.run or "")
        and not args.run.startswith("stability"),"Explicit unique mode/role/run required")
    sources=source_snapshot();config=configuration(args.role);data_dir=args.data_dir.resolve()
    binding,identity,reference=stage_binding(data_dir,sources,args.mode,args.role,args.control_preflight,args.kd_preflight,args.control_run)
    reports=cohort(args.control_preflight,args.kd_preflight,sources) if args.mode=="formal" else None
    account=c.accounting();policy=read(ROOT/"research/campaign.json");base=ROOT/"results/research";folder=base/args.run;lease=base/"local_lease.json"
    require(not folder.exists() and not lease.exists(),"Fresh run and no canonical lease required; unknown state is not reconciled here")
    current=account["now"]();budget(policy,current,current_usage(account,base,current),args.mode,reports[args.role] if reports else None)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();require(re.fullmatch(r"[0-9a-f]{40}",commit),"Actual source commit required")
    from research.process_tree_guard import run_guarded,may_release_lease
    c.claim_lease(lease);began=time.perf_counter();owns=False;receipt={}
    try:
        current=account["now"]();seconds=budget(policy,current,current_usage(account,base,current),args.mode,reports[args.role] if reports else None)
        started=current;folder.mkdir(parents=True,exist_ok=False);owns=True
        manifest={"schema":INPUT_SCHEMA,"run":args.run,"mode":args.mode,"role":args.role,"seed":2026,"data_dir":str(data_dir),"source_sha256":sources,
            "effective_config":config,"effective_config_sha256":c.json_sha(config),"config_sha256":PINS[ROLES[args.role]["config"]],"teacher_reference":reference,
            "input_sha256":binding,"data_identity":identity,"source_git_commit":commit,
            "control_preflight":str(args.control_preflight.resolve()) if args.control_preflight else None,"kd_preflight":str(args.kd_preflight.resolve()) if args.kd_preflight else None,
            "control_run":str(args.control_run.resolve()) if args.control_run else None}
        write(folder/"inputs.json",manifest);digest=sha(folder/"inputs.json")
        receipt={"schema":SCHEMA,"run":args.run,"mode":args.mode,"role":args.role,"engine_role":ROLES[args.role]["engine_role"],"backend":"local",
            "status":"benchmarking" if args.mode=="preflight" else "training","pid":os.getpid(),"process_created_at":psutil.Process().create_time(),
            "started_at":started.isoformat(),"reserved_seconds":seconds,"deadline_unix":started.timestamp()+seconds,"source_sha256":sources,
            "config_sha256":manifest["config_sha256"],"input_sha256":binding,"input_manifest_sha256":digest,"confirmation_evaluated":False,
            "hidden_images_loaded":False,"formal_training_admitted":args.mode=="formal","precision":"bf16","seed":2026}
        write(folder/"supervisor.json",receipt);remaining=receipt["deadline_unix"]-time.time()-2;require(remaining>0,"Setup exhausted owned reservation")
        run_guarded([sys.executable,str(Path(__file__).resolve()),"--_worker",str(folder.resolve()),"--_manifest-sha",digest],folder/"worker.log",remaining,folder/"supervisor.json",receipt,cwd=ROOT)
        name=PREF_ARTIFACT if args.mode=="preflight" else FORMAL_ARTIFACT;result=read(folder/name)
        if args.mode=="preflight":validate_preflight_report(result,manifest,digest)
        else:require(result.get("schema")==FORMAL_SCHEMA and result.get("status")=="complete" and result.get("role")==args.role and result.get("input_manifest_sha256")==digest
            and result.get("source_sha256")==sources and result.get("checkpoint_sha256")==sha(folder/"best.pt") and result.get("benchmark_sha256")==sha(folder/"benchmark.json"),"Actual complete own formal result required")
        require(source_snapshot()==sources and all(sha(path)==value for path,value in binding.items()) and time.time()<=receipt["deadline_unix"],"Bounded unchanged source/input result required")
        receipt.update(status="complete",artifact=name,artifact_sha256=sha(folder/name))
        if args.mode=="preflight":receipt["timing_guard_within_job_cap"]=guard_seconds(result["median_step_seconds"])<=10800
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
                if c.lease_can_release(owner,receipt):lease.unlink()
                else:owner.update(cleanup_requires_reconciliation=True,process_tree_cleanup_verified=False,supervisor=str(folder/"supervisor.json"));write(lease,owner)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run");parser.add_argument("--mode",choices=("preflight","formal"));parser.add_argument("--role",choices=tuple(ROLES))
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    for name in ("control-preflight","kd-preflight","control-run"):parser.add_argument("--"+name,type=Path)
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS);parser.add_argument("--_manifest-sha",help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if args._worker is not None:
        require(args.run is None and args.mode is None and args.role is None and args._manifest_sha is not None
            and args.control_preflight is None and args.kd_preflight is None and args.control_run is None,"Only own authenticated worker argv allowed")
        worker(args._worker,args._manifest_sha)
    elif all(getattr(args,key) is None for key in ("run","mode","role","control_preflight","kd_preflight","control_run","_manifest_sha")):parser.print_help()
    else:
        require(args.run is not None and args.mode is not None and args.role is not None and args._manifest_sha is None,"Explicit fixed mode/role/run required")
        parent(args)


if __name__=="__main__":main()
