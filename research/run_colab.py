"""Prepare and supervise one private Colab T4 development experiment (Issue 7).

Default is prepare-only.  ``--launch`` permits exactly one allocation attempt.
Credentials and CLI responses stay in ignored private files; this never mounts
Drive, installs packages, purchases units, reads hidden images or submits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
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
CORE = {
    "train.py": "4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py": "79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py": "4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py": "75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
}
BASELINE_SHA = "502979bc03301aa2f0273a7414f86799302b908a3abada622a88e4578c4b734f"
CACHE = ("official_train_images.npy", "official_train_labels.npy", "official_train_ids.npy",
         "official_split_indices.npz", "training_manifest.json")
SOURCES = (*CORE, "research/train_shadow.py", "research/augmentations.py", "research/benchmark.py",
           "research/benchmark_shadow.py")
RESIZEMIX_SOURCES = ("research/train_resizemix.py", "research/resizemix.py", "research/benchmark_resizemix.py")
OUTPUTS = ("baseline/benchmark.json", "candidate/best.pt", "candidate/last.pt", "candidate/metrics.json",
           "candidate/history.json", "candidate/config.json", "candidate/environment.json",
           "candidate/provenance.json", "native_amp_smoke.json", "method_preflight.json", "controller.log",
           "candidate/training.log", "candidate/benchmark.log", "candidate/benchmark.json",
           "candidate/benchmark.core.json",
           "baseline/benchmark.log", "remote_receipt.json")
CHUNK_BYTES = 8 * 1024 * 1024


def now():
    return datetime.now(timezone.utc)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_run(run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,70}", run) or run.startswith("stability"):
        raise ValueError("Use a unique exploratory run ID")


def validate_config(config):
    if config.get("train_all") is not False or config.get("split_seed") != 2026:
        raise ValueError("Only the frozen official 40k train / 5k development split is permitted")
    if not isinstance(config.get("epochs"), int) or config["epochs"] < 1:
        raise ValueError("A positive epoch count is required")
    if config.get("num_classes", 10) != 10:
        raise ValueError("This experiment requires the official ten classes")


def trainer_routes(trainer):
    if trainer not in {"shadow", "resizemix"}:
        raise ValueError("Unsupported Colab training method")
    return {"shadow": ("research/train_shadow.py", "research/benchmark_shadow.py"),
            "resizemix": ("research/train_resizemix.py", "research/benchmark_resizemix.py")}[trainer]


def sources_for_trainer(trainer):
    trainer_routes(trainer)
    return SOURCES if trainer == "shadow" else (*SOURCES, *RESIZEMIX_SOURCES)


def parse_usage(output):
    patterns = (r"Current balance:\s*([0-9.]+)\s*compute units",
                r"Usage rate:\s*([0-9.]+)/hr", r"Active assignments:\s*(\d+)")
    matches = [re.search(pattern, output) for pattern in patterns]
    if not all(matches):
        raise ValueError("Colab usage response could not be verified")
    return dict(balance_compute_units=float(matches[0][1]),
                rate_compute_units_per_hour=float(matches[1][1]),
                active_assignments=int(matches[2][1]))


def validate_resource_usage(usage, expected_assignments):
    for key in ("balance_compute_units", "rate_compute_units_per_hour"):
        value = usage.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Colab resource balance/rate must be finite and nonnegative")
    if (type(usage.get("active_assignments")) is not int or
            usage["active_assignments"] != expected_assignments):
        raise RuntimeError("Unexpected number of active Colab assignments")


def wsl_path(path):
    path = Path(path).resolve()
    if os.name != "nt":
        return str(path)
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                        getpass.getuser() + ":(OI)(CI)F"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        path.chmod(0o700)


class ColabCLI:
    def __init__(self, private, distro="Ubuntu", python=None):
        self.private, self.distro = Path(private), distro
        self.counter = 0
        self.python = python
        self.log_prefix = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"

    def command(self, arguments):
        if self.python is None:
            result = subprocess.run(["wsl.exe", "-d", self.distro, "--", "printenv", "HOME"],
                                    capture_output=True, text=True, timeout=15, check=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            user_directory = result.stdout.strip()
            if not re.fullmatch(r"/(?:home/[A-Za-z0-9_.-]+|root)", user_directory):
                raise ValueError("Unexpected WSL home directory")
            self.python = user_directory + "/.local/share/uv/tools/google-colab-cli/bin/python"
        # No shell; paths and arguments remain separate argv throughout.
        return ["wsl.exe", "-d", self.distro, "--", "env", "COLAB_CLI_ASSIGN_RETRIES=0",
                self.python, wsl_path(ROOT / "research/colab_cli.py"),
                *arguments]

    def run(self, arguments, timeout=60):
        self.counter += 1
        try:
            result = subprocess.run(self.command(arguments), stdin=subprocess.DEVNULL,
                                    capture_output=True, encoding="utf-8", errors="replace",
                                    timeout=max(1, timeout),
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            code, output = result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired as error:
            pieces = [error.stdout or "", error.stderr or ""]
            output = "".join(p.decode("utf-8", "replace") if isinstance(p, bytes) else p for p in pieces)
            code = -124
        path = self.private / f"cli_{self.log_prefix}_{self.counter:03d}_{arguments[0]}.private.log"
        path.write_text(output, encoding="utf-8")
        return code, output

    def require(self, arguments, timeout=60):
        code, output = self.run(arguments, timeout)
        if code:
            # Exceptions intentionally omit credential-bearing CLI responses.
            raise RuntimeError(f"Colab {arguments[0]} failed (exit {code}); inspect private log")
        return output


def prepare(root, config_path, folder, run, trainer="shadow"):
    """Create an exact allowlisted bundle; no directory recursion or networks."""
    root, config_path, folder = Path(root), Path(config_path), Path(folder)
    sources = sources_for_trainer(trainer)
    config = json.loads(config_path.read_text())
    validate_config(config)
    for name, expected in CORE.items():
        if sha256(root / name) != expected:
            raise ValueError("Frozen primary training core changed")
    if sha256(root / "results/regularized/best.pt") != BASELINE_SHA:
        raise ValueError("Trusted original baseline checkpoint mismatch")
    manifest = json.loads((root / "data/training_manifest.json").read_text())
    if (manifest.get("source_type") != "official_kaggle_competition_training"
        or manifest.get("external_training_data_loaded") is not False
        or manifest.get("official_test_labels_loaded") is not False
        or manifest.get("split", {}).get("seed") != 2026):
        raise ValueError("Official training provenance is required")
    entries = {name: root / name for name in sources}
    entries.update({"data/" + name: root / "data" / name for name in CACHE})
    entries.update({"baseline/" + name: root / "results/regularized" / name
                    for name in ("best.pt", "metrics.json", "history.json")})
    entries["research/configs/candidate.json"] = config_path
    for name, expected in manifest["files_sha256"].items():
        if name not in CACHE or sha256(root / "data" / name) != expected:
            raise ValueError("Official training cache fingerprint mismatch")
    fingerprints = {name: sha256(path) for name, path in entries.items()}
    bundle = folder / "input.private.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in entries.items():
            archive.write(path, name)
    chunks = []
    with bundle.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            path = folder / f"input_{len(chunks):03d}.private.part"
            path.write_bytes(chunk)
            chunks.append({"name": path.name, "sha256": sha256(path), "bytes": len(chunk)})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    value = {"schema": "private_colab_input_v1", "run": run, "files_sha256": fingerprints,
             "trainer": trainer,
             "source_git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
             "bundle_sha256": sha256(bundle), "chunks": chunks, "config": config,
             "hidden_images_packaged": False, "external_data_packaged": False,
             "confirmation_evaluated": False,
             "split_note": "Frozen loader requires its complete partition file; confirmation indices are validated but never selected/evaluated."}
    write_json(folder / "input_manifest.json", value)
    return value


def remote_guard(deadline):
    """Native Colab self-release survives a disconnected local controller."""
    return f'''import json, threading, time
from datetime import datetime, timezone
from google.colab import runtime as colab_runtime
import torch
_deadline = {deadline!r}
def _release_at_deadline():
    time.sleep(max(0, _deadline-time.time()))
    colab_runtime.unassign()
threading.Thread(target=_release_at_deadline, daemon=True).start()
assert torch.cuda.is_available(), "A real CUDA GPU is required"
assert torch.cuda.get_device_capability() == (7, 5), "Expected the requested T4"
print("SABER_GUARD_JSON=" + json.dumps({{"status":"armed", "deadline_unix":_deadline,
"device":torch.cuda.get_device_name(), "torch_version":torch.__version__, "amp_dtype":"float16"}}))
'''


def enforce_shadow_preflight_budget(method, remaining_seconds):
    """Fixed before measurement: 1.5x SGD-only time +300s, reserve another180s."""
    if (type(method.get("parameters")) is not int or not 0 < method["parameters"] <= 5000000 or
        method.get("warmup_steps") != 3 or method.get("timed_steps") != 10 or
        method.get("real_parameter_updates") != 13 or method.get("finite_parameters_verified") is not True or
        method.get("finite_losses_verified") is not True):
        raise RuntimeError("Shadow method preflight did not prove finite real model updates")
    median, batch_size, epochs = method.get("median_step_seconds"), method.get("batch_size"), method.get("epochs")
    if (type(median) not in (int, float) or not math.isfinite(median) or median <= 0 or
        type(batch_size) is not int or batch_size <= 0 or type(epochs) is not int or epochs <= 0 or
        not math.isfinite(remaining_seconds) or remaining_seconds <= 180):
        raise RuntimeError("Shadow method preflight timing or remaining runtime is invalid")
    estimated = median*math.ceil(40000/batch_size)*epochs
    conservative = 1.5*estimated+300
    method.update(estimated_training_only_seconds=estimated, conservative_pipeline_seconds=conservative,
                  runtime_safety_multiplier=1.5, pipeline_overhead_allowance_seconds=300,
                  transfer_cleanup_reserve_seconds=180, runtime_budget_fit=conservative < remaining_seconds-180)
    if not method["runtime_budget_fit"]:
        method["status"] = "runtime_budget_exceeded"
        raise RuntimeError("Shadow measured speed does not fit the remaining supervised runtime")


def remote_program(manifest, session, deadline):
    # Every interpolated value is a validated ID or a Python literal; no shell.
    trainer = manifest.get("trainer", "shadow")
    train_entry, validation_entry = trainer_routes(trainer)
    return '''import hashlib,json,math,os,subprocess,sys,time,zipfile
from pathlib import Path
''' + inspect.getsource(enforce_shadow_preflight_budget) + '''
M = ''' + repr(manifest) + '''
D = ''' + repr(deadline) + '''
TRAIN = ''' + repr(train_entry) + '''
VALIDATOR = ''' + repr(validation_entry) + '''
P = Path('/content')
R = P / ''' + repr(session) + '''
R.mkdir(exist_ok=False)
bundle = R/'input.zip'
with bundle.open('wb') as out:
    for part in M['chunks']:
        raw=(P/part['name']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==part['sha256']
        out.write(raw)
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==M['bundle_sha256']
with zipfile.ZipFile(bundle) as z:
    assert set(z.namelist()) == set(M['files_sha256'])
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(R)
for n,h in M['files_sha256'].items():
    assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
os.chdir(R)
(R/'candidate').mkdir()
receipt={'status':'running','confirmation_evaluated':False,'hidden_images_loaded':False,
         'source_files_sha256':M['files_sha256'],'run':M['run'],'trainer':M.get('trainer','shadow')}
def run(args, log):
    remaining = D-time.time()-180
    if remaining < 1: raise TimeoutError('No research runtime remains')
    with (R/log).open('w') as output:
        subprocess.run([sys.executable,*args],cwd=R,stdout=output,stderr=subprocess.STDOUT,
                       check=True,timeout=remaining)
try:
    sys.path.insert(0,str(R))
    import torch
    from research.train_shadow import optimizer_step
    p=torch.nn.Parameter(torch.tensor([1.0],device='cuda'))
    opt=torch.optim.SGD([p],lr=.1)
    scaler=torch.amp.GradScaler('cuda')
    original=p.detach().clone()
    assert optimizer_step(p.square().sum(),opt,scaler)
    assert not torch.equal(p.detach(),original)
    updated=p.detach().clone()
    scale_before=scaler.get_scale()
    opt.zero_grad(set_to_none=True)
    overflow_loss=(p*1e38).sum()
    assert torch.isfinite(overflow_loss)
    assert optimizer_step(overflow_loss,opt,scaler) is False
    assert torch.equal(p.detach(),updated)
    assert scaler.get_scale() < scale_before
    smoke={'status':'complete','finite_update_verified':True,'finite_loss_overflow_skipped':True,
           'overflow_parameter_unchanged':True,'scale_before_overflow':scale_before,
           'scale_after_overflow':scaler.get_scale(),'torch_version':torch.__version__,
           'shadow_source_sha256':M['files_sha256']['research/train_shadow.py'],
           'trainer':M.get('trainer','shadow'),'trainer_source_sha256':M['files_sha256'][TRAIN]}
    (R/'native_amp_smoke.json').write_text(json.dumps(smoke,indent=2))
    receipt['native_amp_smoke']=smoke
    del p,opt,scaler
    if M.get('trainer','shadow')=='shadow' and not (M['config'].get('cutmix_alpha',0)>0 and M['config'].get('cutmix_probability',0)>0):
        from train import DEFAULT_CONFIG, augment
        from research.train_shadow import validate_config
        from runtime import seed_all
        from model import build_model, count_parameters
        import random, statistics
        import numpy as np
        seed_all(2026)
        cfg={**DEFAULT_CONFIG,**M['config']}; validate_config(cfg)
        model_cfg={k:cfg[k] for k in ('depth','widen_factor','dropout','num_classes')}
        pre_model=build_model(model_cfg)
        parameters=count_parameters(pre_model); assert 0<parameters<=5000000
        pre_model=pre_model.to(device='cuda',memory_format=torch.channels_last).train()
        pre_opt=torch.optim.SGD(pre_model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'],nesterov=True)
        pre_scaler=torch.amp.GradScaler('cuda',init_scale=256.)
        pre_images=torch.randint(0,256,(cfg['batch_size'],32,32,3),device='cuda',dtype=torch.uint8)
        pre_labels=torch.arange(cfg['batch_size'],device='cuda')%10
        durations=[]; mixed_calls=0; changed_steps=0; probe=next(pre_model.parameters())
        torch.cuda.reset_peak_memory_stats()
        for index in range(13):
            torch.cuda.synchronize(); began=time.perf_counter(); before=probe.detach().clone()
            x=augment(pre_images,cfg['cutout']); y=pre_labels
            mixed=cfg['mixup_alpha']>0 and random.random()<cfg['mixup_probability']
            retained=float(np.random.beta(cfg['mixup_alpha'],cfg['mixup_alpha'])) if mixed else 1.
            permutation=torch.randperm(len(y),device='cuda') if mixed else None
            if mixed: x=x*retained+x[permutation]*(1-retained)
            mixed_calls+=int(mixed); pre_opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=pre_model(x)
                loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'])*retained
                if mixed: loss+=torch.nn.functional.cross_entropy(logits,y[permutation],label_smoothing=cfg['label_smoothing'])*(1-retained)
            assert optimizer_step(loss,pre_opt,pre_scaler)
            assert torch.isfinite(probe).all() and not torch.equal(probe.detach(),before)
            changed_steps+=1; torch.cuda.synchronize()
            if index>=3: durations.append(time.perf_counter()-began)
        assert all(torch.isfinite(p).all() for p in pre_model.parameters())
        assert not (cfg['mixup_alpha']>0 and cfg['mixup_probability']>0) or mixed_calls>0
        method={'status':'complete','method':'shadow_mixup_cutout','synthetic_only':True,'engineering_only':True,
            'training_data_loaded':False,'hidden_images_loaded':False,'native_fp16_verified':True,
            'finite_parameters_verified':True,'finite_losses_verified':True,'real_parameter_updates':changed_steps,
            'actual_mixed_batches':mixed_calls,'mixup_alpha':cfg['mixup_alpha'],'mixup_probability':cfg['mixup_probability'],
            'cutout':cfg['cutout'],'batch_size':cfg['batch_size'],'epochs':cfg['epochs'],'model_config':model_cfg,'parameters':parameters,
            'warmup_steps':3,'timed_steps':10,'seed':2026,'median_step_seconds':statistics.median(durations),
            'p90_step_seconds':statistics.quantiles(durations,n=10,method='inclusive')[8],
            'images_per_second':cfg['batch_size']/statistics.median(durations),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
            'grad_scaler_initial_scale':256.,'grad_scaler_final_scale':pre_scaler.get_scale(),
            'trainer_source_sha256':M['files_sha256'][TRAIN],'augmentation_source_sha256':M['files_sha256']['train.py']}
        try:
            enforce_shadow_preflight_budget(method,D-time.time())
        finally:
            (R/'method_preflight.json').write_text(json.dumps(method,indent=2))
            receipt['method_preflight']=method
        del pre_model,pre_opt,pre_scaler,pre_images,pre_labels,probe,before,x,y,permutation,loss,logits
        torch.cuda.empty_cache()
    if M.get('trainer','shadow')=='resizemix':
        from research.train_resizemix import prepare_batch, optimizer_step
        from train import DEFAULT_CONFIG
        from runtime import seed_all
        from model import build_model, count_parameters
        import math, statistics
        seed_all(2026)
        cfg={**DEFAULT_CONFIG,**M['config']}
        pre_model=build_model({k:cfg[k] for k in ('depth','widen_factor','dropout','num_classes')}).to(device='cuda',memory_format=torch.channels_last).train()
        pre_opt=torch.optim.SGD(pre_model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'],nesterov=True)
        pre_scaler=torch.amp.GradScaler('cuda',init_scale=256.)
        pre_images=torch.randint(0,256,(cfg['batch_size'],32,32,3),device='cuda',dtype=torch.uint8)
        pre_labels=torch.arange(cfg['batch_size'],device='cuda')%10
        geometry=prepare_batch(pre_images,pre_labels,{**cfg,'resizemix_probability':1.})
        assert geometry[2] is not None and 0<geometry[3]<1 and torch.isfinite(geometry[0]).all()
        assert len(geometry[4])==cfg['batch_size'] and all(0<=t<b<=32 and 0<=l<r<=32 and abs(geometry[3]-(1-(b-t)*(r-l)/1024))<1e-12 for t,b,l,r in geometry[4])
        durations=[]; mixed_calls=0; changed_steps=0
        probe=next(pre_model.parameters())
        torch.cuda.reset_peak_memory_stats()
        for index in range(13):
            torch.cuda.synchronize(); began=time.perf_counter(); before=probe.detach().clone()
            x,y,permutation,retained,boxes=prepare_batch(pre_images,pre_labels,cfg)
            mixed_calls+=int(permutation is not None)
            pre_opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=pre_model(x)
                loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'])*retained
                if permutation is not None: loss+=torch.nn.functional.cross_entropy(logits,y[permutation],label_smoothing=cfg['label_smoothing'])*(1-retained)
            assert optimizer_step(loss,pre_opt,pre_scaler)
            assert torch.isfinite(probe).all() and not torch.equal(probe.detach(),before)
            changed_steps+=1
            torch.cuda.synchronize()
            if index>=3: durations.append(time.perf_counter()-began)
        assert mixed_calls>0 and changed_steps==13
        median=statistics.median(durations)
        estimate=median*math.ceil(40000/cfg['batch_size'])*cfg['epochs']
        safe_estimate=2*estimate+300
        method={'status':'complete','method':'resizemix','synthetic_only':True,'training_data_loaded':False,
            'hidden_images_loaded':False,'geometry_verified':True,'native_fp16_verified':True,'real_parameter_updates':changed_steps,
            'actual_mixed_batches':mixed_calls,'batch_size':cfg['batch_size'],'parameters':count_parameters(pre_model),
            'model_config':{k:cfg[k] for k in ('depth','widen_factor','dropout','num_classes')},'seed':2026,
            'warmup_steps':3,'timed_steps':10,'median_step_seconds':median,'p90_step_seconds':statistics.quantiles(durations,n=10,method='inclusive')[8],
            'images_per_second':cfg['batch_size']/median,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
            'grad_scaler_initial_scale':256.,'grad_scaler_final_scale':pre_scaler.get_scale(),'engineering_only':True,
            'trainer_source_sha256':M['files_sha256'][TRAIN],'helper_source_sha256':M['files_sha256']['research/resizemix.py'],
            'estimated_training_only_seconds':estimate,'conservative_pipeline_seconds':safe_estimate,
            'runtime_budget_fit':safe_estimate<D-time.time()-180}
        (R/'method_preflight.json').write_text(json.dumps(method,indent=2))
        receipt['method_preflight']=method
        if not method['runtime_budget_fit']: raise RuntimeError('ResizeMix measured speed does not fit the remaining supervised runtime')
        del pre_model,pre_opt,pre_scaler,pre_images,pre_labels,probe,before,x,y,permutation,loss,logits,geometry
        torch.cuda.empty_cache()
    run(['research/benchmark.py','--checkpoint','baseline/best.pt','--output','baseline/benchmark.json',
         '--device','cuda'], 'baseline/benchmark.log')
    run([TRAIN,'--config','research/configs/candidate.json','--output-dir','candidate',
         '--data-dir','data','--device','cuda','--precision','fp16'], 'candidate/training.log')
    run([VALIDATOR,'--checkpoint','candidate/best.pt','--output','candidate/benchmark.json',
         '--baseline','baseline/benchmark.json','--device','cuda'], 'candidate/benchmark.log')
    receipt['status']='complete'
except Exception as e:
    import traceback
    (R/'controller.log').write_text(traceback.format_exc())
    receipt.update(status='failed',error_type=type(e).__name__)
finally:
    (R/'remote_receipt.json').write_text(json.dumps(receipt,indent=2))
    artifacts=P/'saber_artifacts.zip'
    allowed=''' + repr(OUTPUTS) + '''
    with zipfile.ZipFile(artifacts,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n in allowed:
            if (R/n).is_file(): z.write(R/n,n)
    receipt['artifact_sha256']=hashlib.sha256(artifacts.read_bytes()).hexdigest()
    parts=[]
    with artifacts.open('rb') as source:
        while raw:=source.read(''' + repr(CHUNK_BYTES) + '''):
            name=f'saber_artifacts_{len(parts):03d}.part'
            (P/name).write_bytes(raw)
            parts.append({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    receipt['artifact_chunks']=parts
    print('SABER_RESULT_JSON='+json.dumps(receipt))
'''


def extract_artifacts(archive, destination, expected_sha):
    if sha256(archive) != expected_sha:
        raise ValueError("Downloaded artifacts checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if sum(item.file_size for item in source.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Unexpectedly large result archive")
        if len(set(names)) != len(names) or not set(names) <= set(OUTPUTS):
            raise ValueError("Unapproved result archive member")
        if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
            raise ValueError("Unsafe result archive path")
        source.extractall(destination)


def acquire_lease(path, value):
    # A dead owner may still have a live cloud runtime. Never auto-clear it.
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream)


def stop_and_verify(cli, session, allocation_known=False):
    code, output = cli.run(["stop", "--session", session], 90)
    terminated = "[colab] Session terminated." in output
    usage_code, output = cli.run(["usage"], 30)
    usage = parse_usage(output) if not usage_code else None
    # Official CLI 0.7.5 also exits zero for an unknown session name. An
    # allocation request with an unknown response may still be completing;
    # a no-op stop plus zero current usage cannot reconcile that request.
    return {"stop_exit_code": code, "stop_reported_termination": terminated,
            "allocation_identity_verified": allocation_known, "usage_after": usage,
            "runtime_stop_verified": allocation_known and code == 0 and terminated
            and usage is not None and usage["active_assignments"] == 0}


def watchdog(folder, parent, created, distro):
    folder = Path(folder)
    receipt_path = folder / "supervisor.json"
    while True:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("finished_at"):
            return
        try:
            alive = abs(psutil.Process(parent).create_time() - created) < .01
        except psutil.NoSuchProcess:
            alive = False
        if not alive or time.time() >= receipt["deadline_unix"]:
            cleanup = stop_and_verify(ColabCLI(folder / "private", distro), receipt["session"],
                                      allocation_known=receipt.get("allocation_returned_success") is True)
            write_json(folder / "watchdog_cleanup.json", cleanup)
            return
        time.sleep(2)


def launch(folder, distro):
    manifest = json.loads((folder / "input_manifest.json").read_text())
    trainer = manifest.get("trainer", "shadow")
    trainer_routes(trainer)
    validate_run(manifest["run"])
    validate_config(manifest["config"])
    if sha256(folder / "input.private.zip") != manifest["bundle_sha256"]:
        raise ValueError("Prepared input bundle changed")
    for part in manifest["chunks"]:
        if (not re.fullmatch(r"input_[0-9]{3}\.private\.part", part["name"])
            or sha256(folder / part["name"]) != part["sha256"]):
            raise ValueError("Prepared input chunk changed")
    policy = json.loads((ROOT / "research/campaign.json").read_text())
    if policy.get("colab_additional_purchases_authorized") is not False:
        raise ValueError("This launcher only implements existing/free Colab access")
    deadline = min(datetime.fromisoformat(policy["stops_at"]).timestamp(), time.time()+10800)
    if deadline-time.time() < 600:
        raise RuntimeError("Insufficient research window remains")
    receipt_path = folder / "supervisor.json"
    if receipt_path.exists():
        raise FileExistsError("Existing allocation intent must be reconciled; never retry blindly")
    session = "saber-" + manifest["run"] + "-" + uuid.uuid4().hex[:8]
    lease_path = ROOT / "results/research/cloud_lease.json"
    receipt = {"run": manifest["run"], "session": session, "status": "preparing",
               "trainer": trainer,
               "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
               "started_at": now().isoformat(), "deadline_unix": deadline,
               "assignment_attempts": 0, "no_new_purchases": True, "precision": "fp16",
               "resource_policy": "existing_Colab_entitlement_only_no_purchases",
               "allocation_returned_success": False, "remote_guard_verified": False,
               "confirmation_evaluated": False, "submitted": False,
               "input_bundle_sha256": manifest["bundle_sha256"]}
    acquire_lease(lease_path, receipt)
    private_directory(folder / "private")
    cli = ColabCLI(folder / "private", distro)
    guard = None
    try:
        before = parse_usage(cli.require(["usage"], 30))
        validate_resource_usage(before, expected_assignments=0)
        if before["rate_compute_units_per_hour"] != 0:
            raise RuntimeError("Existing cloud allocation or nonzero consumption requires reconciliation")
        receipt.update(status="allocating", assignment_attempts=1, usage_before=before)
        write_json(receipt_path, receipt)
        guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(folder),
                                  "--parent", str(os.getpid()), "--created", str(receipt["process_created_at"]),
                                  "--distro", distro], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        allocation_output = cli.require(["new", "--gpu", "T4", "--session", session], 150)
        if "[colab] Session READY." not in allocation_output:
            raise RuntimeError("Allocation response did not confirm a persisted session identity")
        receipt["allocation_returned_success"] = True
        write_json(receipt_path, receipt)
        guard_path = folder / "remote_guard.py"
        guard_path.write_text(remote_guard(deadline), encoding="utf-8")
        output = cli.require(["exec", "--session", session, "--file", wsl_path(guard_path), "--timeout", "60"], 90)
        matches = re.findall(r"SABER_GUARD_JSON=(\{[^\n]+\})", output)
        if not matches:
            raise RuntimeError("Remote self-release/GPU guard not verified")
        receipt["runtime_preflight"] = json.loads(matches[-1])
        if (receipt["runtime_preflight"].get("status") != "armed"
            or receipt["runtime_preflight"].get("deadline_unix") != deadline):
            raise RuntimeError("Remote self-release guard does not match this allocation")
        receipt["remote_guard_verified"] = True
        write_json(receipt_path, receipt)
        usage = parse_usage(cli.require(["usage"], 30))
        validate_resource_usage(usage, expected_assignments=1)
        # CLI reports backend CU/h, not a USD charge. Existing entitlement is
        # authorized; FAQ says exhausted balances revert to free-tier policies:
        # https://research.google.com/colaboratory/faq.html
        # A positive rate alone cannot establish a new purchase or dollar cost.
        receipt.update(status="uploading", usage_after_assignment=usage)
        write_json(receipt_path, receipt)
        for part in manifest["chunks"]:
            local = folder / part["name"]
            if sha256(local) != part["sha256"]:
                raise ValueError("Prepared input chunk changed")
            cli.require(["upload", "--session", session, wsl_path(local), "content/" + part["name"]],
                        min(180, deadline-time.time()-120))
        program_path = folder / "remote_program.py"
        program_path.write_text(remote_program(manifest, session, deadline), encoding="utf-8")
        receipt.update(status="training_and_benchmarking")
        write_json(receipt_path, receipt)
        remaining = int(deadline-time.time()-150)
        if remaining < 120:
            raise TimeoutError("No time remains for development training")
        output = cli.require(["exec", "--session", session, "--file", wsl_path(program_path),
                              "--timeout", str(remaining)], remaining+30)
        matches = re.findall(r"SABER_RESULT_JSON=(\{[^\n]+\})", output)
        if not matches:
            raise RuntimeError("Remote execution outcome unknown; do not repeat allocation")
        remote = json.loads(matches[-1])
        receipt["remote_status"] = remote["status"]
        artifacts = folder / "artifacts.private.zip"
        with artifacts.open("xb") as assembled:
            for part in remote["artifact_chunks"]:
                if not re.fullmatch(r"saber_artifacts_[0-9]{3}\.part", part["name"]):
                    raise ValueError("Invalid artifact chunk name")
                local = folder / part["name"]
                cli.require(["download", "--session", session, "content/" + part["name"], wsl_path(local)],
                            min(90, deadline-time.time()-5))
                if sha256(local) != part["sha256"]:
                    raise ValueError("Downloaded artifact chunk changed")
                assembled.write(local.read_bytes())
        extract_artifacts(artifacts, folder, remote["artifact_sha256"])
        receipt.update(status="complete" if remote["status"] == "complete" else "remote_failed",
                       artifact_sha256=remote["artifact_sha256"])
        if remote["status"] == "complete":
            evidence = json.loads((folder / "candidate/benchmark.json").read_text())
            receipt["benchmark_passed"] = any(g["passed"] for g in evidence["promotion"].values())
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if receipt["assignment_attempts"]:
            try:
                receipt.update(stop_and_verify(cli, session,
                               allocation_known=receipt["allocation_returned_success"]))
            except Exception as error:
                receipt.update(runtime_stop_verified=False, cleanup_error_type=type(error).__name__)
        else:
            receipt["runtime_stop_verified"] = True
        receipt["finished_at"] = now().isoformat()
        if receipt["status"] == "complete" and not receipt["runtime_stop_verified"]:
            receipt["status"] = "cleanup_unverified"
        write_json(receipt_path, receipt)
        if receipt["runtime_stop_verified"]:
            own = json.loads(lease_path.read_text()) if lease_path.exists() else {}
            if own.get("session") == session:
                lease_path.unlink()
        if guard and guard.poll() is None:
            guard.terminate()
            guard.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--trainer", choices=("shadow", "resizemix"), default="shadow")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--watchdog", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--created", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog, args.parent, args.created, args.distro)
        return
    if not args.run or not args.config:
        parser.error("--run and --config required")
    validate_run(args.run)
    folder = ROOT / "results/research" / ("colab_" + args.run)
    if folder.exists():
        manifest = json.loads((folder / "input_manifest.json").read_text())
        if manifest.get("trainer", "shadow") != args.trainer:
            raise ValueError("Prepared Colab trainer differs; never reinterpret an existing source bundle")
        if not args.launch or sha256(args.config) != manifest["files_sha256"]["research/configs/candidate.json"]:
            raise FileExistsError("Prepared run exists; inspect it and use --launch once")
    else:
        private_directory(folder)
        try:
            prepare(ROOT, args.config.resolve(), folder, args.run, args.trainer)
        except BaseException:
            write_json(folder / "preparation_failed.json", {"at": now().isoformat()})
            raise
    if args.launch:
        launch(folder, args.distro)
    print(json.dumps({"run": args.run, "prepared": True, "launch_requested": args.launch,
                      "receipt": str(folder / "supervisor.json") if args.launch else None}))


if __name__ == "__main__":
    main()
