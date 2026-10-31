"""Package committed source for ONE private free-quota T4 robustness experiment.

Preparation is local. --launch explicitly checks free quota and makes one push;
an existing launch-attempt receipt forbids retries after an ambiguous outcome.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "kevin1220/saber-wrn22-randaugment-full-20261030"
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
ACCELERATOR = "NvidiaTeslaT4"
SOURCE_KERNELS = ["kevin1220/saber-official-data-export-20261030",
                  "kevin1220/saber-official-test-export-20261030"]
SOURCE_FILES = ("model.py", "data.py", "runtime.py", "train.py", "prepare_data.py",
                "predict.py", "train_robust.py", "configs/robust_full.json")

BOOTSTRAP = r'''
"""Private exploratory robustness experiment; primary recipe stays frozen."""
import base64, hashlib, io, json, pathlib, shutil, subprocess, sys, time, zipfile

LAUNCH = json.loads(__MANIFEST__)
SOURCE = pathlib.Path('/tmp/saber-robust-code')
DATA = pathlib.Path('/tmp/saber-robust-data')
OUTPUT = pathlib.Path('/kaggle/working/robust_full')
for folder in (SOURCE, DATA, OUTPUT): folder.mkdir(parents=True, exist_ok=True)
deadline = time.monotonic() + LAUNCH['runtime_cap_seconds'] - 60

def write(name, value):
    (OUTPUT/name).write_text(json.dumps(value, indent=2) + '\n')

def run(command):
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError('Robustness wrapper runtime cap exhausted')
    print(json.dumps({'event':'robust_command', 'command':command}), flush=True)
    subprocess.run(command, cwd=SOURCE, check=True, timeout=remaining)

def unique_export(name):
    candidates = list(pathlib.Path('/kaggle/input').rglob(name))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one attached official export: ' + name)
    return candidates[0].parent

try:
    write('launch_manifest.json', LAUNCH)
    payload = base64.b64decode(__BUNDLE__)
    if hashlib.sha256(payload).hexdigest() != LAUNCH['source_bundle_sha256']:
        raise RuntimeError('Embedded source-bundle checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if set(archive.namelist()) != set(LAUNCH['source_sha256']):
            raise RuntimeError('Unexpected files in committed source bundle')
        for name in archive.namelist():
            blob = archive.read(name)
            if hashlib.sha256(blob).hexdigest() != LAUNCH['source_sha256'][name]:
                raise RuntimeError('Committed source-file checksum mismatch: ' + name)
            target = SOURCE/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    # The trainer chooses and probes native FP16/BF16 with its own GradScaler.
    # This early check avoids copying/preparing inputs on an unassigned CPU job.
    preflight = """
import json, pathlib, torch
if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU was assigned; robustness experiment aborted')
info = {'gpu':torch.cuda.get_device_name(), 'capability':list(torch.cuda.get_device_capability()),
        'torch':torch.__version__, 'cuda_runtime':torch.version.cuda}
pathlib.Path('/kaggle/working/robust_full/hardware_assignment.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info),flush=True)
"""
    run([sys.executable, '-c', preflight])
    training = unique_export('official_train.npz')
    testing = unique_export('official_test.npz')
    for name in ('official_train.npz', 'official_source_manifest.json', 'train_labels.csv'):
        shutil.copyfile(training/name, DATA/name)
    for name in ('official_test.npz', 'official_test_source_manifest.json', 'test.csv', 'sample_submission.csv'):
        shutil.copyfile(testing/name, DATA/name)
    config = json.loads((SOURCE/'configs/robust_full.json').read_text())
    if config.get('epochs') != 300 or config.get('depth') != 22 or config.get('widen_factor') != 4 or config.get('train_all') is not True:
        raise RuntimeError('Committed robustness recipe must be scratch WRN22-4 / full50k / 300 epochs')
    run([sys.executable, 'prepare_data.py', '--data-dir', str(DATA), '--seed', '2026'])
    write('exploratory_protocol.json', {
        'source_commit':LAUNCH['source_commit'], 'purpose':LAUNCH['purpose'],
        'used_for_primary_selection':False, 'confirmation_evaluated':False,
        'training_examples':50000, 'epochs':300,
        'note':'Separate post-selection exploratory recipe; primary recipe and confirmation remain unchanged',
    })
    run([sys.executable, 'train_robust.py', '--config', 'configs/robust_full.json',
         '--data-dir', str(DATA), '--output-dir', str(OUTPUT), '--device', 'cuda',
         '--max-seconds', str(min(10500, LAUNCH['runtime_cap_seconds']-180))])
    metrics = json.loads((OUTPUT/'metrics.json').read_text())
    if metrics.get('status') != 'complete' or metrics.get('epochs_completed') != 300:
        raise RuntimeError('Robustness experiment did not complete all 300 configured epochs')
    write('robust_launch_status.json', {'status':'complete', 'source_commit':LAUNCH['source_commit'],
        'purpose':LAUNCH['purpose'], 'used_for_primary_selection':False, 'confirmation_evaluated':False})
except BaseException as error:
    write('robust_launch_status.json', {'status':'failed', 'source_commit':LAUNCH['source_commit'],
        'purpose':LAUNCH['purpose'], 'error_type':type(error).__name__,
        'used_for_primary_selection':False, 'confirmation_evaluated':False})
    raise
'''


def build_bundle(commit):
    buffer, hashes = io.BytesIO(), {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            hashes[name] = hashlib.sha256(blob).hexdigest()
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, blob)
    return buffer.getvalue(), hashes


def kaggle_cli(explicit):
    if explicit:
        return explicit
    sibling = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    found = str(sibling) if sibling.is_file() else shutil.which("kaggle")
    if not found:
        raise FileNotFoundError("Use the isolated Kaggle Python or supply --kaggle-cli")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--folder", type=Path, default=ROOT / "tmp/robust-kernel")
    parser.add_argument("--max-seconds", type=int, default=10800)
    parser.add_argument("--kaggle-cli")
    args = parser.parse_args()
    if not 300 <= args.max_seconds <= 10800:
        raise ValueError("Runtime cap must be between 300 and 10800 seconds")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", f"{args.source_commit}^{{commit}}"], cwd=ROOT, text=True).strip()
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    attempt_path = folder / "launch_attempt.json"
    if attempt_path.exists():
        raise RuntimeError("A launch was already attempted; inspect that job instead of retrying")
    payload, hashes = build_bundle(commit)
    launch = {
        "source_commit":commit, "source_sha256":hashes,
        "source_bundle_sha256":hashlib.sha256(payload).hexdigest(),
        "kernel_id":KERNEL_ID, "requested_accelerator":ACCELERATOR,
        "runtime_cap_seconds":args.max_seconds,
        "purpose":"post_selection_exploratory_robustness", "used_for_primary_selection":False,
        "confirmation_evaluated":False, "data_and_weights_in_source_bundle":False,
    }
    code = BOOTSTRAP.replace("__BUNDLE__", repr(base64.b64encode(payload).decode("ascii")))
    code = code.replace("__MANIFEST__", repr(json.dumps(launch)))
    compile(code, "robust_entry.py", "exec")
    (folder / "robust_entry.py").write_text(code, encoding="utf-8")
    (folder / "source_bundle_manifest.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    metadata = {
        "id":KERNEL_ID, "title":"Saber WRN22 RandAugment full50k 20261030",
        "code_file":"robust_entry.py", "language":"python", "kernel_type":"script",
        "is_private":True, "enable_gpu":True, "enable_internet":False,
        "machine_shape":ACCELERATOR, "competition_sources":[COMPETITION],
        "kernel_sources":SOURCE_KERNELS, "dataset_sources":[],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2), flush=True)
    if not args.launch:
        return
    cli = kaggle_cli(args.kaggle_cli)
    quota = json.loads(subprocess.check_output([cli, "quota", "--format", "json"], text=True))
    gpu = next(item for item in quota if item["resource"] == "GPU")
    if float(str(gpu["remaining"]).removesuffix("h")) < args.max_seconds / 3600:
        raise RuntimeError("Insufficient free GPU quota; no launch was attempted")
    receipt = {"status":"launch_attempted", "attempted_at_utc":datetime.now(timezone.utc).isoformat(),
               "free_quota_before":quota, **launch}
    with attempt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    try:
        completed = subprocess.run([cli, "kernels", "push", "-p", str(folder), "-t", str(args.max_seconds),
                                    "--accelerator", ACCELERATOR], check=False, text=True, capture_output=True)
        receipt.update(status="push_accepted" if completed.returncode == 0 else "push_failed",
                       returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        attempt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(completed.stdout, end="", flush=True)
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    except BaseException as error:
        if receipt["status"] == "launch_attempted":
            receipt.update(status="launch_outcome_unknown", error_type=type(error).__name__)
            attempt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
