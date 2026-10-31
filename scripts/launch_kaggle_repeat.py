"""Prepare or launch ONE private, free-quota seed-stability run on Kaggle.

The frozen regularized recipe changes only seed 2026 -> 2027. Split seed 2026,
160 epochs, architecture and optimization stay fixed. This run is excluded from
recipe selection and never evaluates confirmation or predicts hidden test data.

Run with a Python environment containing the official Kaggle CLI. Preparation is
local by default; --launch checks free GPU quota and issues a single push. The
launch-attempt receipt prevents silently starting another version on rerun.
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
SOURCE_COMMIT = "3862ffe5cd2ccf98cc97988bed5c1a8ce7995b08"
KERNEL_ID = "kevin1220/saber-wrn22-repeat-seed2027-20261030"
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
SOURCE_KERNEL = "kevin1220/saber-official-data-export-20261030"
SOURCE_FILES = (
    "train.py", "model.py", "data.py", "runtime.py", "prepare_data.py",
    "predict.py", "configs/wrn_final.json",
)


BOOTSTRAP = r'''
"""Private stability run: no confirmation evaluation, hidden prediction or selection."""
import base64, hashlib, io, json, pathlib, shutil, subprocess, sys, time, zipfile

BUNDLE = __BUNDLE__
LAUNCH = json.loads(__MANIFEST__)
WORK = pathlib.Path('/kaggle/working')
SOURCE = WORK / 'saber'
OUTPUT = WORK / 'repeat-seed2027'
SOURCE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)
deadline = time.monotonic() + LAUNCH['runtime_cap_seconds'] - 180

def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + '\n')

def run(command):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Seed-repeat runtime budget exhausted')
    print(json.dumps({'event': 'repeat_command', 'command': command}), flush=True)
    subprocess.run(command, cwd=SOURCE, check=True, timeout=remaining)

try:
    payload = base64.b64decode(BUNDLE)
    if hashlib.sha256(payload).hexdigest() != LAUNCH['source_bundle_sha256']:
        raise RuntimeError('Embedded frozen-source checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if set(archive.namelist()) != set(LAUNCH['source_sha256']):
            raise RuntimeError('Unexpected files in the frozen-source bundle')
        for name in archive.namelist():
            blob = archive.read(name)
            if hashlib.sha256(blob).hexdigest() != LAUNCH['source_sha256'][name]:
                raise RuntimeError('Frozen-source file checksum mismatch: ' + name)
            target = SOURCE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    write('launch_manifest.json', LAUNCH)
    config = json.loads((SOURCE / 'configs/wrn_final.json').read_text())
    config.update(seed=2027, split_seed=2026)
    if config['epochs'] != 160 or config['depth'] != 22 or config['widen_factor'] != 4:
        raise RuntimeError('The frozen recipe does not match WRN-22-4 / 160 epochs')
    config_path = SOURCE / 'configs/repeat_seed2027.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    # Require native BF16 hardware and test the exact convolution/backward path.
    # Do not silently change precision, use BF16 emulation or change the recipe.
    preflight = """
import runtime
import json, pathlib, torch
if not torch.cuda.is_available(): raise RuntimeError('No GPU was assigned')
capability = torch.cuda.get_device_capability()
if capability[0] < 8: raise RuntimeError('Assigned GPU lacks native BF16; repeat aborted')
if not torch.cuda.is_bf16_supported(): raise RuntimeError('Assigned GPU does not support BF16')
model = torch.nn.Conv2d(3, 8, 3).cuda()
x = torch.randn(2, 3, 32, 32, device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x).square().mean()
loss.backward()
if not torch.isfinite(loss) or not torch.isfinite(model.weight.grad).all(): raise RuntimeError('BF16 smoke failed')
torch.cuda.synchronize()
info = dict(runtime.environment(), native_bf16_preflight_passed=True, cuda_capability=list(capability))
pathlib.Path('/kaggle/working/repeat-seed2027/hardware_preflight.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2), flush=True)
"""
    run([sys.executable, '-c', preflight])
    official = list(pathlib.Path('/kaggle/input').rglob('official_train.npz'))
    if len(official) != 1:
        raise RuntimeError('Expected exactly one official training-export kernel input')
    source_dir = official[0].parent
    data_dir = SOURCE / 'data'
    data_dir.mkdir(exist_ok=True)
    for name in ('official_train.npz', 'official_source_manifest.json', 'train_labels.csv'):
        shutil.copyfile(source_dir / name, data_dir / name)
    run([sys.executable, 'prepare_data.py', '--data-dir', str(data_dir), '--seed', '2026'])
    split = json.loads((data_dir / 'training_manifest.json').read_text())['split']
    if split['indices_sha256'] != LAUNCH['fixed_split_sha256']:
        raise RuntimeError('Seed-repeat split differs from the frozen main campaign')
    if (split['train_count'], split['validation_count'], split['confirmation_count']) != (40000,5000,5000):
        raise RuntimeError('Seed-repeat train/dev/confirmation split sizes are wrong')
    write('stability_protocol.json', dict(
        source_commit=LAUNCH['source_commit'], seed=2027, split_seed=2026,
        fixed_split_sha256=split['indices_sha256'], epochs=160,
        selection_use=False, confirmation_evaluated=False, hidden_predictions_generated=False,
        precision='native BF16 autocast; unchanged frozen training source',
        note='Independent seed stability evidence, never an additional recipe-selection candidate',
    ))
    run([sys.executable, 'train.py', '--config', str(config_path), '--data-dir', str(data_dir),
         '--output-dir', str(OUTPUT), '--device', 'cuda'])
    run([sys.executable, 'predict.py', '--checkpoint', str(OUTPUT / 'best.pt'),
         '--data-dir', str(data_dir), '--output-dir', str(OUTPUT),
         '--device', 'cuda', '--validation-only'])
    evaluation = json.loads((OUTPUT / 'evaluation.json').read_text())
    write('repeat_status.json', dict(
        status='complete', purpose='seed_stability_only', source_commit=LAUNCH['source_commit'],
        seed=2027, validation_accuracy=evaluation['validation_accuracy'],
        flip_tta=evaluation['flip_tta'], parameter_count=evaluation['parameter_count'],
        confirmation_evaluated=False, hidden_predictions_generated=False, used_for_selection=False,
    ))
    print('SEED_REPEAT_COMPLETE', flush=True)
except BaseException as error:
    write('repeat_status.json', dict(status='failed', error_type=type(error).__name__,
          purpose='seed_stability_only', source_commit=LAUNCH['source_commit'],
          confirmation_evaluated=False, hidden_predictions_generated=False))
    raise
'''


def _kaggle_command(explicit: str | None) -> str:
    if explicit:
        return explicit
    sibling = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    found = str(sibling) if sibling.is_file() else shutil.which("kaggle")
    if not found:
        raise FileNotFoundError("Run with the Kaggle CLI environment or supply --kaggle-cli")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--folder", type=Path, default=ROOT / "tmp/kaggle-repeat")
    parser.add_argument("--kaggle-cli")
    parser.add_argument("--accelerator", default="NvidiaL4", choices=("NvidiaL4",))
    parser.add_argument("--max-seconds", type=int, default=9000)
    args = parser.parse_args()
    if not 300 <= args.max_seconds <= 9000:
        raise ValueError("Runtime cap must be between 300 and 9000 seconds")
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    attempt_path = folder / "launch_attempt.json"
    if args.launch and attempt_path.exists():
        raise RuntimeError("A launch was already attempted; inspect its remote status instead of starting another job")
    manifest_path = ROOT / "data/training_manifest.json"
    local_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            blob = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=ROOT)
            hashes[name] = hashlib.sha256(blob).hexdigest()
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, blob)
    payload = bundle.getvalue()
    launch = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": hashes,
        "source_bundle_sha256": hashlib.sha256(payload).hexdigest(),
        "fixed_split_sha256": local_data["split"]["indices_sha256"],
        "kernel_id": KERNEL_ID,
        "runtime_cap_seconds": args.max_seconds,
        "requested_accelerator": args.accelerator,
        "purpose": "independent_seed_stability_only",
        "used_for_selection": False,
        "confirmation_evaluated": False,
        "hidden_predictions_generated": False,
    }
    entry_code = BOOTSTRAP.replace("__BUNDLE__", repr(base64.b64encode(payload).decode("ascii")))
    entry_code = entry_code.replace("__MANIFEST__", repr(json.dumps(launch)))
    compile(entry_code, "repeat_entry.py", "exec")
    (folder / "repeat_entry.py").write_text(entry_code, encoding="utf-8")
    (folder / "source_bundle_manifest.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "Saber WRN22 repeat seed2027 20261030",
        "code_file": "repeat_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": args.accelerator,
        "competition_sources": [COMPETITION],
        "kernel_sources": [SOURCE_KERNEL],
        "dataset_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2), flush=True)
    if not args.launch:
        return
    cli = _kaggle_command(args.kaggle_cli)
    response = subprocess.check_output([cli, "quota", "--format", "json"], text=True)
    quota = json.loads(response)
    gpu = next(item for item in quota if item["resource"] == "GPU")
    remaining_hours = float(gpu["remaining"].removesuffix("h"))
    if remaining_hours < args.max_seconds / 3600:
        raise RuntimeError("Insufficient free GPU quota; no job was launched")
    receipt = {"status": "launch_attempted", "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
               "free_quota_before": quota, **launch}
    attempt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # Exactly one launch request. A timeout or error must be inspected, not retried.
    completed = subprocess.run([cli, "kernels", "push", "-p", str(folder), "-t", str(args.max_seconds),
                                "--accelerator", args.accelerator], check=False, text=True, capture_output=True)
    receipt.update(status="push_accepted" if completed.returncode == 0 else "push_failed",
                   returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    attempt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
