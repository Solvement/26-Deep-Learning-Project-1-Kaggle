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
import json
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
OUTPUTS = ("baseline/benchmark.json", "candidate/best.pt", "candidate/last.pt", "candidate/metrics.json",
           "candidate/history.json", "candidate/config.json", "candidate/environment.json",
           "candidate/provenance.json", "native_amp_smoke.json", "controller.log",
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


def parse_usage(output):
    patterns = (r"Current balance:\s*([0-9.]+)\s*compute units",
                r"Usage rate:\s*([0-9.]+)/hr", r"Active assignments:\s*(\d+)")
    matches = [re.search(pattern, output) for pattern in patterns]
    if not all(matches):
        raise ValueError("Colab usage response could not be verified")
    return dict(balance_compute_units=float(matches[0][1]),
                rate_compute_units_per_hour=float(matches[1][1]),
                active_assignments=int(matches[2][1]))


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


def prepare(root, config_path, folder, run):
    """Create an exact allowlisted bundle; no directory recursion or networks."""
    root, config_path, folder = Path(root), Path(config_path), Path(folder)
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
    entries = {name: root / name for name in SOURCES}
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


def remote_program(manifest, session, deadline):
    # Every interpolated value is a validated ID or a Python literal; no shell.
    return '''import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
M = ''' + repr(manifest) + '''
D = ''' + repr(deadline) + '''
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
         'source_files_sha256':M['files_sha256'],'run':M['run']}
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
           'shadow_source_sha256':M['files_sha256']['research/train_shadow.py']}
    (R/'native_amp_smoke.json').write_text(json.dumps(smoke,indent=2))
    receipt['native_amp_smoke']=smoke
    del p,opt,scaler
    run(['research/benchmark.py','--checkpoint','baseline/best.pt','--output','baseline/benchmark.json',
         '--device','cuda'], 'baseline/benchmark.log')
    run(['research/train_shadow.py','--config','research/configs/candidate.json','--output-dir','candidate',
         '--data-dir','data','--device','cuda','--precision','fp16'], 'candidate/training.log')
    run(['research/benchmark_shadow.py','--checkpoint','candidate/best.pt','--output','candidate/benchmark.json',
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
               "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
               "started_at": now().isoformat(), "deadline_unix": deadline,
               "assignment_attempts": 0, "no_new_purchases": True, "precision": "fp16",
               "allocation_returned_success": False, "remote_guard_verified": False,
               "confirmation_evaluated": False, "submitted": False,
               "input_bundle_sha256": manifest["bundle_sha256"]}
    acquire_lease(lease_path, receipt)
    private_directory(folder / "private")
    cli = ColabCLI(folder / "private", distro)
    guard = None
    try:
        before = parse_usage(cli.require(["usage"], 30))
        if before["active_assignments"] != 0 or before["rate_compute_units_per_hour"] != 0:
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
        if usage["rate_compute_units_per_hour"] != 0:
            raise RuntimeError("Free runtime expected; refusing compute-unit consumption")
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
        if not args.launch or sha256(args.config) != manifest["files_sha256"]["research/configs/candidate.json"]:
            raise FileExistsError("Prepared run exists; inspect it and use --launch once")
    else:
        private_directory(folder)
        try:
            prepare(ROOT, args.config.resolve(), folder, args.run)
        except BaseException:
            write_json(folder / "preparation_failed.json", {"at": now().isoformat()})
            raise
    if args.launch:
        launch(folder, args.distro)
    print(json.dumps({"run": args.run, "prepared": True, "launch_requested": args.launch,
                      "receipt": str(folder / "supervisor.json") if args.launch else None}))


if __name__ == "__main__":
    main()
