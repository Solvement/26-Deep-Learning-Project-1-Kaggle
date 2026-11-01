"""Run one allowlisted GPU diagnostic under the shared local lease and budget.

No training campaign, submission, confirmation access or arbitrary shell command
is available. Every invocation requires a new immutable run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import psutil
from research.run_local import acquire_lease, now, sha256, supervised_command, used_seconds
from runtime import write_json
from research.submit_candidate import validate_native_repeat_receipt

MODES = ("sam-speed", "shadow-benchmark", "sam-benchmark", "resizemix-benchmark",
         "paired", "paired-v2", "predict", "predict-v2", "refit-predict", "cloud-repeat-predict",
         "cloud-shadow-repeat-predict")
POLICIES = ("plain", "flip", "center80_shift20")


def build_command(args, folder, baseline):
    """Validate required files before acquiring a GPU lease or creating output."""
    inputs = []
    if args.mode in ("cloud-repeat-predict", "cloud-shadow-repeat-predict"):
        if not args.manifest or not getattr(args, "attestation", None):
            raise ValueError(f"{args.mode} requires --manifest and --attestation")
        if args.mode == "cloud-repeat-predict":
            from research.repeat_attestation import validate_attestation
            schema, entry = "research_cloud_repeat_attestation_v1", "research/repeat_attestation.py"
        else:
            from research.shadow_repeat_attestation import validate_attestation
            schema, entry = "research_shadow_cloud_repeat_attestation_v1", "research/shadow_repeat_attestation.py"
        manifest, attestation_path = args.manifest.resolve(), args.attestation.resolve()
        attestation = validate_attestation(attestation_path, manifest_path=manifest)
        if (attestation.get("schema") != schema or
                Path(attestation["attestation_path"]).resolve() != attestation_path or
                Path(attestation["frozen_manifest_path"]).resolve() != manifest or
                attestation.get("attestation_sha256") != sha256(attestation_path) or
                attestation.get("frozen_manifest_sha256") != sha256(manifest) or
                not attestation.get("evidence_sha256")):
            raise ValueError("Complete source-bound cloud repeat attestation is required")
        inputs += [manifest, attestation_path, ROOT/"research/submit_candidate.py"]
        inputs += [Path(path) for path in attestation["evidence_sha256"]]
        inputs += [ROOT/name for name in attestation.get("source_sha256", {})]
        output = folder/"delivery.json"
        command = [sys.executable, entry, "--predict", "--manifest", str(manifest),
                   "--attestation", str(attestation_path), "--output-dir", str(folder), "--device", "cuda"]
    elif args.mode == "refit-predict":
        if args.plan is None:
            raise ValueError("refit-predict requires --plan")
        plan = args.plan.resolve()
        inputs.append(plan)
        inputs += [plan.parent/name for name in ("best.pt", "last.pt", "input_config.json", "config.json",
                    "metrics.json", "history.json", "provenance.json", "environment.json", "supervisor.json")]
        document = json.loads(plan.read_text())
        inputs.append(Path(document["parent_delivery_path"]))
        inputs += [ROOT/name for name in document.get("source_sha256", {})]
        output = folder/"delivery.json"
        command = [sys.executable, "research/refit_delivery.py", "--predict", "--plan", str(plan),
                   "--output-dir", str(folder), "--device", "cuda"]
    elif args.mode in ("predict", "predict-v2"):
        if not all((args.manifest, args.repeat_checkpoint, args.repeat_benchmark, args.repeat_receipt)):
            raise ValueError("predict requires --manifest --repeat-checkpoint --repeat-benchmark --repeat-receipt")
        manifest, repeat, benchmark, receipt = [path.resolve() for path in
            (args.manifest, args.repeat_checkpoint, args.repeat_benchmark, args.repeat_receipt)]
        if args.mode == "predict-v2":
            validate_native_repeat_receipt(repeat, benchmark, receipt, ROOT)
            inputs.append(ROOT/"research/submit_candidate.py")
        inputs += [manifest, repeat, benchmark, receipt]
        frozen = json.loads(manifest.read_text())
        inputs += [Path(path) for path in frozen.get("evidence_sha256", {})]
        inputs += [repeat.parent/name for name in ("config.json", "metrics.json", "history.json", "provenance.json", "environment.json")]
        if json.loads(benchmark.read_text()).get("research_training_integrity"):
            inputs.append(benchmark.with_name(benchmark.stem+".core.json"))
        if frozen.get("data_dir"):
            data = Path(frozen["data_dir"])
            inputs += [data/name for name in ("sample_submission.csv", "test.csv", "official_test.npz", "official_test_source_manifest.json")]
            inputs += [data/name for name in json.loads((data/"training_manifest.json").read_text())["files_sha256"]]
        output = folder/"delivery.json"
        entry = "research/delivery_v2.py" if args.mode == "predict-v2" else "research/delivery.py"
        command = [sys.executable, entry, "--predict", "--manifest", str(manifest),
                   "--repeat-checkpoint", str(repeat), "--repeat-benchmark", str(benchmark), "--repeat-receipt", str(receipt),
                   "--output-dir", str(folder), "--device", "cuda"]
    elif args.mode == "sam-speed":
        if args.config is None:
            raise ValueError("sam-speed requires --config")
        config = args.config.resolve()
        inputs.append(config)
        command = [sys.executable, "research/train_sam.py", "--config", str(config),
                   "--output-dir", str(folder), "--device", "cuda", "--precision", "bf16",
                   "--benchmark-steps", "20"]
        output = folder/"engineering_benchmark.json"
    else:
        if args.checkpoint is None:
            raise ValueError("This diagnostic requires --checkpoint")
        checkpoint = args.checkpoint.resolve()
        if "stability_seed2027" in checkpoint.parts:
            raise ValueError("The validation-only stability repeat is excluded")
        inputs += [checkpoint, baseline]
        if args.mode in ("paired", "paired-v2"):
            if args.candidate_benchmark is None or args.policy is None:
                raise ValueError("paired requires --candidate-benchmark and --policy")
            candidate = args.candidate_benchmark.resolve()
            trusted = ROOT/"results/regularized/best.pt"
            inputs += [candidate, trusted]
            output = folder/"paired.json"
            entry = "research/paired_compare_v2.py" if args.mode == "paired-v2" else "research/paired_compare.py"
            command = [sys.executable, entry, "--candidate-checkpoint", str(checkpoint),
                       "--candidate-benchmark", str(candidate), "--baseline-checkpoint", str(trusted),
                       "--baseline-benchmark", str(baseline), "--policy", args.policy, "--output", str(output),
                       "--device", "cuda"]
        else:
            entry = {"shadow-benchmark": "research/benchmark_shadow.py",
                     "sam-benchmark": "research/benchmark_sam.py",
                     "resizemix-benchmark": "research/benchmark_resizemix.py"}[args.mode]
            output = folder/"benchmark.json"
            command = [sys.executable, entry, "--checkpoint", str(checkpoint), "--baseline", str(baseline),
                       "--output", str(output), "--device", "cuda"]
    inputs.append(ROOT/command[1])
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("A required diagnostic input or entrypoint is missing")
    return command, output, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate-benchmark", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--attestation", type=Path)
    for name in ("manifest", "repeat-checkpoint", "repeat-benchmark", "repeat-receipt"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.run.replace("_", "").isalnum() or args.run.startswith("stability"):
        parser.error("Use a unique alphanumeric exploratory diagnostic ID")
    if not 1 <= args.seconds <= 10800:
        parser.error("--seconds must be from 1 through 10800")
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    base = ROOT/"results/research"
    folder = base/args.run
    if folder.exists():
        raise FileExistsError("Use a new diagnostic directory; existing evidence is immutable")
    command, output, inputs = build_command(args, folder, ROOT/policy["baseline_benchmark"])
    lease = base/"local_lease.json"
    acquire_lease(lease)
    started, receipt, owns_folder = time.perf_counter(), {}, False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        owns_folder = True
        receipt = {"run": args.run, "mode": args.mode, "role": "diagnostic", "backend": "local",
                   "status": "starting", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
                   "started_at": now().isoformat(), "confirmation_evaluated": False, "submitted": False,
                   "supervisor_source_sha256": sha256(Path(__file__))}
        receipt["input_sha256"] = {str(path): sha256(path) for path in inputs}
        receipt_path = folder/"supervisor.json"
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=receipt_path)
        budget = min(args.seconds, policy["local_job_timeout_seconds"],
                     policy["local_gpu_seconds_per_utc_day"]-used, int((deadline-now()).total_seconds()))
        if budget < 1:
            raise RuntimeError("No bounded local diagnostic runtime remains")
        receipt.update(status="benchmarking", reserved_seconds=budget, utc_budget_day=day.date().isoformat())
        write_json(receipt_path, receipt)
        remaining = budget-(time.perf_counter()-started)
        if remaining < 1:
            raise TimeoutError("Diagnostic setup exhausted the remaining runtime")
        supervised_command(command, folder/"diagnostic.log", remaining, receipt_path, receipt)
        if args.mode in ("cloud-repeat-predict", "cloud-shadow-repeat-predict") and any(
                sha256(path) != digest for path, digest in receipt["input_sha256"].items()):
            raise RuntimeError("Cloud repeat attestation or its original evidence changed during prediction")
        result = json.loads(output.read_text())
        if result.get("status") != "complete":
            raise ValueError("Diagnostic did not produce complete evidence")
        receipt.update(status="complete", artifact=output.name, artifact_sha256=sha256(output))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        if lease.exists() and json.loads(lease.read_text()).get("pid") == os.getpid():
            lease.unlink()


if __name__ == "__main__":
    main()
