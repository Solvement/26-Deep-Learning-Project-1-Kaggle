"""Supervise fixed SWA engineering or a complete prefix/tail/benchmark job.

GPU work is limited to the shared lease, daily allowance and campaign deadline.
The independent repeat starts its scratch prefix inside this same receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import write_json
import psutil
import torch
from train import DEFAULT_CONFIG, code_provenance, training_source
from research import benchmark as suite
from research.run_local import acquire_lease, now, sha256, supervised_command, used_seconds

SOURCE_FILES = ("train.py", "model.py", "data.py", "runtime.py", "research/swa.py",
    "research/train_swa.py", "research/benchmark_swa.py", "research/run_swa.py",
    "research/benchmark.py", "research/run_local.py", "research/watchdog.py",
    "research/paired_compare.py", "research/benchmark_shadow.py", "research/swa-plan.md",
    "research/train_shadow.py", "research/augmentations.py")


def trainer_api():
    from research import train_swa
    return train_swa


def build_commands(config_path, folder, baseline, parent_last=None, fresh_prefix=False, preflight=False):
    trainer = [sys.executable, "research/train_swa.py", "--config", str(config_path),
        "--output-dir", str(folder), "--data-dir", str(ROOT/"data"), "--device", "cuda", "--precision", "bf16"]
    if preflight:
        return [("preflight", trainer+["--benchmark-steps", "20"])]
    if bool(parent_last) == bool(fresh_prefix):
        raise ValueError("Formal SWA requires exactly one parent-last or fresh-prefix")
    commands = []
    if fresh_prefix:
        prefix = folder/"prefix"
        commands.append(("prefix", [sys.executable, "train.py", "--config", str(prefix/"input_config.json"),
            "--output-dir", str(prefix), "--data-dir", str(ROOT/"data"), "--device", "cuda"]))
        parent_last = prefix/"last.pt"
    commands.append(("tail", trainer+["--parent-last", str(parent_last)]))
    for name, checkpoint in (("swa", "swa.pt"), ("control", "raw_tail.pt")):
        commands.append((name, [sys.executable, "research/benchmark_swa.py", "--checkpoint", str(folder/checkpoint),
            "--output", str(folder/("benchmark.json" if name == "swa" else "control_benchmark.json")),
            "--baseline", str(baseline), "--data-dir", str(ROOT/"data"), "--device", "cuda"]))
    return commands


def validate_preflight(path, config, source):
    path = Path(path).resolve()
    proof = json.loads(path.read_text())
    receipt_path = path.parent/"supervisor.json"
    receipt = json.loads(receipt_path.read_text())
    expected = trainer_api().research_provenance("bf16", config)
    if (proof.get("schema") != "research_swa_engineering_v1" or proof.get("status") != "complete" or
        proof.get("synthetic_only") is not True or proof.get("precision") != "bf16" or
        proof.get("training_data_loaded") is not False or proof.get("hidden_images_loaded") is not False or
        proof.get("confirmation_evaluated") is not False or proof.get("averaging_calls") != 23 or
        proof.get("config_sha256") != trainer_api().config_hash(config) or
        proof.get("research_training") != expected or proof.get("source_sha256") != expected["source_sha256"] or
        proof.get("steps") != 20 or proof.get("timed_steps") != 20 or proof.get("warmup_steps") != 3 or
        proof.get("successful_updates") != 20 or proof.get("total_successful_updates") != 23 or
        proof.get("parameter_changed_steps") != 23 or proof.get("native_bf16_verified") is not True or
        proof.get("finite_losses_verified") is not True or proof.get("finite_parameters_verified") is not True or
        proof.get("ema_tracked") is not True or proof.get("ema_selected") is not False or
        proof.get("parameter_count") != 4166298 or proof.get("model_config") != {
            key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")} or
        receipt.get("mode") != "preflight" or receipt.get("status") != "complete" or
        receipt.get("backend") != "local" or receipt.get("trainer") != "swa" or
        receipt.get("precision") != "bf16" or
        any(type(receipt.get(k)) is not int or receipt[k] <= 0 for k in ("pid", "child_pid", "watchdog_pid")) or
        any(type(receipt.get(k)) not in (int, float) or not math.isfinite(receipt[k]) or receipt[k] <= 0
            for k in ("process_created_at", "child_created_at")) or
        receipt.get("source_sha256") != source or receipt.get("artifact_sha256") != sha256(path)):
        raise ValueError("A complete source/config-bound native BF16 SWA preflight is required")
    median, p90 = proof.get("median_step_seconds"), proof.get("p90_step_seconds")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (median, p90)) or p90 < median:
        raise ValueError("SWA preflight timings are invalid")
    if not receipt.get("finished_at") or not datetime.fromisoformat(receipt["started_at"]) <= datetime.fromisoformat(receipt["finished_at"]) <= now():
        raise ValueError("SWA preflight receipt has invalid completion times")
    return proof, [path, receipt_path]


def required_seconds(proof, fresh_prefix):
    # Fixed before measurement: 50% pipeline margin plus BN/two v3 benchmarks.
    epochs = 200 if fresh_prefix else 40
    return 1.5*proof["median_step_seconds"]*math.ceil(40000/128)*epochs+300


def run(run_id, config_path, mode, parent_last=None, fresh_prefix=False, preflight_path=None, minimum_seconds=900):
    if not run_id.replace("_", "").isalnum() or run_id.startswith("stability"):
        raise ValueError("Use a unique SWA run ID")
    if mode not in {"preflight", "formal"} or type(minimum_seconds) is not int or not 60 <= minimum_seconds <= 10800:
        raise ValueError("SWA mode/minimum runtime is invalid")
    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    config = {**DEFAULT_CONFIG, **json.loads(raw)}
    api = trainer_api()
    api.validate_config(config)
    if mode == "preflight" and (parent_last or fresh_prefix or preflight_path):
        raise ValueError("Synthetic preflight cannot load a training prefix")
    if mode == "formal" and (bool(parent_last) == bool(fresh_prefix) or not preflight_path):
        raise ValueError("Formal SWA requires preflight and exactly one parent-last or fresh-prefix")
    if mode == "formal" and config["seed"] != (2027 if fresh_prefix else 2026):
        raise ValueError("Primary uses its own seed2026 prefix; repeat requires a fresh seed2027 prefix")
    source = {name: sha256(ROOT/name) for name in SOURCE_FILES}
    policy = json.loads((ROOT/"research/campaign.json").read_text())
    deadline = datetime.fromisoformat(policy["stops_at"])
    if now() >= deadline:
        raise RuntimeError("Five-day research window has ended")
    base, folder = ROOT/"results/research", ROOT/"results/research"/run_id
    if folder.exists():
        raise FileExistsError("SWA evidence is immutable; use a new directory")
    manifest = ROOT/"data/training_manifest.json"
    inputs = [config_path, manifest]
    baseline = ROOT/policy["baseline_benchmark"]
    parent_metadata = None
    if mode == "formal":
        baseline_document = json.loads(baseline.read_text())
        signature = {"archive_sha256": training_source(ROOT/"data")["archive_sha256"],
            "split_sha256": json.loads(manifest.read_text())["split"]["indices_sha256"]}
        context = {"device_type": "cuda", "amp_dtype": "torch.bfloat16", "torch_version": torch.__version__,
            "training_core_sha256": code_provenance()["source_sha256"]}
        suite.validate_baseline(baseline_document, baseline_document.get("policies", {}), signature, 128, context,
            sha256(ROOT/"research/benchmark.py"))
        proof, proof_paths = validate_preflight(preflight_path, config, source)
        inputs += [baseline, *proof_paths]
        if parent_last:
            parent_last = Path(parent_last).resolve()
            parent = api.validate_parent(parent_last, config, signature)
            parent_metadata = parent["metadata"]
            inputs += parent["paths"]
    else:
        proof = None
    input_hashes = {str(Path(path).resolve()): sha256(path) for path in inputs}
    commands = build_commands(folder/"input_config.json", folder, baseline, parent_last, fresh_prefix, mode == "preflight")
    lease = base/"local_lease.json"
    acquire_lease(lease)
    started, receipt, owns_folder = time.perf_counter(), {}, False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        owns_folder = True
        receipt = {"run": run_id, "mode": mode, "role": "engineering" if mode == "preflight" else "swa_experiment",
            "backend": "local", "trainer": "swa", "precision": "bf16", "status": "starting", "seed": config["seed"],
            "pid": os.getpid(), "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat(),
            "config_sha256": hashlib.sha256(raw).hexdigest(), "source_config_path": str(config_path),
            "source_sha256": source, "input_sha256": input_hashes, "parent_training": parent_metadata,
            "fresh_prefix": fresh_prefix, "confirmation_evaluated": False, "hidden_images_loaded": False, "submitted": False,
            "stages": []}
        with (folder/"input_config.json").open("xb") as stream:
            stream.write(raw)
        day = now().replace(hour=0, minute=0, second=0, microsecond=0)
        used = used_seconds(base, day, day+timedelta(days=1), exclude=folder/"supervisor.json")
        budget = min(180 if mode == "preflight" else min(10800, policy["local_job_timeout_seconds"]),
            policy["local_gpu_seconds_per_utc_day"]-used, int((deadline-now()).total_seconds()))
        threshold = 60 if mode == "preflight" else max(minimum_seconds, required_seconds(proof, fresh_prefix))
        if budget < threshold:
            raise RuntimeError("Insufficient local budget for the complete fixed SWA job")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("SWA requires actual native CUDA BF16")
        if source != {name: sha256(ROOT/name) for name in SOURCE_FILES} or any(sha256(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("SWA source or immutable input changed before launch")
        receipt.update(status="training", reserved_seconds=budget, utc_budget_day=day.date().isoformat(), required_seconds=threshold)
        write_json(folder/"supervisor.json", receipt)
        if fresh_prefix:
            (folder/"prefix").mkdir()
            write_json(folder/"prefix/input_config.json", api.prefix_config(config))
        for stage, command in commands:
            remaining = budget-(time.perf_counter()-started)
            reserve = 120 if stage in {"prefix", "tail"} else 0
            if remaining-reserve < 1:
                raise TimeoutError("No bounded SWA stage runtime remains")
            receipt["status"] = "benchmarking" if stage in {"swa", "control"} else "training"
            record = {"stage": stage, "command": command, "started_at": now().isoformat()}
            receipt["stages"].append(record)
            write_json(folder/"supervisor.json", receipt)
            supervised_command(command, folder/(stage+".log"), remaining-reserve, folder/"supervisor.json", receipt)
            record.update(finished_at=now().isoformat(), status="complete")
            if stage == "prefix":
                prefix = folder/"prefix"
                origin = {"schema": "research_swa_prefix_origin_v1", "status": "complete", "init": "scratch_seeded",
                    "parent_weights_loaded": False, "resumed": False, "seed": config["seed"],
                    "config_sha256": sha256(prefix/"config.json"), "checkpoint_sha256": sha256(prefix/"last.pt"),
                    "source_sha256": code_provenance()["source_sha256"], "command": command,
                    "rawinputconfig_path": str(prefix/"input_config.json"), "rawinputconfig_sha256": sha256(prefix/"input_config.json"),
                    "started_at": record["started_at"], "finished_at": record["finished_at"]}
                write_json(prefix/"prefix_origin.json", origin)
                parent = api.validate_parent(prefix/"last.pt", config)
                receipt["parent_training"] = parent["metadata"]
        if source != {name: sha256(ROOT/name) for name in SOURCE_FILES} or any(sha256(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("SWA source or original evidence changed during the job")
        if mode == "preflight":
            artifact = folder/"engineering_benchmark.json"
            receipt.update(status="complete", artifact=artifact.name, artifact_sha256=sha256(artifact), finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
            validate_preflight(artifact, config, source)
        else:
            for filename, variant in (("benchmark.json", "swa"), ("control_benchmark.json", "raw_tail_control")):
                document = json.loads((folder/filename).read_text())
                if (document.get("schema") != suite.SCHEMA or document.get("status") != "complete" or
                    document.get("examples") != 5000 or document.get("training_examples") != 40000 or
                    document.get("checkpoint_variant") != variant or document.get("checkpoint_epoch") != 200 or
                    set(document.get("policies", {})) != set(suite.POLICIES) or document.get("config") != config or
                    any(set(report.get("cases", {})) != set(suite.CASES) or any(case.get("examples") != 5000
                        for case in report["cases"].values()) for report in document.get("policies", {}).values()) or
                    document.get("fixed_primary_policy") != "center80_shift20" or
                    not document.get("research_swa_integrity") or document.get("confirmation_evaluated") is not False):
                    raise ValueError("Both fixed SWA/control full development benchmarks are required")
                from research.benchmark_swa import candidate_eligible
                if document.get("candidate_eligible") is not candidate_eligible(variant, document["promotion"], document["promotion_vs_parent"]):
                    raise ValueError("SWA eligibility differs from its two fixed weighted-policy gates")
            receipt.update(status="complete", benchmark_sha256=sha256(folder/"benchmark.json"),
                control_benchmark_sha256=sha256(folder/"control_benchmark.json"),
                benchmark_passed=json.loads((folder/"benchmark.json").read_text())["candidate_eligible"],
                raw_control_prediction_authorized=False, training_examples=40000, validation_examples=5000)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns_folder:
            receipt.update(elapsed_seconds=time.perf_counter()-started, finished_at=now().isoformat())
            write_json(folder/"supervisor.json", receipt)
        if lease.exists() and json.loads(lease.read_text()).get("pid") == os.getpid():
            lease.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--parent-last", type=Path)
    parser.add_argument("--fresh-prefix", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--minimum-seconds", type=int, default=900)
    args = parser.parse_args()
    result = run(args.run, args.config, args.mode, args.parent_last, args.fresh_prefix, args.preflight, args.minimum_seconds)
    print(json.dumps({"run": args.run, "status": result["status"], "mode": args.mode}))


if __name__ == "__main__":
    main()
