"""Compare existing terminal online WRN states on the fixed official dev suite."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import paired_incumbent as supervision
from research import incumbent_compare as incumbent
from research import preflight_core_control as control
from research.process_tree_guard import may_release_lease, run_guarded

BASE = ROOT / "results/research"
LEASE = BASE / "local_lease.json"
ENTRY = Path(__file__).resolve()
GUARD = ROOT / "research/process_tree_guard.py"
PLAN = ROOT / "research/development-architecture-pair-plan.md"
STATS = ROOT / "research/development_error_slices.py"
RUNS = {"dev_architecture_pair_20261103_v1": "terminal_online"}
SOURCES = {"wrn22_4": ROOT / "results/regularized", "wrn34_3": BASE / "wrn34_3_s2026"}
BENCHMARKS = {"wrn22_4": BASE / "baseline_v3_verified.json", "wrn34_3": SOURCES["wrn34_3"] / "benchmark.json"}
PARAMETERS = {"wrn22_4": 4298970, "wrn34_3": 4166298}
ARCHITECTURES = {"wrn22_4": (22, 4), "wrn34_3": (34, 3)}
DIAGNOSTIC_CONTEXT = {"seed": 2026, "device": "cuda", "dtype": "torch.bfloat16", "batch_size": 128,
                      "cudnn_benchmark": False, "cudnn_deterministic": True,
                      "torch_deterministic_algorithms": False}
SCHEMA = "research_development_architecture_pair_supervisor_v1"
INPUT_SCHEMA = "research_development_architecture_pair_inputs_v1"
RESULT_SCHEMA = "research_development_architecture_pair_result_v1"
ARTIFACT = "development_architecture_pair.json"
POLICY = "center80_shift20"
LAST_FIELDS = {"model", "ema", "optimizer", "epoch", "updates", "config", "data_signature", "provenance",
               "history", "best_accuracy", "best_loss", "elapsed_seconds", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
require, sha, read, write = control.require, control.sha256, control.read, control.write
ARTIFACT_PINS = {
    "results/regularized/config.json": "3388cd8c2bde41912bd42b2c47dc11b58a3f360b4a8ab8e3a73aa7b1211af11e",
    "results/regularized/environment.json": "63f182c467c2d2126e6224363f5378d8105a5f6e272e6dbffad67f205e90d84c",
    "results/regularized/metrics.json": "c980910d52e9bb8d751c228ee035cb963df986737ae29072114c49bb561f2cf0",
    "results/regularized/history.json": "867622a8425cb1d402d9b2088d7c2faf2dd51e3275fb96684aa5a61baaa40b71",
    "results/regularized/provenance.json": "bd57c7948630be9f3bab5875f2de68a963e037ef5fb289d5df12d82858f2cdda",
    "results/regularized/last.pt": "9f024a8afde07c2646055c587415aa7adb4a525de678ab52591a10b45bc76f69",
    "results/research/wrn34_3_s2026/config.json": "e4860e764eae3e16eb3f69695becdd76c64c4eb8a977a810b00b2e45ff98af95",
    "results/research/wrn34_3_s2026/environment.json": "63f182c467c2d2126e6224363f5378d8105a5f6e272e6dbffad67f205e90d84c",
    "results/research/wrn34_3_s2026/metrics.json": "0d1dc5b3c1ede77e844ad8d976274fc8f9bfcc5d52357d794cdc481905f603c5",
    "results/research/wrn34_3_s2026/history.json": "21877b3f17750be97680f0af75d546798ad6d34fbca1338b2a8383d956d69816",
    "results/research/wrn34_3_s2026/provenance.json": "4217d9c223264f9b6d88376352ad47c9794b747c851c38c4e256afe2f6759349",
    "results/research/wrn34_3_s2026/last.pt": "0686e9e7698512fb87233586c00e956e05876fdb869633019c1b394a4029bb8b",
    "results/research/baseline_v3_verified.json": "eced314d97fcb5ddef4cad25c6968ba299d0e8a2c79af6bdcbdbf305b79c632d",
    "results/research/wrn34_3_s2026/benchmark.json": "cf8cfa9fbb103db3ad705a5971b993d187e055dff6e6d0a0a3c1ac5964168690",
}


def auth(folder, manifest_sha):
    from research.profile_augmix import verify_os_boundary

    process = psutil.Process()
    parent = process.parent()
    argv = process.cmdline()
    require(len(argv) == 4 and Path(argv[0]).resolve() == Path(sys.executable).resolve()
            and Path(argv[1]).resolve() == GUARD and argv[2] == "--bootstrap",
            "Native diagnostic bootstrap required")
    spec_path = Path(argv[3]).resolve()
    require(spec_path.name == "spec.json" and spec_path.parent.parent == folder
            and folder.parent == BASE.resolve(), "Own bounded diagnostic directory required")
    spec, ready = read(spec_path), read(spec_path.with_name("ready.json"))
    receipt, lease = read(folder / "supervisor.json"), read(LEASE)
    nonce = spec_path.parent.name.removeprefix("process_tree_")
    now = time.time()
    require(parent is not None and receipt.get("schema") == SCHEMA
            and receipt.get("status") == "benchmarking" and receipt.get("run") == folder.name
            and receipt.get("input_manifest_sha256") == manifest_sha == sha(folder / "inputs.json")
            and receipt.get("reserved_seconds") == 180
            and receipt.get("deadline_unix") <= datetime.fromisoformat(receipt["started_at"]).timestamp() + 180.1
            and now < spec.get("deadline_unix") <= receipt["deadline_unix"]
            and parent.pid == receipt.get("pid") == lease.get("pid") == spec.get("host_pid")
            and all(abs(parent.create_time() - value) < .01 for value in (
                receipt.get("process_created_at", 0), lease.get("process_created_at", 0), spec.get("host_created_at", 0)))
            and process.pid == receipt.get("child_pid") == ready.get("pid")
            and all(abs(process.create_time() - value) < .01 for value in (
                receipt.get("child_created_at", 0), ready.get("created_at", 0)))
            and lease.get("cleanup_requires_reconciliation") is not True
            and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1"
            and re.fullmatch(r"[0-9a-f]{32}", nonce) and spec.get("nonce") == ready.get("nonce") == nonce
            and spec.get("job_name") == "Local\\SaberAugMix_" + nonce and ready.get("method") == "windows_job_v1"
            and Path(spec.get("script", "")).resolve() == ENTRY and Path(spec.get("cwd", "")).resolve() == ROOT
            and spec.get("guard_source_sha256") == sha(GUARD) == supervision.PINS["research/process_tree_guard.py"]
            and spec.get("script_sha256") == ready.get("script_sha256") == sha(ENTRY)
            and spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha],
            "Diagnostic script, owner, lease, deadline and nonce must match")
    verify_os_boundary(spec, ready, process)
    return receipt


def training_evidence(role):
    folder = SOURCES[role]
    config, environment, metrics, history, provenance = (
        read(folder / name) for name in ("config.json", "environment.json", "metrics.json", "history.json", "provenance.json")
    )
    benchmark = read(BENCHMARKS[role])
    require((config["depth"], config["widen_factor"]) == ARCHITECTURES[role]
            and config["epochs"] == 160 and config["batch_size"] == 128 and config["train_all"] is False
            and config["seed"] == config["split_seed"] == 2026
            and metrics["config"] == config and metrics["status"] == "complete" and metrics["epochs_completed"] == 160
            and metrics["training_examples"] == provenance["train_examples"] == 40000
            and metrics["validation_examples"] == provenance["validation_examples"] == 5000
            and metrics["parameter_count"] == provenance["parameter_count"] == PARAMETERS[role]
            and [row.get("epoch") for row in history] == list(range(1, 161))
            and provenance["code"]["source_sha256"] == incumbent.CORE
            and provenance["data_signature"] == incumbent.DATA
            and benchmark["config"] == config
            and benchmark["training_metrics_sha256"] == sha(folder / "metrics.json")
            and benchmark["training_history_sha256"] == sha(folder / "history.json"),
            "Complete original 160-epoch source-bound training evidence required")
    incumbent.validate_document(benchmark)
    return config, environment, metrics, history, provenance, benchmark


def evidence():
    sources = {**supervision.source_snapshot(), **{
        p.relative_to(ROOT).as_posix(): sha(p) for p in (ENTRY, PLAN, STATS)
    }}
    data_paths, _ = control.official_inputs(ROOT / "data")
    inputs = {str(path.resolve()): sha(path) for path in data_paths}
    for name, digest in ARTIFACT_PINS.items():
        path = ROOT / name
        require(sha(path) == digest, "Pinned terminal or recorded benchmark bytes differ: " + name)
        inputs[str(path.resolve())] = digest
    a, b = (training_evidence(role) for role in SOURCES)
    require(set(a[0]) == set(b[0]) and {k for k in a[0] if a[0][k] != b[0][k]} == {"depth", "widen_factor"}
            and a[1] == b[1], "Only architecture may differ in recorded config; recorded environments must match")
    return inputs, sources


def terminal_model(role, config, metrics, history, provenance, device):
    import torch
    from model import build_model, count_parameters

    state = torch.load(SOURCES[role] / "last.pt", map_location="cpu", weights_only=False)
    require(set(state) == LAST_FIELDS and state["epoch"] == 160 and state["updates"] == 50080
            and state["config"] == config and state["data_signature"] == incumbent.DATA
            and state["provenance"] == provenance["code"] and state["history"] == history
            and state["best_accuracy"] == metrics["best_validation_accuracy"]
            and state["best_loss"] == metrics["best_validation_loss"]
            and state["elapsed_seconds"] == metrics["elapsed_seconds"]
            and isinstance(state["optimizer"], dict) and set(state["optimizer"]) == {"state", "param_groups"}
            and len(state["optimizer"]["param_groups"]) == 1
            and isinstance(state["python_rng"], tuple) and isinstance(state["numpy_rng"], tuple)
            and isinstance(state["torch_rng"], torch.Tensor) and isinstance(state["cuda_rng"], list)
            and len(state["cuda_rng"]) >= 1 and all(isinstance(row, torch.Tensor) for row in state["cuda_rng"]),
            "Terminal original training state schema, optimizer/RNG metadata or chronology differs")
    model = build_model({key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")})
    require(count_parameters(model) == count_parameters(model, True) == PARAMETERS[role] <= 5000000
            and isinstance(state["model"], dict) and set(state["model"]) == set(state["ema"])
            and all(bool(torch.isfinite(value).all()) for value in state["model"].values()),
            "Exact finite terminal online residual state required")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device=device, memory_format=torch.channels_last).eval()


def collect(model, images, labels, device, benchmark, role):
    import numpy as np
    import torch
    from research import benchmark as suite

    vectors = np.empty((5000, len(suite.CASES)), dtype=np.bool_)
    diagnostics = []
    with torch.inference_mode():
        for column, case in enumerate(suite.CASES):
            digest = hashlib.sha256()
            for start in range(0, len(labels), 128):
                stop = start + 128
                pixels = suite.stress(images[start:stop], case, start)
                digest.update(pixels.cpu().numpy().tobytes())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = suite.infer(model, pixels, POLICY)
                require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()),
                        "Actual finite native BF16 logits required")
                vectors[start:start + len(pixels), column] = logits.argmax(1).eq(labels[start:stop]).cpu().numpy()
            pixels_sha = digest.hexdigest()
            require(pixels_sha == benchmark["policies"][POLICY]["cases"][case]["input_pixel_sha256"],
                    "Regenerated transformed pixels differ from frozen benchmark")
            row = {"case": case, "examples": 5000, "input_pixel_sha256": pixels_sha,
                   "terminal_online_correct": int(vectors[:, column].sum())}
            diagnostics.append(row)
            print(json.dumps({"event": "terminal_architecture_case", "role": role, **row}), flush=True)
    return vectors, diagnostics


def worker(folder, manifest_sha):
    receipt = auth(folder, manifest_sha)
    manifest = read(folder / "inputs.json")
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == folder.name
            and manifest.get("model") == RUNS[folder.name] and manifest.get("source_sha256") == receipt.get("source_sha256")
            and manifest.get("input_sha256") == receipt.get("input_sha256")
            and all(sha(ROOT / path) == value for path, value in manifest["source_sha256"].items())
            and all(sha(path) == value for path, value in manifest["input_sha256"].items()),
            "Frozen diagnostic inputs changed")
    import runtime  # Windows DLL isolation before Torch.
    import numpy as np
    import torch
    from data import load_training_arrays
    from train import code_provenance, tensor_images
    from research import benchmark as suite
    from research.development_error_slices import summarize_slices
    from research.paired_compare import paired_bootstrap

    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.__version__ == incumbent.CONTEXT["torch_version"]
            and code_provenance()["source_sha256"] == incumbent.CORE,
            "Pinned native CUDA BF16 and core source required")
    runtime.seed_all(DIAGNOSTIC_CONTEXT["seed"])
    historical_environment = runtime.environment()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    environment = runtime.environment()
    require(environment["cudnn_benchmark"] is False and torch.backends.cudnn.deterministic is True
            and torch.are_deterministic_algorithms_enabled() is False,
            "Fixed process-local diagnostic backend required")
    device = torch.device("cuda")
    images, labels, train_ids, val_ids = load_training_arrays(ROOT / "data")
    require(len(train_ids) == 40000 and len(val_ids) == len(set(val_ids.tolist())) == 5000
            and not set(train_ids.tolist()) & set(val_ids.tolist()), "Original independent 5000 dev required")
    x, dev_labels = tensor_images(images[val_ids], device), labels[val_ids]
    y = torch.as_tensor(dev_labels, device=device)
    vectors, diagnostics, training = {}, {}, {}
    for role in SOURCES:
        config, recorded, metrics, history, provenance, benchmark = training_evidence(role)
        removed = "process_local_cudnn_path_entries_removed"
        require(set(historical_environment) == set(recorded)
                and all(historical_environment[key] == value for key, value in recorded.items() if key != removed)
                and type(historical_environment[removed]) is int and historical_environment[removed] >= 0,
                "Native diagnostic hardware/software differs from recorded training")
        model = terminal_model(role, config, metrics, history, provenance, device)
        vectors[role], diagnostics[role] = collect(model, x, y, device, benchmark, role)
        training[role] = {"architecture": list(ARCHITECTURES[role]), "parameters": PARAMETERS[role],
            "last_sha256": ARTIFACT_PINS[(SOURCES[role] / "last.pt").relative_to(ROOT).as_posix()],
            "historical_benchmark_sha256": ARTIFACT_PINS[BENCHMARKS[role].relative_to(ROOT).as_posix()],
            "recorded_training_commit": provenance["code"]["git_commit"],
            "recorded_training_environment_sha256": ARTIFACT_PINS[
                (SOURCES[role] / "environment.json").relative_to(ROOT).as_posix()], "epochs": 160,
            "updates": 50080, "weight_variant": "terminal_online_model"}
        del model
    cases = []
    for a, b in zip(diagnostics["wrn22_4"], diagnostics["wrn34_3"], strict=True):
        require(a["case"] == b["case"] and a["input_pixel_sha256"] == b["input_pixel_sha256"],
                "Matched transformed pixels required")
        cases.append({"case": a["case"], "examples": 5000, "input_pixel_sha256": a["input_pixel_sha256"],
                      "wrn22_4_correct": a["terminal_online_correct"], "wrn34_3_correct": b["terminal_online_correct"]})
    output = folder / ARTIFACT
    require(not output.exists(), "Diagnostic output must be new")
    write(output, {"schema": RESULT_SCHEMA, "status": "complete", "model": RUNS[folder.name],
        "input_manifest_sha256": manifest_sha, "development_examples": 5000,
        "development_index_order_sha256": hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest(),
        "policy": POLICY, "actual_environment": environment, "diagnostic_context": DIAGNOSTIC_CONTEXT,
        "data_signature": incumbent.DATA, "same_transformed_pixels_verified": True, "cases": cases,
        "training": training, "delta_orientation": "wrn34_3 minus wrn22_4",
        "training_runtime_scope": "Separate historical jobs with equal recorded environment; exact runtime states and trajectories are unobserved.",
        "architecture_scope": "Depth and width change together; one historical seed each; no causal or seed-variance estimate.",
        "paired_clean_and_mean_stress": paired_bootstrap(vectors["wrn34_3"], vectors["wrn22_4"]),
        "slices": summarize_slices(dev_labels, vectors["wrn34_3"], vectors["wrn22_4"], suite.CASES),
        "raw_flags_persisted": False, "hidden_images_loaded": False,
        "confirmation_evaluated": False, "promotion_authorized": False, "submission_authorized": False})


def parent(run):
    require(os.name == "nt" and run in RUNS, "Use a predeclared Windows diagnostic run ID")
    folder = BASE / run
    require(not folder.exists() and not LEASE.exists(), "Existing diagnostic or local lease requires review")
    inputs, sources = evidence()
    policy, account = read(ROOT / "research/campaign.json"), control.accounting()
    require(datetime.fromisoformat(policy["stops_at"]).astimezone(timezone.utc) ==
            datetime(2026, 11, 5, 5, tzinfo=timezone.utc), "Campaign stop differs")
    control.claim_lease(LEASE)
    receipt = {}
    owns = False
    try:
        started = account["now"]()
        day = started.replace(hour=0, minute=0, second=0, microsecond=0)
        budget = control.bounded_seconds(policy, started, account["used_seconds"](BASE, day, day + timedelta(days=1)))
        folder.mkdir(exist_ok=False)
        owns = True
        manifest_path = folder / "inputs.json"
        write(manifest_path, {"schema": INPUT_SCHEMA, "run": run, "model": RUNS[run], "source_sha256": sources, "input_sha256": inputs})
        manifest_sha = sha(manifest_path)
        receipt = {"schema": SCHEMA, "run": run, "backend": "local", "status": "benchmarking",
            "role": "diagnostic_only", "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
            "started_at": started.isoformat(), "reserved_seconds": int(budget), "deadline_unix": started.timestamp() + budget,
            "source_sha256": sources, "input_sha256": inputs, "input_manifest_sha256": manifest_sha,
            "confirmation_evaluated": False, "submitted": False}
        receipt_path = folder / "supervisor.json"
        write(receipt_path, receipt)
        remaining = receipt["deadline_unix"] - time.time() - 2
        require(remaining > 0, "Diagnostic setup exceeded reservation")
        run_guarded([sys.executable, str(ENTRY), "--_worker", str(folder.resolve()), "--_manifest-sha", manifest_sha],
                    folder / "diagnostic.log", remaining, receipt_path, receipt, cwd=ROOT)
        result = read(folder / ARTIFACT)
        require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "complete"
                and result.get("input_manifest_sha256") == manifest_sha and result.get("diagnostic_context") == DIAGNOSTIC_CONTEXT
                and result.get("model") == RUNS[run] and len(result.get("cases", [])) == 6
                and result.get("same_transformed_pixels_verified") is True
                and all(result.get(key) is False for key in ("raw_flags_persisted", "hidden_images_loaded",
                    "confirmation_evaluated", "promotion_authorized", "submission_authorized"))
                and all(sha(ROOT / path) == digest for path, digest in sources.items())
                and all(sha(path) == digest for path, digest in inputs.items())
                and time.time() <= receipt["deadline_unix"], "Complete bounded unchanged diagnostic required")
        receipt.update(status="complete", artifact=ARTIFACT, artifact_sha256=sha(folder / ARTIFACT))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns and "schema" in receipt:
            finished = account["now"]()
            receipt.update(finished_at=finished.isoformat(), elapsed_seconds=(finished - started).total_seconds())
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            write(folder / "supervisor.json", receipt)
        if LEASE.exists():
            owner = read(LEASE)
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at", 0) - psutil.Process().create_time()) < .01:
                if control.lease_can_release(owner, receipt):
                    LEASE.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False,
                                 supervisor=str(folder / "supervisor.json"))
                    write(LEASE, owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", choices=RUNS)
    mode.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker.resolve(), args._manifest_sha)
    else:
        require(args._manifest_sha is None, "Manifest is reserved for authenticated worker")
        parent(args.run)


if __name__ == "__main__":
    main()
