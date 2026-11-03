"""Bounded native development-only class slices for the frozen TA comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import trivialaugment_paired_v2 as pair
from research import trivialaugment_delivery as delivery
from research import incumbent_compare as incumbent
from research import preflight_core_control as control
from research.process_tree_guard import may_release_lease, run_guarded

BASE = ROOT / "results/research"
LEASE = BASE / "local_lease.json"
PAIR = BASE / "ta_dualpair_20261103_v2"
ENTRY = Path(__file__).resolve()
GUARD = ROOT / "research/process_tree_guard.py"
STATS = ROOT / "research/development_error_slices.py"
PLAN = ROOT / "research/development-error-slices-plan.md"
SCHEMA = "research_development_error_slices_supervisor_v1"
INPUT_SCHEMA = "research_development_error_slices_inputs_v1"
RESULT_SCHEMA = "research_development_error_slices_result_v1"
ARTIFACT = "development_error_slices.json"
require, sha, read, write = control.require, control.sha256, control.read, control.write


def evidence():
    _, original, paired_inputs = pair.completed_pair(PAIR)
    frozen, documents, ta_inputs, sources = delivery.qualified_inputs()
    sources = {**sources, **pair.source_snapshot(), **{
        p.relative_to(ROOT).as_posix(): sha(p) for p in (ENTRY, STATS, PLAN, ROOT / "research/campaign.json")
    }}
    inputs = {**ta_inputs, **paired_inputs}
    require(not any(Path(path).name in {"official_test.npz", "test.csv", "sample_submission.csv"}
                    for path in inputs), "Diagnostic inputs must not include hidden data")
    return frozen, documents, original, inputs, sources


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
            and spec.get("guard_source_sha256") == sha(GUARD) == pair.PINS["research/process_tree_guard.py"]
            and spec.get("script_sha256") == ready.get("script_sha256") == sha(ENTRY)
            and spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha],
            "Diagnostic script, owner, lease, deadline and nonce must match")
    verify_os_boundary(spec, ready, process)
    return receipt


def ta_checkpoint(path, document, role, train_ids, val_ids):
    import torch
    from research import benchmark as suite
    from research import benchmark_trivialaugment as typed

    folder = path.parent
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    metrics, history, provenance, environment = (
        read(folder / name) for name in ("metrics.json", "history.json", "provenance.json", "environment.json")
    )
    integrity = typed.validate_training(checkpoint, metrics, history, provenance, environment, last,
                                        (folder / "input_config.json").read_bytes())
    require(integrity == document["research_training_integrity"]
            and checkpoint["config"] == read(folder / "config.json") == document["config"]
            and checkpoint["config"]["seed"] == role, "Actual TA typed tensor integrity differs")
    suite.validate_checkpoint(checkpoint, train_ids, val_ids, incumbent.DATA, metrics, incumbent.CORE, history)
    return checkpoint


def worker(folder, manifest_sha):
    receipt = auth(folder, manifest_sha)
    manifest = read(folder / "inputs.json")
    # Parent established the complete typed lineage before allocation. Rebind exact bytes here.
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == folder.name
            and manifest.get("source_sha256") == receipt.get("source_sha256")
            and manifest.get("input_sha256") == receipt.get("input_sha256")
            and all(sha(ROOT / path) == value for path, value in manifest["source_sha256"].items())
            and all(sha(path) == value for path, value in manifest["input_sha256"].items()),
            "Frozen diagnostic inputs changed")

    import runtime  # Windows DLL isolation before Torch.
    import numpy as np
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from train import code_provenance, tensor_images
    from research import benchmark as suite
    from research import paired_compare_v2
    from research.development_error_slices import compare_directions, summarize_slices

    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.__version__ == incumbent.CONTEXT["torch_version"]
            and code_provenance()["source_sha256"] == incumbent.CORE,
            "Actual pinned native CUDA BF16 and original core required")
    runtime.seed_all(2026)
    device = torch.device("cuda")
    require(suite.benchmark_dtype(device) == torch.bfloat16, "Actual BF16 evaluation required")
    frozen = read(delivery.FROZEN_PATH)
    primary = Path(frozen["checkpoint_path"])
    repeat = Path(frozen["repeat_run_path"]) / "best.pt"
    environment = runtime.environment()
    pair.validate_environment(environment, read(primary.parent / "environment.json"))
    data_dir = Path(frozen["data_dir"])
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    require(len(train_ids) == 40000 and len(val_ids) == len(set(val_ids.tolist())) == 5000
            and not set(train_ids.tolist()) & set(val_ids.tolist()), "Original independent 5000 dev required")
    original = read(PAIR / "paired_incumbent.json")
    require(hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest() ==
            original["development_index_order_sha256"], "Exact original development order required")
    x = tensor_images(images[val_ids], device)
    dev_labels = labels[val_ids]
    y = torch.as_tensor(dev_labels, device=device)

    diagnostics = []
    vectors = {}
    flag_hashes = {}
    for role, path, reference, seed in (
        ("candidate", primary, Path(frozen["benchmark_path"]), 2026),
        ("incumbent", pair.INCUMBENT_CHECKPOINT, incumbent.INCUMBENT_PATH, None),
        ("repeat", repeat, repeat.parent / "benchmark.json", 2027),
    ):
        if seed is None:
            checkpoint, document = paired_compare_v2.load_experiment(
                path, reference, incumbent.DATA, incumbent.CONTEXT, incumbent.CORE, train_ids, val_ids)
        else:
            document = read(reference)
            require(sha(path) == document["checkpoint_sha256"], "TA checkpoint/benchmark hash differs")
            checkpoint = ta_checkpoint(path, document, seed, train_ids, val_ids)
        model = build_model(checkpoint["model_config"])
        require(count_parameters(model) == count_parameters(model, True) == checkpoint["parameter_count"] <= 5000000,
                "Single residual-model parameter budget differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),
                "Selected tensor state is nonfinite")
        model = model.to(device=device, memory_format=torch.channels_last).eval()
        # The frozen collector permits only original pairing role labels. Collect
        # repeat as a separate candidate, then attach its actual diagnostic role.
        role_diagnostics = []
        vector = pair.collect_correctness_exact(
            model, x, y, pair.POLICY, 128, device, document["policies"][pair.POLICY],
            "candidate" if role == "repeat" else role, role_diagnostics, folder / (role + "_case_diagnostics.json"))
        diagnostics.extend({**row, "comparator": role} for row in role_diagnostics)
        digest = hashlib.sha256(vector.astype(np.uint8).tobytes()).hexdigest()
        if role != "repeat":
            require(digest == original[role + "_correctness_sha256"],
                    "Per-image correctness differs from authenticated completed pair")
        vectors[role], flag_hashes[role] = vector, digest
        del model, checkpoint
    primary_stats = summarize_slices(dev_labels, vectors["candidate"], vectors["incumbent"], suite.CASES)
    repeat_stats = summarize_slices(dev_labels, vectors["repeat"], vectors["incumbent"], suite.CASES)
    output = folder / ARTIFACT
    require(not output.exists(), "Diagnostic output must be new")
    write(output, {"schema": RESULT_SCHEMA, "status": "complete", "input_manifest_sha256": manifest_sha,
        "policy": pair.POLICY, "development_examples": 5000, "cases": list(suite.CASES),
        "actual_environment": environment, "data_signature": incumbent.DATA, "evaluation_context": incumbent.CONTEXT,
        "correctness_sha256": flag_hashes, "case_diagnostics": diagnostics,
        "primary": primary_stats, "repeat": repeat_stats, "direction_check": compare_directions(primary_stats, repeat_stats),
        "same_development_images": True, "raw_flags_persisted": False, "hidden_images_loaded": False,
        "confirmation_evaluated": False, "promotion_authorized": False, "submission_authorized": False})


def parent(run):
    require(os.name == "nt" and re.fullmatch(r"dev_error_slices_[A-Za-z0-9_]+", run),
            "Use a unique Windows native diagnostic ID with dev_error_slices_ prefix")
    folder = BASE / run
    require(not folder.exists() and not LEASE.exists(), "Existing diagnostic or local lease requires review")
    _, _, _, inputs, sources = evidence()
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
        write(manifest_path, {"schema": INPUT_SCHEMA, "run": run, "source_sha256": sources, "input_sha256": inputs})
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
                and result.get("input_manifest_sha256") == manifest_sha and len(result.get("case_diagnostics", [])) == 18
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
    mode.add_argument("--run")
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
