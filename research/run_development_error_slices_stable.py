"""Bounded development-only class slices under a fixed cuDNN diagnostic context."""
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
from research import run_development_error_slices as original_entry
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
PLAN = ROOT / "research/development-error-slices-stable-plan.md"
ORIGINAL_ENTRY_SHA = "c8149119f3ae966ef75794395f1912bd30441828b5863bf674afd00fb8ec3bf0"
FAILED = BASE / "dev_error_slices_20261103_v1"
RUNS = ("dev_error_slices_stable_20261103_a", "dev_error_slices_stable_20261103_b")
DIAGNOSTIC_CONTEXT = {"seed": 2026, "device": "cuda", "dtype": "torch.bfloat16", "batch_size": 128,
                      "cudnn_benchmark": False, "cudnn_deterministic": True,
                      "torch_deterministic_algorithms": False}
SCHEMA = "research_development_error_slices_stable_supervisor_v2"
INPUT_SCHEMA = "research_development_error_slices_stable_inputs_v2"
RESULT_SCHEMA = "research_development_error_slices_stable_result_v2"
ARTIFACT = "development_error_slices.json"
require, sha, read, write = control.require, control.sha256, control.read, control.write


def evidence():
    original_entry_path = Path(original_entry.__file__).resolve()
    require(sha(original_entry_path) == ORIGINAL_ENTRY_SHA, "Frozen v1 diagnostic source differs")
    frozen, documents, original, inputs, sources = original_entry.evidence()
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ENTRY, PLAN)})
    sources[original_entry_path.relative_to(ROOT).as_posix()] = ORIGINAL_ENTRY_SHA
    failed_receipt = FAILED / "supervisor.json"
    failed_cases = FAILED / "candidate_case_diagnostics.json"
    require(sha(failed_receipt) == "5a634a128917d84395ca2415d2cbb9778a11570aaae7eb936f84744fbe4b91ab"
            and sha(failed_cases) == "802ba140b3f2f3ae76bfd056daaecbcc77fadaef090fc954760e67ab07ef8d49",
            "Frozen v1 failure files changed")
    receipt, cases = read(failed_receipt), read(failed_cases)["rows"]
    require(receipt.get("schema") == original_entry.SCHEMA and receipt.get("status") == "failed"
            and receipt.get("process_tree_cleanup_verified") is True
            and len(cases) == 2 and cases[-1].get("case") == "noise_004"
            and cases[-1].get("pixels_match") is True and cases[-1].get("accuracy_matches") is False
            and cases[-1].get("actual_correct_count") == 4253
            and cases[-1].get("expected_correct_count") == 4254,
            "Observed v1 exact-case failure and completed cleanup required")
    inputs.update({str(p.resolve()): sha(p) for p in (failed_receipt, failed_cases)})
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


def collect(model, images, labels, device, reference, role):
    import numpy as np
    import torch
    from research import benchmark as suite

    vectors = np.empty((5000, len(suite.CASES)), dtype=np.bool_)
    diagnostics = []
    with torch.inference_mode():
        for column, case in enumerate(suite.CASES):
            digest = hashlib.sha256()
            for start in range(0, len(labels), DIAGNOSTIC_CONTEXT["batch_size"]):
                stop = start + DIAGNOSTIC_CONTEXT["batch_size"]
                batch = suite.stress(images[start:stop], case, start)
                digest.update(batch.cpu().numpy().tobytes())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = suite.infer(model, batch, pair.POLICY)
                require(bool(torch.isfinite(logits).all()), "Nonfinite diagnostic logits")
                vectors[start:start + len(batch), column] = logits.argmax(1).eq(labels[start:stop]).cpu().numpy()
            row = pair.case_diagnostic("candidate" if role == "repeat" else role, pair.POLICY, case,
                                       digest.hexdigest(), int(vectors[:, column].sum()), reference["cases"][case])
            require(row["pixels_match"], "Diagnostic stress pixels differ from frozen benchmark")
            diagnostics.append({**row, "comparator": role,
                "actual_minus_historical_correct_count": row["actual_correct_count"] - row["expected_correct_count"]})
    return vectors, diagnostics


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
    runtime.seed_all(DIAGNOSTIC_CONTEXT["seed"])
    historical_environment = runtime.environment()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    require(suite.benchmark_dtype(device) == torch.bfloat16, "Actual BF16 evaluation required")
    frozen = read(delivery.FROZEN_PATH)
    primary = Path(frozen["checkpoint_path"])
    repeat = Path(frozen["repeat_run_path"]) / "best.pt"
    environment = runtime.environment()
    pair.validate_environment(historical_environment, read(primary.parent / "environment.json"))
    require(environment["cudnn_benchmark"] is False and torch.backends.cudnn.deterministic is True
            and torch.are_deterministic_algorithms_enabled() is False, "Fixed process-local diagnostic backend required")
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
            checkpoint = original_entry.ta_checkpoint(path, document, seed, train_ids, val_ids)
        model = build_model(checkpoint["model_config"])
        require(count_parameters(model) == count_parameters(model, True) == checkpoint["parameter_count"] <= 5000000,
                "Single residual-model parameter budget differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),
                "Selected tensor state is nonfinite")
        model = model.to(device=device, memory_format=torch.channels_last).eval()
        vector, role_diagnostics = collect(model, x, y, device, document["policies"][pair.POLICY], role)
        diagnostics.extend(role_diagnostics)
        digest = hashlib.sha256(vector.astype(np.uint8).tobytes()).hexdigest()
        vectors[role], flag_hashes[role] = vector, digest
        del model, checkpoint
    primary_stats = summarize_slices(dev_labels, vectors["candidate"], vectors["incumbent"], suite.CASES)
    repeat_stats = summarize_slices(dev_labels, vectors["repeat"], vectors["incumbent"], suite.CASES)
    output = folder / ARTIFACT
    require(not output.exists(), "Diagnostic output must be new")
    write(output, {"schema": RESULT_SCHEMA, "status": "complete", "input_manifest_sha256": manifest_sha,
        "policy": pair.POLICY, "development_examples": 5000, "cases": list(suite.CASES),
        "actual_environment": environment, "data_signature": incumbent.DATA,
        "diagnostic_context": DIAGNOSTIC_CONTEXT, "historical_evaluation_context": incumbent.CONTEXT,
        "historical_correctness_sha256": {role: original[role + "_correctness_sha256"]
                                           for role in ("candidate", "incumbent")},
        "correctness_sha256": flag_hashes, "case_diagnostics": diagnostics,
        "primary": primary_stats, "repeat": repeat_stats, "direction_check": compare_directions(primary_stats, repeat_stats),
        "same_development_images": True, "raw_flags_persisted": False, "hidden_images_loaded": False,
        "confirmation_evaluated": False, "promotion_authorized": False, "submission_authorized": False})


def parent(run):
    require(os.name == "nt" and run in RUNS, "Use a predeclared Windows diagnostic run ID")
    if run == RUNS[1]:
        completed(RUNS[0])
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
                and result.get("input_manifest_sha256") == manifest_sha and result.get("diagnostic_context") == DIAGNOSTIC_CONTEXT
                and len(result.get("case_diagnostics", [])) == 18
                and all(row.get("pixels_match") is True for row in result["case_diagnostics"])
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


def completed(run):
    from research.process_tree_guard import matching_process

    folder = BASE / run
    manifest, receipt, result = (read(folder / n) for n in ("inputs.json", "supervisor.json", ARTIFACT))
    manifest_sha = sha(folder / "inputs.json")
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete"
            and receipt.get("role") == "diagnostic_only" and receipt.get("run") == run
            and receipt.get("process_tree_cleanup_verified") is True and receipt.get("reserved_seconds") == 180
            and 0 < receipt.get("elapsed_seconds", 0) <= 180.1
            and receipt.get("artifact") == ARTIFACT and receipt.get("artifact_sha256") == sha(folder / ARTIFACT)
            and receipt.get("input_manifest_sha256") == result.get("input_manifest_sha256") == manifest_sha
            and receipt.get("source_sha256") == manifest.get("source_sha256")
            and receipt.get("input_sha256") == manifest.get("input_sha256")
            and manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == run
            and result.get("schema") == RESULT_SCHEMA and result.get("status") == "complete"
            and result.get("diagnostic_context") == DIAGNOSTIC_CONTEXT,
            "Complete immutable diagnostic receipt required")
    proofs = receipt.get("process_tree_guard_results", [])
    require(len(proofs) == 1, "Exactly one diagnostic native guard required")
    proof = proofs[0]
    directory = Path(proof["guard_directory"]).resolve()
    require(directory.parent == folder.resolve() and directory.name == "process_tree_" + proof["nonce"],
            "Own diagnostic guard proof directory required")
    spec, ready, completion, host = [read(directory / name) for name in
        ("spec.json", "ready.json", "completion.json", "host_result.json")]
    require(host == proof and proof.get("method") == "windows_job_v1"
            and proof.get("cleanup_verified") is True and proof.get("completion_authenticated") is True
            and proof.get("timed_out") is False and proof.get("completion_exit_code") == proof.get("child_returncode") == 0
            and spec.get("guard_source_sha256") == proof.get("guard_source_sha256")
            == pair.PINS["research/process_tree_guard.py"]
            and spec.get("nonce") == ready.get("nonce") == completion.get("nonce") == proof["nonce"]
            and spec.get("script") == str(ENTRY)
            and spec.get("script_sha256") == ready.get("script_sha256") == completion.get("script_sha256")
            == manifest["source_sha256"][ENTRY.relative_to(ROOT).as_posix()] == sha(ENTRY)
            and spec.get("arguments") == ["--_worker", str(folder.resolve()), "--_manifest-sha", manifest_sha]
            and spec.get("host_pid") == receipt["pid"] and spec.get("host_created_at") == receipt["process_created_at"]
            and ready.get("pid") == completion.get("pid") == proof.get("child_pid") == receipt["child_pid"]
            and ready.get("created_at") == completion.get("created_at") == proof.get("child_created_at")
            == receipt["child_created_at"] and completion.get("exit_code") == 0
            and matching_process(receipt["pid"], receipt["process_created_at"]) is None
            and matching_process(receipt["child_pid"], receipt["child_created_at"]) is None,
            "Authenticated stopped diagnostic process and cleanup required")
    return manifest, receipt, result


def verify_replication():
    manifest_a, receipt_a, result_a = completed(RUNS[0])
    manifest_b, receipt_b, result_b = completed(RUNS[1])
    require(datetime.fromisoformat(receipt_a["finished_at"]) < datetime.fromisoformat(receipt_b["started_at"])
            and (receipt_a["child_pid"], receipt_a["child_created_at"]) != (receipt_b["child_pid"], receipt_b["child_created_at"])
            and manifest_a["source_sha256"] == manifest_b["source_sha256"]
            and manifest_a["input_sha256"] == manifest_b["input_sha256"]
            and all(sha(ROOT / path) == digest for path, digest in manifest_b["source_sha256"].items())
            and all(sha(path) == digest for path, digest in manifest_b["input_sha256"].items()),
            "Sequential independent runs with identical source/input bytes required")
    require({k: v for k, v in result_a.items() if k != "input_manifest_sha256"} ==
            {k: v for k, v in result_b.items() if k != "input_manifest_sha256"},
            "Independent stable-context flags or aggregates differ")
    print("Exact independently repeated development-only flags, diagnostics and class aggregates verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", choices=RUNS)
    mode.add_argument("--verify-replication", action="store_true")
    mode.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker.resolve(), args._manifest_sha)
    else:
        require(args._manifest_sha is None, "Manifest is reserved for authenticated worker")
        if args.verify_replication:
            verify_replication()
        else:
            parent(args.run)


if __name__ == "__main__":
    main()
