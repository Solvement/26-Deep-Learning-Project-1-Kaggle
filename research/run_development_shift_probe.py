"""Guarded single-forward phase/translation probe on the original official dev split."""
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
PLAN = ROOT / "research/development-shift-probe-plan.md"
STATS = ROOT / "research/development_shift_stats.py"
RUNS = {"dev_shift_probe_20261103_incumbent": "incumbent",
        "dev_shift_probe_20261103_resizemix": "resizemix"}
RESIZEMIX = BASE / "colab_wrn22_resizemix_s2026_v3/candidate/best.pt"
RESIZEMIX_BENCHMARK = BASE / "wrn22_resizemix_s2026_local_v3/benchmark.json"
MODELS = {
    "incumbent": (supervision.INCUMBENT_CHECKPOINT, incumbent.INCUMBENT_PATH,
                  incumbent.INCUMBENT_CHECKPOINT_SHA, incumbent.INCUMBENT_BENCHMARK_SHA),
    "resizemix": (RESIZEMIX, RESIZEMIX_BENCHMARK,
                  "c39436412b454f37f030e49beb977ab8e4a6dc60f8027a0e9942c3c3ad534894",
                  "b489b856bcef815209a72212dd2f3e5b4251a4332335f819f922020ecff7ba93"),
}
DIAGNOSTIC_CONTEXT = {"seed": 2026, "device": "cuda", "dtype": "torch.bfloat16", "batch_size": 128,
                      "cudnn_benchmark": False, "cudnn_deterministic": True,
                      "torch_deterministic_algorithms": False}
OFFSETS = {"reflect": range(1, 5), "canvas": range(2, 5)}
OFFSET_CONVENTION = "Positive dx moves content right; positive dy moves content down."
CANVAS_FILL = (125, 123, 114)
SCHEMA = "research_development_shift_probe_supervisor_v1"
INPUT_SCHEMA = "research_development_shift_probe_inputs_v1"
RESULT_SCHEMA = "research_development_shift_probe_result_v1"
ARTIFACT = "development_shift_probe.json"
require, sha, read, write = control.require, control.sha256, control.read, control.write


def evidence():
    sources = {**supervision.source_snapshot(), **{
        p.relative_to(ROOT).as_posix(): sha(p) for p in (ENTRY, PLAN, STATS, ROOT / "research/development_error_slices.py")
    }}
    for checkpoint, benchmark, checkpoint_sha, benchmark_sha in MODELS.values():
        require(sha(checkpoint) == checkpoint_sha and sha(benchmark) == benchmark_sha,
                "Exact incumbent and ResizeMix scored checkpoint and local benchmark required")
    inputs, _, _ = supervision.binding_for(ROOT / "data", RESIZEMIX, RESIZEMIX_BENCHMARK, sources)
    require(not any(Path(path).name in {"official_test.npz", "test.csv", "sample_submission.csv"}
                    for path in inputs), "No hidden data in diagnostic closure")
    return inputs, sources


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


def shift(batch, border, dx, dy):
    """Transform uint8 NHWC images without interpolation."""
    import torch
    import torch.nn.functional as F

    if border == "reflect":
        if (dx, dy) == (0, 0):
            return batch.contiguous()
        pad = 4
        padded = F.pad(batch.permute(0, 3, 1, 2).float(), (pad,) * 4, mode="reflect").to(torch.uint8)
        left, top = pad - dx, pad - dy
        return padded[:, :, top:top + 32, left:left + 32].permute(0, 2, 3, 1).contiguous()
    require(border == "canvas", "Fixed reflect or canvas family required")
    canvas = torch.empty_like(batch)
    canvas[:] = torch.tensor(CANVAS_FILL, dtype=batch.dtype, device=batch.device)
    left, top = 5 + dx, 5 + dy
    canvas[:, top:top + 22, left:left + 22, :] = batch[:, 5:27, 5:27, :]
    return canvas.contiguous()


def collect(model, images, border, dx, dy):
    import numpy as np
    import torch
    from research import benchmark as suite

    digest = hashlib.sha256()
    predictions = np.empty(5000, dtype=np.int64)
    batch_size = DIAGNOSTIC_CONTEXT["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            pixels = shift(images[start:start + batch_size], border, dx, dy)
            digest.update(pixels.cpu().numpy().tobytes())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = suite.infer(model, pixels, "plain")
            require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()),
                    "Actual finite native BF16 logits required")
            predictions[start:start + len(pixels)] = logits.argmax(1).cpu().numpy()
    return predictions, digest.hexdigest()


def worker(folder, manifest_sha):
    receipt = auth(folder, manifest_sha)
    manifest = read(folder / "inputs.json")
    role = RUNS[folder.name]
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == folder.name
            and manifest.get("model") == role and manifest.get("source_sha256") == receipt.get("source_sha256")
            and manifest.get("input_sha256") == receipt.get("input_sha256")
            and all(sha(ROOT / path) == value for path, value in manifest["source_sha256"].items())
            and all(sha(path) == value for path, value in manifest["input_sha256"].items()),
            "Frozen diagnostic inputs changed")
    import runtime  # Windows DLL isolation before Torch.
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from train import MEAN, code_provenance, tensor_images
    from research import benchmark as suite
    from research import paired_compare_v2
    from research.development_shift_stats import summarize_shift_predictions

    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.__version__ == incumbent.CONTEXT["torch_version"]
            and code_provenance()["source_sha256"] == incumbent.CORE
            and tuple(round(channel * 255) for channel in MEAN) == CANVAS_FILL,
            "Pinned native CUDA BF16, core source and RGB mean required")
    runtime.seed_all(DIAGNOSTIC_CONTEXT["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    environment = runtime.environment()
    require(environment["cudnn_benchmark"] is False and torch.backends.cudnn.deterministic is True
            and torch.are_deterministic_algorithms_enabled() is False and environment["cudnn"] == 92000
            and environment["cuda_runtime"] == "13.0", "Fixed process-local diagnostic backend required")
    device = torch.device("cuda")
    require(suite.benchmark_dtype(device) == torch.bfloat16, "Actual BF16 evaluation required")
    images, labels, train_ids, val_ids = load_training_arrays(ROOT / "data")
    require(len(train_ids) == 40000 and len(val_ids) == len(set(val_ids.tolist())) == 5000
            and not set(train_ids.tolist()) & set(val_ids.tolist()), "Original independent 5000 dev required")
    checkpoint_path, benchmark_path, checkpoint_sha, benchmark_sha = MODELS[role]
    checkpoint, benchmark = paired_compare_v2.load_experiment(checkpoint_path, benchmark_path,
        incumbent.DATA, incumbent.CONTEXT, incumbent.CORE, train_ids, val_ids)
    model = build_model(checkpoint["model_config"])
    require(count_parameters(model) == count_parameters(model, True) == checkpoint["parameter_count"] <= 5000000,
            "Single residual-model parameter budget differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
            "Selected tensor state is nonfinite")
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    x, dev_labels = tensor_images(images[val_ids], device), labels[val_ids]
    clean, shifted, cases = {}, {}, []
    for border, distances in OFFSETS.items():
        shifted[border] = {}
        offsets = [(0, 0)] + [offset for d in distances for offset in ((d, 0), (-d, 0), (0, d), (0, -d))]
        for dx, dy in offsets:
            predictions, pixels = collect(model, x, border, dx, dy)
            row = {"border": border, "dx": dx, "dy": dy, "examples": 5000, "input_pixel_sha256": pixels,
                   "correct": int((predictions == dev_labels).sum())}
            cases.append(row)
            if dx == dy == 0:
                clean[border] = predictions
                if border == "reflect":
                    expected_pixels = benchmark["policies"]["plain"]["cases"]["clean"]["input_pixel_sha256"]
                    require(pixels == expected_pixels, "Original clean development pixels differ")
            else:
                shifted[border][dx, dy] = predictions
            print(json.dumps({"event": "shift_probe_case", **row}), flush=True)
    output = folder / ARTIFACT
    require(not output.exists(), "Diagnostic output must be new")
    write(output, {"schema": RESULT_SCHEMA, "status": "complete", "model": role,
        "input_manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha, "benchmark_sha256": benchmark_sha,
        "development_examples": 5000, "development_index_order_sha256": hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest(),
        "policy": "plain", "signed_offset_convention": OFFSET_CONVENTION,
        "canvas": {"central_patch_size": 22, "frame_size": 32, "fill_rgb": list(CANVAS_FILL)},
        "actual_environment": environment, "data_signature": incumbent.DATA, "diagnostic_context": DIAGNOSTIC_CONTEXT,
        "historical_evaluation_context": incumbent.CONTEXT, "historical_clean_pixel_hash_matches": True,
        "historical_plain_clean_correct_count": round(5000 * benchmark["policies"]["plain"]["cases"]["clean"]["accuracy"]),
        "cases": cases, "stats": summarize_shift_predictions(dev_labels, clean, shifted),
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
                and result.get("model") == RUNS[run] and len(result.get("cases", [])) == 30
                and result.get("historical_clean_pixel_hash_matches") is True
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
