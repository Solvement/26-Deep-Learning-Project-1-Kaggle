"""Guarded peripheral-context probe on the original official development split."""
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
from research import run_development_shift_probe as shift_probe
from research import incumbent_compare as incumbent
from research import preflight_core_control as control
from research.process_tree_guard import may_release_lease, run_guarded

BASE, LEASE, GUARD = shift_probe.BASE, shift_probe.LEASE, shift_probe.GUARD
ENTRY = Path(__file__).resolve()
PLAN = ROOT / "research/development-periphery-probe-plan.md"
SHIFT_SOURCE_SHA = "2f1027ca40b11a126dd7fbc277d69e9d284c4ad85380a6279afbe9690c1ca703"
RUNS = {"dev_periphery_probe_20261103_incumbent": "incumbent",
        "dev_periphery_probe_20261103_resizemix": "resizemix"}
MODELS = shift_probe.MODELS
SHIFT_RUNS = {
    "incumbent": ("27af437c9979007c64e31507437017917b5fdc578ab17035b9caba2be447765d",
                  "0972b2df104a2a9ada964ca59ab5a801c55f12890883f40b1c9913629e57b4d8", 4818),
    "resizemix": ("4a559692fd3f0674b100fe732518fd580534cb989b416bcfdc3d7e67341af5af",
                  "7870372ba421da7b525920d4b58e4e8f4d9f11c4d73f2bedb3e4e6f51df89a1a", 4810),
}
DIAGNOSTIC_CONTEXT = shift_probe.DIAGNOSTIC_CONTEXT
WIDTHS, FILLS = (30, 28, 26, 24, 22), ("mean", "edge")
CANVAS_FILL = shift_probe.CANVAS_FILL
SCHEMA = "research_development_periphery_probe_supervisor_v1"
INPUT_SCHEMA = "research_development_periphery_probe_inputs_v1"
RESULT_SCHEMA = "research_development_periphery_probe_result_v1"
ARTIFACT = "development_periphery_probe.json"
require, sha, read, write = control.require, control.sha256, control.read, control.write


def evidence():
    shift_path = Path(shift_probe.__file__).resolve()
    require(sha(shift_path) == SHIFT_SOURCE_SHA, "Actual bound shift source required")
    inputs, sources = shift_probe.evidence()
    sources = {**sources, **{p.relative_to(ROOT).as_posix(): sha(p) for p in (ENTRY, PLAN)}}
    for role in SHIFT_RUNS:
        receipt, result, _ = shift_evidence(role)
        inputs.update({str(path.resolve()): sha(path) for path in (receipt, result)})
    return inputs, sources


def shift_evidence(role):
    folder = BASE / f"dev_shift_probe_20261103_{role}"
    receipt_path, result_path = folder / "supervisor.json", folder / shift_probe.ARTIFACT
    receipt_sha, result_sha, clean_correct = SHIFT_RUNS[role]
    require(sha(receipt_path) == receipt_sha and sha(result_path) == result_sha,
            "Complete bound shift evidence required")
    receipt, result = read(receipt_path), read(result_path)
    row = result["cases"][0]
    require(receipt.get("schema") == shift_probe.SCHEMA and receipt.get("status") == "complete"
            and receipt.get("process_tree_cleanup_verified") is True
            and receipt.get("artifact_sha256") == result_sha
            and result.get("schema") == shift_probe.RESULT_SCHEMA and result.get("status") == "complete"
            and result.get("model") == role and result.get("diagnostic_context") == DIAGNOSTIC_CONTEXT
            and result.get("checkpoint_sha256") == MODELS[role][2]
            and row.get("border") == "reflect" and row.get("dx") == row.get("dy") == 0
            and row.get("correct") == clean_correct, "Actual prior stable-context clean evidence differs")
    return receipt_path, result_path, result


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


def mask_periphery(batch, fill, width):
    """Leave the central uint8 patch at its original pixel coordinates."""
    import torch

    if fill == "clean" and width == 32:
        return batch.contiguous()
    require(width in WIDTHS and fill in FILLS, "Predeclared center width and fill required")
    low = (32 - width) // 2
    high = low + width
    if fill == "edge":
        index = torch.arange(32, device=batch.device).clamp(low, high - 1)
        return batch.index_select(1, index).index_select(2, index).contiguous()
    canvas = torch.empty_like(batch)
    canvas[:] = torch.tensor(CANVAS_FILL, dtype=batch.dtype, device=batch.device)
    canvas[:, low:high, low:high, :] = batch[:, low:high, low:high, :]
    return canvas.contiguous()


def summarize_periphery(labels, clean, masked):
    import numpy as np
    from research.development_shift_stats import CLASSES, REPLICATES, SEED, _joint_intervals, _predictions, _summarize

    labels, clean = _predictions(labels), _predictions(clean)
    require(set(labels.tolist()) == set(range(10)) and set(masked) ==
            {(fill, width) for width in WIDTHS for fill in FILLS}, "Exact 5k/class/fill/width design required")
    clean_correct = (clean == labels).astype(np.int8)
    predictions = {key: _predictions(value) for key, value in masked.items()}
    correct = {key: (value == labels).astype(np.int8) for key, value in predictions.items()}
    contrasts = {
        f"clean_minus_{fill}_{width}": (clean_correct - correct[fill, width], 1)
        for width in WIDTHS for fill in FILLS
    }
    contrasts.update({
        f"mean_minus_edge_{width}": (correct["mean", width] - correct["edge", width], 1)
        for width in WIDTHS
    })
    rng = np.random.default_rng(SEED)
    result = {"schema": "official_development_periphery_stats_v1", "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES, "resampling_unit": "original_development_image",
        "interval_scope": "pointwise descriptive intervals on reused development images",
        "clean": _summarize(clean[:, None], labels, clean),
        "by_fill_width": {f"{fill}_{width}": _summarize(value[:, None], labels, clean)
                          for (fill, width), value in predictions.items()},
        "paired_contrasts": _joint_intervals(contrasts, rng, REPLICATES), "per_true_class_contrasts": {}}
    for index, name in enumerate(CLASSES):
        mask = labels == index
        result["per_true_class_contrasts"][name] = _joint_intervals(
            {contrast: (values[mask], scale) for contrast, (values, scale) in contrasts.items()}, rng, REPLICATES)
    return result


def collect(model, images, fill, width):
    import numpy as np
    import torch
    from research import benchmark as suite

    digest = hashlib.sha256()
    predictions = np.empty(5000, dtype=np.int64)
    batch_size = DIAGNOSTIC_CONTEXT["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            pixels = mask_periphery(images[start:start + batch_size], fill, width)
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
    clean, pixels = collect(model, x, "clean", 32)
    expected_pixels = benchmark["policies"]["plain"]["cases"]["clean"]["input_pixel_sha256"]
    _, _, prior = shift_evidence(role)
    clean_correct = int((clean == dev_labels).sum())
    require(pixels == expected_pixels == prior["cases"][0]["input_pixel_sha256"]
            and clean_correct == prior["cases"][0]["correct"]
            and environment == prior["actual_environment"],
            "Original stable-context clean pixel hash, correct count or runtime differs")
    cases = [{"fill": "clean", "width": 32, "examples": 5000, "input_pixel_sha256": pixels,
              "correct": clean_correct}]
    masked = {}
    for width in WIDTHS:
        for fill in FILLS:
            predictions, pixels = collect(model, x, fill, width)
            masked[fill, width] = predictions
            row = {"fill": fill, "width": width, "examples": 5000, "input_pixel_sha256": pixels,
                   "correct": int((predictions == dev_labels).sum())}
            cases.append(row)
            print(json.dumps({"event": "periphery_probe_case", **row}), flush=True)
    output = folder / ARTIFACT
    require(not output.exists(), "Diagnostic output must be new")
    write(output, {"schema": RESULT_SCHEMA, "status": "complete", "model": role,
        "input_manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha, "benchmark_sha256": benchmark_sha,
        "development_examples": 5000, "development_index_order_sha256": hashlib.sha256(val_ids.astype("<i8").tobytes()).hexdigest(),
        "policy": "plain", "center_widths": list(WIDTHS), "fills": list(FILLS), "mean_fill_rgb": list(CANVAS_FILL),
        "edge_fill": "nearest unchanged interior pixel, including corner clamping",
        "actual_environment": environment, "data_signature": incumbent.DATA, "diagnostic_context": DIAGNOSTIC_CONTEXT,
        "historical_evaluation_context": incumbent.CONTEXT, "historical_clean_pixel_hash_matches": True,
        "same_context_clean_match": True, "shift_probe_result_sha256": SHIFT_RUNS[role][1],
        "historical_plain_clean_correct_count": round(5000 * benchmark["policies"]["plain"]["cases"]["clean"]["accuracy"]),
        "cases": cases, "stats": summarize_periphery(dev_labels, clean, masked),
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
                and result.get("model") == RUNS[run] and len(result.get("cases", [])) == 11
                and result.get("historical_clean_pixel_hash_matches") is True and result.get("same_context_clean_match") is True
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
