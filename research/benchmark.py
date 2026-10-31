"""Benchmark a 40k scratch checkpoint on fixed official development inputs only.

No confirmation indices, hidden evaluation images or external data are loaded.
Synthetic stress cases are engineering proxies, not the CIFAR-C benchmark and
not evidence of the cause of the hidden-test gap.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime import seed_all, write_json
import torch
import torch.nn.functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from train import code_provenance, normalize, tensor_images, training_source

SCHEMA = "official_development_stress_v1"
CASES = ("clean", "noise_004", "boxblur3", "contrast065", "brightness080", "resize24")
POLICIES = ("plain", "flip", "center80_shift20")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint(checkpoint, train_ids, val_ids, signature):
    config = checkpoint["config"]
    if config.get("train_all") is not False:
        raise ValueError("Held-out benchmark rejects full-data checkpoints")
    if len(train_ids) != 40000 or len(val_ids) != 5000:
        raise ValueError("Benchmark requires the original official 40k/5k partition")
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise ValueError("Training and benchmark examples overlap")
    if checkpoint.get("data_signature") != signature:
        raise ValueError("Checkpoint official source/split mismatch")


def stress(batch, case, offset=0):
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 NHWC benchmark inputs")
    x = batch.permute(0, 3, 1, 2).float() / 255
    if case == "noise_004":
        generator = torch.Generator(device=batch.device).manual_seed(99173 + offset)
        x = x + torch.randn(x.shape, generator=generator, device=x.device) * .04
    elif case == "boxblur3":
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
    elif case == "contrast065":
        center = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - center) * .65 + center
    elif case == "brightness080":
        x = x * .8
    elif case == "resize24":
        x = F.interpolate(F.interpolate(x, (24, 24), mode="bilinear", align_corners=False),
                          (32, 32), mode="bilinear", align_corners=False)
    elif case != "clean":
        raise ValueError("Unknown fixed development stress case")
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def infer(model, batch, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown inference policy")
    center = normalize(batch)
    logits = model(center)
    if policy == "plain":
        return logits
    logits = (logits + model(center.flip(-1))) * .5
    if policy == "flip":
        return logits
    # A conservative translation policy: 80% center/flip, 20% mean small shifts.
    padded = F.pad(batch.permute(0, 3, 1, 2).float(), (2, 2, 2, 2), mode="reflect").to(torch.uint8)
    shifts = []
    for top, left in ((0, 2), (4, 2), (2, 0), (2, 4)):
        x = normalize(padded[:, :, top:top+32, left:left+32].permute(0, 2, 3, 1))
        shifts.append((model(x) + model(x.flip(-1))) * .5)
    return logits * .8 + torch.stack(shifts).mean(0) * .2


def recalibrate_training_bn(model, train_images, batch_size):
    """Only the 40k optimization images may set the new BN buffers."""
    calibrated = copy.deepcopy(model).eval()
    modules = [m for m in calibrated.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    original_momenta = [m.momentum for m in modules]
    for module in modules:
        module.reset_running_stats()
        module.train()
    with torch.inference_mode():
        seen = 0
        for start in range(0, len(train_images), batch_size):
            batch = train_images[start:start+batch_size]
            # SWA authors' sample-weighted estimator avoids overweighting tails.
            for module in modules:
                module.momentum = len(batch)/(seen+len(batch))
            calibrated(normalize(batch))
            seen += len(batch)
    for module, momentum in zip(modules, original_momenta):
        module.momentum = momentum
    return calibrated.eval()


def score(model, images, labels, device, policy, batch_size, limit=None):
    reports = {}
    examples = min(len(labels), limit) if limit else len(labels)
    dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for case in CASES:
            correct = 0
            loss = 0.0
            started = time.perf_counter()
            for start in range(0, examples, batch_size):
                batch = stress(images[start:min(start+batch_size, examples)], case, start)
                target = labels[start:start+len(batch)]
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logits = infer(model, batch, policy)
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite benchmark logits")
                loss += F.cross_entropy(logits.float(), target, reduction="sum").item()
                correct += (logits.argmax(1) == target).sum().item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            reports[case] = {"accuracy": correct/examples, "loss": loss/examples,
                             "examples": examples, "seconds": time.perf_counter()-started}
    return {"cases": reports, "clean_accuracy": reports["clean"]["accuracy"],
            "clean_loss": reports["clean"]["loss"],
            "stress_accuracy": sum(reports[c]["accuracy"] for c in CASES[1:])/(len(CASES)-1)}


def promotion_gate(candidate, baseline):
    clean = candidate["clean_accuracy"] - baseline["clean_accuracy"]
    robust = candidate["stress_accuracy"] - baseline["stress_accuracy"]
    # Predeclared exploratory engineering thresholds, not a significance test.
    passed = ((clean >= .001-1e-12 and robust >= -.002-1e-12) or
              (clean >= -.002-1e-12 and robust >= .005-1e-12))
    return {"passed": passed, "clean_delta": clean, "stress_delta": robust,
            "thresholds": {"clean_improvement": .001, "stress_noninferiority": -.002,
                           "stress_improvement": .005, "clean_noninferiority": -.002}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recalibrate-bn", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke only; a partial benchmark cannot pass promotion")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Positive batch size and optional smoke limit required")
    if "stability_seed2027" in args.checkpoint.parts:
        raise ValueError("The validation-only stability repeat is excluded from candidate selection")
    if args.output.exists():
        raise FileExistsError("Use a new benchmark output path; never overwrite evidence")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_all(2026)
    device = torch.device(args.device)
    started = time.perf_counter()
    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    images, labels, train_ids, val_ids = load_training_arrays(args.data_dir)
    manifest = json.loads((args.data_dir/"training_manifest.json").read_text())
    signature = {"archive_sha256": training_source(args.data_dir)["archive_sha256"],
                 "split_sha256": manifest["split"]["indices_sha256"]}
    validate_checkpoint(checkpoint, train_ids, val_ids, signature)
    model = build_model(checkpoint["model_config"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if args.recalibrate_bn:
        train_images = tensor_images(images[train_ids], device)
        model = recalibrate_training_bn(model, train_images, args.batch_size)
        del train_images
    val_images = tensor_images(images[val_ids], device)
    val_labels = torch.as_tensor(labels[val_ids], device=device)
    reports = {}
    for policy in args.policies:
        reports[policy] = score(model, val_images, val_labels, device, policy, args.batch_size, args.limit)
        print(json.dumps({"policy": policy, **reports[policy]}), flush=True)
    baseline_doc = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline_doc and (baseline_doc.get("schema") != SCHEMA or baseline_doc.get("data_signature") != signature
                         or baseline_doc.get("examples") != 5000 or baseline_doc.get("stress_cases") != list(CASES)
                         or baseline_doc.get("bn_recalibrated") or baseline_doc.get("batch_size") != args.batch_size):
        raise ValueError("Baseline must be a complete, identical-suite original-BN benchmark")
    baseline = (baseline_doc["policies"]["flip"] if baseline_doc else reports.get("flip"))
    gates = {p: promotion_gate(report, baseline) for p, report in reports.items()} if baseline else {}
    if args.limit:
        for gate in gates.values():
            gate["passed"] = False
    if sha256(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("Original checkpoint changed during benchmark")
    result = {"schema": SCHEMA, "status": "smoke" if args.limit else "complete", "examples": min(5000, args.limit or 5000),
              "training_examples": 40000, "parameter_count": count_parameters(model),
              "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_variant": checkpoint["variant"], "model_config": checkpoint["model_config"],
              "config": checkpoint["config"], "data_signature": signature,
              "stress_cases": list(CASES), "batch_size": args.batch_size,
              "bn_recalibrated": args.recalibrate_bn, "policies": reports, "promotion": gates,
              "baseline_sha256": sha256(args.baseline) if args.baseline else None,
              "elapsed_seconds": time.perf_counter()-started, "source": code_provenance(),
              "benchmark_source_sha256": sha256(Path(__file__)), "confirmation_evaluated": False,
              "hidden_images_loaded": False, "external_data_loaded": False,
              "interpretation": "Exploratory reused development and synthetic stress proxies; no hidden-test or independent confirmation guarantee."}
    write_json(args.output, result)
    print(json.dumps({"event": "benchmark_complete", "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
