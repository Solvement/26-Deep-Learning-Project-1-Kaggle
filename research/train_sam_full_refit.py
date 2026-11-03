"""Native BF16 scratch SAM refit engine for the authenticated full-refit controller."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

SCHEMA = "research_sam_full_refit_training_v1"
PREFLIGHT_SCHEMA = "research_sam_full_refit_actual_preflight_v1"
TRAIN_RESULT_SCHEMA = "research_sam_full_refit_train_result_v1"
PARAMETER_COUNT, EPOCHS, STEPS, BATCH, TAIL = 4298970, 160, 391, 128, 80
TRAIN_EXAMPLES = 50000
VALIDATION_KEYS = ("validation_loss", "validation_accuracy", "ema_validation_loss", "ema_validation_accuracy")
ARTIFACTS = ("best.pt", "last.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json")
SIGNATURE = {
    "archive_sha256": "d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
    "split_sha256": "73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_config():
    """Exact audited primary26 recipe, with full data and its selected online variant."""
    return {"depth": 22, "widen_factor": 4, "dropout": 0.0, "num_classes": 10,
        "epochs": EPOCHS, "batch_size": BATCH, "lr": 0.1, "min_lr": 0.0001,
        "warmup_epochs": 5, "momentum": 0.9, "weight_decay": 0.0005,
        "label_smoothing": 0.0, "cutout": 16, "mixup_alpha": 0.2, "mixup_probability": 0.5,
        "ema_decay": 0.995, "seed": 2026, "split_seed": 2026, "train_all": True,
        "final_variant": "model", "fixed_flip_tta": False, "sam_rho": 0.05}


def validate_config(config):
    expected = fixed_config()
    require(config == expected and all(type(config[key]) is type(value) for key, value in expected.items()),
            "Exact primary26 full50k BF16 160-epoch SAM online recipe required")


def setup(config, data_dir):
    """Seed and stage the official training archive without any evaluation tensors."""
    validate_config(config)
    import runtime  # The controller authenticates the worker before calling this engine.
    import numpy as np
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from research import train_sam as original
    from train import tensor_images, training_source

    device = torch.device("cuda")
    precision, dtype = original.resolve_sam_precision("bf16", device)
    require(precision == "bf16" and dtype == torch.bfloat16, "Actual native BF16 CUDA required")
    original.validate_config({**config, "train_all": False})
    runtime.seed_all(config["seed"])
    data_dir = Path(data_dir)
    images, labels, old_train, old_dev = load_training_arrays(data_dir)
    manifest = json.loads((data_dir / "training_manifest.json").read_text(encoding="utf-8"))
    original.validate_partition(old_train, old_dev, manifest, config)
    source = training_source(data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(signature == SIGNATURE and images.shape == (TRAIN_EXAMPLES, 32, 32, 3)
            and images.dtype == np.uint8 and labels.shape == (TRAIN_EXAMPLES,) and labels.dtype.kind in "iu"
            and np.all((labels >= 0) & (labels < 10)), "Actual official ordered 50k RGB training pool required")
    train_x, train_y = tensor_images(images, device), torch.as_tensor(labels, device=device)
    del images, labels
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    require(count_parameters(model) == count_parameters(model, trainable_only=True) == PARAMETER_COUNT,
            "Original WRN22-4 parameter count required")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    parameters, averages = list(model.parameters()), list(ema.parameters())
    bns = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    require(len(parameters) == len(averages) == 62
            and len(bns) == 19 and all(param.dtype == torch.float32 for param in parameters + averages),
            "Original FP32 masters/EMA/BN topology required")
    return {"device": device, "dtype": dtype, "model": model, "ema": ema, "optimizer": optimizer,
        "parameters": parameters, "averages": averages, "bns": bns,
        "train_x": train_x, "train_y": train_y, "model_config": model_config,
        "source": source, "signature": signature, "config": config}


def step(engine, order, epoch, batch, successful_updates, observe_closure=None):
    """Original two-pass SAM update, with optional read-only preflight observation."""
    import torch
    from research import train_sam as original

    config = engine["config"]
    model, optimizer, device = (engine[key] for key in ("model", "optimizer", "device"))
    start = batch * BATCH
    ids = order[start:start + BATCH]
    labels = engine["train_y"][ids]
    lr = original.schedule((epoch - 1) * STEPS + batch, EPOCHS * STEPS, STEPS, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
    prepared = original.prepare_batch(engine["train_x"][ids], labels, config)
    _, y, permutation, lam = prepared
    first = {}
    closure = original.loss_closure(model, prepared, config, device, engine["dtype"], first)
    if observe_closure is not None:
        original_closure = closure
        def closure():
            loss = original_closure()
            observe_closure()
            return loss
    report = original.sam_step(model, optimizer, closure, rho=config["sam_rho"], precision="bf16")
    require(report["updated"] is True and report["forward_backward_passes"] == 2,
            "Original SAM must perform both passes and one update")
    with torch.no_grad():
        updates = successful_updates + 1
        decay = min(config["ema_decay"], (1 + updates) / (10 + updates))
        torch._foreach_lerp_(engine["averages"], engine["parameters"], 1 - decay)
        logits = first["logits"]
        predicted = logits.argmax(1)
        correct = predicted.eq(y).sum() * lam
        if permutation is not None:
            correct += predicted.eq(y[permutation]).sum() * (1 - lam)
    return {**report, "correct": correct, "mixed": permutation is not None,
            "lr": lr, "source_examples": len(y), "logits_dtype": str(logits.dtype)}


def proof_step(engine, order, batch, successful_updates, template):
    """Time the original update with explicit two-pass gradient and BN proofs."""
    import torch
    from research import batched_finite_proof as proof
    from research.preflight_batched_core_control import full_state_values

    model, ema, optimizer = (engine[key] for key in ("model", "ema", "optimizer"))
    params, averages, bns = (engine[key] for key in ("parameters", "averages", "bns"))
    counters = [module.num_batches_tracked for module in bns]
    before_bn = proof.snapshot_bn_counters(counters)
    before, ema_before = params[0].detach().clone(), averages[0].detach().clone()
    first_bn, counts = [], [0, 0]
    forwards = 0

    def observe_closure():
        nonlocal forwards
        forwards += 1
        require(forwards <= 2, "One prepared batch may run only two SAM forwards")
        if forwards == 1:
            first_bn.extend(value.detach().clone() for module in bns for value in module.buffers())

    def count_gradient(gradient):
        require(forwards in (1, 2), "Gradient outside the two observed SAM passes")
        counts[forwards - 1] += 1

    hooks = [param.register_hook(count_gradient) for param in params]
    try:
        row = step(engine, order, 1, batch, successful_updates, observe_closure)
    finally:
        for hook in hooks:
            hook.remove()
    values = full_state_values(model, ema, optimizer, template, torch)
    finite = bool(proof.all_finite(values).item())
    model_delta = float((params[0] - before).abs().max().item())
    ema_delta = float((averages[0] - ema_before).abs().max().item())
    bn = proof.prove_bn_increment(before_bn, counters)
    final_bn = [value for module in bns for value in module.buffers()]
    bn_preserved = len(first_bn) == len(final_bn) == 57 and all(
        torch.equal(before, after) for before, after in zip(first_bn, final_bn, strict=True))
    require(finite and bn["all_incremented"] and bn_preserved and forwards == 2
            and counts == [62, 62] and row["logits_dtype"] == "torch.bfloat16"
            and all(math.isfinite(row[key]) and row[key] > 0 for key in
                ("first_loss", "second_loss", "grad_norm", "second_grad_norm", "perturbation_norm"))
            and math.isclose(row["perturbation_norm"], row["rho"], rel_tol=5e-4, abs_tol=1e-6)
            and model_delta > 0 and ema_delta > 0,
            "Native BF16 finite SGD/EMA, two complete passes and exact first-pass BN required")
    return {**row, "correct": float(row["correct"]), "all_state_finite": finite, "gradient_counts": counts,
        "state_tensor_count": len(values), "bn_single_forward": True,
        "bn_first_pass_state_preserved": bn_preserved,
        "restore_before_sgd_contract": "unchanged original sam_step copies exact saved parameters before optimizer.step",
        "bn_counters_before": bn["before"], "bn_counters_after": bn["after"],
        "parameter_probe_max_delta": model_delta, "ema_probe_max_delta": ema_delta}


def _environment():
    from runtime import environment
    return {**environment(), "sam_training_precision": "bf16", "grad_scaler_enabled": False,
            "actual_autocast_dtype": "torch.bfloat16"}


def preflight(config, data_dir):
    """23 timed actual-image SAM batches and the separate true 80-row tail shape."""
    began = time.perf_counter()
    engine = setup(config, data_dir)
    import numpy as np
    import torch
    from research.preflight_batched_core_control import inventory_template

    model, ema, device = (engine[key] for key in ("model", "ema", "device"))
    model.train()
    template = inventory_template(model, ema, torch)
    require(len(template) == 300 and sum(row[0] == "torch.float32" for row in template.values()) == 262,
            "Complete original model/EMA/SGD inventory required")
    order = torch.randperm(TRAIN_EXAMPLES, device=device)
    selected = order[:23 * BATCH].cpu().numpy()
    ids = np.load(Path(data_dir) / "official_train_ids.npy", allow_pickle=False)
    require(len(ids) == TRAIN_EXAMPLES and len(np.unique(selected)) == 2944,
            "Actual unique official sources required")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - began
    rows, durations = [], []
    for batch in range(23):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        row = proof_step(engine, order, batch, batch, template)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        require(row["source_examples"] == BATCH, "Full measured source batch required")
        rows.append(row)
    require({row["mixed"] for row in rows} == {False, True}, "Both original Mixup branches required")
    torch.cuda.synchronize()
    tick = time.perf_counter()
    tail = proof_step(engine, order, STEPS - 1, 23, template)
    torch.cuda.synchronize()
    tail_seconds = time.perf_counter() - tick
    require(tail["source_examples"] == TAIL, "Actual full-epoch 80-row tail required")
    return {"schema": PREFLIGHT_SCHEMA, "status": "complete", "precision": "bf16", "training_examples": TRAIN_EXAMPLES,
        "validation_examples": 0, "parameter_count": PARAMETER_COUNT, "batch_size": BATCH, "steps_per_epoch": STEPS,
        "attempted_updates": 23, "successful_updates": 23, "skipped_updates": 0, "sam_forward_passes": 46,
        "steps": rows, "step_seconds": durations, "median_step_seconds": statistics.median(durations[3:]),
        "tail_step": tail, "tail_step_seconds": tail_seconds, "environment": _environment(),
        "setup_seconds": setup_seconds, "source_examples_seen": 2944, "no_weights_saved": True,
        "warmup_attempts": 3, "timed_attempts": 20, "bn_modules": 19,
        "full_state_inventory_count": len(template), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "selected_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "selected_official_ids_sha256": hashlib.sha256(np.ascontiguousarray(ids[selected]).tobytes()).hexdigest(),
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def validate_history(history, epochs=EPOCHS):
    """Check the full scalar accounting, including tail and null validation."""
    require(type(epochs) is int and 1 <= epochs <= EPOCHS and len(history) == epochs,
            "Complete ordered refit history required")
    updates = skipped = forwards = 0
    for epoch, row in enumerate(history, 1):
        expected = {"epoch": epoch, "attempted_updates": STEPS, "source_examples": TRAIN_EXAMPLES,
                    "forward_calls": 2 * STEPS, "last_source_batch_size": TAIL}
        require(all(type(row.get(key)) is int and row[key] == value for key, value in expected.items())
                and type(row.get("successful_updates")) is int and row["successful_updates"] == STEPS
                and type(row.get("skipped_updates")) is int and row["skipped_updates"] == 0
                and type(row.get("sam_forward_passes")) is int and row["sam_forward_passes"] == 2 * STEPS
                and row.get("selected_variant") == "model"
                and all(key in row and row[key] is None for key in VALIDATION_KEYS),
                "Epoch coverage, BF16 two-pass accounting, fixed online model or zero validation differs")
        digest = row.get("source_order_sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Actual ordered source indices hash required")
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                    for key in ("train_loss", "train_accuracy", "lr", "epoch_seconds", "elapsed_seconds",
                                "sam_loss", "mean_perturbation_norm", "mean_first_grad_norm"))
                and row["train_accuracy"] <= 1 and row["mean_perturbation_norm"] > 0
                and row["mean_first_grad_norm"] > 0,
                "Finite native training history scalars required")
        updates += row["successful_updates"]
        skipped += row["skipped_updates"]
        forwards += row["sam_forward_passes"]
    return updates, skipped, forwards


def train(config, folder, data_dir, lineage):
    """Execute only a new 160-epoch scratch refit with fixed terminal online weights."""
    validate_config(config)
    folder = Path(folder)
    require(folder.is_dir() and not any((folder / name).exists() for name in ARTIFACTS),
            "Authenticated fresh folder with no prior training artifacts required")
    require(isinstance(lineage, dict) and bool(lineage), "Controller lineage required")
    engine = setup(config, data_dir)
    import runtime
    import numpy as np
    import torch
    from research import train_sam as original

    model, ema, optimizer, device = (engine[key] for key in ("model", "ema", "optimizer", "device"))
    marker = {**original.research_provenance("bf16", config), "schema": SCHEMA,
        "method": "scratch_full50k_fixed_final_sam", "lineage": lineage,
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS,
        "steps_per_epoch": STEPS, "final_variant": "model",
        "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "repeat_weights_loaded": False, "independent_development_evaluated": False,
        "source_order": "official_archive_order_before_seeded_epoch_permutations"}
    provenance = {**original.code_provenance(), "research_training": marker}
    runtime.write_json(folder / "config.json", config)
    runtime.write_json(folder / "environment.json", {**_environment(), "research_training": marker})
    runtime.write_json(folder / "provenance.json", {"code": provenance, "official_data": engine["source"],
        "data_signature": engine["signature"], "train_examples": TRAIN_EXAMPLES, "validation_examples": 0,
        "parameter_count": PARAMETER_COUNT, "normalization": {"mean": original.MEAN, "std": original.STD}})
    print(json.dumps({"event": "sam_full_refit_started", "precision": "bf16",
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS,
        "parameter_count": PARAMETER_COUNT, "final_variant": "model"}), flush=True)
    history, updates, total_forwards = [], 0, 0
    began = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(TRAIN_EXAMPLES, device=device)
        order_cpu = order.cpu().numpy()
        require(len(np.unique(order_cpu)) == TRAIN_EXAMPLES and int(order_cpu.min()) == 0
                and int(order_cpu.max()) == TRAIN_EXAMPLES - 1, "Complete original source permutation required")
        order_sha = hashlib.sha256(order_cpu.astype("<i8").tobytes()).hexdigest()
        loss_sum = sam_loss_sum = perturbation_sum = grad_norm_sum = 0.0
        correct = torch.zeros((), device=device)
        epoch_updates = epoch_forwards = seen = 0
        for batch in range(STEPS):
            row = step(engine, order, epoch, batch, updates)
            require(row["source_examples"] == (TAIL if batch == STEPS - 1 else BATCH)
                    and row["logits_dtype"] == "torch.bfloat16", "Actual BF16 full/tail source batch required")
            size = row["source_examples"]
            updates += 1
            epoch_updates += 1
            epoch_forwards += row["forward_backward_passes"]
            loss_sum += row["first_loss"] * size
            sam_loss_sum += row["second_loss"] * size
            perturbation_sum += row["perturbation_norm"] * size
            grad_norm_sum += row["grad_norm"] * size
            correct += row["correct"]
            seen += size
        require(seen == TRAIN_EXAMPLES and epoch_updates == STEPS and epoch_forwards == 2 * STEPS,
                "Complete 50k epoch with exact two-pass updates required")
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers(), strict=True):
                average.copy_(online)
        elapsed = time.perf_counter() - began
        record = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": float(correct) / seen,
            "sam_loss": sam_loss_sum / seen, "mean_perturbation_norm": perturbation_sum / seen,
            "mean_first_grad_norm": grad_norm_sum / seen,
            **{key: None for key in VALIDATION_KEYS}, "selected_variant": "model", "lr": row["lr"],
            "epoch_seconds": time.perf_counter() - tick, "elapsed_seconds": elapsed,
            "attempted_updates": STEPS, "successful_updates": epoch_updates, "skipped_updates": 0,
            "sam_forward_passes": epoch_forwards, "source_examples": seen, "forward_calls": epoch_forwards, "last_source_batch_size": TAIL, "source_order_sha256": order_sha}
        history.append(record)
        total_forwards += epoch_forwards
        require(validate_history(history, epoch) == (updates, 0, total_forwards), "Cumulative update accounting differs")
        final = epoch == EPOCHS
        if final:
            require(all(int(module.num_batches_tracked) == STEPS * EPOCHS for module in engine["bns"]),
                    "Every actual online BN must account for one retained pass per SAM update")
            require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
                    "Nonfinite final online tensor")
            original.save_checkpoint(folder / "best.pt", {"model": model.state_dict(),
                "model_config": engine["model_config"], "config": config, "epoch": EPOCHS, "variant": "model",
                "validation_accuracy": None, "validation_loss": None, "parameter_count": PARAMETER_COUNT,
                "data_signature": engine["signature"], "provenance": provenance})
        runtime.write_json(folder / "history.json", history)
        runtime.write_json(folder / "metrics.json", {"status": "complete" if final else "running",
            "epochs_completed": epoch, "best_validation_accuracy": None, "best_validation_loss": None,
            "parameter_count": PARAMETER_COUNT, "elapsed_seconds": elapsed, "training_examples": TRAIN_EXAMPLES,
            "validation_examples": 0, "config": config, "research_training": marker,
            "attempted_updates": epoch * STEPS, "successful_updates": updates, "skipped_updates": 0,
            "sam_forward_passes": total_forwards, "source_examples_seen": epoch * TRAIN_EXAMPLES})
        original.save_checkpoint(folder / "last.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch,
            "updates": updates, "skipped_updates": 0, "attempted_updates": epoch * STEPS,
            "sam_forward_passes": total_forwards,
            "source_examples_seen": epoch * TRAIN_EXAMPLES, "config": config, "data_signature": engine["signature"],
            "provenance": provenance, "history": history, "best_accuracy": None, "best_loss": None,
            "elapsed_seconds": elapsed, **original.capture_rng(device)})
        print(json.dumps(record), flush=True)
    report = {"schema": TRAIN_RESULT_SCHEMA, "status": "complete", "precision": "bf16",
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "parameter_count": PARAMETER_COUNT,
        "epochs_completed": EPOCHS, "attempted_updates": EPOCHS * STEPS, "successful_updates": updates,
        "skipped_updates": 0, "source_examples_seen": EPOCHS * TRAIN_EXAMPLES,
        "sam_forward_passes": total_forwards, "final_variant": "model", "lineage": lineage,
        "independent_development_evaluated": False, "confirmation_evaluated": False,
        "hidden_images_loaded": False, "external_data_loaded": False, "parent_weights_loaded": False,
        "repeat_weights_loaded": False, "environment": _environment()}
    print(json.dumps({"event": "sam_full_refit_complete", "epochs_completed": EPOCHS,
        "successful_updates": updates, "skipped_updates": 0}), flush=True)
    return report
