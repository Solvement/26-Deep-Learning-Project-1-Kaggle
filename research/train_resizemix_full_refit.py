"""FP16 scratch ResizeMix refit engine for the authenticated full-refit controller."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

SCHEMA = "research_resizemix_full_refit_training_v1"
PREFLIGHT_SCHEMA = "research_resizemix_full_refit_actual_preflight_v1"
TRAIN_RESULT_SCHEMA = "research_resizemix_full_refit_train_result_v1"
PARAMETER_COUNT, EPOCHS, STEPS, BATCH, TAIL = 4298970, 160, 391, 128, 80
TRAIN_EXAMPLES, INITIAL_SCALE = 50000, 65536.0
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
    """Exact audited primary26 EMA recipe, with only train_all changed."""
    return {"depth": 22, "widen_factor": 4, "dropout": 0.0, "num_classes": 10,
        "epochs": EPOCHS, "batch_size": BATCH, "lr": 0.1, "min_lr": 0.0001,
        "warmup_epochs": 5, "momentum": 0.9, "weight_decay": 0.0005,
        "label_smoothing": 0.0, "cutout": 0, "mixup_alpha": 0.0, "mixup_probability": 0.0,
        "ema_decay": 0.995, "seed": 2026, "split_seed": 2026, "train_all": True,
        "final_variant": "ema", "fixed_flip_tta": False, "resizemix_probability": 0.5,
        "resizemix_tau_min": 0.1, "resizemix_tau_max": 0.8}


def validate_config(config):
    expected = fixed_config()
    require(config == expected and all(type(config[key]) is type(value) for key, value in expected.items()),
            "Exact primary26 full50k FP16 160-epoch ResizeMix EMA recipe required")


def setup(config, data_dir):
    """Seed and stage the official training archive without any evaluation tensors."""
    validate_config(config)
    import runtime  # The controller authenticates the worker before calling this engine.
    import numpy as np
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from research import train_resizemix as original
    from train import tensor_images, training_source

    device = torch.device("cuda")
    precision, dtype = original.resolve_precision("fp16", device)
    require(precision == "fp16" and dtype == torch.float16, "Actual native FP16 CUDA required")
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
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE, enabled=True)
    parameters, averages = list(model.parameters()), list(ema.parameters())
    bns = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    require(scaler.is_enabled() and scaler.get_scale() == INITIAL_SCALE and len(parameters) == len(averages) == 62
            and len(bns) == 19 and all(param.dtype == torch.float32 for param in parameters + averages),
            "Actual FP16 GradScaler and original FP32 masters/EMA/BN topology required")
    return {"device": device, "dtype": dtype, "model": model, "ema": ema, "optimizer": optimizer,
        "scaler": scaler, "parameters": parameters, "averages": averages, "bns": bns,
        "train_x": train_x, "train_y": train_y, "model_config": model_config,
        "source": source, "signature": signature, "config": config}


def step(engine, order, epoch, batch, successful_updates):
    """One shared original ResizeMix FP16 train step used by both modes."""
    import torch
    from torch.nn import functional as F
    from research import train_resizemix as original

    config = engine["config"]
    optimizer, scaler = engine["optimizer"], engine["scaler"]
    start = batch * BATCH
    ids = order[start:start + BATCH]
    labels = engine["train_y"][ids]
    lr = original.schedule((epoch - 1) * STEPS + batch, EPOCHS * STEPS, STEPS, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y, permutation, lam, rectangle = original.prepare_batch(engine["train_x"][ids], labels, config)
    mixed = permutation is not None
    optimizer.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast("cuda", dtype=engine["dtype"]):
        logits = engine["model"](x)
        loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"]) * lam
        if mixed:
            loss += F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"]) * (1 - lam)
    updated = original.optimizer_step(loss, optimizer, scaler)
    with torch.no_grad():
        if updated:
            updates = successful_updates + 1
            decay = min(config["ema_decay"], (1 + updates) / (10 + updates))
            torch._foreach_lerp_(engine["averages"], engine["parameters"], 1 - decay)
        predicted = logits.detach().argmax(1)
        correct = predicted.eq(y).sum() * lam
        if mixed:
            correct += predicted.eq(y[permutation]).sum() * (1 - lam)
    return {"loss": loss.detach(), "correct": correct, "updated": updated, "mixed": mixed, "lr": lr,
        "source_examples": len(y), "logits_dtype": str(logits.dtype),
        "scale_before": scale_before, "scale_after": float(scaler.get_scale()),
        "geometry": (x, y, labels, permutation, lam, rectangle)}


def _state_values(engine, template, successful_updates):
    import torch
    from research.preflight_batched_core_control import full_state_values

    model, ema, optimizer = (engine[key] for key in ("model", "ema", "optimizer"))
    if successful_updates:
        return full_state_values(model, ema, optimizer, template, torch)
    require(not optimizer.state, "SGD state must be empty before the first successful update")
    entries = [(role + ":" + name, value) for role, module in (("model", model), ("ema", ema))
               for name, value in module.state_dict().items()]
    require(len(entries) == 238 and all((str(value.dtype), tuple(value.shape), str(value.device)) == template[name]
            for name, value in entries), "Original complete model/EMA inventory required before first update")
    return [value for _, value in entries]


def proof_step(engine, order, batch, successful_updates, template):
    """Measure original step plus full geometry, state, finite and BN proofs."""
    import torch
    from research import batched_finite_proof as proof
    from research.run_resizemix300_primary import geometry_proof

    params, averages = engine["parameters"], engine["averages"]
    counters = [m.num_batches_tracked for m in engine["bns"]]
    before_bn = proof.snapshot_bn_counters(counters)
    before, ema_before = params[0].detach().clone(), averages[0].detach().clone()
    momentum = engine["optimizer"].state.get(params[0], {}).get("momentum_buffer")
    momentum_before = momentum.detach().clone() if momentum is not None else None
    row = step(engine, order, 1, batch, successful_updates)
    geometry = geometry_proof(*row.pop("geometry"), torch)
    values = _state_values(engine, template, successful_updates + int(row["updated"]))
    grads = [param.grad for param in params]
    require(all(g is not None for g in grads), "All 62 original parameter gradients required")
    norm = None
    if row["updated"]:
        norm = float(torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt().item())
    finite = bool(proof.all_finite(values + (grads if row["updated"] else [])).item())
    model_delta = float((params[0] - before).abs().max().item())
    ema_delta = float((averages[0] - ema_before).abs().max().item())
    momentum_after = engine["optimizer"].state.get(params[0], {}).get("momentum_buffer")
    momentum_same = ((momentum_before is None and momentum_after is None) or
                     (momentum_before is not None and momentum_after is not None
                      and torch.equal(momentum_before, momentum_after)))
    bn = proof.prove_bn_increment(before_bn, counters)
    require(finite and bn["all_incremented"] and row["logits_dtype"] == "torch.float16",
            "Native FP16 finite state and one actual BN forward required")
    if row["updated"]:
        require(math.isfinite(norm) and norm > 0 and model_delta > 0 and ema_delta > 0
                and row["scale_after"] >= row["scale_before"], "Actual finite FP16 SGD and EMA updates required")
    else:
        require(model_delta == ema_delta == 0 and momentum_same and row["scale_after"] < row["scale_before"],
                "Native skipped update must back off and preserve SGD/EMA probes")
    return {**row, "loss": float(row["loss"]), "correct": float(row["correct"]), **geometry,
        "gradient_norm": norm, "all_state_finite": finite, "gradient_count": len(grads),
        "state_tensor_count": len(values), "bn_single_forward": True,
        "bn_counters_before": bn["before"], "bn_counters_after": bn["after"],
        "parameter_probe_max_delta": model_delta, "ema_probe_max_delta": ema_delta,
        "skip_momentum_probe_preserved": momentum_same if not row["updated"] else None}


def overflow_smoke(engine, order, successful_updates, template):
    """Force one disposable overflow and verify all parameter/SGD/EMA preservation."""
    import torch

    model, ema, optimizer = (engine[key] for key in ("model", "ema", "optimizer"))
    params = [param.detach().clone() for param in model.parameters()]
    averages = [param.detach().clone() for param in ema.parameters()]
    momentum = [optimizer.state.get(param, {}).get("momentum_buffer") for param in model.parameters()]
    momentum = [value.detach().clone() if value is not None else None for value in momentum]
    handle = next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        result = proof_step(engine, order, 23, successful_updates, template)
    finally:
        handle.remove()
    final_momentum = [optimizer.state.get(param, {}).get("momentum_buffer") for param in model.parameters()]
    checks = {"all_parameters_unchanged": all(torch.equal(p, old) for p, old in zip(model.parameters(), params)),
              "ema_parameters_unchanged": all(torch.equal(p, old) for p, old in zip(ema.parameters(), averages)),
              "optimizer_state_unchanged": all((a is None and b is None) or
                 (a is not None and b is not None and torch.equal(a, b)) for a, b in zip(momentum, final_momentum))}
    require(result["updated"] is False and result["scale_after"] < result["scale_before"] and all(checks.values()),
            "Injected overflow failed complete SGD/EMA preservation check")
    return {**result, **checks, "status": "passed", "infinite_gradient_injected": True}


def _environment():
    from runtime import environment
    return {**environment(), "resizemix_training_precision": "fp16", "grad_scaler_enabled": True,
            "actual_autocast_dtype": "torch.float16"}


def preflight(config, data_dir):
    """23 actual full batches plus a distinct actual 80-row tail and overflow smoke."""
    began = time.perf_counter()
    engine = setup(config, data_dir)
    import numpy as np
    import torch
    from research.preflight_batched_core_control import inventory_template

    model, ema, device = (engine[key] for key in ("model", "ema", "device"))
    model.train()
    template = inventory_template(model, ema, torch)
    require(len(template) == 300 and sum(row[0] == "torch.float32" for row in template.values()) == 262,
            "Complete 300-entry original model/EMA/SGD inventory required")
    order = torch.randperm(TRAIN_EXAMPLES, device=device)
    selected = order[:23 * BATCH].cpu().numpy()
    ids = np.load(Path(data_dir) / "official_train_ids.npy", allow_pickle=False)
    require(len(ids) == TRAIN_EXAMPLES and len(np.unique(selected)) == 2944, "Actual unique official sources required")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - began
    rows, durations, updates = [], [], 0
    for batch in range(23):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        row = proof_step(engine, order, batch, updates, template)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        require(row["source_examples"] == BATCH, "Full measured official source batch required")
        rows.append(row)
        updates += int(row["updated"])
    require(updates > 0 and any(row["mixed"] for row in rows) and any(not row["mixed"] for row in rows),
            "Actual successful ResizeMix and both augmentation branches required")
    torch.cuda.synchronize()
    tick = time.perf_counter()
    tail = proof_step(engine, order, STEPS - 1, updates, template)
    torch.cuda.synchronize()
    tail_seconds = time.perf_counter() - tick
    require(tail["source_examples"] == TAIL, "Actual full-epoch 80-row tail required")
    overflow = overflow_smoke(engine, order, updates + int(tail["updated"]), template)
    torch.cuda.synchronize()
    return {"schema": PREFLIGHT_SCHEMA, "status": "complete", "precision": "fp16", "training_examples": TRAIN_EXAMPLES,
        "validation_examples": 0, "parameter_count": PARAMETER_COUNT, "batch_size": BATCH, "steps_per_epoch": STEPS,
        "attempted_updates": 23, "successful_updates": updates, "skipped_updates": 23 - updates,
        "steps": rows, "step_seconds": durations, "median_step_seconds": statistics.median(durations[3:]),
        "tail_step": tail, "tail_step_seconds": tail_seconds, "overflow_smoke": overflow, "environment": _environment(),
        "setup_seconds": setup_seconds, "source_examples_seen": 2944, "no_weights_saved": True,
        "warmup_attempts": 3, "timed_attempts": 20, "bn_modules": 19, "initial_grad_scaler_scale": INITIAL_SCALE,
        "full_state_inventory_count": len(template), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "selected_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "selected_official_ids_sha256": hashlib.sha256(np.ascontiguousarray(ids[selected]).tobytes()).hexdigest(),
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}



def validate_history(history, epochs=EPOCHS):
    """Check the full scalar accounting, including tail and null validation."""
    require(type(epochs) is int and 1 <= epochs <= EPOCHS and len(history) == epochs,
            "Complete ordered refit history required")
    updates = skipped = mixed = 0
    for epoch, row in enumerate(history, 1):
        expected = {"epoch": epoch, "attempted_updates": STEPS, "source_examples": TRAIN_EXAMPLES,
                    "forward_calls": STEPS, "last_source_batch_size": TAIL}
        require(all(type(row.get(key)) is int and row[key] == value for key, value in expected.items())
                and type(row.get("successful_updates")) is int and row["successful_updates"] > 0
                and type(row.get("skipped_updates")) is int and row["skipped_updates"] >= 0
                and row["successful_updates"] + row["skipped_updates"] == STEPS
                and type(row.get("resizemix_batches")) is int and 0 <= row["resizemix_batches"] <= STEPS
                and row.get("selected_variant") == "ema"
                and all(key in row and row[key] is None for key in VALIDATION_KEYS),
                "Epoch coverage, FP16 accounting, fixed EMA or zero validation differs")
        digest = row.get("source_order_sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Actual ordered source indices hash required")
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                    for key in ("train_loss", "train_accuracy", "lr", "epoch_seconds", "elapsed_seconds",
                                "grad_scaler_scale"))
                and row["train_accuracy"] <= 1 and row["grad_scaler_scale"] > 0,
                "Finite native training history scalars required")
        updates += row["successful_updates"]
        skipped += row["skipped_updates"]
        mixed += row["resizemix_batches"]
    return updates, skipped, mixed


def train(config, folder, data_dir, lineage):
    """Execute only a new 160-epoch scratch refit with fixed final EMA weights."""
    validate_config(config)
    folder = Path(folder)
    require(folder.is_dir() and not any((folder / name).exists() for name in ARTIFACTS),
            "Authenticated fresh folder with no prior training artifacts required")
    require(isinstance(lineage, dict) and bool(lineage), "Controller lineage required")
    engine = setup(config, data_dir)
    import runtime
    import numpy as np
    import torch
    from research import train_resizemix as original

    model, ema, optimizer, scaler, device = (engine[key] for key in ("model", "ema", "optimizer", "scaler", "device"))
    marker = {**original.research_provenance("fp16", config), "schema": SCHEMA,
        "method": "scratch_full50k_fixed_final_resizemix", "lineage": lineage,
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS,
        "steps_per_epoch": STEPS, "final_variant": "ema", "initial_grad_scaler_scale": INITIAL_SCALE,
        "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "repeat_weights_loaded": False, "independent_development_evaluated": False,
        "source_order": "official_archive_order_before_seeded_epoch_permutations"}
    provenance = {**original.code_provenance(), "research_training": marker}
    runtime.write_json(folder / "config.json", config)
    runtime.write_json(folder / "environment.json", {**_environment(), "research_training": marker})
    runtime.write_json(folder / "provenance.json", {"code": provenance, "official_data": engine["source"],
        "data_signature": engine["signature"], "train_examples": TRAIN_EXAMPLES, "validation_examples": 0,
        "parameter_count": PARAMETER_COUNT, "normalization": {"mean": original.MEAN, "std": original.STD}})
    print(json.dumps({"event": "resizemix_full_refit_started", "precision": "fp16",
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS,
        "parameter_count": PARAMETER_COUNT, "final_variant": "ema"}), flush=True)
    history, updates, skipped, total_mixed = [], 0, 0, 0
    began = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(TRAIN_EXAMPLES, device=device)
        order_cpu = order.cpu().numpy()
        require(len(np.unique(order_cpu)) == TRAIN_EXAMPLES and int(order_cpu.min()) == 0
                and int(order_cpu.max()) == TRAIN_EXAMPLES - 1, "Complete original source permutation required")
        order_sha = hashlib.sha256(order_cpu.astype("<i8").tobytes()).hexdigest()
        loss_sum, correct = torch.zeros((), device=device), torch.zeros((), device=device)
        epoch_updates = epoch_skipped = mixed = seen = 0
        for batch in range(STEPS):
            row = step(engine, order, epoch, batch, updates)
            require(row["source_examples"] == (TAIL if batch == STEPS - 1 else BATCH)
                    and row["logits_dtype"] == "torch.float16", "Actual FP16 original full/tail source batch required")
            size = row["source_examples"]
            updated = int(row["updated"])
            updates += updated
            skipped += 1 - updated
            epoch_updates += updated
            epoch_skipped += 1 - updated
            mixed += int(row["mixed"])
            loss_sum += row["loss"] * size
            correct += row["correct"]
            seen += size
        require(seen == TRAIN_EXAMPLES and epoch_updates > 0, "Complete 50k epoch with successful updates required")
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers(), strict=True):
                average.copy_(online)
        elapsed = time.perf_counter() - began
        record = {"epoch": epoch, "train_loss": float(loss_sum) / seen, "train_accuracy": float(correct) / seen,
            **{key: None for key in VALIDATION_KEYS}, "selected_variant": "ema", "lr": row["lr"],
            "epoch_seconds": time.perf_counter() - tick, "elapsed_seconds": elapsed,
            "attempted_updates": STEPS, "successful_updates": epoch_updates, "skipped_updates": epoch_skipped,
            "grad_scaler_scale": float(scaler.get_scale()), "resizemix_batches": mixed, "source_examples": seen,
            "forward_calls": STEPS, "last_source_batch_size": TAIL, "source_order_sha256": order_sha}
        history.append(record)
        total_mixed += mixed
        require(validate_history(history, epoch) == (updates, skipped, total_mixed), "Cumulative update accounting differs")
        final = epoch == EPOCHS
        if final:
            require(all(int(module.num_batches_tracked) == STEPS * EPOCHS for module in engine["bns"]),
                    "Every actual online BN must account for all full-refit forward passes")
            require(all(bool(torch.isfinite(value).all()) for value in ema.state_dict().values()),
                    "Nonfinite final EMA tensor")
            original.save_checkpoint(folder / "best.pt", {"model": ema.state_dict(),
                "model_config": engine["model_config"], "config": config, "epoch": EPOCHS, "variant": "ema",
                "validation_accuracy": None, "validation_loss": None, "parameter_count": PARAMETER_COUNT,
                "data_signature": engine["signature"], "provenance": provenance})
        runtime.write_json(folder / "history.json", history)
        runtime.write_json(folder / "metrics.json", {"status": "complete" if final else "running",
            "epochs_completed": epoch, "best_validation_accuracy": None, "best_validation_loss": None,
            "parameter_count": PARAMETER_COUNT, "elapsed_seconds": elapsed, "training_examples": TRAIN_EXAMPLES,
            "validation_examples": 0, "config": config, "research_training": marker,
            "attempted_updates": epoch * STEPS, "successful_updates": updates, "skipped_updates": skipped,
            "resizemix_batches": total_mixed, "source_examples_seen": epoch * TRAIN_EXAMPLES})
        original.save_checkpoint(folder / "last.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "grad_scaler": scaler.state_dict(), "epoch": epoch,
            "updates": updates, "skipped_updates": skipped, "attempted_updates": epoch * STEPS,
            "source_examples_seen": epoch * TRAIN_EXAMPLES, "config": config, "data_signature": engine["signature"],
            "provenance": provenance, "history": history, "best_accuracy": None, "best_loss": None,
            "elapsed_seconds": elapsed, **original.capture_rng(device)})
        print(json.dumps(record), flush=True)
    report = {"schema": TRAIN_RESULT_SCHEMA, "status": "complete", "precision": "fp16",
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "parameter_count": PARAMETER_COUNT,
        "epochs_completed": EPOCHS, "attempted_updates": EPOCHS * STEPS, "successful_updates": updates,
        "skipped_updates": skipped, "source_examples_seen": EPOCHS * TRAIN_EXAMPLES,
        "resizemix_batches": total_mixed, "final_variant": "ema", "lineage": lineage,
        "independent_development_evaluated": False, "confirmation_evaluated": False,
        "hidden_images_loaded": False, "external_data_loaded": False, "parent_weights_loaded": False,
        "repeat_weights_loaded": False, "environment": _environment()}
    print(json.dumps({"event": "resizemix_full_refit_complete", "epochs_completed": EPOCHS,
        "successful_updates": updates, "skipped_updates": skipped}), flush=True)
    return report
