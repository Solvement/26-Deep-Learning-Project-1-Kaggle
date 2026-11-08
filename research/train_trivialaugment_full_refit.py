"""Authenticated-controller-only native BF16 TrivialAugment 50k scratch refit."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

SCHEMA = "research_trivialaugment_full_refit_training_v1"
PREFLIGHT_SCHEMA = "research_trivialaugment_full_refit_actual_preflight_v1"
TRAIN_RESULT_SCHEMA = "research_trivialaugment_full_refit_train_result_v1"
PARAMETER_COUNT, EPOCHS, STEPS, BATCH, TAIL = 4166298, 160, 391, 128, 80
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
    from research.run_trivialaugment_primary import configuration
    return {**configuration(2026), "train_all": True, "final_variant": "ema"}


def validate_config(config):
    expected = fixed_config()
    require(type(config) is dict and config == expected
            and all(type(config[key]) is type(value) for key, value in expected.items()),
            "Exact primary26 TA160 full50k BF16 terminal EMA recipe required")


def setup(config, data_dir):
    validate_config(config)
    import runtime  # Controller authentication and native containment precede this import.
    import numpy as np
    import torch
    from data import load_training_arrays
    from model import build_model, count_parameters
    from research import train_trivialaugment as ta
    from research import batched_finite_proof as proof

    device = torch.device("cuda")
    precision, dtype = ta.resolve_precision("bf16", device)
    require(precision == "bf16" and dtype == torch.bfloat16
            and torch.cuda.is_bf16_supported(including_emulation=False), "Native CUDA BF16 required")
    ta.validate_config({**config, "train_all": False})
    runtime.seed_all(config["seed"])
    data_dir = Path(data_dir)
    images, labels, old_train, old_dev = load_training_arrays(data_dir)
    manifest = json.loads((data_dir / "training_manifest.json").read_text(encoding="utf-8"))
    ta.validate_partition(old_train, old_dev, manifest, config)
    source = ta.training_source(data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(signature == SIGNATURE and images.shape == (TRAIN_EXAMPLES, 32, 32, 3)
            and images.dtype == np.uint8 and labels.shape == (TRAIN_EXAMPLES,) and labels.dtype.kind in "iu"
            and np.all((labels >= 0) & (labels < 10)), "Actual official ordered 50k training images required")
    train_pixels = np.ascontiguousarray(images)
    train_y = torch.as_tensor(labels, device=device)
    model_config = {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    require(count_parameters(model) == count_parameters(model, trainable_only=True) == PARAMETER_COUNT,
            "Original WRN34-3 parameter count required")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    params, averages = list(model.parameters()), list(ema.parameters())
    bns = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    require(len(params) == len(averages) == 98 and len(bns) == 31
            and proof.snapshot_bn_counters([module.num_batches_tracked for module in bns]).cpu().tolist() == [0] * 31
            and all(param.dtype == torch.float32 for param in params + averages),
            "Original scratch BN and FP32 model/EMA topology required")
    return {"device": device, "dtype": dtype, "model": model, "ema": ema, "optimizer": optimizer,
            "scaler": scaler, "params": params, "averages": averages, "bns": bns,
            "rng": ta.ta_rng(2026), "train_pixels": train_pixels, "train_y": train_y,
            "model_config": model_config, "source": source, "signature": signature, "config": config}


def step(engine, order, epoch, batch, updates, metrics):
    """Original one-view TA, crop/flip/Cutout/Mixup, BF16 SGD and FP32 EMA."""
    import torch
    from research import train_trivialaugment as ta
    config = engine["config"]
    start = batch * BATCH
    ids = order[start:start + BATCH]
    y = engine["train_y"][ids]
    lr = ta.schedule((epoch - 1) * STEPS + batch, EPOCHS * STEPS, STEPS, config)
    optimizer = engine["optimizer"]
    for group in optimizer.param_groups:
        group["lr"] = lr
    result = ta.train_step(engine["model"], engine["ema"], optimizer, engine["scaler"],
                           engine["train_pixels"][ids.cpu().numpy()], y, config, engine["rng"],
                           updates, engine["device"], engine["dtype"])
    require(result["updated"] is True and result["logits"].dtype == torch.bfloat16
            and len(y) == (TAIL if batch == STEPS - 1 else BATCH), "Actual complete native BF16 TA batch required")
    with torch.no_grad():
        predicted = result["logits"].argmax(1)
        lam, permutation = result["lam"], result["permutation"]
        correct = predicted.eq(y).sum() * lam
        if permutation is not None:
            correct += predicted.eq(y[permutation]).sum() * (1 - lam)
        metrics["loss"] += result["loss"] * len(y)
        metrics["correct"] += correct
        metrics["seen"] += len(y)
    return {"loss": result["loss"], "correct": correct, "lr": lr, "mixed": permutation is not None,
            "source_examples": len(y), "logits_dtype": str(result["logits"].dtype),
            "ta_operation_counts": result["ta_operation_counts"]}


def proof_step(engine, order, batch, updates, template, metrics):
    import torch
    from research import train_trivialaugment as ta
    from research import batched_finite_proof as proof
    from research.preflight_batched_core_control import full_state_values

    model, ema, optimizer = (engine[key] for key in ("model", "ema", "optimizer"))
    params, averages, bns = (engine[key] for key in ("params", "averages", "bns"))
    counters = [module.num_batches_tracked for module in bns]
    before_bn = proof.snapshot_bn_counters(counters)
    before, ema_before = params[0].detach().clone(), averages[0].detach().clone()
    rng_before = ta.json_sha(engine["rng"].bit_generator.state)
    result = step(engine, order, 1, batch, updates, metrics)
    grads = [parameter.grad for parameter in params]
    require(len(grads) == 98 and all(grad is not None for grad in grads),
            "Actual gradients for every original parameter required")
    norm = torch.stack([grad.detach().float().square().sum() for grad in grads]).sum().sqrt().item()
    values = full_state_values(model, ema, optimizer, template, torch)
    finite = bool(proof.all_finite([*values, *grads]).item())
    model_delta = float((params[0] - before).abs().max().item())
    ema_delta = float((averages[0] - ema_before).abs().max().item())
    bn = proof.prove_bn_increment(before_bn, counters)
    rng_after = ta.json_sha(engine["rng"].bit_generator.state)
    loss = float(result["loss"])
    scaler = engine["scaler"]
    require(not scaler.is_enabled() and scaler.get_scale() == 1.0 and len(values) == 480
            and finite and bn["all_incremented"] and math.isfinite(norm) and norm > 0
            and math.isfinite(loss) and loss > 0 and model_delta > 0 and ema_delta > 0 and rng_before != rng_after,
            "Actual finite native TA/SGD/EMA/BN/independent-RNG proof required")
    return {**result, "loss": loss, "correct": float(result["correct"]), "gradient_norm": norm,
            "updates": updates + 1, "gradient_count": len(grads), "state_tensor_count": len(values),
            "all_parameters_finite": True, "bn_single_forward": True,
            "bn_counters_before": bn["before"], "bn_counters_after": bn["after"],
            "parameter_probe_max_delta": model_delta, "ema_probe_max_delta": ema_delta,
            "ta_rng_before_sha256": rng_before, "ta_rng_after_sha256": rng_after}


def _environment():
    from runtime import environment
    return {**environment(), "trivialaugment_training_precision": "bf16", "grad_scaler_enabled": False,
            "actual_autocast_dtype": "torch.bfloat16"}


def metric_state(torch, device):
    return {"loss": torch.zeros((), device=device), "correct": torch.zeros((), device=device), "seen": 0}


def preflight(config, data_dir):
    """Measure the complete 50k proof path, including a separate real 80-row tail."""
    began = time.perf_counter()
    engine = setup(config, data_dir)
    import numpy as np
    import torch
    from research.preflight_batched_core_control import inventory_template
    from research.run_trivialaugment_primary import OPERATIONS, validate_operation_counts

    model, ema, device = (engine[key] for key in ("model", "ema", "device"))
    model.train()
    template = inventory_template(model, ema, torch)
    floats = sum(row[0] == "torch.float32" for row in template.values())
    ints = sum(row[0] == "torch.int64" for row in template.values())
    require(len(template) == 480 and floats == 418 and ints == 62, "Complete original FP32/int64 inventory required")
    order = torch.randperm(TRAIN_EXAMPLES, device=device)
    metrics = metric_state(torch, device)
    selected = order[:23 * BATCH].cpu().numpy()
    tail_selected = order[-TAIL:].cpu().numpy()
    official_ids = np.load(Path(data_dir) / "official_train_ids.npy", allow_pickle=False)
    require(len(official_ids) == TRAIN_EXAMPLES and len(np.unique(np.concatenate((selected, tail_selected)))) == 3024,
            "Disjoint full and tail official sources required")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - began
    rows, durations, counts = [], [], dict.fromkeys(OPERATIONS, 0)

    def measure(batch, updates, size):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        row = proof_step(engine, order, batch, updates, template, metrics)
        require(row["source_examples"] == size, "Actual complete source batch required")
        validate_operation_counts(row["ta_operation_counts"], size)
        for name, value in row["ta_operation_counts"].items():
            counts[name] += value
        torch.cuda.synchronize()
        return row, time.perf_counter() - tick

    for batch in range(23):
        row, duration = measure(batch, batch, BATCH)
        rows.append(row)
        durations.append(duration)
    require({row["mixed"] for row in rows} == {False, True}, "Both original Mixup branches required")
    full_counts = counts.copy()
    validate_operation_counts(full_counts, 2944)
    tail, tail_seconds = measure(STEPS - 1, 23, TAIL)
    validate_operation_counts(counts, 3024)
    require(metrics["seen"] == 3024, "Full and tail metric coverage differs")
    return {"schema": PREFLIGHT_SCHEMA, "status": "complete", "precision": "bf16",
            "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "parameter_count": PARAMETER_COUNT,
            "batch_size": BATCH, "steps_per_epoch": STEPS, "attempted_updates": 23, "successful_updates": 23,
            "skipped_updates": 0, "forward_passes": 23, "steps": rows, "step_seconds": durations,
            "median_step_seconds": statistics.median(durations[3:]), "tail_step": tail, "tail_step_seconds": tail_seconds,
            "environment": _environment(), "setup_seconds": setup_seconds, "source_examples_seen": 2944,
            "model_views_seen": 2944, "views_per_source": 1, "ta_operation_counts": full_counts,
            "ta_rng_algorithm": "PCG64", "ta_rng_seed": 2026, "no_weights_saved": True,
            "warmup_attempts": 3, "timed_attempts": 20, "bn_modules": 31,
            "full_state_inventory_count": len(template), "state_float32_count": floats, "state_int64_count": ints,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "selected_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
            "selected_official_ids_sha256": hashlib.sha256(np.ascontiguousarray(official_ids[selected]).tobytes()).hexdigest(),
            "tail_indices_sha256": hashlib.sha256(tail_selected.astype("<i8").tobytes()).hexdigest(),
            "tail_official_ids_sha256": hashlib.sha256(np.ascontiguousarray(official_ids[tail_selected]).tobytes()).hexdigest(),
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def validate_history(history, epochs=EPOCHS):
    from research.run_trivialaugment_primary import validate_operation_counts
    require(type(epochs) is int and 1 <= epochs <= EPOCHS and type(history) is list and len(history) == epochs,
            "Complete ordered TA50k history required")
    previous_elapsed = 0.
    for epoch, row in enumerate(history, 1):
        expected = {"epoch": epoch, "attempted_updates": STEPS, "successful_updates": STEPS,
                    "skipped_updates": 0, "source_examples": TRAIN_EXAMPLES, "model_views_seen": TRAIN_EXAMPLES,
                    "forward_passes": STEPS, "forward_calls": STEPS, "last_source_batch_size": TAIL}
        require(all(type(row.get(key)) is int and row[key] == value for key, value in expected.items())
                and row.get("selected_variant") == "ema" and row.get("grad_scaler_scale") == 1.0
                and all(key in row and row[key] is None for key in VALIDATION_KEYS),
                "TA50k epoch coverage, fixed terminal EMA or zero validation differs")
        for key in ("source_order_sha256", "ta_rng_state_sha256"):
            digest = row.get(key)
            require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "Actual ordered source and independent RNG hashes required")
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                    for key in ("train_loss", "train_accuracy", "lr", "epoch_seconds", "elapsed_seconds"))
                and row["train_accuracy"] <= 1 and row["elapsed_seconds"] > previous_elapsed,
                "Finite chronological native training history required")
        validate_operation_counts(row.get("ta_operation_counts"), TRAIN_EXAMPLES)
        previous_elapsed = row["elapsed_seconds"]
    return epochs * STEPS, 0, epochs * STEPS


def train(config, folder, data_dir, lineage):
    validate_config(config)
    folder = Path(folder)
    require(folder.is_dir() and not any((folder / name).exists() for name in ARTIFACTS),
            "Authenticated fresh folder with no training artifacts required")
    require(isinstance(lineage, dict) and bool(lineage), "Controller lineage required")
    engine = setup(config, data_dir)
    import runtime
    import numpy as np
    import torch
    from research import train_trivialaugment as ta
    from research.run_trivialaugment_primary import OPERATIONS, validate_operation_counts

    model, ema, optimizer, scaler, device = (engine[key] for key in ("model", "ema", "optimizer", "scaler", "device"))
    marker = {"schema": SCHEMA, "method": "scratch_full50k_fixed_final_trivialaugment", "lineage": lineage,
              "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS, "steps_per_epoch": STEPS,
              "precision": "bf16", "final_variant": "ema", "initialization": "scratch_seeded",
              "ta_rng_algorithm": "PCG64", "ta_rng_seed": 2026, "views_per_source": 1,
              "parent_weights_loaded": False, "repeat_weights_loaded": False, "independent_development_evaluated": False,
              "source_order": "official_archive_order_before_seeded_epoch_permutations"}
    provenance = {**ta.code_provenance(), "research_training": marker}
    runtime.write_json(folder / "config.json", config)
    runtime.write_json(folder / "environment.json", {**_environment(), "research_training": marker})
    runtime.write_json(folder / "provenance.json", {"schema": SCHEMA, "code": provenance, "official_data": engine["source"],
        "data_signature": engine["signature"], "train_examples": TRAIN_EXAMPLES, "validation_examples": 0,
        "parameter_count": PARAMETER_COUNT, "normalization": {"mean": ta.MEAN, "std": ta.STD}})
    print(json.dumps({"event": "trivialaugment_full_refit_started", "precision": "bf16",
        "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "epochs": EPOCHS,
        "parameter_count": PARAMETER_COUNT, "final_variant": "ema"}), flush=True)
    history, updates, totals = [], 0, dict.fromkeys(OPERATIONS, 0)
    began = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(TRAIN_EXAMPLES, device=device)
        order_cpu = order.cpu().numpy()
        require(len(np.unique(order_cpu)) == TRAIN_EXAMPLES and int(order_cpu.min()) == 0
                and int(order_cpu.max()) == TRAIN_EXAMPLES - 1, "Complete original source permutation required")
        order_sha = hashlib.sha256(order_cpu.astype("<i8").tobytes()).hexdigest()
        metrics, counts = metric_state(torch, device), dict.fromkeys(OPERATIONS, 0)
        for batch in range(STEPS):
            row = step(engine, order, epoch, batch, updates, metrics)
            updates += 1
            for name, value in row["ta_operation_counts"].items():
                counts[name] += value
                totals[name] += value
        seen = metrics["seen"]
        require(seen == TRAIN_EXAMPLES and updates == epoch * STEPS, "Complete TA50k one-view epoch required")
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers(), strict=True):
                average.copy_(online)
        ta_state = copy.deepcopy(engine["rng"].bit_generator.state)
        state_sha = ta.json_sha(ta_state)
        elapsed = time.perf_counter() - began
        record = {"epoch": epoch, "train_loss": float(metrics["loss"]) / seen,
                  "train_accuracy": float(metrics["correct"]) / seen,
                  **{key: None for key in VALIDATION_KEYS}, "selected_variant": "ema", "lr": row["lr"],
                  "epoch_seconds": time.perf_counter() - tick, "elapsed_seconds": elapsed,
                  "attempted_updates": STEPS, "successful_updates": STEPS, "skipped_updates": 0,
                  "source_examples": seen, "model_views_seen": seen, "forward_passes": STEPS, "forward_calls": STEPS,
                  "last_source_batch_size": TAIL, "source_order_sha256": order_sha, "grad_scaler_scale": scaler.get_scale(),
                  "ta_operation_counts": counts, "ta_rng_state_sha256": state_sha}
        history.append(record)
        require(validate_history(history, epoch) == (updates, 0, updates), "Cumulative TA50k history differs")
        validate_operation_counts(totals, epoch * TRAIN_EXAMPLES)
        final = epoch == EPOCHS
        common = {"schema": SCHEMA, "model_config": engine["model_config"], "config": config, "epoch": epoch,
                  "parameter_count": PARAMETER_COUNT, "data_signature": engine["signature"], "provenance": provenance}
        if final:
            require(all(int(module.num_batches_tracked) == updates for module in engine["bns"])
                    and all(bool(torch.isfinite(value).all()) for value in ema.state_dict().values()),
                    "Complete finite EMA and exactly one BN forward per update required")
            ta.save_checkpoint(folder / "best.pt", {**common, "model": ema.state_dict(), "variant": "ema",
                "validation_accuracy": None, "validation_loss": None, "selected_ta_rng_state_sha256": state_sha})
        runtime.write_json(folder / "history.json", history)
        runtime.write_json(folder / "metrics.json", {"schema": SCHEMA, "status": "complete" if final else "running",
            "epochs_completed": epoch, "best_validation_accuracy": None, "best_validation_loss": None,
            "parameter_count": PARAMETER_COUNT, "elapsed_seconds": elapsed, "training_examples": TRAIN_EXAMPLES,
            "validation_examples": 0, "config": config, "research_training": marker,
            "attempted_updates": updates, "successful_updates": updates, "skipped_updates": 0,
            "forward_passes": updates, "source_examples_seen": epoch * TRAIN_EXAMPLES,
            "model_views_seen": epoch * TRAIN_EXAMPLES, "ta_operation_counts": totals, "ta_rng_state_sha256": state_sha})
        ta.save_checkpoint(folder / "last.pt", {**common, "formal_training_state": True, "preflight_state": False,
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(), "updates": updates, "attempted_updates": updates, "skipped_updates": 0,
            "forward_passes": updates, "source_examples_seen": epoch * TRAIN_EXAMPLES, "model_views_seen": epoch * TRAIN_EXAMPLES,
            "history": history, "best_accuracy": None, "best_loss": None, "elapsed_seconds": elapsed,
            "ta_operation_counts": totals, "ta_rng_state": ta_state, "ta_rng_state_sha256": state_sha, **ta.capture_rng(device)})
        print(json.dumps(record), flush=True)
    return {"schema": TRAIN_RESULT_SCHEMA, "status": "complete", "precision": "bf16",
            "training_examples": TRAIN_EXAMPLES, "validation_examples": 0, "parameter_count": PARAMETER_COUNT,
            "epochs_completed": EPOCHS, "attempted_updates": updates, "successful_updates": updates, "skipped_updates": 0,
            "source_examples_seen": EPOCHS * TRAIN_EXAMPLES, "model_views_seen": EPOCHS * TRAIN_EXAMPLES,
            "forward_passes": updates, "final_variant": "ema", "lineage": lineage, "ta_operation_counts": totals,
            "ta_rng_state_sha256": state_sha, "independent_development_evaluated": False, "confirmation_evaluated": False,
            "hidden_images_loaded": False, "external_data_loaded": False, "parent_weights_loaded": False,
            "repeat_weights_loaded": False, "environment": _environment()}
