"""Bounded actual-image native-BF16 preflight for two fixed WRN controls.

No arguments show help. This never trains a formal candidate, evaluates images,
loads hidden inputs, saves weights, or submits. All preflight state is discarded.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "research_core_control_preflight_supervisor_v1"
REPORT_SCHEMA = "research_core_control_actual_image_preflight_v1"
INPUT_SCHEMA = "research_core_control_preflight_inputs_v1"
CONFIGS = {
    2026: ("research/configs/wrn16_5.json", "d2745cb8d36e7105cf13b0d32f4467024d36e69d9cc316368c45065063c18528"),
    2027: ("research/configs/wrn16_5_seed2027.json", "2b4d0e98b8357d930994957857522579d04c3b4af6af6c0bbbebfc2372ed49a7"),
}
WRN300_CONFIGS = {
    2026:("research/configs/wrn34_3_300_bf16.json", "cc55380bd03bd0ca2147c7e4a34c31de37345480d71a9721b65737717fc2f348"),
    2027:("research/configs/wrn34_3_300_bf16_seed2027.json", "4ce28d4d6f479addb1d471b21c38bc1ef2ae204256c04b1fbd2f2aed7c1b5363"),
}
RECIPES = {
    "wrn16_5":{"configs":CONFIGS, "parameters":4289754, "bn_modules":13, "epochs":160, "future_trainer":"frozen"},
    "wrn34_3_300_bf16":{"configs":WRN300_CONFIGS, "parameters":4166298, "bn_modules":31, "epochs":300, "future_trainer":"shadow"},
}
PINS = {
    "train.py": "4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py": "79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py": "4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py": "75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
    "research/run_local.py": "6c42cfd08aaad6b39c1acd23ed788044a68c31b047a677a0218308a1dd15a069",
    "research/profile_augmix.py": "82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66",
    "research/process_tree_guard.py": "5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80",
    "research/train_shadow.py": "9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
    "research/augmentations.py": "718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
    "research/wrn16-5-plan.md": "1d7485c64f4911defad68ecc0b5d3b14ac4e35c65f3dfc68e07734404c558559",
    "research/wrn34-3-300-bf16-plan.md": "2678357b601d7cfcbb2eb6efc9920633920601b66f6551cba3a6b79c1384fd2e",
    "research/configs/wrn34_3.json": "2901f0a7d717083c8811aac9cff1910c7a7493eb3c474c260bb9b7f7eb45df02",
}
ARRAYS = {"official_train_images.npy", "official_train_labels.npy", "official_train_ids.npy", "official_split_indices.npz"}
DATA_IDENTITY = {
    "archive_sha256":"d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
    "split_sha256":"73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e",
    "train_indices_sha256":"b8472b8b293ad1aacad02abb7fcb7bfe8d37fd8065f280f675dee6ae8f9d2afd",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_snapshot(root=ROOT):
    actual = {name: sha256(root/name) for name in PINS}
    require(actual == PINS, "Frozen WRN16-5 dependencies changed")
    for recipe in RECIPES.values():
        for name, digest in recipe["configs"].values():
            require(sha256(root/name) == digest, "Frozen control configuration changed")
            actual[name] = digest
    actual["research/preflight_core_control.py"] = sha256(root/"research/preflight_core_control.py")
    actual["research/campaign.json"] = sha256(root/"research/campaign.json")
    return actual


def configuration(seed, root=ROOT, recipe="wrn16_5"):
    require(recipe in RECIPES and type(seed) is int and seed in (2026,2027), "Only the two fixed controls and primary/repeat seeds are supported")
    configs = RECIPES[recipe]["configs"]
    for name, digest in configs.values():
        require(sha256(root/name) == digest, "The exact raw control configurations are required")
    require(sha256(root/"research/configs/wrn34_3.json") == PINS["research/configs/wrn34_3.json"], "Reference recipe changed")
    primary, repeat = [read(root/configs[s][0]) for s in (2026, 2027)]
    changes = {"depth":16,"widen_factor":5} if recipe == "wrn16_5" else {"epochs":300}
    require(primary == {**read(root/"research/configs/wrn34_3.json"), **changes} and
            repeat == {**primary, "seed":2027}, "Only depth/width and the declared repeat seed may differ")
    tree = ast.parse((root/"train.py").read_text())
    defaults = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in node.targets))
    return {**defaults, **(primary if seed == 2026 else repeat)}


def accounting():
    """Reuse only frozen stdlib/psutil functions, without importing its runtime."""
    path = ROOT/"research/run_local.py"
    require(sha256(path) == PINS["research/run_local.py"], "Frozen accounting source changed")
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"now", "used_seconds"}]
    require(len(nodes) == 2, "Frozen accounting primitives are incomplete")
    namespace = dict(datetime=datetime, timezone=timezone, json=json, psutil=psutil, os=os, time=time)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def claim_lease(path):
    """Atomic create only: a stale, foreign or unknown lease is never renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid":os.getpid(), "process_created_at":psutil.Process().create_time(),
                   "started_at":datetime.now(timezone.utc).isoformat()}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def lease_can_release(owner, receipt):
    from research.process_tree_guard import may_release_lease
    return (may_release_lease(receipt) and owner.get("cleanup_requires_reconciliation") is not True and
            owner.get("process_tree_cleanup_verified") is not False and
            owner.get("status") not in ("unknown", "failed", "cleanup_unverified"))


def bounded_seconds(policy, current, used):
    from research.profile_augmix import budget_seconds
    seconds = budget_seconds(policy, current, used)
    require(seconds == 180, "Reserve the complete 180-second actual-image preflight or wait")
    return seconds


def official_inputs(data_dir):
    manifest = read(data_dir/"training_manifest.json")
    require(manifest.get("competition") == "cifar-10-res-net-architecture-challenge-dl" and
            manifest.get("source_type") == "official_kaggle_competition_training" and
            manifest.get("split", {}).get("schema") == "train_dev_confirmation_v1" and
            manifest["split"].get("seed") == 2026 and set(manifest.get("files_sha256", {})) == ARRAYS,
            "Only the audited official training cache and fixed split are supported")
    paths = [data_dir/"training_manifest.json", *(data_dir/name for name in sorted(ARRAYS))]
    require(all(sha256(data_dir/name) == digest for name, digest in manifest["files_sha256"].items()), "Official training cache changed")
    source = manifest["source_provenance"]
    identity = {"archive_sha256":source.get("official_train_npz_sha256", source.get("ordered_training_png_sha256")),
                "split_sha256":manifest["split"]["indices_sha256"],
                "train_indices_sha256":manifest["split"]["train_indices_sha256"]}
    require(identity == DATA_IDENTITY and source.get("train_labels_csv_sha256") ==
            "b596f4cbb12107ecac987e3a61123af40408a6bb83719a1dbb5cc3ce70968e93", "Exact audited official archive/split/train pool required")
    return paths, identity


def validate_worker(folder, manifest_sha):
    """Authenticate the real OS boundary before importing runtime/Torch/data."""
    folder = Path(folder).resolve()
    require(folder.parent == (ROOT/"results/research").resolve() and re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") and
            sha256(folder/"inputs.json") == manifest_sha, "Controlled immutable preflight inputs required")
    manifest, receipt = read(folder/"inputs.json"), read(folder/"supervisor.json")
    process, parent = psutil.Process(), psutil.Process(os.getppid())
    started = datetime.fromisoformat(receipt.get("started_at", ""))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "benchmarking" and
            receipt.get("input_manifest_sha256") == manifest_sha and started.tzinfo is not None and
            type(receipt.get("reserved_seconds")) in (int, float) and 0 < receipt["reserved_seconds"] <= 180 and
            type(receipt.get("deadline_unix")) in (int, float) and math.isfinite(receipt["deadline_unix"]) and
            started.timestamp() <= time.time()+.1 and time.time() < receipt["deadline_unix"] <= started.timestamp()+receipt["reserved_seconds"]+.1 and
            receipt.get("pid") == parent.pid and abs(receipt.get("process_created_at", 0)-parent.create_time()) < .01 and
            receipt.get("child_pid") == process.pid and abs(receipt.get("child_created_at", 0)-process.create_time()) < .01,
            "Exact live bounded preflight supervisor required")
    lease = read(ROOT/"results/research/local_lease.json")
    require(lease.get("pid") == parent.pid and abs(lease.get("process_created_at", 0)-parent.create_time()) < .01 and
            lease.get("cleanup_requires_reconciliation") is not True and lease.get("process_tree_cleanup_verified") is not False and
            lease.get("status") not in ("unknown", "failed", "cleanup_unverified"), "Canonical live local lease required")
    matches = []
    for directory in folder.glob("process_tree_*"):
        try:
            spec, ready = read(directory/"spec.json"), read(directory/"ready.json")
        except (OSError, ValueError):
            continue
        if ready.get("pid") == process.pid and abs(ready.get("created_at", 0)-process.create_time()) < .01:
            matches.append((directory, spec, ready))
    require(len(matches) == 1, "Exactly one actual preflight OS guard required")
    directory, spec, ready = matches[0]
    nonce = directory.name.removeprefix("process_tree_")
    guard = (ROOT/"research/process_tree_guard.py").resolve()
    require(re.fullmatch(r"[0-9a-f]{32}", nonce) and spec.get("schema") == ready.get("schema") == "owned_python_process_tree_v1" and
            spec.get("nonce") == ready.get("nonce") == nonce and spec.get("host_pid") == parent.pid and
            abs(spec.get("host_created_at", 0)-parent.create_time()) < .01 and Path(spec.get("cwd", "")).resolve() == ROOT.resolve() and
            Path(spec.get("script", "")).resolve() == Path(__file__).resolve() and
            spec.get("script_sha256") == ready.get("script_sha256") == sha256(__file__) and
            spec.get("guard_source_sha256") == sha256(guard) == PINS["research/process_tree_guard.py"] and
            spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", manifest_sha] and
            ready.get("method") == ("windows_job_v1" if os.name == "nt" else "posix_session_v1") and
            type(spec.get("deadline_unix")) in (int, float) and time.time() < spec["deadline_unix"] <= receipt["deadline_unix"]+.1 and
            spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if os.name == "nt" else None), "Actual guard identity/source/deadline differs")
    command = process.cmdline()
    require(len(command) == 4 and Path(command[0]).resolve() == Path(sys.executable).resolve() and Path(command[1]).resolve() == guard and
            command[2] == "--bootstrap" and Path(command[3]).resolve() == directory/"spec.json", "Actual guarded bootstrap required")
    from research.profile_augmix import verify_os_boundary
    verify_os_boundary(spec, ready, process)
    sources = source_snapshot()
    recipe = manifest.get("recipe")
    config = configuration(manifest.get("seed"), recipe=recipe)
    expected_config = RECIPES[recipe]["configs"][manifest["seed"]][1]
    paths, identity = official_inputs(Path(manifest["data_dir"]))
    expected_binding = {str(path.resolve()):sha256(path) for path in [*paths, ROOT/RECIPES[recipe]["configs"][manifest["seed"]][0], *(ROOT/name for name in sources)]}
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("run") == receipt.get("run") == folder.name and
            receipt.get("recipe") == recipe and
            receipt.get("mode") == "preflight" and receipt.get("backend") == "local" and receipt.get("precision") == "bf16" and
            receipt.get("confirmation_evaluated") is False and receipt.get("submitted") is False and
            isinstance(manifest.get("source_git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", manifest["source_git_commit"]) and
            manifest.get("source_sha256") == receipt.get("source_sha256") == sources and
            manifest.get("config_sha256") == receipt.get("config_sha256") == expected_config and manifest.get("effective_config") == config and
            manifest.get("effective_config_sha256") == json_sha(config) and manifest.get("data_identity") == identity and
            manifest.get("input_sha256") == receipt.get("input_sha256") == expected_binding,
            "Exact source/config/official-input preflight binding differs")
    return manifest


def core_step(core, torch, model, ema, optimizer, train_x, train_y, order, batch, config, updates, future_trainer="frozen", scaler=None):
    """Original train.py per-batch math; proofs wrap it without changing RNG."""
    import random
    import numpy as np
    from torch.nn import functional as F
    require(future_trainer in ("frozen","shadow"), "Fixed core/shadow batch primitive required")
    lr = core.schedule(batch, config["epochs"]*313, 313, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
    ids = order[batch*config["batch_size"]:(batch+1)*config["batch_size"]]
    y = train_y[ids]
    x = core.augment(train_x[ids], config["cutout"])
    mix = config["mixup_alpha"] > 0 and random.random() < config["mixup_probability"]
    lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mix else 1.0
    permutation = torch.randperm(len(y), device=y.device) if mix else None
    if mix:
        x = x*lam+x[permutation]*(1-lam)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=y.device.type == "cuda"):
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=config["label_smoothing"])*lam
        if mix:
            extra = F.cross_entropy(logits, y[permutation], label_smoothing=config["label_smoothing"])*(1-lam)
            if future_trainer == "shadow":
                loss += extra
            else:
                loss = loss+extra
    require(logits.dtype == (torch.bfloat16 if y.device.type == "cuda" else torch.float32), "Actual forward dtype differs from the original native BF16 protocol")
    require(bool(torch.isfinite(loss).item()) and loss.item() > 0, "Finite positive actual loss required")
    if future_trainer == "shadow":
        from research.train_shadow import optimizer_step
        require(scaler is not None and not scaler.is_enabled() and scaler.get_scale() == 1., "Original BF16 shadow disabled-scaler branch required")
        require(optimizer_step(loss, optimizer, scaler) is True, "BF16 shadow may not skip updates")
    else:
        require(scaler is None, "Original core BF16 uses direct backward/SGD")
        loss.backward()
    grads = [p.grad for p in model.parameters()]
    require(all(g is not None for g in grads), "Every trainable parameter requires its real gradient")
    norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt()
    require(bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item()) and
            math.isfinite(norm.item()) and norm.item() > 0, "All gradients must be finite with positive norm")
    if future_trainer == "frozen":
        optimizer.step()
    updates += 1
    with torch.no_grad():
        decay = min(config["ema_decay"], (1+updates)/(10+updates))
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
        # Original train.py also accounts training metrics before ending a step.
        correct = logits.detach().argmax(1).eq(y).sum()*lam
        if mix:
            correct += logits.detach().argmax(1).eq(y[permutation]).sum()*(1-lam)
        loss_sum = loss.detach()*len(y)
    return {"loss":loss.item(), "gradient_norm":norm.item(), "lr":lr, "mixed":bool(mix), "updates":updates, "logits_dtype":str(logits.dtype)}


def worker(folder, manifest_sha):
    setup_began = time.perf_counter()
    manifest = validate_worker(folder, manifest_sha)
    import runtime  # DLL isolation precedes Torch; never imported by the host.
    import copy
    import numpy as np
    import torch
    import train as core
    from data import load_training_arrays
    from model import build_model, count_parameters
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False), "Actual native CUDA BF16 required; emulation/FP16/CPU are forbidden")
    config = manifest["effective_config"]
    recipe = RECIPES[manifest["recipe"]]
    runtime.seed_all(config["seed"])
    images, labels, train_ids, _ = load_training_arrays(Path(manifest["data_dir"]))
    require(len(train_ids) == 40000, "Only the original 40k training pool is allowed")
    require(hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest() == DATA_IDENTITY["train_indices_sha256"], "Actual training row identities differ")
    device = torch.device("cuda")
    train_x, train_y = core.tensor_images(images[train_ids], device), torch.as_tensor(labels[train_ids], device=device)
    model = build_model({k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}).to(device=device, memory_format=torch.channels_last)
    require(count_parameters(model) == count_parameters(model, True) == recipe["parameters"], "Actual total/trainable parameter count differs")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False) if recipe["future_trainer"] == "shadow" else None
    model.train()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    require(len(bns) == recipe["bn_modules"] and all(m.num_batches_tracked.item() == 0 for m in bns), "Original scratch BN topology/state required")
    order = torch.randperm(len(train_y), device=device)
    selected = train_ids[order[:23*128].cpu().numpy()]
    original_ids = np.load(Path(manifest["data_dir"])/"official_train_ids.npy", allow_pickle=False)[selected]
    require(len(np.unique(selected)) == 2944, "Preflight must use 2944 unique official training rows")
    rows, durations = [], []
    parameters, averages = list(model.parameters()), list(ema.parameters())
    require(all(p.dtype == torch.float32 for p in [*parameters, *averages]), "Original FP32 model/EMA master parameters required")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_began
    for batch in range(23):
        torch.cuda.synchronize()
        began = time.perf_counter()
        before, before_ema = parameters[0].detach().clone(), averages[0].detach().clone()
        counters = [m.num_batches_tracked.item() for m in bns]
        row = core_step(core, torch, model, ema, optimizer, train_x, train_y, order, batch, config, batch, recipe["future_trainer"], scaler)
        state_values = [*model.state_dict().values(), *ema.state_dict().values(),
                        *(value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))]
        finite = bool(torch.stack([torch.isfinite(p).all() for p in state_values]).all().item())
        changed = (parameters[0]-before).abs().max().item()
        ema_changed = (averages[0]-before_ema).abs().max().item()
        require(finite and changed > 0 and ema_changed > 0 and
                all(m.num_batches_tracked.item() == old+1 for m, old in zip(bns, counters)), "Finite real SGD/EMA updates and exactly one BN forward required")
        row.update(parameter_probe_max_delta=changed, ema_probe_max_delta=ema_changed, all_parameters_finite=True, bn_single_forward=True)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-began)
        rows.append(row)
    env = runtime.environment()
    report = {"schema":REPORT_SCHEMA, "status":"complete", "engineering_only":True, "synthetic_data":False,
        "accuracy_evaluated":False, "formal_training_admitted":False, "hidden_images_loaded":False, "confirmation_evaluated":False,
        "training_precision":"bf16", "native_bf16_without_emulation":True, "grad_scaler_enabled":False,
        "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
        "recipe":manifest["recipe"], "future_trainer":recipe["future_trainer"], "epochs":config["epochs"],
        "planned_formal_attempted_updates":313*config["epochs"], "disabled_scaler_scale":scaler.get_scale() if scaler is not None else None,
        "seed":config["seed"], "parameter_count":recipe["parameters"], "trainable_parameter_count":recipe["parameters"],
        "config_sha256":manifest["config_sha256"], "effective_config_sha256":manifest["effective_config_sha256"],
        "source_sha256":manifest["source_sha256"], "input_manifest_sha256":manifest_sha,
        "source_git_commit":manifest["source_git_commit"], "data_identity":manifest["data_identity"],
        "environment":env, "environment_sha256":json_sha(env), "training_pool_examples":40000, "evaluation_examples":0,
        "batch_size":128, "warmup_updates":3, "timed_updates":20, "successful_updates":23, "skipped_updates":0,
        "source_examples_seen":2944, "unique_source_examples":2944,
        "ordered_training_indices_sha256":hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "ordered_official_ids_sha256":hashlib.sha256(np.ascontiguousarray(original_ids).tobytes()).hexdigest(),
        "bn_modules":recipe["bn_modules"], "bn_forward_counts":[int(m.num_batches_tracked.item()) for m in bns],
        "ema_buffer_rule":"Original epoch-end-only copy; no epoch boundary occurs in this preflight.", "ema_buffer_copies":0,
        "steps":rows, "step_seconds":durations, "median_step_seconds":statistics.median(durations[3:]),
        "setup_seconds":setup_seconds, "full_step_seconds":sum(durations),
        "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(), "preflight_state_discarded":True}
    report["conservative_pipeline_seconds"] = 1.5*report["median_step_seconds"]*313*config["epochs"]+300
    report["timing_scope"] = "Actual uint8 gather/augment/MixUp/transfer-resident model/loss/backward/SGD/EMA/metrics and all finite/update/BN proofs; 3 warmup plus 20 timed full batches. Initial official input staging is charged by supervisor wall time."
    validate_report(report, manifest, manifest_sha)
    write(folder/"core_control_preflight.json", report)


def validate_report(report, manifest, manifest_sha):
    require(manifest.get("recipe") in RECIPES, "Fixed named control required")
    recipe = RECIPES[manifest["recipe"]]
    config = configuration(manifest.get("seed"), recipe=manifest["recipe"])
    require(manifest.get("effective_config") == config and manifest.get("effective_config_sha256") == json_sha(config) and
            manifest.get("config_sha256") == recipe["configs"][manifest["seed"]][1] and manifest.get("data_identity") == DATA_IDENTITY,
            "Predeclared raw/effective configuration and official data identity required")
    expected = {"schema":REPORT_SCHEMA, "status":"complete", "engineering_only":True, "synthetic_data":False,
        "accuracy_evaluated":False, "formal_training_admitted":False, "hidden_images_loaded":False, "confirmation_evaluated":False,
        "training_precision":"bf16", "native_bf16_without_emulation":True, "grad_scaler_enabled":False,
        "master_parameter_dtype":"torch.float32", "ema_parameter_dtype":"torch.float32",
        "recipe":manifest["recipe"], "future_trainer":recipe["future_trainer"], "epochs":recipe["epochs"],
        "planned_formal_attempted_updates":313*recipe["epochs"], "disabled_scaler_scale":1. if recipe["future_trainer"] == "shadow" else None,
        "seed":manifest["seed"], "parameter_count":recipe["parameters"], "trainable_parameter_count":recipe["parameters"],
        "training_pool_examples":40000, "evaluation_examples":0, "batch_size":128, "warmup_updates":3, "timed_updates":20,
        "successful_updates":23, "skipped_updates":0, "source_examples_seen":2944, "unique_source_examples":2944,
        "bn_modules":recipe["bn_modules"], "bn_forward_counts":[23]*recipe["bn_modules"], "ema_buffer_copies":0, "preflight_state_discarded":True,
        "config_sha256":manifest["config_sha256"], "effective_config_sha256":manifest["effective_config_sha256"],
        "source_sha256":manifest["source_sha256"], "input_manifest_sha256":manifest_sha,
        "source_git_commit":manifest["source_git_commit"], "data_identity":manifest["data_identity"]}
    require(all(type(report.get(k)) is type(v) and report[k] == v for k,v in expected.items()), "Complete typed actual-image BF16 evidence required")
    rows, durations = report.get("steps", []), report.get("step_seconds", [])
    require(len(rows) == len(durations) == 23 and all(type(s) in (int,float) and math.isfinite(s) and s > 0 for s in durations), "23 positive complete full-step timings required")
    for index, row in enumerate(rows, 1):
        require(type(row.get("updates")) is int and row["updates"] == index and row.get("all_parameters_finite") is True and row.get("bn_single_forward") is True and
                type(row.get("mixed")) is bool and row.get("logits_dtype") == "torch.bfloat16" and
                all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k] > 0 for k in
                    ("loss", "gradient_norm", "lr", "parameter_probe_max_delta", "ema_probe_max_delta")), "Finite positive actual update proof required")
        require(row["lr"] == manifest["effective_config"]["lr"]*index/(5*313), "Original initial-epoch warmup schedule required")
    median = statistics.median(durations[3:])
    require(report.get("median_step_seconds") == median and report.get("conservative_pipeline_seconds") == 1.5*median*313*recipe["epochs"]+300,
            "Fixed actual full-pipeline median/guard differs")
    require(type(report.get("setup_seconds")) in (int,float) and math.isfinite(report["setup_seconds"]) and report["setup_seconds"] > 0 and
            report.get("full_step_seconds") == sum(durations), "Complete setup/full-step wall accounting required")
    env = report.get("environment", {})
    require(env.get("device") == "cuda" and bool(env.get("gpu")) and env.get("amp_dtype") == "torch.bfloat16" and
            report.get("environment_sha256") == json_sha(env) and type(report.get("peak_gpu_memory_bytes")) is int and report["peak_gpu_memory_bytes"] > 0,
            "Actual native CUDA environment and peak allocation required")
    for key in ("ordered_training_indices_sha256", "ordered_official_ids_sha256"):
        require(isinstance(report.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", report[key]), "Private source identity hashes required")
    json.dumps(report, allow_nan=False)


def parent(args):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,69}", args.run or "") and not args.run.startswith("stability"), "Fresh controlled run ID required")
    sources, config = source_snapshot(), configuration(args.seed, recipe=args.recipe)
    configs = RECIPES[args.recipe]["configs"]
    policy = read(ROOT/"research/campaign.json")
    account = accounting()
    data_dir = args.data_dir.resolve()
    paths, identity = official_inputs(data_dir)
    binding = {str(path.resolve()):sha256(path) for path in [*paths, ROOT/configs[args.seed][0], *(ROOT/name for name in sources)]}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Actual source commit required")
    base, folder = ROOT/"results/research", ROOT/"results/research"/args.run
    lease = base/"local_lease.json"
    require(not folder.exists() and not lease.exists(), "Existing artifacts/lease require explicit reconciliation; no restart/resume here")
    day = account["now"]().replace(hour=0,minute=0,second=0,microsecond=0)
    bounded_seconds(policy, account["now"](), account["used_seconds"](base, day, day+timedelta(days=1)))
    from research.process_tree_guard import run_guarded, may_release_lease
    claim_lease(lease)
    began, owns, receipt = time.perf_counter(), False, {}
    try:
        seconds = bounded_seconds(policy, account["now"](), account["used_seconds"](base, day, day+timedelta(days=1)))
        folder.mkdir(parents=True, exist_ok=False)
        owns = True
        started = account["now"]()
        manifest = {"schema":INPUT_SCHEMA, "recipe":args.recipe, "seed":args.seed, "run":args.run, "data_dir":str(data_dir),
            "source_sha256":sources, "config_sha256":configs[args.seed][1], "effective_config":config,
            "effective_config_sha256":json_sha(config), "source_git_commit":commit, "data_identity":identity, "input_sha256":binding}
        write(folder/"inputs.json", manifest)
        manifest_sha = sha256(folder/"inputs.json")
        receipt = {"schema":SCHEMA, "backend":"local", "run":args.run, "status":"benchmarking", "mode":"preflight",
            "trainer":"core_control_engineering", "recipe":args.recipe, "precision":"bf16", "pid":os.getpid(), "process_created_at":psutil.Process().create_time(),
            "started_at":started.isoformat(), "reserved_seconds":seconds, "deadline_unix":started.timestamp()+seconds,
            "source_sha256":sources, "config_sha256":configs[args.seed][1], "input_sha256":binding, "input_manifest_sha256":manifest_sha,
            "confirmation_evaluated":False, "submitted":False}
        write(folder/"supervisor.json", receipt)
        remaining = receipt["deadline_unix"]-time.time()-2
        require(remaining > 0, "Preflight preparation exhausted its reservation")
        run_guarded([sys.executable, str(Path(__file__).resolve()), "--_worker", str(folder.resolve()), "--_manifest-sha", manifest_sha],
                    folder/"preflight.log", remaining, folder/"supervisor.json", receipt, cwd=ROOT)
        result = read(folder/"core_control_preflight.json")
        validate_report(result, manifest, manifest_sha)
        require(source_snapshot() == sources and all(sha256(path) == digest for path,digest in binding.items()) and time.time() <= receipt["deadline_unix"], "Bounded unchanged complete preflight required")
        receipt.update(status="complete", artifact="core_control_preflight.json", artifact_sha256=sha256(folder/"core_control_preflight.json"))
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if owns:
            receipt.update(elapsed_seconds=time.perf_counter()-began, finished_at=account["now"]().isoformat())
            if not may_release_lease(receipt):
                receipt["status"] = "cleanup_unverified"
            write(folder/"supervisor.json", receipt)
        if lease.exists():
            owner = read(lease)
            if owner.get("pid") == os.getpid() and abs(owner.get("process_created_at", 0)-psutil.Process().create_time()) < .01:
                if lease_can_release(owner, receipt):
                    lease.unlink()
                else:
                    owner.update(cleanup_requires_reconciliation=True, process_tree_cleanup_verified=False, supervisor=str(folder/"supervisor.json"))
                    write(lease, owner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--recipe", choices=tuple(RECIPES))
    parser.add_argument("--seed", type=int, choices=(2026,2027))
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest-sha", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        require(args.run is None and args.recipe is None and args.seed is None and args._manifest_sha is not None, "Only the guarded worker signature is allowed")
        worker(args._worker, args._manifest_sha)
    elif args.run is None and args.recipe is None and args.seed is None and args._manifest_sha is None:
        parser.print_help()
    else:
        require(args.run is not None and args.recipe is not None and args.seed is not None and args._manifest_sha is None, "Explicit fresh --run, --recipe and --seed are required")
        parent(args)


if __name__ == "__main__":
    main()
