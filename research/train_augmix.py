"""Independent scratch AugMix+JSD160 training and actual-image pipeline preflight.

CPU/PIL reflect crop/flip and two AugMix views precede one view-major 3B
forward. Only official train IDs are used; confirmation and hidden inputs are
never selected. Native CUDA is explicitly FP16+GradScaler. CPU FP32 exists for
synthetic mathematical tests, not GPU admission or performance claims.
The explicit init_scale256 numerical protocol differs from frozen shadow's
formal default65536; cross-method changes are not solely augmentation effects.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import environment, seed_all, write_json
import numpy as np
import torch
import psutil
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.nn import functional as F
from data import load_training_arrays
from model import build_model, count_parameters
from research.augmix import AugMixConfig, augmix_loss, augmix_views, rgb_image
from research.train_shadow import (capture_rng, evaluate, optimizer_step,
    resolve_precision, restore_rng, validate_partition)
from train import (DEFAULT_CONFIG, MEAN, STD, code_provenance, save_checkpoint,
    schedule, sha256, tensor_images, training_source)

SCHEMA = "research_augmix_training_v1"
PREFLIGHT_SCHEMA = "research_augmix_preflight_v1"
PARAMETER_COUNT = 4_298_970
OFFICIAL_SIGNATURE = {"archive_sha256":"d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
                      "split_sha256":"73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e"}
SOURCE_KEYS = ("research/train_augmix.py", "research/augmix.py", "research/train_shadow.py", "research/augmentations.py")


def require(condition, message):
    if not condition: raise ValueError(message)


def fixed_config(seed=2026):
    require(type(seed) is int and seed in (2026, 2027), "AugMix engine supports exactly integer primary2026 or future frozen-repeat2027")
    return {**DEFAULT_CONFIG, "cutout": 0, "mixup_alpha": 0.0, "mixup_probability": 0.0,
        "augmix_width": 3, "augmix_depth": -1, "augmix_severity": 3, "augmix_alpha": 1.0,
        "augmix_all_ops": False, "augmix_jsd_weight": 12.0, "augmix_num_workers": 2,
        "augmix_prefetch_factor": 2, "augmix_multiprocessing_context": "spawn", "augmix_persistent_workers": False,
        "amp_init_scale":256.0, "seed":seed}


def validate_config(config):
    require(isinstance(config, dict), "Exact AugMix configuration mapping required")
    fixed = fixed_config(config.get("seed"))
    require(config == fixed and set(config) == set(fixed) and
            all(type(config[k]) is type(v) for k, v in fixed.items()),
            "Only seed2026/2027 fixed WRN22-4/AugMix-JSD160/sourceB128 recipe is supported")


def raw_config(config, raw=None):
    validate_config(config)
    raw = raw if raw is not None else (json.dumps(config, indent=2)+"\n").encode()
    require(isinstance(raw, bytes) and {**DEFAULT_CONFIG, **json.loads(raw)} == config,
            "Exact raw AugMix input differs from the effective fixed recipe")
    return raw


def research_provenance(precision, config, input_config_sha256=None):
    validate_config(config)
    require(precision in ("fp16", "fp32"), "AugMix requires explicit CUDA FP16 or CPU FP32")
    raw_sha = input_config_sha256 or hashlib.sha256(raw_config(config)).hexdigest()
    require(isinstance(raw_sha, str) and len(raw_sha) == 64 and all(c in "0123456789abcdef" for c in raw_sha),
            "AugMix raw configuration SHA256 is required")
    return {"schema": SCHEMA, "method": "augmix_jsd", "source_sha256": {key:sha256(ROOT/key) for key in SOURCE_KEYS},
        "precision": precision, "grad_scaler_enabled": precision == "fp16", "grad_scaler_init_scale":config["amp_init_scale"],
        "numeric_protocol":"explicit_FP16_init256_differs_frozen_shadow_formal_default65536",
        "training_seed": config["seed"], "split_seed": 2026,
        "input_config_sha256": raw_sha,
        "effective_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_batch_size": 128, "model_batch_size": 384, "forward": "one_view_major_3B_forward",
        "augmentation_rng": "PCG64_epoch_source_seeds_then_per_source_PCG64_two_child_views_v2",
        "data_pipeline": {"num_workers": 2, "prefetch_factor": 2, "context": "spawn", "persistent_workers": False,
            "sampler_rng": "independent_torch_CPU_generator", "loader_rng": "independent_torch_CPU_generator"},
        "common_preprocessing": "reflect4_crop32_xy0to8_independent_flip0.5_once",
        "loss": "CEclean_plus12_mean_KL_three_to_clamped_mixture_FP32_logtarget",
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}


def _generator(rng):
    require(isinstance(rng, np.random.Generator) and isinstance(rng.bit_generator, np.random.PCG64),
            "An explicit saved PCG64 parent generator is required")


def common_crop_flip(image, rng):
    """Distribution matches the campaign reflect crop/flip; RNG sequence differs."""
    _generator(rng)
    require(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.shape == (32, 32, 3),
            "Common preprocessing requires uint8 RGB32 NHWC")
    top, left = (int(v) for v in rng.integers(0, 9, size=2))
    padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="reflect")
    cropped = padded[top:top+32, left:left+32]
    if float(rng.random()) < .5: cropped = cropped[:, ::-1]
    return np.ascontiguousarray(cropped).copy()


def prepare_batch(images, labels, rng, device, *, sample_seeds=None):
    """CPU/PIL -> three normalized view-major blocks -> one device transfer."""
    require(isinstance(images, np.ndarray) and images.dtype == np.uint8 and images.ndim == 4 and
            images.shape[1:] == (32, 32, 3) and 1 <= len(images) <= 128,
            "AugMix source minibatch must contain1..128 uint8 RGB32 images")
    if isinstance(labels, torch.Tensor):
        require(labels.device.type == "cpu" and labels.dtype == torch.int64, "Source labels must be CPU int64")
        labels = labels.numpy()
    require(isinstance(labels, np.ndarray) and labels.dtype == np.int64 and labels.shape == (len(images),) and
            np.all((labels >= 0) & (labels < 10)), "Source labels must be matching ten-class int64 IDs")
    if sample_seeds is None:
        _generator(rng)
        sample_seeds = rng.integers(0, 2**64, size=len(images), dtype=np.uint64)
    require(isinstance(sample_seeds, np.ndarray) and sample_seeds.dtype == np.uint64 and sample_seeds.shape == (len(images),),
            "Every source image requires its predeclared uint64 seed")
    columns = [[], [], []]
    for image, seed in zip(images, sample_seeds):
        per_source = np.random.Generator(np.random.PCG64(int(seed)))
        base = common_crop_flip(image, per_source)
        for bucket, view in zip(columns, augmix_views(base, per_source)): bucket.append(view)
    normalized = torch.cat([torch.stack(bucket) for bucket in columns], dim=0)
    normalized = normalized.to(device=device, memory_format=torch.channels_last)
    return normalized, torch.as_tensor(labels.copy(), dtype=torch.int64, device=device)


class SeededSourceDataset(Dataset):
    """CPU-only rows: all transform randomness comes from a saved source seed."""
    def __init__(self, images, labels, seeds):
        self.images, self.labels, self.seeds = images, labels, seeds

    def __len__(self): return len(self.labels)

    def __getitem__(self, index):
        if get_worker_info() is not None:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
                    "Spawned AugMix workers must remain CPU-only without CUDA initialization")
        return self.images[index], int(self.labels[index]), self.seeds[index]


def collate_seeded_sources(samples):
    images, labels, seeds = zip(*samples)
    return prepare_batch(np.stack(images), np.asarray(labels, dtype=np.int64), None, torch.device("cpu"),
                         sample_seeds=np.asarray(seeds, dtype=np.uint64))


class EpochPipeline:
    """Ordered spawned workers with explicit source seeds and bounded teardown."""
    def __init__(self, images, labels, train_ids, rng, sampler_rng, loader_rng, *, workers=2, limit_batches=None, output=None, tag="epoch0001"):
        _generator(rng)
        require(type(workers) is int and workers in (0, 2), "Only serial CPU audit or fixed two-worker pipeline is supported")
        require(isinstance(sampler_rng, torch.Generator) and isinstance(loader_rng, torch.Generator), "Independent sampler and loader generators required")
        ids = np.asarray(train_ids)
        require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) > 0 and len(np.unique(ids)) == len(ids), "Unique source IDs required")
        order = torch.randperm(len(ids), generator=sampler_rng, device="cpu").numpy()
        count = len(ids) if limit_batches is None else min(len(ids), limit_batches*128)
        self.source_ids = ids[order[:count]].copy()
        self.sample_seeds = rng.integers(0, 2**64, size=count, dtype=np.uint64)
        self.workers, self.output, self.tag = workers, Path(output) if output is not None else None, tag
        require(tag.replace("_", "").isalnum(), "Safe pipeline receipt tag required")
        dataset = SeededSourceDataset(np.asarray(images)[self.source_ids].copy(), np.asarray(labels)[self.source_ids].copy(), self.sample_seeds)
        kwargs = {"batch_size":128, "shuffle":False, "drop_last":False, "num_workers":workers,
                  "collate_fn":collate_seeded_sources, "generator":loader_rng, "pin_memory":False}
        if workers: kwargs.update(prefetch_factor=2, multiprocessing_context="spawn", persistent_workers=False)
        self.loader = DataLoader(dataset, **kwargs)
        self.iterator, self.worker_records, self.closed = None, [], False
        self.owner_pid, self.owner_created_at = os.getpid(), psutil.Process().create_time()

    def _receipt(self, status, stopped):
        if self.output is not None:
            write_json(self.output/f"worker_pool_{self.tag}.json", {"schema":"research_augmix_worker_pool_v1", "status":status,
                "owner_pid":self.owner_pid, "owner_created_at":self.owner_created_at, "workers":self.worker_records,
                "num_workers":self.workers, "prefetch_factor":2 if self.workers else None, "context":"spawn" if self.workers else None,
                "worker_cuda_initialized":False, "all_workers_stopped":stopped,
                "source_ids_sha256":hashlib.sha256(np.asarray(self.source_ids,dtype="<i8").tobytes()).hexdigest(),
                "sample_seeds_sha256":hashlib.sha256(self.sample_seeds.astype("<u8").tobytes()).hexdigest()})

    def __enter__(self):
        if self.output is not None:
            path = self.output/f"worker_pool_{self.tag}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                for item in previous.get("workers", []):
                    try:
                        worker = psutil.Process(item["pid"])
                        require(abs(worker.create_time()-item["created_at"]) >= .01, "An old exact AugMix worker is still alive")
                    except psutil.NoSuchProcess: pass
                with path.with_name(path.stem+f"_preserved_{time.time_ns()}.json").open("xb") as stream: stream.write(path.read_bytes())
        saved_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        before_children = {(child.pid, child.create_time()) for child in psutil.Process().children()} if self.workers else set()
        try:
            if self.workers: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.iterator = iter(self.loader)
            # A worker may exit immediately after the iterator is returned.
            # Identity collection and the first durable receipt therefore
            # belong to the same cleanup-on-failure boundary as spawning.
            self.worker_records = []
            for worker in getattr(self.iterator, "_workers", []):
                self.worker_records.append({"pid":worker.pid, "created_at":psutil.Process(worker.pid).create_time()})
            self._receipt("running", False)
        except BaseException:
            # A partially constructed DataLoader iterator may not be returned.
            # Only newly spawned direct children of this exact trainer whose
            # command identifies multiprocessing.spawn can belong to this pool.
            for child in psutil.Process().children() if self.workers else []:
                try:
                    identity = (child.pid, child.create_time())
                    recorded = {(item["pid"],item["created_at"]) for item in self.worker_records}
                    if identity not in before_children and identity not in recorded and "spawn_main" in " ".join(child.cmdline()):
                        self.worker_records.append({"pid":identity[0], "created_at":identity[1]})
                except psutil.NoSuchProcess: pass
            self.closed = cleanup_workers(self.worker_records)
            self._receipt("startup_failed", self.closed)
            raise
        finally:
            if self.workers:
                if saved_mask is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else: os.environ["CUDA_VISIBLE_DEVICES"] = saved_mask
        return self

    def __iter__(self): return self
    def __next__(self): return next(self.iterator)

    def close(self):
        if self.closed: return
        try:
            if self.iterator is not None and self.workers: self.iterator._shutdown_workers()
        finally:
            stopped = cleanup_workers(self.worker_records)
            self.closed = stopped
            self._receipt("closed" if stopped else "cleanup_unverified", stopped)
        require(stopped, "Owned AugMix worker cleanup did not complete")

    def __exit__(self, *_): self.close()


def cleanup_workers(records):
    """Only act on exact owned PID/create identities; never enumerate other jobs."""
    stopped = True
    for item in records:
        try:
            process = psutil.Process(item["pid"])
            if abs(process.create_time()-item["created_at"]) >= .01: continue
            try: process.wait(timeout=3)
            except psutil.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        except psutil.NoSuchProcess: pass
        except (psutil.AccessDenied, psutil.TimeoutExpired): stopped = False
    return stopped


def build_epoch_loader(*args, **kwargs): return EpochPipeline(*args, **kwargs)


def train_step(model, optimizer, scaler, prepared, device, dtype, *, ema=None, successful_updates=0, ema_decay=.995):
    images, targets = prepared
    batch = len(targets)
    require(images.shape == (3*batch, 3, 32, 32) and images.dtype == torch.float32 and
            images.device == targets.device and images.device.type == device.type and
            (device.index is None or images.device.index == device.index) and 1 <= batch <= 128,
            "A single normalized view-major3B minibatch is required")
    require(type(successful_updates) is int and successful_updates >= 0, "Successful-update counter is required")
    optimizer.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        logits = model(images)
    require(logits.shape == (3*batch, 10), "AugMix requires one ten-class3B forward")
    clean, first, second = logits.split(batch)
    loss = augmix_loss(clean, first, second, targets)
    cross_entropy = F.cross_entropy(clean.float(), targets)
    jsd = (loss-cross_entropy)/12
    updated = optimizer_step(loss, optimizer, scaler)
    after = successful_updates+int(updated)
    if updated and ema is not None:
        with torch.no_grad():
            decay = min(ema_decay, (1+after)/(10+after))
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
    result = {"loss": float(loss.detach()), "cross_entropy": float(cross_entropy.detach()), "jsd": float(jsd.detach()),
        "correct_clean": int(clean.detach().argmax(1).eq(targets).sum()), "updated": updated,
        "source_examples": batch, "model_views": 3*batch, "forward_calls": 1,
        "scale_before": scale_before, "scale_after": float(scaler.get_scale()), "successful_updates_after": after}
    require(all(math.isfinite(result[k]) for k in ("loss", "cross_entropy", "jsd", "scale_before", "scale_after")) and
            result["loss"] >= 0 and result["cross_entropy"] >= 0 and result["jsd"] >= -1e-5,
            "AugMix step produced invalid loss/scaler evidence")
    return result


def pipeline_generators(seed=2026):
    require(type(seed) is int and seed in (2026, 2027), "Pipeline seed must be exact integer2026 or2027")
    return (torch.Generator(device="cpu").manual_seed(seed+1001), torch.Generator(device="cpu").manual_seed(seed+2001))


def capture_state_rng(rng, device, sampler_rng=None, loader_rng=None, *, seed=2026):
    _generator(rng)
    defaults = pipeline_generators(seed)
    sampler_rng, loader_rng = sampler_rng or defaults[0], loader_rng or defaults[1]
    return {**capture_rng(device), "augmix_rng": copy.deepcopy(rng.bit_generator.state),
            "sampler_rng":sampler_rng.get_state().clone(), "loader_rng":loader_rng.get_state().clone()}


def validate_rng(state, precision):
    require(all(k in state for k in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "augmix_rng", "sampler_rng", "loader_rng")), "Exact AugMix RNG state is incomplete")
    try:
        random.Random().setstate(state["python_rng"])
        np.random.RandomState().set_state(state["numpy_rng"])
        require(isinstance(state["torch_rng"], torch.Tensor) and state["torch_rng"].dtype == torch.uint8, "Torch RNG must be bytes")
        torch.Generator(device="cpu").set_state(state["torch_rng"].cpu())
        for key in ("sampler_rng", "loader_rng"):
            require(isinstance(state[key], torch.Tensor) and state[key].dtype == torch.uint8, "Sampler/loader RNG must be bytes")
            torch.Generator(device="cpu").set_state(state[key].cpu())
        require(state["augmix_rng"].get("bit_generator") == "PCG64", "AugMix RNG must retain PCG64 identity")
        np.random.PCG64().state = copy.deepcopy(state["augmix_rng"])
    except (TypeError, KeyError, RuntimeError) as error:
        raise ValueError("Malformed saved AugMix RNG state") from error
    require(isinstance(state["cuda_rng"], list) and (precision != "fp16" or bool(state["cuda_rng"])) and
            all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8 for v in state["cuda_rng"]), "Native FP16 CUDA RNG state is incomplete")


def restore_state_rng(state, rng, device, sampler_rng=None, loader_rng=None, *, seed=2026):
    _generator(rng)
    validate_rng(state, "fp16" if device.type == "cuda" else "fp32")
    restore_rng(state, device)
    rng.bit_generator.state = copy.deepcopy(state["augmix_rng"])
    defaults = pipeline_generators(seed)
    (sampler_rng or defaults[0]).set_state(state["sampler_rng"].cpu())
    (loader_rng or defaults[1]).set_state(state["loader_rng"].cpu())


def validate_history(history, epochs):
    require(type(epochs) is int and 1 <= epochs <= 160 and len(history) == epochs, "Complete ordered AugMix epoch history required")
    successful = skipped = 0
    for epoch, row in enumerate(history, 1):
        require(type(row.get("epoch")) is int and row["epoch"] == epoch and
                all(type(row.get(k)) is int for k in ("successful_updates", "skipped_updates", "attempted_updates", "source_examples", "model_views", "forward_calls")) and
                row["successful_updates"] >= 1 and row["skipped_updates"] >= 0 and
                row["successful_updates"]+row["skipped_updates"] == row["attempted_updates"] == row["forward_calls"] == 313 and
                row["source_examples"] == 40000 and row["model_views"] == 120000 and row.get("last_source_batch_size") == 64,
                "AugMix history source/view/update/forward counts differ")
        for key in ("train_loss", "cross_entropy", "jsd", "validation_loss", "ema_validation_loss", "epoch_seconds", "elapsed_seconds", "grad_scaler_scale", "lr"):
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= (-1e-5 if key == "jsd" else 0), "AugMix history contains invalid finite metrics")
        require(row["grad_scaler_scale"] > 0 and all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and 0 <= row[k] <= 1
                    for k in ("train_accuracy", "validation_accuracy", "ema_validation_accuracy")), "Invalid AugMix accuracy/scaler")
        successful += row["successful_updates"]
        skipped += row["skipped_updates"]
    return successful, skipped


def validate_scaler(state, precision):
    if precision == "fp32":
        require(state == {}, "CPU FP32 must have a disabled scaler state")
        return
    require(isinstance(state, dict) and set(state) == {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"} and
            type(state["scale"]) in (int, float) and math.isfinite(state["scale"]) and state["scale"] > 0 and
            state["growth_factor"] == 2.0 and state["backoff_factor"] == .5 and state["growth_interval"] == 2000 and
            type(state["_growth_tracker"]) is int and 0 <= state["_growth_tracker"] < 2000, "Native AugMix GradScaler state differs")


def validate_saved_state(last, config, precision):
    reference = build_model({k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")})
    expected = reference.state_dict()
    for kind in ("model", "ema"):
        state = last.get(kind, {})
        require(set(state) == set(expected), "AugMix saved model/EMA keys differ")
        require(all(isinstance(state[k], torch.Tensor) and state[k].shape == v.shape and state[k].dtype == v.dtype and
                    torch.isfinite(state[k]).all() for k, v in expected.items()), "AugMix saved model/EMA shape/dtype/finite values differ")
    parameters = list(reference.parameters())
    optimizer = last.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    require(len(groups) == 1 and groups[0].get("params") == list(range(len(parameters))) and
            groups[0].get("momentum") == .9 and groups[0].get("weight_decay") == .0005 and groups[0].get("nesterov") is True and
            set(optimizer.get("state", {})) == set(range(len(parameters))), "AugMix exact SGD topology/state is incomplete")
    for index, parameter in enumerate(parameters):
        item = optimizer["state"][index]
        momentum = item.get("momentum_buffer")
        require(set(item) == {"momentum_buffer"} and isinstance(momentum, torch.Tensor) and momentum.shape == parameter.shape and
                momentum.dtype == torch.float32 and torch.isfinite(momentum).all(), "AugMix saved SGD momentum is invalid")
    validate_scaler(last.get("grad_scaler"), precision)
    validate_rng(last, precision)


def validate_resume(last, config, signature, provenance):
    validate_config(config)
    previous = last.get("provenance", {})
    require(last.get("schema") == SCHEMA and last.get("config") == config and last.get("data_signature") == signature and
            previous.get("source_sha256") == provenance.get("source_sha256") and
            previous.get("research_training") == provenance.get("research_training"),
            "AugMix resume typed source/config/raw/precision/data identity differs")
    epoch = last.get("epoch")
    require(type(epoch) is int and 1 <= epoch < 160 and last.get("parameter_count") == PARAMETER_COUNT and
            last.get("model_config") == {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}, "AugMix resume architecture or epoch differs")
    updates, skipped = validate_history(last.get("history", []), epoch)
    require(last.get("updates") == updates and last.get("skipped_updates") == skipped and last.get("attempted_updates") == 313*epoch and
            last.get("source_examples_seen") == 40000*epoch and last.get("model_views_seen") == 120000*epoch,
            "AugMix resume total counts differ from its exact epoch boundary")
    validate_saved_state(last, config, provenance["research_training"]["precision"])


def resolve_augmix_precision(requested, device):
    require((device.type == "cpu" and requested == "fp32") or (device.type == "cuda" and requested == "fp16"),
            "AugMix requires explicit CUDA FP16 or CPU FP32; no implicit precision changes")
    return resolve_precision(requested, device)


def train(config, output, data_dir, device, precision="fp16", stop_after_epoch=None, resume=False, input_config_bytes=None):
    validate_config(config)
    precision, dtype = resolve_augmix_precision(precision, device)
    require(stop_after_epoch is None or type(stop_after_epoch) is int and 1 <= stop_after_epoch <= 160, "Invalid stop-after-epoch")
    raw = raw_config(config, input_config_bytes)
    output, data_dir = Path(output), Path(data_dir)
    artifacts = ("last.pt", "best.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json", "input_config.json")
    if not resume and any((output/name).exists() for name in artifacts): raise FileExistsError("AugMix artifacts are immutable; resume explicitly")
    if resume and not (output/"last.pt").is_file(): raise FileNotFoundError("AugMix resume needs last.pt")
    seed_all(config["seed"])
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    sampler_rng, loader_rng = pipeline_generators(config["seed"])
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    manifest = json.loads((data_dir/"training_manifest.json").read_text())
    validate_partition(train_ids, val_ids, manifest, config)
    source = training_source(data_dir)
    signature = {"archive_sha256": source["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(signature == OFFICIAL_SIGNATURE, "AugMix must retain the registered official archive and frozen split")
    marker = research_provenance(precision, config, hashlib.sha256(raw).hexdigest())
    provenance = {**code_provenance(), "research_training": marker}
    val_x, val_y = tensor_images(images[val_ids], device), torch.as_tensor(labels[val_ids], device=device)
    model_config = {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last)
    require(count_parameters(model) == count_parameters(model, True) == PARAMETER_COUNT, "AugMix parameter count differs")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005, nesterov=True)
    scaler = torch.amp.GradScaler(device.type, init_scale=config["amp_init_scale"], enabled=precision == "fp16")
    history, start_epoch, updates, skipped, previous_seconds = [], 1, 0, 0, 0.0
    best_accuracy, best_loss = -1.0, float("inf")
    if resume:
        require((output/"input_config.json").read_bytes() == raw, "Raw AugMix resume configuration changed")
        last = torch.load(output/"last.pt", map_location=device, weights_only=False)
        validate_resume(last, config, signature, provenance)
        model.load_state_dict(last["model"], strict=True)
        ema.load_state_dict(last["ema"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["grad_scaler"])
        history, start_epoch, updates, skipped = last["history"], last["epoch"]+1, last["updates"], last["skipped_updates"]
        best_accuracy, best_loss, previous_seconds = last["best_accuracy"], last["best_loss"], last["elapsed_seconds"]
        restore_state_rng(last, rng, device, sampler_rng, loader_rng)
        del last
    end_epoch = min(160, stop_after_epoch or 160)
    require(end_epoch >= start_epoch, "Stop epoch precedes the resumed state")
    output.mkdir(parents=True, exist_ok=True)
    (output/"input_config.json").write_bytes(raw)
    write_json(output/"config.json", config)
    write_json(output/"environment.json", {**environment(), "schema": SCHEMA, "amp_dtype": str(dtype),
        "device": str(device), "augmix_training_precision": precision, "grad_scaler_enabled": scaler.is_enabled(), "research_training": marker})
    write_json(output/"provenance.json", {"schema": SCHEMA, "code": provenance, "official_data": source, "data_signature": signature,
        "train_examples": 40000, "validation_examples": 5000, "parameter_count": PARAMETER_COUNT, "normalization": {"mean": MEAN, "std": STD}})
    started = time.perf_counter()
    for epoch in range(start_epoch, end_epoch+1):
        began = time.perf_counter()
        model.train()
        sums = {k:0.0 for k in ("loss", "cross_entropy", "jsd")}
        correct = epoch_updates = epoch_skipped = seen = views = forwards = 0
        with build_epoch_loader(images, labels, train_ids, rng, sampler_rng, loader_rng, output=output, tag=f"epoch{epoch:04d}") as pipeline:
            for batch, (cpu_images, cpu_targets) in enumerate(pipeline):
                lr = schedule((epoch-1)*313+batch, 160*313, 313, config)
                for group in optimizer.param_groups: group["lr"] = lr
                prepared = (cpu_images.to(device=device, memory_format=torch.channels_last), cpu_targets.to(device))
                report = train_step(model, optimizer, scaler, prepared, device, dtype, ema=ema, successful_updates=updates)
                updates += int(report["updated"])
                skipped += int(not report["updated"])
                epoch_updates += int(report["updated"])
                epoch_skipped += int(not report["updated"])
                for key in sums: sums[key] += report[key]*len(cpu_targets)
                correct += report["correct_clean"]
                seen += len(cpu_targets)
                views += report["model_views"]
                forwards += report["forward_calls"]
        with torch.no_grad():
            for average, online in zip(ema.buffers(), model.buffers()): average.copy_(online)
        regular, average = evaluate(model, val_x, val_y, device, dtype), evaluate(ema, val_x, val_y, device, dtype)
        choice = "ema" if (average["accuracy"], -average["loss"]) >= (regular["accuracy"], -regular["loss"]) else "model"
        metric = average if choice == "ema" else regular
        elapsed = previous_seconds+time.perf_counter()-started
        record = {"epoch": epoch, "train_loss": sums["loss"]/seen, "cross_entropy": sums["cross_entropy"]/seen, "jsd": sums["jsd"]/seen,
            "train_accuracy": correct/seen, "validation_loss": regular["loss"], "validation_accuracy": regular["accuracy"],
            "ema_validation_loss": average["loss"], "ema_validation_accuracy": average["accuracy"], "selected_variant": choice,
            "lr": lr, "epoch_seconds": time.perf_counter()-began, "elapsed_seconds": elapsed, "successful_updates": epoch_updates,
            "skipped_updates": epoch_skipped, "attempted_updates": 313, "source_examples": seen, "model_views": views,
            "forward_calls": forwards, "last_source_batch_size": len(cpu_targets), "grad_scaler_scale": float(scaler.get_scale()),
            "worker_pool_receipt_sha256":sha256(output/f"worker_pool_epoch{epoch:04d}.json")}
        history.append(record)
        validate_history(history, epoch)
        if (metric["accuracy"], -metric["loss"]) > (best_accuracy, -best_loss):
            best_accuracy, best_loss = metric["accuracy"], metric["loss"]
            save_checkpoint(output/"best.pt", {"schema": SCHEMA, "model": (ema if choice == "ema" else model).state_dict(),
                "model_config": model_config, "config": config, "epoch": epoch, "variant": choice,
                "validation_accuracy": best_accuracy, "validation_loss": best_loss, "parameter_count": PARAMETER_COUNT,
                "data_signature": signature, "provenance": provenance})
        write_json(output/"history.json", history)
        write_json(output/"metrics.json", {"schema": SCHEMA, "status": "complete" if epoch == 160 else "running",
            "epochs_completed": epoch, "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
            "parameter_count": PARAMETER_COUNT, "elapsed_seconds": elapsed, "training_examples": 40000, "validation_examples": 5000,
            "config": config, "research_training": marker, "successful_updates": updates, "skipped_updates": skipped,
            "attempted_updates": epoch*313, "source_examples_seen": epoch*40000, "model_views_seen": epoch*120000,
            "forward_calls": epoch*313})
        save_checkpoint(output/"last.pt", {"schema": SCHEMA, "model": model.state_dict(), "ema": ema.state_dict(),
            "model_config": model_config, "parameter_count": PARAMETER_COUNT, "optimizer": optimizer.state_dict(), "grad_scaler": scaler.state_dict(),
            "epoch": epoch, "updates": updates, "skipped_updates": skipped, "attempted_updates": epoch*313,
            "source_examples_seen": epoch*40000, "model_views_seen": epoch*120000, "config": config, "data_signature": signature,
            "provenance": provenance, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss,
            "elapsed_seconds": elapsed, **capture_state_rng(rng, device, sampler_rng, loader_rng)})
        print(json.dumps(record), flush=True)


def current_sources():
    return {**code_provenance()["source_sha256"], **{key:sha256(ROOT/key) for key in SOURCE_KEYS}}


def _state_equal(left, right):
    if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict): return isinstance(right, dict) and set(left) == set(right) and all(_state_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)): return isinstance(right, type(left)) and len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def overflow_smoke(model, optimizer, scaler, prepared, device, dtype, ema):
    """Finite loss/infinite gradient must skip SGD and EMA; separate diagnostic."""
    require(scaler.is_enabled(), "Overflow smoke requires an enabled native GradScaler")
    before = [p.detach().clone() for p in model.parameters()]
    ema_before = [p.detach().clone() for p in ema.parameters()]
    state_before = copy.deepcopy(optimizer.state_dict())
    probe = next(model.parameters())
    handle = probe.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        report = train_step(model, optimizer, scaler, prepared, device, dtype, ema=ema, successful_updates=23)
    finally:
        handle.remove()
    result = {"status": "passed", "finite_loss": math.isfinite(report["loss"]), "infinite_gradient_injected": True,
        "updated": report["updated"], "scale_before": report["scale_before"], "scale_after": report["scale_after"],
        "all_parameters_unchanged": all(torch.equal(p, old) for p, old in zip(model.parameters(), before)),
        "optimizer_state_unchanged": _state_equal(optimizer.state_dict(), state_before),
        "ema_parameters_unchanged": all(torch.equal(p, old) for p, old in zip(ema.parameters(), ema_before)),
        "source_examples": report["source_examples"], "model_views": report["model_views"], "forward_calls": 1}
    require(result["finite_loss"] and result["updated"] is False and result["scale_after"] < result["scale_before"] and
            result["all_parameters_unchanged"] and result["optimizer_state_unchanged"] and result["ema_parameters_unchanged"],
            "Native AugMix overflow skip did not preserve all parameters/SGD/EMA and lower the scaler")
    return result


def preflight(config, output, data_dir, device, precision="fp16", input_config_bytes=None):
    """Measure the actual official train-image CPU/PIL/transfer/3B/SGD pipeline."""
    validate_config(config)
    precision, dtype = resolve_augmix_precision(precision, device)
    require(device.type == "cuda" and precision == "fp16", "Admission preflight must run actual native CUDA FP16")
    raw = raw_config(config, input_config_bytes)
    path, data_dir = Path(output)/"engineering_benchmark.json", Path(data_dir)
    if path.exists(): raise FileExistsError("Existing AugMix preflight is immutable")
    source_hashes = current_sources()
    seed_all(config["seed"])
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    sampler_rng, loader_rng = pipeline_generators(config["seed"])
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    manifest = json.loads((data_dir/"training_manifest.json").read_text())
    validate_partition(train_ids, val_ids, manifest, config)
    signature = {"archive_sha256": training_source(data_dir)["archive_sha256"], "split_sha256": manifest["split"]["indices_sha256"]}
    require(signature == OFFICIAL_SIGNATURE, "Preflight must use the registered official archive and frozen split")
    model_config = {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}
    model = build_model(model_config).to(device=device, memory_format=torch.channels_last).train()
    require(count_parameters(model) == count_parameters(model, True) == PARAMETER_COUNT, "Preflight architecture exceeds or changes fixed parameters")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005, nesterov=True)
    scaler = torch.amp.GradScaler(device.type, init_scale=config["amp_init_scale"], enabled=True)
    pipeline = build_epoch_loader(images, labels, train_ids, rng, sampler_rng, loader_rng,
                                 limit_batches=23, output=output, tag="preflight")
    selected = pipeline.source_ids
    require(len(np.unique(selected)) == 23*128 and np.isin(selected, train_ids).all(), "Only unique official train IDs may enter the preflight")
    durations, reports, changed = [], [], 0
    torch.cuda.reset_peak_memory_stats(device)
    probe = next(model.parameters())
    with pipeline:
        for index in range(23):
            torch.cuda.synchronize(device)
            before = probe.detach().clone()
            began = time.perf_counter()
            cpu_images, cpu_targets = next(pipeline)
            prepared = (cpu_images.to(device=device, memory_format=torch.channels_last), cpu_targets.to(device))
            report = train_step(model, optimizer, scaler, prepared, device, dtype, ema=ema, successful_updates=index)
            require(report["updated"] is True and report["forward_calls"] == 1 and report["source_examples"] == 128 and
                    report["model_views"] == 384, "Every normal preflight step must make one actual single3B update")
            changed += int(not torch.equal(probe.detach(), before))
            maximum = float(torch.stack(torch._foreach_norm([p.detach() for p in model.parameters()], float("inf"))).amax())
            require(math.isfinite(maximum), "Nonfinite preflight master parameter")
            # Include proof work rather than granting the CPU producers an
            # unmeasured gap that would make queue-wait timing optimistic.
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-began
            reports.append(report)
            if index >= 3: durations.append(elapsed)
    smoke = overflow_smoke(model, optimizer, scaler, prepared, device, dtype, ema)
    median = float(np.median(durations))
    require(changed == 23 and current_sources() == source_hashes, "Real updates or immutable preflight source changed")
    marker = research_provenance(precision, config, hashlib.sha256(raw).hexdigest())
    result = {"schema": PREFLIGHT_SCHEMA, "status": "complete", "synthetic_only": False, "official_training_only": True,
        "precision": precision, "grad_scaler_enabled": True, "initial_grad_scaler_scale":config["amp_init_scale"],
        "parameters": PARAMETER_COUNT, "model_config": model_config,
        "num_workers":2, "prefetch_factor":2, "multiprocessing_context":"spawn", "persistent_workers":False,
        "worker_cuda_initialized":False, "all_workers_stopped":pipeline.closed,
        "worker_pool_receipt_sha256":sha256(Path(output)/"worker_pool_preflight.json"),
        "config": config, "input_config_sha256": hashlib.sha256(raw).hexdigest(), "batch_size": 128, "model_batch_size": 384,
        "warmup_steps": 3, "timed_steps": 20, "attempted_updates": 23, "successful_updates": 23, "skipped_updates": 0,
        "parameter_changed_steps": changed, "forward_calls": 23, "single_forward_per_step": True,
        "source_examples_seen": 2944, "model_views_seen": 8832, "training_examples": 40000, "validation_examples": 5000,
        "validation_evaluated": False, "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
        "training_data_loaded": True, "data_signature": signature, "source_ids_are_training_only": True,
        "source_ids_unique_count": 2944, "preflight_source_ids_sha256": hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
        "median_step_seconds": median, "p90_step_seconds": float(np.quantile(durations, .9)), "images_per_second": 128/median,
        "required_training_and_benchmark_seconds": 1.5*median*313*160+300,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device), "overflow_smoke": smoke,
        "mean_loss": float(np.mean([r["loss"] for r in reports])), "mean_cross_entropy": float(np.mean([r["cross_entropy"] for r in reports])),
        "mean_jsd": float(np.mean([r["jsd"] for r in reports])), "research_training": marker, "source_sha256": source_hashes,
        "environment": {**environment(), "device": str(device), "amp_dtype": str(dtype), "grad_scaler_enabled": True},
        "formal_initialization": f"discard_preflight_then_reseed{config['seed']}_scratch_model_SGD_scaler_PCG64",
        "interpretation": "Measured actual official-train image CPU/PIL, transfer, single3B forward, full loss, backward, SGD and EMA. Explicit init256 differs from old shadow formal default65536. No accuracy or qualification evidence."}
    validate_preflight(result, config, hashlib.sha256(raw).hexdigest())
    write_json(path, result)
    print(json.dumps(result), flush=True)
    return result


def validate_preflight(report, config, input_config_sha256, expected_source_sha256=None, require_t4=False):
    validate_config(config)
    marker = research_provenance("fp16", config, input_config_sha256)
    require(report.get("schema") == PREFLIGHT_SCHEMA and report.get("status") == "complete" and
            report.get("synthetic_only") is False and report.get("official_training_only") is True and report.get("training_data_loaded") is True and
            report.get("precision") == "fp16" and report.get("grad_scaler_enabled") is True and report.get("initial_grad_scaler_scale") == 256.0 and
            report.get("parameters") == PARAMETER_COUNT and report.get("model_config") == {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")} and
            report.get("config") == config and report.get("input_config_sha256") == input_config_sha256 and report.get("research_training") == marker,
            "AugMix preflight source/config/raw/native FP16 architecture identity differs")
    require(report.get("source_sha256") == (expected_source_sha256 or current_sources()), "AugMix preflight source binding differs")
    require(report.get("num_workers") == 2 and report.get("prefetch_factor") == 2 and report.get("multiprocessing_context") == "spawn" and
            report.get("persistent_workers") is False and report.get("worker_cuda_initialized") is False and report.get("all_workers_stopped") is True,
            "AugMix preflight must use the fixed CPU-only two-worker pipeline and complete teardown")
    counts = {"batch_size":128, "model_batch_size":384, "warmup_steps":3, "timed_steps":20, "attempted_updates":23,
              "successful_updates":23, "skipped_updates":0, "parameter_changed_steps":23, "forward_calls":23,
              "source_examples_seen":2944, "model_views_seen":8832, "training_examples":40000, "validation_examples":5000,
              "source_ids_unique_count":2944}
    require(all(type(report.get(k)) is int and report[k] == value for k, value in counts.items()) and
            report.get("single_forward_per_step") is True and report.get("source_ids_are_training_only") is True,
            "AugMix preflight must prove all23 normal sourceB128/single3B actual updates")
    require(all(report.get(k) is False for k in ("validation_evaluated", "confirmation_evaluated", "hidden_images_loaded", "external_data_loaded")),
            "AugMix preflight may select only official training IDs")
    require(isinstance(report.get("preflight_source_ids_sha256"), str) and len(report["preflight_source_ids_sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in report["preflight_source_ids_sha256"]), "Actual preflight train ID fingerprint required")
    signature = report.get("data_signature", {})
    require(signature == OFFICIAL_SIGNATURE, "Registered official preflight archive/frozen split signature required")
    smoke = report.get("overflow_smoke", {})
    require(smoke.get("status") == "passed" and smoke.get("updated") is False and all(smoke.get(k) is True for k in
            ("finite_loss", "infinite_gradient_injected", "all_parameters_unchanged", "optimizer_state_unchanged", "ema_parameters_unchanged")) and
            smoke.get("source_examples") == 128 and smoke.get("model_views") == 384 and smoke.get("forward_calls") == 1 and
            all(type(smoke.get(k)) in (int, float) and math.isfinite(smoke[k]) and smoke[k] > 0 for k in ("scale_before", "scale_after")) and
            smoke["scale_after"] < smoke["scale_before"], "Separate native finite-loss/overflow skip proof required")
    median, p90 = report.get("median_step_seconds"), report.get("p90_step_seconds")
    require(type(median) in (int,float) and math.isfinite(median) and median > 0 and type(p90) in (int,float) and math.isfinite(p90) and p90 >= median and
            type(report.get("required_training_and_benchmark_seconds")) in (int,float) and
            math.isclose(report["required_training_and_benchmark_seconds"], 1.5*median*313*160+300, rel_tol=0, abs_tol=1e-9),
            "AugMix actual pipeline speed or fixed guard is invalid")
    require(all(type(report.get(k)) in (int,float) and math.isfinite(report[k]) and report[k] >= (-1e-5 if k == "mean_jsd" else 0)
                for k in ("mean_loss", "mean_cross_entropy", "mean_jsd")), "AugMix preflight losses must remain finite")
    env = report.get("environment", {})
    require(env.get("device") in ("cuda", "cuda:0") and env.get("amp_dtype") == "torch.float16" and bool(env.get("gpu")) and
            env.get("packages", {}).get("torch") and (not require_t4 or env["gpu"] == "Tesla T4"), "Actual native FP16 hardware/Torch preflight required")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Alias for actual-image preflight; must be20")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = {**DEFAULT_CONFIG, **json.loads(raw)}
    validate_config(config)
    if args.preflight or args.benchmark_steps:
        require(args.benchmark_steps in (0,20) and not args.resume and args.stop_after_epoch is None, "Preflight has fixed20timedsteps and cannot resume training")
        preflight(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision, raw)
    else:
        train(config, args.output_dir, args.data_dir, torch.device(args.device), args.precision, args.stop_after_epoch, args.resume, raw)


if __name__ == "__main__": main()
