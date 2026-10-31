"""FP32 equal-parameter SWA and isolated training-only BatchNorm recomputation.

Algorithm reference: https://arxiv.org/abs/1803.05407 and
https://github.com/timgaripov/swa. This independent helper loads no datasets,
checkpoints, labels, confirmation examples, hidden images or optimizer state.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401  # Windows DLL isolation before torch.
import numpy as np
import torch
from train import augment

BN_SCHEMA = "research_swa_bn_recompute_v1"


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _all_finite(values):
    # For FP32 tensors, max(abs(x)) is finite iff every element is finite.
    # Infinity norm cannot overflow like a summed norm. Aggregate on-device so
    # each device needs one host decision, rather than one per parameter.
    # Empty tensors are vacuously finite; their infinity norm is undefined.
    groups = {}
    for value in values:
        if value.numel():
            groups.setdefault(value.device, []).append(value)
    return all(torch.isfinite(torch.stack(torch._foreach_norm(group, float("inf")))).all().item()
               for group in groups.values())


def _parameters(model):
    values = dict(model.named_parameters())
    for name, value in values.items():
        if value.dtype != torch.float32:
            raise ValueError(f"SWA requires finite FP32 parameters: {name}")
    if not _all_finite(values.values()):
        raise ValueError("SWA requires finite FP32 parameters")
    return values


@torch.no_grad()
def average_parameters_(averaged_model, sampled_model, count):
    """Add one equally weighted parameter sample; return the new sample count.

    Every named parameter, including BN affine weight/bias, is averaged. All
    buffers (running means/variances and integer counters) remain untouched.
    Models must be independent storage copies with matching names/shapes/device
    and finite FP32 parameters. No gradient or optimizer state is modified.
    """
    if type(count) is not int or count < 0:
        raise ValueError("SWA sample count must be a nonnegative integer")
    target, source = _parameters(averaged_model), _parameters(sampled_model)
    if not target or target.keys() != source.keys():
        raise ValueError("SWA named parameter sets must match and be nonempty")
    for name in target:
        if target[name].shape != source[name].shape or target[name].device != source[name].device:
            raise ValueError(f"SWA parameter shape/device differs: {name}")
    target_storage = {value.untyped_storage().data_ptr() for value in target.values() if value.numel()}
    if target_storage & {value.untyped_storage().data_ptr() for value in source.values() if value.numel()}:
        raise ValueError("SWA average and sampled parameters must not alias storage")
    names = list(target)
    if count == 0:
        updated = [source[name].detach().clone(memory_format=torch.preserve_format) for name in names]
    else:
        # A convex combination avoids overflow in source-target for large but
        # finite FP32 inputs, while retaining equal weights across samples.
        weight = 1.0/(count+1)
        updated = [target[name].detach()*(1.0-weight)+source[name].detach()*weight for name in names]
    if not _all_finite(updated):
        raise FloatingPointError("SWA parameter average became nonfinite; no parameters were changed")
    for name, value in zip(names, updated):
        target[name].copy_(value)
    return count+1


@contextmanager
def preserved_seed(seed, device):
    """Use a reproducible RNG stream, then restore caller RNGs on every exit.

    Python, NumPy and torch CPU states are preserved. On CUDA, only the selected
    device is seeded/used/restored; other CUDA devices are never touched. CPU
    execution never initializes CUDA. Model dropout is disabled by BN helper.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("BN recomputation seed must be an integer in [0, 2**32)")
    device = torch.device(device)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _precision(device, precision):
    if device.type == "cpu" and precision == "fp32":
        return torch.float32
    if (device.type == "cuda" and precision == "bf16" and torch.cuda.is_available() and
            torch.cuda.is_bf16_supported(including_emulation=False)):
        return torch.bfloat16
    raise ValueError("SWA BN recomputation supports CPU FP32 or native CUDA BF16 only")


@torch.no_grad()
def recompute_training_bn_(model, train_images, *, batch_size=128, seed=2026, precision="fp32", device=None):
    """Recompute only tracked BN statistics on exactly 40,000 training images.

    Caller must supply ONLY the prefiltered official train_ids in fixed order as
    uint8 NHWC32x32x3 tensors. The helper never obtains indices or labels.
    Frozen augment(...,cutout=0) supplies crop/flip/normalization, with no Mixup
    or Cutout. Every BN momentum is batch_examples/(seen+batch_examples).

    All non-BN modules run in eval mode, and no gradients are created. Original
    training modes/momenta and RNGs are restored even on error. Parameters are
    checked unchanged; failed recomputation restores the original BN buffers.
    Use the same seed/order for SWA and raw controls, and compare returned
    augmented_input_sha256 to verify that both consumed identical augmentations.
    """
    if (not isinstance(train_images, torch.Tensor) or train_images.dtype != torch.uint8 or
            tuple(train_images.shape) != (40000, 32, 32, 3)):
        raise ValueError("SWA BN recomputation requires exactly 40,000 uint8 NHWC32x32x3 training images")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("BN recomputation batch size must be a positive integer")
    parameters = _parameters(model)
    buffers = list(model.buffers())
    inferred = next(iter(parameters.values())).device if parameters else buffers[0].device if buffers else torch.device("cpu")
    device = torch.device(device) if device is not None else inferred
    if device.type == "cuda" and device.index is None and inferred.type == "cuda":
        device = inferred
    elif device.type == "cpu":
        device = torch.device("cpu")
    if any(value.device != device for value in [*parameters.values(), *buffers]):
        raise ValueError("SWA BN model parameters/buffers must match the requested device")
    dtype = _precision(device, precision)
    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not batchnorms or any(not module.track_running_stats or module.running_mean is None or
                            module.running_var is None or module.num_batches_tracked is None for module in batchnorms):
        raise ValueError("SWA BN recomputation requires tracked BatchNorm statistics")
    saved_parameters = {name: value.detach().clone() for name, value in parameters.items()}
    saved_modes = [(module, module.training) for module in model.modules()]
    saved_momenta = [module.momentum for module in batchnorms]
    saved_buffers = [(module, {name: getattr(module, name).detach().clone() for name in
                              ("running_mean", "running_var", "num_batches_tracked")}) for module in batchnorms]
    digest = hashlib.sha256()
    successful = False
    seen, batches, last_batch = 0, 0, 0
    try:
        model.eval()
        for module in batchnorms:
            module.reset_running_stats()
            module.training = True
        with preserved_seed(seed, device):
            for start in range(0, len(train_images), batch_size):
                batch = train_images[start:start+batch_size].to(device=device)
                for module in batchnorms:
                    module.momentum = len(batch)/(seen+len(batch))
                inputs = augment(batch, cutout=0)
                digest.update(inputs.detach().cpu().numpy().tobytes())
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    output = model(inputs)
                if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
                    raise FloatingPointError("SWA BN recomputation produced nonfinite outputs")
                seen += len(batch)
                batches += 1
                last_batch = len(batch)
        if any(not torch.equal(value, saved_parameters[name]) for name, value in parameters.items()):
            raise RuntimeError("SWA BN recomputation changed model parameters")
        if any(not torch.isfinite(module.running_mean).all() or not torch.isfinite(module.running_var).all()
               for module in batchnorms):
            raise FloatingPointError("SWA BN running statistics became nonfinite")
        successful = True
    finally:
        if not successful:
            for name, value in parameters.items():
                value.copy_(saved_parameters[name])
            for module, record in saved_buffers:
                for name, value in record.items():
                    getattr(module, name).copy_(value)
        for module, mode in saved_modes:
            module.training = mode
        for module, momentum in zip(batchnorms, saved_momenta):
            module.momentum = momentum
    return {"schema": BN_SCHEMA, "training_examples": seen, "batch_size": batch_size, "batches": batches,
            "last_batch_examples": last_batch, "seed": seed, "precision": precision, "device_type": device.type,
            "shuffle": False, "augmentation": "frozen_crop_flip_cutout0", "mixup": False, "cutout": 0,
            "momentum": "sample_weighted_batch_size_over_seen", "grad_enabled": False,
            "parameters_unchanged": True, "training_modes_restored": True, "momenta_restored": True,
            "rng_restored": True, "parameter_count": sum(value.numel() for value in parameters.values()),
            "augmented_input_sha256": digest.hexdigest(), "helper_source_sha256": source_sha256()}
