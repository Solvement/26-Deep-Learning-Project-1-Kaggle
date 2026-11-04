"""Independent, CPU/PIL AugMix transforms and differentiable three-view JSD.

Algorithm: Hendrycks et al., https://arxiv.org/abs/1912.02781
Mechanics checked against google-research/augmix commit
9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4, cifar.py and augmentations.py.
No author code, dataset loader, trainer or checkpoint is imported/executed.
Inputs are already crop/flipped RGB32 training images. This module does not
perform crop/flip, allocate a GPU, train, evaluate or predict hidden images.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

AUTHOR_COMMIT = "9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4"
OPERATIONS = ("autocontrast", "equalize", "posterize", "rotate", "solarize",
              "shear_x", "shear_y", "translate_x", "translate_y")
MEAN = (.4914, .4822, .4465)
STD = (.2470, .2435, .2616)
JSD_WEIGHT = 12.0
_ARRAY_MEAN = np.asarray(MEAN, dtype=np.float32)
_ARRAY_STD = np.asarray(STD, dtype=np.float32)
_ARRAY_LOW = -_ARRAY_MEAN/_ARRAY_STD
_ARRAY_HIGH = (np.float32(1)-_ARRAY_MEAN)/_ARRAY_STD
for _constant in (_ARRAY_MEAN, _ARRAY_STD, _ARRAY_LOW, _ARRAY_HIGH):
    _constant.flags.writeable = False


@dataclass(frozen=True)
class AugMixConfig:
    width: int = 3
    depth: int = -1
    severity: int = 3
    alpha: float = 1.0
    all_ops: bool = False

    def validate(self):
        if (any(type(v) is not int for v in (self.width, self.depth, self.severity)) or
                (self.width, self.depth, self.severity) != (3, -1, 3) or
                type(self.alpha) not in (float, int) or self.alpha != 1.0 or
                self.all_ops is not False):
            raise ValueError("Only the registered width3/depth1-3/severity3/alpha1/default-nine AugMix is supported")


DEFAULT_CONFIG = AugMixConfig()


@dataclass(frozen=True)
class Operation:
    name: str
    value: int | float | None = None


@dataclass(frozen=True)
class AugMixTrace:
    weights: tuple[float, ...]
    mixing: float
    chains: tuple[tuple[Operation, ...], ...]


def _rng(rng):
    if not isinstance(rng, np.random.Generator):
        raise TypeError("An explicit, checkpointable numpy.random.Generator is required")
    return rng


def rgb_image(image):
    """Copy a strict uint8 NHWC array or RGB32 PIL image; never convert modes."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.shape != (32, 32, 3):
            raise ValueError("AugMix arrays must be uint8 RGB32 NHWC")
        return Image.fromarray(np.array(image, copy=True))
    if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("AugMix requires an already preprocessed RGB32 PIL image")
    return image.copy()


def normalize_image(image):
    """Return CPU float32 CHW using the unchanged campaign normalization."""
    return _chw_tensor(_normalized_array(rgb_image(image)))


def _normalized_array(image):
    """Internal float32 HWC math, avoiding tiny repeated Torch dispatches."""
    if image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("Every generated AugMix branch must stay RGB32")
    values = np.asarray(image, dtype=np.uint8).astype(np.float32)/np.float32(255)
    return (values-_ARRAY_MEAN)/_ARRAY_STD


def _chw_tensor(values):
    if (values.shape != (32, 32, 3) or values.dtype != np.float32 or not np.isfinite(values).all() or
            np.any(values < _ARRAY_LOW-np.float32(2e-6)) or np.any(values > _ARRAY_HIGH+np.float32(2e-6))):
        raise ValueError("Generated normalized AugMix pixels are not finite RGB32 values")
    return torch.from_numpy(values.transpose(2, 0, 1).copy())


def sample_operation(name, rng, config=DEFAULT_CONFIG):
    config.validate()
    _rng(rng)
    if name not in OPERATIONS: raise ValueError("Unknown or disabled AugMix operation")
    if name in ("autocontrast", "equalize"): return Operation(name)
    level = float(rng.uniform(.1, config.severity))
    if name == "posterize": return Operation(name, 4-int(level*4/10))
    if name == "solarize": return Operation(name, 256-int(level*256/10))
    if name == "rotate": value = int(level*30/10)
    elif name.startswith("shear_"): value = level*.3/10
    else: value = int(level*(32/3)/10)
    if float(rng.random()) > .5: value = -value
    return Operation(name, value)


def apply_operation(image, operation):
    """Apply a sampled operator with PIL bilinear geometry and black fill."""
    pil = rgb_image(image)
    if not isinstance(operation, Operation) or operation.name not in OPERATIONS:
        raise ValueError("A registered Operation is required")
    name, value = operation.name, operation.value
    if name in ("autocontrast", "equalize"):
        if value is not None: raise ValueError("Histogram operations have no magnitude")
        return (ImageOps.autocontrast if name == "autocontrast" else ImageOps.equalize)(pil)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Operation magnitude must be finite")
    if name == "posterize":
        if type(value) is not int or value not in (3, 4): raise ValueError("Severity3 posterize retains three or four bits")
        return ImageOps.posterize(pil, value)
    if name == "solarize":
        if type(value) is not int or not 180 <= value <= 254: raise ValueError("Severity3 solarize threshold is180..254")
        return ImageOps.solarize(pil, value)
    if name == "rotate":
        if type(value) is not int or abs(value) > 8: raise ValueError("Severity3 rotation is truncated to -8..8 degrees")
        return pil.rotate(value, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    if name.startswith("shear_"):
        if abs(value) > .09: raise ValueError("Severity3 shear magnitude exceeds .09")
        coefficients = (1, value, 0, 0, 1, 0) if name == "shear_x" else (1, 0, 0, value, 1, 0)
    else:
        if type(value) is not int or abs(value) > 3: raise ValueError("Severity3 translation is truncated to -3..3 pixels")
        coefficients = (1, 0, value, 0, 1, 0) if name == "translate_x" else (1, 0, 0, 0, 1, value)
    return pil.transform((32, 32), Image.Transform.AFFINE, coefficients,
                         resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def _normalized_view(value):
    if (not isinstance(value, torch.Tensor) or value.shape != (3, 32, 32) or
            value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all()):
        raise ValueError("Each normalized view must be finite CPU float32 CHW RGB32")
    low = -value.new_tensor(MEAN)[:, None, None]/value.new_tensor(STD)[:, None, None]
    high = (1-value.new_tensor(MEAN)[:, None, None])/value.new_tensor(STD)[:, None, None]
    if torch.any(value < low-2e-6) or torch.any(value > high+2e-6):
        raise ValueError("Normalized view lies outside the RGB pixel range")


def mix_normalized(clean, chains, weights, mixing):
    """Convex float32 mixture; normalized values are never quantized to uint8."""
    _normalized_view(clean)
    if not isinstance(chains, (tuple, list)) or len(chains) != 3:
        raise ValueError("Exactly three transformed chains are required")
    for chain in chains: _normalized_view(chain)
    w = np.asarray(weights)
    if (w.shape != (3,) or w.dtype.kind not in "fi" or not np.isfinite(w).all() or
            np.any(w < 0) or not np.isclose(w.sum(dtype=np.float64), 1.0, rtol=0, atol=2e-7) or
            type(mixing) not in (float, int, np.float32, np.float64) or not math.isfinite(mixing) or not 0 <= mixing <= 1):
        raise ValueError("AugMix requires finite convex weights and mixing in[0,1]")
    result = torch.zeros_like(clean)
    for weight, chain in zip(w.astype(np.float32), chains): result = result+float(weight)*chain
    result = (1-float(mixing))*clean+float(mixing)*result
    _normalized_view(result)
    return result


def augmix_view(image, rng, config=DEFAULT_CONFIG, return_trace=False):
    """Sample one AugMix view from one already crop/flipped base image."""
    config.validate()
    _rng(rng)
    base = rgb_image(image)
    weights = rng.dirichlet(np.ones(3)).astype(np.float32)
    mixing = np.float32(rng.beta(1, 1))
    transformed, traces = [], []
    for _ in range(3):
        chain = base.copy()
        sampled = []
        for _ in range(int(rng.integers(1, 4))):
            name = OPERATIONS[int(rng.integers(len(OPERATIONS)))]
            operation = sample_operation(name, rng, config)
            sampled.append(operation)
            chain = apply_operation(chain, operation)
        transformed.append(_normalized_array(chain))
        traces.append(tuple(sampled))
    # Retain the reference's separate float32 multiply/add order, including
    # rounding the Python clean coefficient to the Torch scalar's float32.
    clean = _normalized_array(base)
    mixed = np.zeros_like(clean)
    for weight, branch in zip(weights, transformed):
        mixed = mixed+weight*branch
    result = _chw_tensor(np.float32(1-float(mixing))*clean+mixing*mixed)
    trace = AugMixTrace(tuple(float(w) for w in weights), float(mixing), tuple(traces))
    return (result, trace) if return_trace else result


def augmix_views(image, rng, config=DEFAULT_CONFIG, return_trace=False):
    """Clean plus two independent child streams; parent state alone replays all."""
    config.validate()
    _rng(rng)
    base = rgb_image(image)
    seeds = rng.integers(0, 2**64, size=2, dtype=np.uint64)
    first, first_trace = augmix_view(base, np.random.default_rng(int(seeds[0])), config, True)
    second, second_trace = augmix_view(base, np.random.default_rng(int(seeds[1])), config, True)
    views = (normalize_image(base), first, second)
    trace = {"child_seeds": tuple(int(seed) for seed in seeds), "view_traces": (first_trace, second_trace)}
    return (views, trace) if return_trace else views


def augmix_no_jsd(image, rng, config=DEFAULT_CONFIG):
    """Separate single-view ablation; never an implicit three-view fallback."""
    return augmix_view(image, rng, config)


def _logits(*values):
    first = values[0]
    if (not isinstance(first, torch.Tensor) or first.ndim != 2 or first.shape[0] < 1 or first.shape[1] < 2 or
            not first.is_floating_point() or first.device.type not in ("cpu", "cuda")):
        raise ValueError("Logits must be a nonempty floating N x K tensor with K>=2")
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.shape != first.shape or value.dtype != first.dtype or
                value.device != first.device or not torch.isfinite(value).all()):
            raise ValueError("All logits require identical shape/device/dtype and finite values")


def _targets(logits, targets):
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64 or targets.shape != (logits.shape[0],) or
            targets.device != logits.device or torch.any(targets < 0) or torch.any(targets >= logits.shape[1])):
        raise ValueError("Targets must be matching-device int64 class IDs for the source batch")


def jsd_three_view(clean_logits, first_logits, second_logits):
    """Mean KL(p_view || clamp(mean_views,1e-7,1)); FP32, no detach."""
    _logits(clean_logits, first_logits, second_logits)
    with torch.autocast(clean_logits.device.type, enabled=False):
        # log_target=True is algebraically the author KL direction/reduction,
        # while avoiding 0*log(0) target gradients for saturated softmax.
        log_probabilities = [F.log_softmax(value.float(), dim=1) for value in (clean_logits, first_logits, second_logits)]
        if any(not torch.isfinite(value).all() for value in log_probabilities):
            raise FloatingPointError("Logits exceed the finite FP32 log-probability range")
        mixture = torch.stack([value.exp() for value in log_probabilities]).mean(dim=0).clamp(1e-7, 1)
        result = torch.stack([F.kl_div(mixture.log(), value, reduction="batchmean", log_target=True)
                              for value in log_probabilities]).mean()
    if not torch.isfinite(result): raise FloatingPointError("Nonfinite three-view JSD")
    return result


def augmix_loss(clean_logits, first_logits, second_logits, targets):
    """Clean-only mean CE plus exactly12 times three-distribution JSD."""
    _logits(clean_logits, first_logits, second_logits)
    _targets(clean_logits, targets)
    with torch.autocast(clean_logits.device.type, enabled=False):
        result = F.cross_entropy(clean_logits.float(), targets)+JSD_WEIGHT*jsd_three_view(clean_logits, first_logits, second_logits)
    if not torch.isfinite(result): raise FloatingPointError("Nonfinite AugMix objective")
    return result


def no_jsd_loss(single_view_logits, targets):
    """Single AugMix-view CE only: the separately registered no-JSD ablation."""
    _logits(single_view_logits)
    _targets(single_view_logits, targets)
    with torch.autocast(single_view_logits.device.type, enabled=False):
        result = F.cross_entropy(single_view_logits.float(), targets)
    if not torch.isfinite(result): raise FloatingPointError("Nonfinite no-JSD objective")
    return result
