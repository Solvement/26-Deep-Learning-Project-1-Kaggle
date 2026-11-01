"""Independent ResizeMix adaptation: whole donor, uniform side ratio, full fit.

Primary paper: https://arxiv.org/abs/2012.11101, section4.
Author algorithm references (read, never imported):
https://github.com/JieQin-AI/ResizeMix/blob/main/run_apis/trainer.py
https://github.com/JieQin-AI/ResizeMix/blob/main/tools/utils.py

The author samples one side ratio per batch and independent centers per image,
then shifts complete resized patches into bounds; edges do not crop donors.
This controlled experiment uses bilinear interpolation (align_corners=False)
and probability0.5. The author code instead defaults to nearest interpolation
and enables ResizeMix on every non-CutMix batch. No author files, external data,
pretrained weights, or external augmentation packages are used here.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401  # Windows DLL isolation before torch.
import numpy as np
import torch
from torch.nn import functional as F


def validate_resizemix(tau_min=.1, tau_max=.8, probability=.5):
    for name, value in (("tau_min", tau_min), ("tau_max", tau_max), ("probability", probability)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"ResizeMix {name} must be a finite number")
    if not 0 < tau_min <= tau_max <= 1:
        raise ValueError("ResizeMix requires 0 < tau_min <= tau_max <= 1 for SIDE ratios")
    if not 0 <= probability <= 1:
        raise ValueError("ResizeMix probability must be within [0, 1]")


def resizemix(images, tau_min=.1, tau_max=.8, probability=.5):
    """Return (new_images, donor_indices_or_None, retained_weight, boxes_or_None).

    Input is floating NCHW, already cropped/flipped and normalized. A shared
    ``tau ~ Uniform(tau_min,tau_max)`` scales BOTH side lengths, not patch area.
    Integer patch sizes follow floor(H*tau), floor(W*tau). Every whole donor
    image is resized to that size with bilinear sampling before pasting.

    ``boxes`` is one (top,bottom,left,right) Python tuple per image. Independently
    sampled centers are shifted into the valid full-patch range, matching the
    author's getbox_new semantics. All boxes therefore have equal actual area;
    returned retained label weight is 1 - patch_height*patch_width/(H*W).

    Labels are untouched; mix target labels with labels[donor_indices]. Python,
    NumPy and torch RNG streams are sufficient for checkpoint/replay. When
    probability=0, return the original input without consuming any RNG state.
    """
    validate_resizemix(tau_min, tau_max, probability)
    if probability == 0 or random.random() >= probability:
        return images, None, 1.0, None
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or
            not images.is_floating_point() or min(images.shape) < 1):
        raise ValueError("ResizeMix requires a nonempty floating NCHW batch")
    batch, _, height, width = images.shape
    tau = random.uniform(tau_min, tau_max)
    patch_height, patch_width = int(height*tau), int(width*tau)
    if min(patch_height, patch_width) < 1:
        raise ValueError("ResizeMix sampled side ratio creates an empty patch on these image dimensions")

    # Equivalent to the author's clipping/end/shift-back sequence. Complete
    # patches fit even at a boundary; do not clip away source content or area.
    tops = np.clip(np.random.randint(0, height, size=batch)-patch_height//2, 0, height-patch_height)
    lefts = np.clip(np.random.randint(0, width, size=batch)-patch_width//2, 0, width-patch_width)
    boxes = tuple((int(top), int(top)+patch_height, int(left), int(left)+patch_width)
                  for top, left in zip(tops, lefts))
    permutation = torch.randperm(batch, device=images.device)
    # Source reads complete original images before writing any destination.
    patches = F.interpolate(images[permutation], size=(patch_height, patch_width),
                            mode="bilinear", align_corners=False)
    mixed = images.clone(memory_format=torch.preserve_format)
    image_ids = torch.arange(batch, device=images.device)[:, None, None]
    rows = torch.as_tensor(tops, device=images.device)[:, None, None]+torch.arange(patch_height, device=images.device)[None, :, None]
    columns = torch.as_tensor(lefts, device=images.device)[:, None, None]+torch.arange(patch_width, device=images.device)[None, None, :]
    mixed.permute(0, 2, 3, 1)[image_ids, rows, columns] = patches.permute(0, 2, 3, 1)
    retained_weight = 1.0-patch_height*patch_width/(height*width)
    return mixed, permutation, retained_weight, boxes
