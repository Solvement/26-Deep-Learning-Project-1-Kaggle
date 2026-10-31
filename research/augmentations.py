"""Independent CutMix implementation for normalized NCHW training batches.

Algorithm reference: Yun et al., ICCV 2019, https://arxiv.org/abs/1905.04899
and https://github.com/clovaai/CutMix-PyTorch. No pretrained models, external
datasets or author implementation files are imported.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch


def validate_cutmix(alpha, probability):
    for name, value in (("alpha", alpha), ("probability", probability)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"CutMix {name} must be a finite number")
    if alpha < 0 or not 0 <= probability <= 1:
        raise ValueError("CutMix requires alpha >= 0 and probability within [0, 1]")
    if probability > 0 and alpha <= 0:
        raise ValueError("Positive CutMix probability requires positive alpha")


def apply_cutmix(images, permutation, rectangle):
    """Return a new batch and the retained-label weight from actual pasted area.

    Rectangle coordinates are (top, bottom, left, right), already clipped to
    image bounds. Donor patches always come from the original, unmodified batch.
    """
    if images.ndim != 4 or not images.is_floating_point() or min(images.shape) < 1:
        raise ValueError("CutMix requires a nonempty floating NCHW batch")
    if (permutation.shape != (len(images),) or permutation.dtype != torch.int64 or
            permutation.device != images.device):
        raise ValueError("CutMix donor permutation must be matching device int64 batch indices")
    if (len(rectangle) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in rectangle)):
        raise ValueError("CutMix rectangle requires four integer coordinates")
    top, bottom, left, right = rectangle
    height, width = images.shape[-2:]
    if not (0 <= top <= bottom <= height and 0 <= left <= right <= width):
        raise ValueError("CutMix rectangle must be clipped to image bounds")
    mixed = images.clone(memory_format=torch.preserve_format)
    mixed[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
    retained_weight = 1.0-(bottom-top)*(right-left)/(height*width)
    return mixed, retained_weight


def cutmix(images, alpha=1.0, probability=.5):
    """Sample one shared rectangle and donor permutation per training minibatch.

    Returns (mixed_images, donor_permutation_or_None, retained_label_weight,
    clipped_rectangle_or_None). Labels are never modified by this helper.
    Python/NumPy/torch global RNGs are captured by shadow resume checkpoints.
    A probability of zero consumes no random numbers and returns the input.
    """
    validate_cutmix(alpha, probability)
    if probability == 0 or random.random() >= probability:
        return images, None, 1.0, None
    if images.ndim != 4 or not images.is_floating_point() or min(images.shape) < 1:
        raise ValueError("CutMix requires a nonempty floating NCHW batch")
    sampled_weight = float(np.random.beta(alpha, alpha))
    ratio = math.sqrt(1.0-sampled_weight)
    height, width = images.shape[-2:]
    patch_height, patch_width = int(height*ratio), int(width*ratio)
    cy, cx = int(np.random.randint(height)), int(np.random.randint(width))
    # Preserve the sampled integer size when odd; clip only at actual borders.
    rectangle = (max(0, cy-patch_height//2), min(height, cy+(patch_height+1)//2),
                 max(0, cx-patch_width//2), min(width, cx+(patch_width+1)//2))
    permutation = torch.randperm(len(images), device=images.device)
    mixed, retained_weight = apply_cutmix(images, permutation, rectangle)
    return mixed, permutation, retained_weight, rectangle
