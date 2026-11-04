"""One CPU/PIL TrivialAugment operation in the fixed_standard/31 space.

Independent implementation of Mueller & Hutter, ICCV2021, Algorithm1.
Author reference (MIT, Samuel Gabriel Mueller2021): automl/trivialaugment
commit e6545d80affcc18b7024185a096e93e21010023e, aug_lib.py.
Upstream copyright/permission: research/licenses/TrivialAugment-MIT.txt.
No reference code, data loader, trainer, Torch or GPU is imported/executed.
Inputs are original RGB32 images, before the parent's crop/flip/normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

AUTHOR_COMMIT = "e6545d80affcc18b7024185a096e93e21010023e"
AUTHOR_SOURCE_SHA256 = "8f5f9febb671fd12be067e311dbdcb837a2a29c672861f20c2cf03732fdd2868"
SPACE = "fixed_standard"
STRENGTHS = 31
OPERATIONS = ("identity", "autocontrast", "equalize", "rotate", "solarize", "color", "posterize",
              "contrast", "brightness", "sharpness", "shear_x", "shear_y", "translate_x", "translate_y")
SIGNED = frozenset(("rotate", "shear_x", "shear_y", "translate_x", "translate_y"))
_ENHANCERS = {"color": ImageEnhance.Color, "contrast": ImageEnhance.Contrast,
              "brightness": ImageEnhance.Brightness, "sharpness": ImageEnhance.Sharpness}


@dataclass(frozen=True)
class Operation:
    """A replayable sample; level is a discrete index, never a Wide magnitude."""
    name: str
    level: int
    sign: int = 1
    space: str = SPACE


def _validate(operation):
    if (not isinstance(operation, Operation) or operation.space != SPACE or operation.name not in OPERATIONS
            or type(operation.level) is not int or not 0 <= operation.level < STRENGTHS
            or type(operation.sign) is not int or operation.sign not in (-1, 1)
            or (operation.name not in SIGNED and operation.sign != 1)):
        raise ValueError("A fixed_standard operation, integer level0..30 and canonical geometric sign are required")


def rgb_image(image):
    """Copy strict RGB32 PIL or uint8 HWC input; never convert modes or mutate."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.shape != (32, 32, 3):
            raise ValueError("TA arrays must be uint8 HWC RGB32")
        return Image.fromarray(np.array(image, copy=True))
    if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("TA requires original RGB32 pixels")
    return image.copy()


def magnitude(operation):
    """Author fixed_standard scaling, including integer truncation and fix."""
    _validate(operation)
    name, level = operation.name, operation.level
    if name in ("identity", "autocontrast", "equalize"):
        return None
    if name.startswith("shear_"):
        return operation.sign * (float(level) * .3 / 30)
    if name.startswith("translate_"):
        return operation.sign * int(level * 10 / 30)
    if name == "rotate":
        return operation.sign * int(level * 30 / 30)
    if name == "solarize":
        return 256 - int(level * 256 / 30)
    if name == "posterize":
        # fixed_standard uses8..4 bits; the uncorrected0..4 space is excluded.
        return 8 - int(level * 4 / 30)
    return float(level) * (1.9 - .1) / 30 + .1


def sample_operation(rng):
    """Uniform op/level, independent sign, using only the caller's RNG object.

    NumPy Generator and random.Random are checkpointable. The module-global RNG
    objects and SystemRandom are rejected. Author seed-stream equivalence is
    not claimed: no unnecessary probability-one Bernoulli draw is consumed.
    """
    if (not isinstance(rng, (np.random.Generator, random.Random)) or isinstance(rng, random.SystemRandom)):
        raise TypeError("An explicit numpy.random.Generator or checkpointable random.Random is required")
    if isinstance(rng, np.random.Generator):
        name = OPERATIONS[int(rng.integers(len(OPERATIONS)))]
        level = int(rng.integers(STRENGTHS))
    else:
        name = OPERATIONS[rng.randrange(len(OPERATIONS))]
        level = rng.randrange(STRENGTHS)
    sign = -1 if name in SIGNED and float(rng.random()) > .5 else 1
    return Operation(name, level, sign)


def apply_operation(image, operation):
    """Apply a trace with explicit PIL nearest-neighbor geometry and black fill."""
    value = magnitude(operation)
    image = rgb_image(image)
    name = operation.name
    if name == "identity":
        return image
    if name == "autocontrast":
        return ImageOps.autocontrast(image)
    if name == "equalize":
        return ImageOps.equalize(image)
    if name == "solarize":
        return ImageOps.solarize(image, value)
    if name == "posterize":
        return ImageOps.posterize(image, value)
    if name in _ENHANCERS:
        return _ENHANCERS[name](image).enhance(value)
    if name == "rotate":
        return image.rotate(value, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))
    coefficients = {
        "shear_x": (1, value, 0, 0, 1, 0), "shear_y": (1, 0, 0, value, 1, 0),
        "translate_x": (1, 0, value, 0, 1, 0), "translate_y": (1, 0, 0, 0, 1, value),
    }[name]
    return image.transform(image.size, Image.Transform.AFFINE, coefficients,
                           resample=Image.Resampling.NEAREST, fillcolor=(0, 0, 0))


def augment(image, rng, *, return_trace=False):
    """Return one RGB32 PIL view; optionally also return the immutable trace.

    There is no crop, flip, Cutout, Mixup, normalization or loss in this module.
    The future trainer applies the original parent augmentation after TA.
    """
    if type(return_trace) is not bool:
        raise TypeError("return_trace must be boolean")
    image = rgb_image(image)  # Invalid input does not advance the caller's RNG.
    operation = sample_operation(rng)
    output = apply_operation(image, operation)
    return (output, operation) if return_trace else output
