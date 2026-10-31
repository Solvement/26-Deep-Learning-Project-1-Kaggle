"""Scratch-only SE-WRN22-4/r16 candidate, separate from the frozen builder.

Hu et al.: https://arxiv.org/abs/1709.01507
Author implementation: https://github.com/hujie-frank/SENet
SE gates act on conv2 output before the residual sum. No shortcut is gated.
This module provides an architecture, not a trainer or a measured result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import runtime  # Windows DLL isolation before torch; no GPU is initialized here.
import torch
from torch import Tensor, nn
from model import PARAMETER_LIMIT, PreActWideBlock, WideResNet as FrozenWideResNet, count_parameters

ARCHITECTURE = "se_wrn22_4_r16_v1"
PARAMETER_COUNT = 4_332_654


@dataclass(frozen=True)
class SEModelConfig:
    architecture: str = ARCHITECTURE
    depth: int = 22
    widen_factor: int = 4
    dropout: float = 0.0
    num_classes: int = 10
    reduction: int = 16

    def validate(self) -> None:
        values = (self.depth, self.widen_factor, self.num_classes, self.reduction)
        if (self.architecture != ARCHITECTURE or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values)
                or values != (22, 4, 10, 16) or isinstance(self.dropout, bool)
                or not isinstance(self.dropout, (float, int)) or self.dropout != 0.0):
            raise ValueError("Only the predeclared SE-WRN22-4/r16/dropout0/ten-class architecture is supported")


class SqueezeExcitation(nn.Module):
    """Global spatial mean -> bias FC -> ReLU -> bias FC -> sigmoid -> scale."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (channels, reduction)) or channels % reduction:
            raise ValueError("SE channels/reduction must be positive integers with exact divisibility")
        self.channels, self.reduction = int(channels), int(reduction)
        self.hidden_channels = self.channels//self.reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(self.channels, self.hidden_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Linear(self.hidden_channels, self.channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("SE requires NCHW features with its declared channel count")
        descriptor = self.pool(features).flatten(1)
        weights = self.sigmoid(self.expand(self.relu(self.reduce(descriptor))))
        return features*weights[:, :, None, None]


class SEPreActWideBlock(nn.Module):
    """Take scratch-constructed frozen block modules under their original names.

    The frozen block and its class are never patched. Keeping names preserves
    all shared state_dict keys, including the preactivated projection shortcut.
    """

    def __init__(self, original: PreActWideBlock, reduction: int = 16) -> None:
        super().__init__()
        if not isinstance(original, PreActWideBlock):
            raise TypeError("SE blocks require a freshly constructed frozen WRN block")
        for name in ("bn1", "relu1", "conv1", "bn2", "relu2", "dropout", "conv2", "projection"):
            setattr(self, name, getattr(original, name))
        self.se = SqueezeExcitation(self.conv2.out_channels, reduction)

    def forward(self, inputs: Tensor) -> Tensor:
        activated = self.relu1(self.bn1(inputs))
        shortcut = inputs if self.projection is None else self.projection(activated)
        features = self.conv1(activated)
        features = self.dropout(self.relu2(self.bn2(features)))
        return self.se(self.conv2(features))+shortcut


class SEWideResNet(nn.Module):
    """Fixed 4,332,654-parameter residual architecture, randomly initialized.

    Construct the complete frozen WRN first, then add nine SE gates. Therefore
    a matching seed produces identical shared initial tensors and state keys.
    New gates use nn.Linear's default initialization after base construction;
    they consume subsequent CPU RNG normally. No pretrained weights are read.
    """

    def __init__(self, depth=22, widen_factor=4, dropout=0.0, num_classes=10,
                 reduction=16, architecture=ARCHITECTURE) -> None:
        super().__init__()
        self.config = SEModelConfig(architecture, depth, widen_factor, dropout, num_classes, reduction)
        self.config.validate()
        original = FrozenWideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout, num_classes=num_classes)
        self.stem = original.stem
        self.stage1 = nn.Sequential(*(SEPreActWideBlock(block, reduction) for block in original.stage1))
        self.stage2 = nn.Sequential(*(SEPreActWideBlock(block, reduction) for block in original.stage2))
        self.stage3 = nn.Sequential(*(SEPreActWideBlock(block, reduction) for block in original.stage3))
        self.final_bn, self.final_relu = original.final_bn, original.final_relu
        self.pool, self.classifier = original.pool, original.classifier
        if count_parameters(self) != PARAMETER_COUNT or count_parameters(self, True) != PARAMETER_COUNT or PARAMETER_COUNT > PARAMETER_LIMIT:
            raise ValueError("SE actual total/trainable parameter budget differs from the registered architecture")

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        features = self.stage3(self.stage2(self.stage1(features)))
        features = self.final_relu(self.final_bn(features))
        return self.classifier(self.pool(features).flatten(1))

    def export_config(self) -> dict[str, Any]: return asdict(self.config)

    def count_parameters(self, trainable_only=False) -> int: return count_parameters(self, trainable_only)


def build_model(config: SEModelConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> SEWideResNet:
    if config is not None and not isinstance(config, SEModelConfig) and "architecture" not in config:
        raise ValueError("A saved SE model configuration must explicitly identify its architecture")
    arguments = asdict(config) if isinstance(config, SEModelConfig) else dict(config or {})
    arguments.update(kwargs)
    try: record = SEModelConfig(**arguments)
    except TypeError as error: raise ValueError("Unknown SE architecture constructor fields") from error
    record.validate()
    return SEWideResNet(**asdict(record))


def source_provenance() -> dict[str, Any]:
    return {"architecture": ARCHITECTURE, "initialization": "scratch_base_then_default_linear_gates",
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("model.py", "runtime.py", "research/se_model.py")}}
