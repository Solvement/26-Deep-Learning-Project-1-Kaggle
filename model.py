"""CIFAR WideResNet, initialized from scratch and limited to 5M parameters.

Architecture reference:
    Sergey Zagoruyko and Nikos Komodakis. Wide Residual Networks. BMVC, 2016.
    https://arxiv.org/abs/1605.07146
    Authors' implementation: https://github.com/szagoruyko/wide-residual-networks

This independent PyTorch implementation follows the paper's preactivation basic
blocks and three CIFAR stages. The default WRN-22-4 chooses a smaller depth/width
than the paper's large models to satisfy the project budget. No pretrained
weights, downloads, or external model libraries are used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from typing import Any, Mapping

import torch
from torch import Tensor, nn


PARAMETER_LIMIT = 5_000_000


@dataclass(frozen=True)
class ModelConfig:
    """Serializable constructor arguments; depth uses the WRN 6n+4 convention."""

    depth: int = 22
    widen_factor: int = 4
    dropout: float = 0.0
    num_classes: int = 10

    def validate(self) -> None:
        for name in ("depth", "widen_factor", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.depth < 10 or (self.depth - 4) % 6:
            raise ValueError("depth must be 6n+4 with n >= 1 (10, 16, 22, 28, ...)")
        if self.widen_factor < 1:
            raise ValueError("widen_factor must be >= 1")
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters exactly; BatchNorm running statistics are buffers."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


class PreActWideBlock(nn.Module):
    """BN-ReLU-Conv-BN-ReLU-Dropout-Conv, followed by a residual sum.

    Identity shortcuts preserve the original input. When spatial size or channel
    count changes, a learned 1x1 projection consumes the preactivated input, as
    in the authors' CIFAR WideResNet implementation.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dropout: float
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        activated = self.relu1(self.bn1(inputs))
        shortcut = inputs if self.projection is None else self.projection(activated)
        features = self.conv1(activated)
        features = self.dropout(self.relu2(self.bn2(features)))
        return self.conv2(features) + shortcut


class WideResNet(nn.Module):
    """A scratch-trained CIFAR classifier; defaults to WRN-22-4.

    The 3x3 stride-one stem retains all 32x32 pixels. Only stages two and three
    downsample, giving feature maps of 32x32, 16x16 and 8x8. Adaptive average
    pooling makes the classifier independent of the final spatial dimensions.
    Both total and trainable parameter counts are checked against 5,000,000 at
    construction, so freezing parameters cannot hide an oversized model.
    """

    def __init__(
        self,
        depth: int = 22,
        widen_factor: int = 4,
        dropout: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(depth, widen_factor, dropout, num_classes)
        self.config.validate()
        blocks_per_stage = (depth - 4) // 6
        widths = (16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor)
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.stage1 = self._make_stage(
            widths[0], widths[1], blocks_per_stage, 1, dropout
        )
        self.stage2 = self._make_stage(
            widths[1], widths[2], blocks_per_stage, 2, dropout
        )
        self.stage3 = self._make_stage(
            widths[2], widths[3], blocks_per_stage, 2, dropout
        )
        self.final_bn = nn.BatchNorm2d(widths[3])
        self.final_relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], num_classes)
        total = count_parameters(self)
        if total > PARAMETER_LIMIT:
            raise ValueError(
                f"WRN-{depth}-{widen_factor} has {total:,} total parameters; "
                f"the project limit is {PARAMETER_LIMIT:,}. "
                "Choose a smaller depth or widen_factor."
            )
        self.apply(self._initialize_from_scratch)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        blocks = [PreActWideBlock(in_channels, out_channels, stride, dropout)]
        blocks.extend(
            PreActWideBlock(out_channels, out_channels, 1, dropout)
            for _ in range(num_blocks - 1)
        )
        return nn.Sequential(*blocks)

    @staticmethod
    def _initialize_from_scratch(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        features = self.stage3(self.stage2(self.stage1(features)))
        features = self.final_relu(self.final_bn(features))
        return self.classifier(self.pool(features).flatten(1))

    def export_config(self) -> dict[str, Any]:
        """Return JSON-compatible arguments for exact architecture recreation."""

        return asdict(self.config)

    def count_parameters(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)


def build_model(
    config: ModelConfig | Mapping[str, Any] | None = None, **kwargs: Any
) -> WideResNet:
    """Construct from a saved config or explicit arguments; never load weights."""

    arguments = asdict(config) if isinstance(config, ModelConfig) else dict(config or {})
    arguments.update(kwargs)
    return WideResNet(**arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a CPU forward/backward check"
    )
    args = parser.parse_args()
    model = WideResNet(args.depth, args.widen_factor, args.dropout, args.num_classes)
    summary: dict[str, Any] = {
        "architecture": f"WRN-{args.depth}-{args.widen_factor}",
        "config": model.export_config(),
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "parameter_limit": PARAMETER_LIMIT,
        "within_parameter_limit": count_parameters(model) <= PARAMETER_LIMIT,
        "initialization": "random_from_scratch",
    }
    if args.smoke_test:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        model.train()
        logits = model(torch.randn(2, 3, 32, 32))
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, args.num_classes - 1]))
        loss.backward()
        if logits.shape != (2, args.num_classes) or not torch.isfinite(loss):
            raise RuntimeError("CPU forward check failed")
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("A parameter has missing or nonfinite gradients")
        summary["cpu_smoke_test"] = {
            "logits_shape": list(logits.shape),
            "loss": loss.detach().item(),
            "all_gradients_finite": True,
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
