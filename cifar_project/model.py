"""CIFAR-sized residual CNN; architecture reference: He et al., arXiv:1512.03385."""
import torch
from torch import nn

PARAMETER_LIMIT = 5_000_000


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return torch.relu(self.residual(x) + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, base_width=32, blocks_per_stage=3):
        super().__init__()
        if not isinstance(base_width, int) or not 1 <= base_width <= 128:
            raise ValueError("base_width must be an integer from 1 to 128")
        if not isinstance(blocks_per_stage, int) or not 1 <= blocks_per_stage <= 16:
            raise ValueError("blocks_per_stage must be an integer from 1 to 16")
        self.stem = nn.Sequential(nn.Conv2d(3, base_width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        blocks = []
        channels = base_width
        for stage, width in enumerate([base_width, base_width * 2, base_width * 4]):
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(ResidualBlock(channels, width, stride))
                channels = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.parameter_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.parameter_count > PARAMETER_LIMIT:
            raise ValueError(f"Model has {self.parameter_count:,} trainable parameters; limit is {PARAMETER_LIMIT:,}")

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected N x 3 x 32 x 32 CIFAR images")
        return self.classifier(self.pool(self.blocks(self.stem(x))).flatten(1))
