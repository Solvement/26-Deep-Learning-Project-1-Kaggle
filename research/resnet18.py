"""4-stage pre-activation ResNet-18 for 32x32 inputs (He et al. 2016, identity mappings).

Different inductive bias from the 3-stage WRNs: four stages down to 4x4. `base` sets the
stage widths (base, 2b, 4b, 8b); base=40 gives ~4.4M params (submittable), base=64 ~11.2M.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PreActBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False) if stride != 1 or cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        skip = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        return out + skip


class PreActResNet18(nn.Module):
    def __init__(self, base: int = 40, num_classes: int = 10) -> None:
        super().__init__()
        widths = [base, base*2, base*4, base*8]
        self.stem = nn.Conv2d(3, base, 3, 1, 1, bias=False)
        layers, cin = [], base
        for i, w in enumerate(widths):
            for j in range(2):
                layers.append(PreActBlock(cin, w, 2 if (i > 0 and j == 0) else 1)); cin = w
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.layers(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


if __name__ == "__main__":
    for b in (40, 64):
        m = PreActResNet18(b)
        n = sum(p.numel() for p in m.parameters())
        assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
        print(b, n)
