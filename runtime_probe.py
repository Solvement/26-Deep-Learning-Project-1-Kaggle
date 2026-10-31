"""Process-local CUDA DLL isolation and a real CNN training smoke test.

Run with ``python Project1/runtime_probe.py``. Training entrypoints should call
``prepare_torch_environment()`` before importing torch. This changes no global
environment variables or installed packages.
"""
from __future__ import annotations

import json
import os
import platform
import sys


def prepare_torch_environment() -> list[str]:
    """Prevent unrelated NVIDIA pip cuDNN DLLs from overriding torch's DLLs.

    On this Windows machine, torch bundles cuDNN 9.20 while a separate NVIDIA
    package adds cuDNN 9.23 to PATH. cuDNN's dynamically loaded sublibraries must
    all come from the same release. Only this process's PATH is modified.
    """
    if os.name != "nt":
        return []
    entries = os.environ.get("PATH", "").split(os.pathsep)
    removed = [p for p in entries if "nvidia" in p.lower() and "cudnn" in p.lower()]
    os.environ["PATH"] = os.pathsep.join(p for p in entries if p not in removed)
    return removed


def main() -> None:
    removed = prepare_torch_environment()
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this probe requires an NVIDIA GPU.")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10),
    ).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(64, 3, 32, 32, device="cuda").to(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (64,), device="cuda")
    losses = []
    for dtype in [torch.float32, torch.bfloat16]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError("CNN loss or gradients are nonfinite.")
        optimizer.step()
        torch.cuda.synchronize()
        losses.append({"dtype": str(dtype), "loss": loss.detach().item()})
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({
        "status": "CUDA_CNN_TRAINING_OK",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "free_GiB": round(free / 2**30, 3),
        "total_GiB": round(total / 2**30, 3),
        "removed_conflicting_path_entries": len(removed),
        "smoke_tests": losses,
    }, indent=2))


if __name__ == "__main__":
    main()
