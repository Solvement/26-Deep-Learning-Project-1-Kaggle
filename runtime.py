"""Reproducible training setup and process-local Windows CUDA DLL isolation."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys


def configure_runtime() -> dict:
    """Resolve the local cuDNN conflict before importing torch.

    The NVIDIA cuDNN pip directory on this machine contains a different release
    from torch/lib. Removing that directory from this process's PATH makes all
    dynamically loaded cuDNN libraries use torch's bundled release. No installed
    files or system/user environment variables are changed.
    """
    removed = []
    if os.name == "nt":
        paths = os.environ.get("PATH", "").split(os.pathsep)
        removed = [p for p in paths if "nvidia" in p.lower() and "cudnn" in p.lower()]
        os.environ["PATH"] = os.pathsep.join(p for p in paths if p not in removed)
    return {"process_local_cudnn_path_entries_removed": len(removed)}


RUNTIME_CONFIGURATION = configure_runtime()

# Keep these imports below configure_runtime(): torch loads Windows DLLs here.
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


def seed_all(seed: int = 2026, deterministic: bool = False, threads: int = 4) -> None:
    """Seed Python, NumPy and torch; document speed/determinism explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def environment() -> dict:
    packages = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "matplotlib", "nbformat", "reportlab"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(DEVICE),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if DEVICE.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None,
        "gpu_memory_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
        if DEVICE.type == "cuda" else None,
        "amp_dtype": str(AMP_DTYPE),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        **RUNTIME_CONFIGURATION,
    }


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def amp():
    return torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda")


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    print(json.dumps(environment(), indent=2))
