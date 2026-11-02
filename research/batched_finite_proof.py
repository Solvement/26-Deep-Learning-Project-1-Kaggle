"""Standalone proof primitives; no trainer, timing, admission or GPU launch.

Callers remain responsible for the complete expected tensor/counter inventory.
These helpers inspect detached values and never change the supplied tensors.
"""
from __future__ import annotations

import torch


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
FINITE_DTYPES = (*FLOAT_DTYPES, torch.int64, torch.bool)


def _checked_tensors(values, dtypes):
    tensors = tuple(values)
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Proof inputs must be tensors")
        if (tensor.layout != torch.strided or tensor.is_nested or
                tensor.device.type not in ("cpu", "cuda") or tensor.dtype not in dtypes):
            raise ValueError("Unsupported proof dtype, layout or device")
    if tensors and any(t.device != tensors[0].device for t in tensors):
        raise ValueError("Proof tensors must share one device")
    return tensors


def all_finite(values):
    """Return one detached scalar-bool tensor for all input elements.

    FP16/BF16/FP32 values are exactly representable in FP32; if any FP64 input
    exists use FP64, which exactly represents all the supported float formats.
    Thus promotion cannot turn finite values into Inf or hide NaN/Inf. Int64
    and bool have no nonfinite values and are never cast to a floating dtype.
    Empty tensors and an empty collection have the usual all=True identity;
    an empty collection returns a CPU flag and does not certify an inventory.
    """
    tensors = _checked_tensors(values, FINITE_DTYPES)
    floating = [t for t in tensors if t.dtype in FLOAT_DTYPES]
    if not floating:
        device = tensors[0].device if tensors else torch.device("cpu")
        return torch.ones((), dtype=torch.bool, device=device)
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in floating) else torch.float32
    flat = torch.cat([t.detach().reshape(-1).to(dtype=dtype) for t in floating])
    return torch.isfinite(flat).all()


def snapshot_bn_counters(values):
    """Copy scalar int64 counters into an independent ordered device vector."""
    counters = _checked_tensors(values, (torch.int64,))
    if any(t.ndim != 0 for t in counters):
        raise ValueError("BN counters must be scalar int64 tensors")
    if not counters:
        return torch.empty((0,), dtype=torch.int64)
    return torch.stack(counters).detach().clone()


def prove_bn_increment(before, counters, expected_increment=1):
    """Copy both complete counter vectors to CPU together and audit each layer.

    Take ``before`` with snapshot_bn_counters before the forward/update. Python
    integer comparison preserves the old scalar-item semantics even at int64
    limits: no wrapping device addition can falsely prove an increment.
    The returned before/after lists retain one auditable entry per input layer.
    """
    if type(expected_increment) is not int or expected_increment < 0:
        raise ValueError("Expected BN increment must be a nonnegative Python int")
    (before,) = _checked_tensors((before,), (torch.int64,))
    if before.ndim != 1:
        raise ValueError("A one-dimensional BN snapshot is required")
    after = snapshot_bn_counters(counters)
    if after.shape != before.shape or after.device != before.device:
        raise ValueError("BN counter count or device changed")
    rows = torch.stack((before.detach(), after)).to(device="cpu").tolist()
    previous, current = rows
    return {
        "all_incremented": all(new == old + expected_increment for old, new in zip(previous, current)),
        "before": previous,
        "after": current,
        "expected_increment": expected_increment,
    }
