"""Independent, stateless ordinary SAM update using a caller-owned SGD optimizer.

Algorithm: Foret et al., https://arxiv.org/abs/2010.01412 and the authors'
https://github.com/google-research/sam. This implementation does not import
author code, datasets or weights. It supports CPU FP32 / native CUDA BF16 with
FP32 master parameters; FP16 and GradScaler are intentionally unsupported.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Preserve process-local Windows DLL isolation before importing torch.
import runtime  # noqa: F401
import torch


def _tensor_groups(tensors):
    """Group by device/dtype so mixed BN counters do not disable foreach fusion."""
    groups = {}
    for index, tensor in enumerate(tensors):
        groups.setdefault((tensor.device, tensor.dtype), []).append((index, tensor))
    return groups.values()


@torch.no_grad()
def _clone_tensors(tensors):
    tensors = list(tensors)
    saved = [None]*len(tensors)
    for group in _tensor_groups(tensors):
        # empty_like allocates without a fill kernel; foreach_copy_ batches the
        # actual copies and exists in the repository's PyTorch2.9 CI version.
        copies = [torch.empty_like(tensor, memory_format=torch.preserve_format) for _, tensor in group]
        torch._foreach_copy_(copies, [tensor.detach() for _, tensor in group])
        for (index, _), value in zip(group, copies):
            saved[index] = value
    return saved


@torch.no_grad()
def _copy_tensors(destinations, sources):
    for group in _tensor_groups(destinations):
        torch._foreach_copy_([tensor for _, tensor in group], [sources[index] for index, _ in group])


def _bn_snapshot(model):
    states, locations, tensors = [], [], []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            buffers = {}
            states.append((module, module.training, module.momentum, module.track_running_stats, buffers))
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name)
                buffers[name] = None
                if value is not None:
                    locations.append((buffers, name))
                    tensors.append(value)
    for (buffers, name), saved in zip(locations, _clone_tensors(tensors)):
        buffers[name] = saved
    return states


@torch.no_grad()
def _restore_bn(states):
    destinations, sources = [], []
    for module, training, momentum, tracking, buffers in states:
        module.training, module.momentum, module.track_running_stats = training, momentum, tracking
        for name, saved in buffers.items():
            if saved is None:
                setattr(module, name, None)
            else:
                destinations.append(getattr(module, name))
                sources.append(saved)
    _copy_tensors(destinations, sources)


@contextmanager
def _second_pass_bn(model):
    """Keep training batch normalization but discard every second-pass BN update."""
    states = _bn_snapshot(model)
    try:
        for module, *_ in states:
            # momentum=0 alone still increments num_batches_tracked; restore all
            # buffers afterwards, including in the nonfinite/exception path.
            module.momentum = 0.0
        yield
    finally:
        _restore_bn(states)


def _global_l2(tensors, stage):
    values = list(tensors)
    if not values:
        raise ValueError(f"SAM {stage}: no optimizer parameter has a gradient")
    if any(value.is_sparse for value in values):
        raise ValueError("SAM requires dense gradients")
    # Same global L2 formula as individual norms, using multi-tensor kernels.
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([value.detach().float() for value in values], 2)))
    scalar = float(norm.item())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"SAM {stage}: nonfinite gradient or global L2 norm")
    return scalar


def _backward(closure, stage):
    with torch.enable_grad():
        loss = closure()
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.is_floating_point() or not loss.requires_grad:
            raise ValueError("SAM closure must return a differentiable scalar loss without calling backward")
        scalar = float(loss.detach().item())
        if not math.isfinite(scalar):
            raise FloatingPointError(f"SAM {stage}: nonfinite loss")
        loss.backward()
    return scalar


def _parameters(model, optimizer, precision, scaler):
    if precision not in ("fp32", "bf16") or scaler is not None:
        raise ValueError("SAM supports only fp32/bf16 without any GradScaler; FP16 is unsupported")
    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("SAM requires a caller-owned torch.optim.SGD optimizer")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    identities = [id(parameter) for parameter in parameters]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if not parameters or len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError("SAM optimizer must contain every trainable model parameter exactly once")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("SAM requires FP32 master parameters, including for BF16 autocast")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("SAM supports one device per model")
    device = next(iter(devices))
    if precision == "fp32" and device.type not in ("cpu", "cuda"):
        raise ValueError("SAM FP32 supports CPU or CUDA only")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("SAM BF16 requires a natively supported CUDA GPU")
    # Linf cannot overflow for finite FP32 weights and propagates NaN/Inf.
    nonempty = [parameter.detach() for parameter in parameters if parameter.numel()]
    maximum = float(torch.stack(torch._foreach_norm(nonempty, float("inf"))).amax().item()) if nonempty else 0.0
    if not math.isfinite(maximum):
        raise FloatingPointError("SAM model contains a nonfinite parameter before the update")
    return parameters


def sam_step(model, optimizer, closure, *, rho=.05, precision="fp32", scaler=None):
    """Perform two forward/backward passes and one ordinary nonadaptive SAM step.

    ``closure()`` must compute and return a scalar loss on the SAME already
    augmented inputs and mixed labels on both calls. It must not call backward,
    zero_grad, optimizer.step, mutate parameters/modes, or resample augmentation.
    Autocast belongs inside the closure. Separate dropout draws are permitted.

    On success, first-pass BN statistics are retained, gradients are cleared,
    and JSON-safe loss/norm/update evidence is returned. Parameters with None
    gradients are skipped; all-None is an error; all-zero gives zero perturbation
    and a normal second SGD step (including configured momentum/weight decay).

    Failed passes restore exact original parameters and pre-call BN buffers,
    clear gradients and never invoke SGD, so its momentum state is unchanged.
    An exception inside optimizer.step itself may partially change SGD state;
    recover the outer training checkpoint rather than retrying that minibatch.
    """
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("SAM rho must be positive and finite")
    parameters = _parameters(model, optimizer, precision, scaler)
    originals = _clone_tensors(parameters)
    original_bn = _bn_snapshot(model)
    updated = False
    try:
        model.zero_grad(set_to_none=True)
        first_loss = _backward(closure, "first pass")
        grad_norm = _global_l2((parameter.grad for parameter in parameters if parameter.grad is not None), "first pass")
        with torch.no_grad():
            if grad_norm > 0:
                active = [parameter for parameter in parameters if parameter.grad is not None]
                torch._foreach_add_(active, [parameter.grad for parameter in active], alpha=rho/grad_norm)
        perturbation_norm = _global_l2(torch._foreach_sub([parameter.detach() for parameter in parameters], originals), "perturbation")
        with _second_pass_bn(model):
            model.zero_grad(set_to_none=True)
            second_loss = _backward(closure, "second pass")
            second_grad_norm = _global_l2((parameter.grad for parameter in parameters if parameter.grad is not None), "second pass")
        # Copying saved originals avoids subtractive floating point drift.
        _copy_tensors(parameters, originals)
        evidence = {"first_loss": first_loss, "second_loss": second_loss, "grad_norm": grad_norm,
                    "second_grad_norm": second_grad_norm, "perturbation_norm": perturbation_norm,
                    "rho": float(rho), "updated": True, "forward_backward_passes": 2}
        optimizer.step()
        updated = True
        return evidence
    finally:
        if not updated:
            _copy_tensors(parameters, originals)
            _restore_bn(original_bn)
        model.zero_grad(set_to_none=True)
