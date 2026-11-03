"""Numerical and teacher-isolation primitives for the new scratch KD experiment.

Importing this module does not import Torch/NumPy, initialize CUDA, load a model,
or authorize any teacher/checkpoint. The caller owns actual lineage, the fixed
recipe, timing, native-process boundary, student rollback and artifact audit.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

TEACHER_STATE_TENSORS = 191
TEACHER_BN_MODULES = 31
TEACHER_PARAMETERS = 4166298


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(value, name, lower, upper=None, *, inclusive=True):
    require(type(value) in (int, float) and math.isfinite(value), "Finite non-boolean "+name+" required")
    require((value >= lower if inclusive else value > lower) and (upper is None or value <= upper), "Invalid "+name)
    return float(value)


def _hyperparameters(temperature, alpha):
    return _scalar(temperature, "temperature", 0, inclusive=False), _scalar(alpha, "alpha", 0, 1)


def _finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all().item())


def mixed_hard_loss(student_logits, labels, paired_labels, lam, label_smoothing=0.0):
    """Preserve the frozen CE*lam then out-of-place optional mixed-label addition."""
    lam = _scalar(lam, "Mixup lambda", 0, 1)
    smoothing = _scalar(label_smoothing, "label smoothing", 0, 1)
    require(paired_labels is not None or lam == 1., "Unmixed hard loss requires lambda one")
    from torch.nn import functional as F
    hard = F.cross_entropy(student_logits, labels, label_smoothing=smoothing)*lam
    if paired_labels is not None:
        hard = hard+F.cross_entropy(student_logits, paired_labels, label_smoothing=smoothing)*(1-lam)
    return hard


def distillation_components(student_logits, teacher_logits, hard_loss, temperature, alpha):
    """FP32 forward KL(teacher || student), batchmean*T², with no clipping.

    Alpha zero deliberately inspects no tensors and imports no numerical package:
    it returns the original hard-loss object. The unchanged parent trainer still
    owns the ordinary hard-loss/gradient finite checks on that control branch.
    """
    temperature, alpha = _hyperparameters(temperature, alpha)
    if alpha == 0.:
        return {"loss": hard_loss, "hard_loss": hard_loss, "kd_loss": None}
    import torch
    from torch.nn import functional as F
    require(isinstance(student_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor), "Actual student/teacher logits required")
    require(student_logits.ndim == teacher_logits.ndim == 2 and student_logits.shape == teacher_logits.shape and
        student_logits.shape[0] > 0 and student_logits.shape[1] == 10, "Matching nonempty (B,10) logits required")
    require(student_logits.is_floating_point() and teacher_logits.is_floating_point() and
        student_logits.device == teacher_logits.device and student_logits.device.type in ("cpu", "cuda"), "Floating logits on the same supported device required")
    require(isinstance(hard_loss, torch.Tensor) and hard_loss.ndim == 0 and hard_loss.is_floating_point() and
        hard_loss.device == student_logits.device, "Original scalar hard-loss tensor required")
    require(_finite(student_logits) and _finite(teacher_logits) and _finite(hard_loss), "Nonfinite logits/hard loss are forbidden")
    square = temperature*temperature
    require(math.isfinite(square) and 0 < square <= torch.finfo(torch.float32).max, "Temperature square must be representable in FP32")
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        student = student_logits.float()/temperature
        teacher = teacher_logits.detach().float()/temperature
        require(_finite(student) and _finite(teacher), "Scaled FP32 logits are not representable")
        log_student = F.log_softmax(student, dim=1)
        log_teacher = F.log_softmax(teacher, dim=1).detach()
        require(_finite(log_student) and _finite(log_teacher), "FP32 log probabilities must remain finite")
        kd = F.kl_div(log_student, log_teacher, reduction="batchmean", log_target=True)*square
        require(_finite(kd), "FP32 forward KL is not representable")
        total = (1-alpha)*hard_loss+alpha*kd
        require(_finite(total), "Combined KD loss is not finite")
    return {"loss": total, "hard_loss": hard_loss, "kd_loss": kd}


def distillation_loss(student_logits, teacher_logits, hard_loss, temperature, alpha):
    """Return only the training tensor; alpha0 is the identical hard-loss object."""
    temperature, alpha = _hyperparameters(temperature, alpha)
    if alpha == 0.:
        return hard_loss
    return distillation_components(student_logits, teacher_logits, hard_loss, temperature, alpha)["loss"]


def capture_parent_rng():
    """Capture Python, NumPy, Torch CPU and every available CUDA generator."""
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None}


def restore_parent_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def assert_parent_rng_equal(left, right):
    import numpy as np
    import torch
    a, b = left["numpy"], right["numpy"]
    require(left["python"] == right["python"] and a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:] and
        torch.equal(left["torch_cpu"], right["torch_cpu"]), "Parent Python/NumPy/TorchCPU RNG changed")
    a, b = left["torch_cuda"], right["torch_cuda"]
    require((a is None) == (b is None) and (a is None or len(a) == len(b) and all(torch.equal(x,y) for x,y in zip(a,b))), "Parent all-CUDA RNG changed")


@contextmanager
def isolated_teacher_rng():
    """Restore every parent RNG even if teacher construction/load/forward fails."""
    state = capture_parent_rng()
    try:
        yield
    finally:
        restore_parent_rng(state)
        assert_parent_rng_equal(state, capture_parent_rng())


def _teacher_structure(teacher, expected_state_tensors, expected_bn_modules, expected_parameters):
    import torch
    require(isinstance(teacher, torch.nn.Module), "Separate teacher module required")
    require(all(not module.training for module in teacher.modules()), "Every teacher module must remain in eval mode")
    parameters = dict(teacher.named_parameters())
    require(all(not parameter.requires_grad and parameter.grad is None for parameter in parameters.values()), "Teacher parameters may not require or retain gradients")
    require(sum(parameter.numel() for parameter in parameters.values()) == expected_parameters, "Actual teacher parameter count differs")
    state = teacher.state_dict()
    bns = tuple(name for name,module in teacher.named_modules() if isinstance(module, torch.nn.BatchNorm2d))
    require(len(state) == expected_state_tensors and len(bns) == expected_bn_modules, "Complete teacher state/BN topology differs")
    require(all(isinstance(value,torch.Tensor) for value in state.values()), "Every teacher state entry must be a tensor")
    counters=[name for name,value in state.items() if not value.is_floating_point()]
    require(len(counters) == expected_bn_modules and all(name.endswith("num_batches_tracked") and state[name].dtype == torch.int64 for name in counters) and
        all(value.dtype == torch.float32 for value in state.values() if value.is_floating_point()), "Teacher FP32 master/buffer and int64 BN-counter dtypes required")
    return state, bns, tuple(parameters)


def teacher_state_snapshot(teacher, expected_state_tensors=TEACHER_STATE_TENSORS,
                           expected_bn_modules=TEACHER_BN_MODULES, expected_parameters=TEACHER_PARAMETERS):
    """Clone the full already-frozen teacher once after its guarded setup/load."""
    import torch
    require(all(type(value) is int and value >= 0 for value in (expected_state_tensors,expected_bn_modules,expected_parameters)) and
        expected_state_tensors > 0 and expected_parameters > 0, "Explicit nonnegative teacher topology counts required")
    state,bns,parameters = _teacher_structure(teacher,expected_state_tensors,expected_bn_modules,expected_parameters)
    require(bool(torch.stack([torch.isfinite(value).all() for value in state.values()]).all().item()), "Initial teacher state must be finite")
    return {"module_identity": id(teacher), "state": {name:value.detach().clone() for name,value in state.items()},
        "bn_names": bns, "parameter_names": parameters, "expected_state_tensors": expected_state_tensors,
        "expected_bn_modules": expected_bn_modules, "expected_parameters": expected_parameters}


def assert_teacher_immutable(teacher, snapshot):
    """Full exact state equality, finiteness, eval/BN topology and no gradients."""
    import torch
    require(id(teacher) == snapshot["module_identity"], "Teacher object identity changed")
    state,bns,parameters = _teacher_structure(teacher,snapshot["expected_state_tensors"],snapshot["expected_bn_modules"],snapshot["expected_parameters"])
    frozen = snapshot["state"]
    require(tuple(state) == tuple(frozen) and bns == snapshot["bn_names"] and parameters == snapshot["parameter_names"], "Full teacher state/module coverage changed")
    checks=[]
    for name,value in state.items():
        original=frozen[name]
        require(value.shape == original.shape and value.dtype == original.dtype and value.device == original.device and value.stride() == original.stride(), "Teacher tensor metadata changed")
        checks.append(torch.isfinite(value).all() & value.eq(original).all())
    require(bool(torch.stack(checks).all().item()), "Teacher state values changed or became nonfinite")


def restore_teacher_state(teacher, snapshot):
    """Failure recovery only; a mutation still raises and cannot qualify a step."""
    import torch
    require(id(teacher) == snapshot["module_identity"], "Cannot restore another teacher object")
    with torch.no_grad():
        teacher.load_state_dict(snapshot["state"], strict=True)
        teacher.eval().requires_grad_(False)
        for parameter in teacher.parameters():
            parameter.grad = None
    assert_teacher_immutable(teacher,snapshot)


def frozen_teacher_forward(teacher, mixed_inputs, snapshot):
    """No-grad forward on the identical final mixed tensor; restore RNG always.

    The caller's autocast context controls teacher precision. Both complete
    teacher-state checks and all RNG work execute inside this function, so its
    entire call belongs inside the preflight/training step timer.
    """
    import torch
    require(isinstance(mixed_inputs,torch.Tensor) and mixed_inputs.ndim >= 1 and mixed_inputs.shape[0] > 0, "Actual nonempty final mixed input tensor required")
    parent_rng=capture_parent_rng()
    try:
        assert_teacher_immutable(teacher,snapshot)
        with torch.no_grad():
            logits=teacher(mixed_inputs)
        require(isinstance(logits,torch.Tensor) and logits.ndim == 2 and logits.shape == (mixed_inputs.shape[0],10) and
            logits.device == mixed_inputs.device and logits.is_floating_point() and _finite(logits), "Complete finite teacher (B,10) logits required")
        assert_teacher_immutable(teacher,snapshot)
        return logits.detach()
    except BaseException:
        restore_teacher_state(teacher,snapshot)
        raise
    finally:
        restore_parent_rng(parent_rng)
        assert_parent_rng_equal(parent_rng,capture_parent_rng())
