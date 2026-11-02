# Batched finite/BN proof equivalence study

Issue7 / PR8, 2026-11-02. `batched_finite_proof.py` is a small independent
primitive library. It is not connected to a trainer, formal preflight, profiler,
supervisor or uploader. No speed, accuracy or admission improvement is claimed.
Preserve all old bound source, negative timing receipts and gates byte-for-byte.

## Exact finite-value contract

The old predicate is the conjunction of `torch.isfinite(t).all()` for every
tensor in the complete state/gradient inventory. The new predicate detaches and
flattens every floating tensor, promotes FP16/BF16/FP32 to FP32, or all floats
to FP64 if a FP64 input exists, concatenates and applies one full-element
`isfinite(...).all()`. All values of each input format are exactly representable
in its chosen destination. A maximum finite FP64 is never narrowed to FP32;
promotion cannot introduce Inf or turn NaN/Inf finite. This is not a norm,
sum, sample, probe or numerical bound, which could overflow on finite elements
or miss a bad element.

Explicit supported dtypes are FP16, BF16, FP32, FP64, int64 and bool. Int64 and
bool contain only finite values by definition and are not converted to floats.
Complex, other integer/float formats, quantized, sparse, nested and non-CPU/CUDA
inputs are rejected. Inputs must share one device; no implicit cross-device
copy is performed. Empty tensors are checked with all=True identity, and an
empty collection returns a CPU True flag. Thus callers must independently prove
the expected nonempty tensor inventory, count, names/shapes/dtypes and full
optimizer/EMA membership; this helper does not establish completeness.

Reshape may copy noncontiguous or channels-last tensors. Cast/concatenation
allocations and every finite check must remain in any future whole-step timer,
with peak memory measured. No host `.item()` is performed by `all_finite`;
the caller decides when to combine or consume its scalar device flag. This
changes scheduling/temporary storage and could be slower; mathematical
equivalence is not a performance measurement.

## Exact BN counter contract

`snapshot_bn_counters` accepts ordered scalar int64 counters on one device and
returns an independent stacked/cloned vector **before** the update. The clone
must not alias live BN buffers. `prove_bn_increment` makes the corresponding
after snapshot, stacks before/after and copies that pair to CPU together. It
then compares every pair with Python integer arithmetic and returns every
before/after value plus the common expected increment and all-layer result.
This avoids per-layer GPU `.item()` synchronization without losing the original
layer-by-layer audit. It also matches scalar Python semantics at int64 limits:
device `before+1` could wrap and falsely accept max-int64 -> min-int64, whereas
the Python comparison rejects it. Input counter order, topology/count and
initial-zero/specific training-forward rule remain the caller's proof duties.
An empty inventory is the explicit vacuous identity, never evidence of31 BNs.

The helper does not change BN updates or counter buffers, batch size, RNG,
model/EMA/SGD state, optimizer semantics, learning math or overflow handling.
Positive gradient norm, real parameter/EMA changes, scaler rules and all other
existing proof obligations remain separate and cannot be removed by this work.

## Verification and any future use

The dedicated CPU tests compare against the original per-tensor full-element
predicate across all supported dtypes, NaN/both infinities in each position,
largest finite values, mixed dtypes, empty collections/tensors, noncontiguous
and channels-last inputs, detached gradient-bearing tensors, and seeded mixed
inventories. They verify unsupported input rejection and no input mutation.
BN tests cover before-clone aliasing, missed/double updates, exact full lists,
int64 bounds/overflow, empty inventory and invalid dtype/shape/count/device.

Only execute local CPU Torch tests when the owner confirms no active local GPU
job; set `CUDA_VISIBLE_DEVICES=-1` for that test process only. Linux CI may run
the same tests independently. No official data, model, CUDA allocation or CUDA
capability probe is needed. Record actual executed results separately from
this plan; the presence of tests is not a pass claim.

Actual profile evidence must first support the suspected repeated reduction/
host-read overhead. Any future integration needs a new independent source path
and plan, complete inventory/type/state/RNG/math equivalence review, and a fresh
uninstrumented real-image23-step whole-pipeline preflight including every old
proof obligation plus all new allocation/check costs. Existing reservation,
native-precision, OS-tree/lease, daily/job/hard-stop limits and conservative
admission formula stay fixed. Neither a CPU pass nor an instrumented profile is
formal timing admission or authorization to start training.

## Actual bounded CPU verification, 2026-11-02

After the owner confirmed the GPU lease was absent and deferred the next GPU
job, all16 dedicated tests passed locally in0.704 seconds (0.711752 seconds
including discovery,3.36 seconds total process). Environment: Python3.14,
Torch2.12.0+cu130, one CPU thread, `CUDA_VISIBLE_DEVICES=-1` set only inside
that test process before importing Torch. No CUDA capability probe, model,
official data or GPU operation ran. These results establish the tested CPU
value/state equivalence cases only; Linux CI, CUDA equivalence/timing and any
future formal admission are not claimed.
