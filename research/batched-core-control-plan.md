# Uninstrumented batched outer-proof WRN300 control

Issue7 / PR8, 2026-11-02. New independent engineering entry:
`preflight_batched_core_control.py`. No formal training, accuracy, checkpoint,
prediction or submission is produced. The script does not launch a trainer or
authorize formal admission. Preserve every old source, gate and receipt.

The completed v2 profile
`results/research/wrn300_core_profile_20261102_v2/profile_summary.json`, SHA256
`1aca18a6632bbecb20888ff0e2d1e6cb8b2c2f0495f65a11e12c9966c7455af4`,
has20 captured real updates,77,828 CUDA launches,13,560 isfinite calls and3,420
local scalar extractions. The instrumented outer state/probe range median was
151.58ms and core-step range median160.59ms. These sequential boundary wall
ranges include instrumentation and asynchronous waits; they are not an
uninstrumented or causal speed estimate. Nested operator/CUDA annotations must
not be added to those wall ranges. The evidence supports a new measured proof
schedule, not a claim of formal speed improvement.

## Fixed experiment and allowed difference

Use only `wrn34_3_300_bf16` primary seed2026 with the exact existing raw config
SHA256 `cc55380bd03bd0ca2147c7e4a34c31de37345480d71a9721b65737717fc2f348`.
WRN34-3 has4,166,298 total/trainable parameters,98 parameter tensors and31 BNs.
Use the complete resident official40k training pool, batch128,3 warmup plus20
timed successful real updates,2944 unique official source IDs, native BF16
(emulation excluded), FP32 model/EMA/SGD momentum and disabled GradScaler1.
Do not read development/confirmation/hidden images for an accuracy evaluation.

Call the original `preflight_core_control.core_step` unchanged, with its exact
arguments and `future_trainer="shadow"`. Its source SHA256 is
`85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761`.
The original loss, augment/MixUp RNG, SGD, EMA, per-parameter shadow gradient
finite checks, subsequent full gradient finite/norm checks and metric/probe
math remain intact. Do not remove the98 original shadow checks or replace a
norm, optimizer, trainer or model primitive.

Only two outer proofs change scheduling. Full model+EMA+momentum finite checks
use `batched_finite_proof.all_finite`; before/after BN counters use its stacked
clone and one paired CPU transfer, retaining all per-layer values and exact
Python integer increment comparison. That frozen helper's SHA256 is
`c7bb0c161fab8b3a0361c1179f44d59d991428dd9d7bc8485f28d7cd61ee59ff`.
Its separate CPU equivalence suite covers NaN/Inf, maximum finite floats,
noncontiguous/empty/mixed formats, no mutation and BN alias/overflow boundaries.
This worker additionally proves the full expected inventory every step:
191 model state tensors,191 EMA tensors and98 momentum buffers,480 total,
418 FP32 plus62 scalar int64 BN counters. Names, shapes, dtypes, device and
optimizer ownership/keys must match the fresh initial model-derived template;
an empty/partial/extra inventory is rejected. All allocations, metadata
inventory checks, elementwise finite proofs and BN/probe proofs stay timed.

## Identity, timing and interpretation

The new script has its own supervisor/input/result schemas. Authenticate its
own actual script/source, worker argv, nonce, bootstrap, native Job/session
membership, PID creation times, live canonical lease,180s reservation/deadline,
config/source/data hashes and existing profile evidence before importing
runtime, Torch, the batched helper or official arrays. Do not delegate identity
to the old worker validator or masquerade as an old preflight. Parent uses
exclusive fresh lease/folder creation and the frozen accounting/OS guard
primitives; unknown ownership/cleanup retains the lease for reconciliation.

No profiler is used. Each of23 steps starts/ends with actual CUDA synchronization
around the entire original core step and every original/new proof. Three
warmups remain in resource accounting; report the20-step median, all23 timing
rows, setup and full supervisor elapsed time, all BN before/after counts and
real finite-positive gradient/parameter/EMA changes. Record allocated and
reserved peak CUDA memory; concatenation storage is a measured cost.

Compute the unchanged conservative formula
`1.5 * median_step_seconds * 313 * 300 + 300`.
Keep the3h/job,6h/UTC-day,one local GPU lease and2026-11-05 05:00UTC hard stop.
The <=180s experiment requires its entire reservation to fit. A successful
engineering report does not start or admit formal training; root must audit
all evidence and remaining budget first. CPU or instrumented profile timings
cannot substitute for this fresh uninstrumented measurement. No formal speed
or guard pass exists before an actual completed receipt is reviewed.

## Pre-execution checks

Use stdlib contract tests for own identity/source/schema/lease/budget handling,
strict complete rows/inventory and unchanged guard formula. AST checks pin the
original core-step call, single23-step loop, complete inventory, no profiler
and authentication before runtime/Torch. Dedicated CPU numerical tests execute
the old outer loop's AST independently against the new wrapper on synthetic
inputs, comparing model/EMA/SGD/BN and Python/NumPy/Torch RNG exactly. They also
reject missing/extra/mistyped inventory and injected NaN/Inf while preserving
the helper's int64 overflow and maximum-finite boundaries. Local Torch tests
need root's GPU-idle confirmation and per-process CUDA_VISIBLE_DEVICES=-1;
otherwise leave numerical execution to Linux CI. Preserve failures as failures.

## Actual CPU/stdlib verification before any GPU run

On2026-11-02, root confirmed an idle GPU/no lease and deferred other jobs for
one CPU test process with CUDA_VISIBLE_DEVICES=-1 set before Torch import.
Four numerical tests passed in4.933s (5.004618s including discovery), using
one CPU thread and no official inputs. Independently executed original outer
loop AST and the new wrapper produced identical model/EMA/SGD/BN state and
Python/NumPy/Torch RNG across four synthetic steps. Actual CPU WRN34-3 metadata
confirmed98 parameters/191 state tensors per model and480 combined entries;
empty initial momentum, missing/extra/mistyped state and injected NaN/Inf were
rejected, while maximum finite FP32 was accepted. Ten separate stdlib tests
passed in0.384s without importing Torch/runtime, including own OS identity,
lease/budget, complete reports, non-bool finite timing and full native-context
metadata. These are CPU correctness checks, not CUDA equivalence, timing,
formal admission or training results. Root must review Linux CI and this new
source before the separately authorized bounded GPU preflight.
