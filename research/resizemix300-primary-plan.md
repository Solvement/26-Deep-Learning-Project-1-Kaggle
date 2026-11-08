# WRN22-4/300 ResizeMix native-BF16 primary, Issue7 / PR8

Predeclared2026-11-02. The160-epoch ResizeMix primary scored0.875 publicly,
below the incumbent WRN34-3/300 at0.878. Its matched local noise accuracy is
6.88 percentage points below that incumbent. A longer schedule might improve
generalization, but cannot be assumed to repair that regression. Rejected SE,
AugMix and WRN34 timing results remain rejected; do not rescue them by renaming
an artifact, changing a gate or shortening its required proof work.

## Fixed method, lineage and prospective decision

The new raw primary config `configs/wrn22_resizemix_300_bf16.json` changes only
epochs160->300 from the existing parsed `configs/wrn22_resizemix.json`.
`configs/wrn22_resizemix_300_bf16_seed2027.json` differs only in seed2027; freeze
its exact raw SHA before primary training. Both use official splitseed2026,
40k train/5k dev, WRN22-4/dropout0/4,298,970 parameters, batch128, original
SGD/schedule/EMA, probability0.5 ResizeMix with uniform side ratio0.1-0.8,
bilinear full-donor resize and shifted full-fit rectangles. MixUp/Cutout remain
disabled. Training precision is explicitly native BF16 with FP32 masters/EMA
and disabled GradScaler1; precision is a supervisor argument/receipt field,
not an extra key rejected by the frozen trainer config validator.

Relative to the scored160-epoch T4/FP16 experiment this new recipe also changes
training precision/platform to local native BF16, with no GradScaler overflow
skips. Only epochs differ in the parsed config, not in the entire training
protocol. A matched local-BF16 evaluation does not make the training protocols
matched. Attribute any result to this complete new exploratory recipe, not
solely to longer training. Record actual source/config/precision/environment;
the future independent27 must use this same new BF16 protocol. No additional
160-epoch control is required to execute the proposed recipe exploration.

The new `run_resizemix300_primary.py` has distinct supervisor/input/preflight/
primary-result schemas and roles. It admits only fresh primaryseed2026. It
does not masquerade as a native repeat, change the original ResizeMix marker,
load old weights, resume partial runs or create a delivery authorization.
Original stability2027 and every repeat remain excluded from hidden inference.

Apply **all** [incumbent promotion conditions](incumbent-promotion-plan.md):
complete typed5000dev/all18 cases/nativeBF16, fixed weighted old-baseline gate,
same-policy incumbent mean/clean gate, **each stress delta>=-0.01**, actual
paired-original-ID10000-bootstrap evidence, primary/policy/exact27 freeze, then
fresh scratch27 and the same gates. Noise is reported separately. Failure at
the scalar screen stops pairing, repeat and submission; no policy or threshold
rescue. Scalar gate arithmetic uses the exact integer correct counts of each
5000-image case (the five stress counts sum over25000 views), preserving
inclusive threshold equality without floating subtraction artifacts. This does
not treat those views as statistically independent. A completed primary is not
a qualified uploader input. The existing
native repeat may support the later27 training only after its required freeze
and a separate audit of the300 recipe. Its future native27 scalar screen needs
an explicit typed extension because the current comparator is primary-only;
old160 repeat evidence cannot substitute.

## One controlled adapter, two explicit modes

No arguments show help. `--mode preflight --run FRESH` performs only a bounded
<=180s actual-image engineering test. `--mode formal --run FRESH --preflight
PATH` requires the complete immutable result and stopped/clean supervisor of
this exact new preflight, unchanged source/config/data and matching native
hardware/runtime. Never allocate or launch formal automatically after preflight.

Before runtime/Torch imports, the worker validates its own script hash,
schema/role/mode, actual PID creation times, parent/child argv, unique nonce,
bootstrap and native OSJob/session membership, canonical exclusive lease,
reservation/deadline, all source/config/data bindings and primaryseed2026.
Reuse frozen accounting/official-input/OSguard primitives; do not invoke an old
worker validator. Unknown or unverified cleanup retains the matching lease;
fresh-only directories preserve every attempt. Concurrency1,6h/UTC-day,3h/job
and the Nov5 05:00UTC hard stop remain fixed. No paid/cloud resource is used.

The preflight keeps all official40k images resident, sourcebatch128,3 warmup
plus20 timed successful SGD/EMA steps (2944 unique training rows). Original
`prepare_batch` and original `optimizer_step` execute directly; its62 actual
per-parameter finite-gradient checks and disabled scaler are not removed.
The loss/SGD/EMA/metric/RNG order matches the frozen ResizeMix batch-loop AST.
Additional proofs check all gradients/positive finite norm, real parameter/EMA
changes, complete model/EMA/momentum finite state and all19 BN increments.
WRN22 has62 parameter tensors,119 state tensors per model and300 total proof
tensors:262 FP32+38 scalar int64. Validate actual names/shapes/dtypes/device and
momentum ownership. Use frozen batched finite/BN primitives; they add no math
or RNG and never replace finite checks by a norm.

Every actual batch verifies unchanged labels, valid donor permutation, full-fit
same-area square rectangles and retained weight exactly `1-area/1024`, or the
complete unmixed identity. Record per-step mixed flag/area/retained weight and
all BN before/after counts, not labels/IDs/pixels. Fixedseed2026 and the original
sampler must execute mixing without forcing or resampling a branch; zero mixed
batches reject the preflight. Record the actual23-step total instead of a
hardcoded predicted mixed count. Geometry, allocations, state/gradient/norm/probes and
all synchronization stay inside whole-step timing. Record all23 durations,
20-step median, setup, allocated/reserved peak memory, source/config/data and
environment hashes. Do not save preflight tensors/weights or reuse its state.

Formal admission uses only that fresh uninstrumented measurement and the
unchanged formula `1.5*median_step_seconds*313*300 + 300`, including300 seconds
for finalization. It must fit actual remaining day budget,10800s cap and hard
stop. No reduced safety factor, omitted proof, synthetic/old timing, split job
or implicit restart may bypass rejection. WRN34-300's failed timing is not
evidence about this different model, and this preflight is not a quality result.

After admission, the same authenticated OSJob worker calls the **unchanged**
`train_resizemix.train(..., precision="bf16", resume=False)` from fresh scratch,
then the unchanged `benchmark_resizemix.main` with all3 policies and full5000
dev, original BN, batch128 and the pinned original native-v3 baseline. No
module builder, marker or validator is monkeypatched. A new result links the
primary role, preflight/receipt, actual checkpoint, complete300 history/93900
successful updates/zero skips, original typed ResizeMix integrity and benchmark.
Scalar incumbent comparison is recorded under the predeclared screen, with no
automatic pairing/repeat/prediction/submission.

## Required verification before actual GPU execution

Stdlib tests cover raw/effective config+seed-only repeat, own identity/lease,
budget/preflight-source linkage, strict reports/geometry, complete benchmark
and gate boundary behavior. CPU tests independently execute the original
ResizeMix batch-loop AST and compare model/EMA/SGD/BN/metrics and Python/NumPy/
Torch RNG with the new engineering step. Real CPU model metadata must confirm
62/19/119/300 counts. No official data or CUDA is needed. Ask root for an idle
GPU window before local Torch tests, then use CUDA_VISIBLE_DEVICES=-1 only in
that process. Root alone launches the actual GPU preflight after source review
and CI; until then no throughput, admission, training or score is claimed.

## Actual pre-GPU validation, 2026-11-02

Root reserved an idle/no-lease window. Three CPU tests passed in3.486s
(3.549367s including discovery), with one CPU thread and CUDA_VISIBLE_DEVICES=-1
set only inside that process before importing Torch. Four original ResizeMix
batch-loop AST executions matched the new engineering step exactly in model,
EMA, SGD, BN, accumulated training metrics and Python/NumPy/Torch RNG. The
actual CPU WRN22 constructor confirmed4,298,970 parameters,62 parameter
tensors,19BN,119 state tensors/model and300 full-state entries (262FP32+38int64).
Geometry tests rejected invalid donor permutations, non-square rectangles and
wrong retained weights. No official data or GPU was used.

Twelve separate stdlib tests passed in0.384s without importing Torch/runtime.
They include own OS identity before imports, fixed config/seed/source, strict
native metadata and timing types, complete geometry/inventory rows, integer
gate boundaries, unaltered hard stop for both modes, source-bound cleaned
preflight/complete elapsed accounting, active-owner rejection/PID reuse,
existing-lease refusal and unverified-child cleanup retaining the lease.
The copied test fixture initially used a shifted campaign clock; it was
corrected to one controlled wall clock and the exact original deadline. No
production threshold or budget was relaxed to pass. These results establish
CPU/control correctness only; root review, Linux CI and a fresh actual native
GPU preflight remain required before formal admission.
