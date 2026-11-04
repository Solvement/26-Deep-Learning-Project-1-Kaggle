# Fixed-standard TrivialAugment primary controller, Issue7 / PR8

Registered2026-11-02 before actual TA preflight or formal training. This
independent controller has its own source, worker identity, supervisor/input/
preflight/primary schemas and roles. It reuses the separately audited TA
trainer's existing `train_step`; it does not replace a model builder, marker,
optimizer or old validator. No resume, repeat, prediction or submission path.

## Recipe and interpretation

Freeze the exact primary and planned27 raw configurations from the trainer:
`configs/wrn34_3_trivialaugment.json` and its `_seed2027.json` counterpart.
They differ only in training seed. Relative to the parsed original WRN34-3/160
parent they add only fixed-standard TA with31 strengths. Architecture34-3,
4,166,298 total/trainable parameters,160 epochs,40k/5k splitseed2026, batch128,
original crop/flip/normalization/Cutout/Mixup, SGD/schedule/EMA and selection
remain fixed. Training is explicitly native BF16 with FP32 masters/EMA and
disabled GradScaler1. The new typed TA marker must never be represented as
ordinary shadow training or a native repeat.

The standard14-operation registry and author attribution/license are in the
component/training plans. Every source image receives one sampled PIL operation
on raw32x32 RGB CPU uint8 pixels before the original GPU augmentation. An
independent NumPy PCG64 generator is seeded2026; it does not replace or reseed
the original Python/NumPy-global/Torch streams. Evaluation has no TA. This is
new recipe exploration, with no assumed hidden-score gain and no tuning of an
operation/strength or noise-specific policy after observing a benchmark.

## Controlled preflight and timing

No arguments do nothing. Explicit `--mode preflight --run FRESH` reserves the
complete180 seconds under the canonical exclusive GPU lease; it never launches
formal automatically. The worker authenticates its own script/source, argv,
manifest, role/mode, parent/child PID creation times, unique nonce, actual
bootstrap and native Job/session membership before runtime/Torch imports.
Source/config/official-input/baseline/incumbent hashes must remain stable.
Unknown owner, allocation or cleanup fails closed and retains the lease.

Stage the exact official40k CPU uint8 pool and device labels, construct a fresh
WRN34-3, EMA, original SGD and disabled scaler, then the same GPU randperm as
the trainer and an independent PCG64 stream. Run3 warmup+20 timed successful
real updates. Every whole-step timer includes GPU-index-to-CPU transfer,
CPUpool gather, every PIL operation, batch packing, H2D transfer, original
core augmentation/Mixup, the trainer's complete `train_step` with its98 original
per-parameter gradient checks, SGD, EMA, original accumulated metrics and all
extra proofs/allocations/synchronization. Nothing is prefetched out of timing.

Check real finite-positive gradients/norm, first-parameter/EMA changes, all31
BN increments and the complete480-state inventory (418FP32+62scalar-int64:
191model+191EMA+98momentum tensors), with names/shapes/dtypes/device and SGD
ownership. Frozen batched finite/BN helpers preserve full-element checks and
exact counter arithmetic. Record all23 timings,20-step median, setup, peak
allocated/reserved CUDA memory and complete native environment metadata.

Record every operation count for every step using all14 registry names, each
batch total128 and aggregate total2944, plus independent RNG-state hashes.
Zero counts for an individual operation are legitimate observations; do not
force an operation, resample or alter PCG64. Never persist pixels, labels,
logits, permutations, weights or raw RNG state from preflight. Its state is
discarded, with no formal-state marker or checkpoint.

Keep the guard `1.5*median_step_seconds*313*160 + 300`, including300 seconds
finalization,10800s/job,21600s/UTC-day, concurrency1 and the fixed
2026-11-05 05:00UTC hard stop. Use a new uninstrumented actual-image preflight,
not paper speed, synthetic timing, old experiments or an instrumented profile.
Rejection cannot be bypassed by changing the multiplier, deleting proofs,
shortening the run, split/resume or moving CPU/PIL/gather work outside timing.

## Formal primary and promotion boundary

Explicit `--mode formal --run OTHER --preflight PATH` requires this entry's
complete immutable cleaned/stopped preflight, exact unchanged source/config/
data, full timing coverage and matching native hardware/runtime. Only seed2026
is runnable here. Under the same independently authenticated OSJob, call the
unchanged new TA trainer from fresh scratch with explicit BF16 and exact raw
config bytes, then its typed full5000/all18-case benchmark with original BN,
batch128 and pinned original baseline. The trainer supplies the new TA marker
and independently audits formal saved tensors, one-view/update/operation/RNG
accounting; this controller cannot substitute a shadow marker.

Record actual160 epochs/50080 successful updates/zero skips,6.4M sources/views,
all14 actual operation totals, checkpoint/source/config/benchmark hashes and
own primary role. Apply the unchanged fixed-weighted baseline gate and every
[prospective incumbent condition](incumbent-promotion-plan.md), including each
stress delta>=-0.01. Scalar thresholds use verified5000-image integer counts
to preserve inclusive equality, not arbitrary rounded accuracies. A failed
scalar screen stops pair/repeat/prediction/submission. Passing only permits
root review for required paired-original-ID evidence, frozen primary/policy/
exact27, same-protocol fresh27 and explicit typed delivery support. There is
no automatic promotion or upload. Current primary-only scalar tools are not
silently reused for a later27; that route needs its own explicit audit.

## Verification status and execution boundary

Controller tests must cover own early identity/nonce/argv/lease, strict complete
report/context/count/RNG hashes,180s/3h/6h/deadline/source-bound admission,
preflight-state rejection, scalar boundaries and no automatic launch. Linux
CPU tests independently execute the trainer's actual batch-loop AST against
the wrapper, comparing model/EMA/SGD/BN, accumulated metrics, operation totals,
PCG64 and all original RNG streams. Injected NaN/Inf, malformed registry counts
and wrong RNG state must fail. Original trainer/component numerical tests
remain separate evidence. No local Torch or GPU test is authorized by writing
these files. Root owns integration, review, CI and every actual GPU launch.

## Actual controller validation before integration

On2026-11-02,12 stdlib contract tests passed locally in0.494s (0.572127s
including discovery). The test process confirmed neither Torch nor runtime
was imported. These checks cover fixed registry/config/source, early own OS
identity, all14 operation totals and chained RNG hashes, strict full typed
reports/context, the unchanged160 guard/day/job/deadline, cleaned immutable
preflight with stopped-owner/PID-reuse checks, unknown cleanup retaining the
lease, and exact inclusive incumbent count thresholds. AST checks verify that
CPU gather and the complete original train_step execute inside the synchronized
whole-step timer, and formal uses exact raw config bytes and the typed TA
benchmark. No local Torch, GPU, formal training or native timing ran.

Two separate Torch CPU test methods are written for Linux CI: actual trainer
batch-loop AST versus the wrapper compares model/EMA/SGD/BN/metrics/operation
counts, PCG64 and parent RNG; injected NaN/Inf loss, malformed registry and
nonadvancing PCG64 state must fail. They have **not been run locally** and their
mere presence is not a pass claim. New-controller CI and actual native
preflight/admission remain pending root integration and execution.
