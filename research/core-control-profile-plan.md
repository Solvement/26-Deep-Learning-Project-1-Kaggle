# Single diagnostic: WRN34-3/300 native-BF16 proof costs

Issue 7 / PR 8. This is a predeclared engineering diagnostic, not training
admission, an accuracy experiment, a prediction or an upload. The completed
unprofiled preflight remains negative for admission: its 0.0888533500-second
median gives 12,814.994 seconds under the unchanged fixed guard. Static source
inspection suggests repeated finite reductions and scalar reads; it does not
establish the actual bottleneck or promise an improvement.

Use only the existing `wrn34_3_300_bf16` primary-seed2026 configuration, official
40k training pool, source batch128, native BF16 and FP32 master/EMA parameters.
The frozen `preflight_core_control.py` source is pinned to
`85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761`.
Call its unchanged `core_step` for three warmup and twenty timed updates. Retain
the real shadow optimizer helper with disabled scaler/scale1, the original
schedule, augmentation, MixUp, SGD, EMA, complete gradient/state/optimizer
finite checks, parameter/EMA probes and every BN-counter proof. Do not optimize,
remove or move proof work outside timing. There must be23 actual successful
updates, zero skips,31 BN layers and4,166,298 total/trainable parameters.
Run the three complete warmup steps before entering the profiler, then capture
exactly twenty complete steps. Both windows call one unchanged local step body;
all23 steps retain their whole-step timer and count toward the supervisor's
reservation. Do not proportionally subtract warmup costs from an aggregate.

The new entry authenticates its own source, manifest, process identity,
bootstrap arguments, nonce and actual native Job/session membership before
runtime/Torch/data imports. Use the canonical local lease by exclusive creation;
an existing or unknown lease is never reconciled automatically. The complete
reservation is180 seconds, charged to the same6h UTC-day budget and campaign
stop2026-11-05 05:00UTC. Verified cleanup is required before lease release.
Current training has priority; implementation and tests do not authorize GPU
execution. Root reviews source/CI and resource availability before one run.

Instrument only existing boundaries: probe snapshots, BN-before, the unchanged
core step, state/probe proofs and BN-after. Keep the original whole-step timer,
including every proof and final synchronization. CPU+CUDA profiler activities
and record-function overhead are additional work. Save bounded key-average
operator/range aggregates, timed wall aggregates, hardware/package hashes and
observed positive-duration CUDA-activity counts. Capability is not evidence of
captured device activity; an empty capture explicitly disables device
decomposition. CPU inclusive ranges overlap and are not additive GPU costs.
Operator aggregates and reported wall aggregates both cover only the twenty
post-warmup steps. Each of the five named CPU ranges must have exactly twenty
captured calls, independently of whether CUDA activities were captured. This
prevents cuDNN autotuning and first-step allocation from being attributed to the
steady-state operator breakdown. Warmup still performs all original proofs.
Never export traces, tensors, images, labels, IDs, predictions or weights.

Every result states `formal_admission_evidence=False`; there is no training
launch, modified threshold or calculated admission decision in this entry.
If profiling supports a proof-batching hypothesis, it needs separately reviewed
new source, numerical/proof equivalence tests and a fresh uninstrumented actual
23-step preflight. Preserve all existing source and negative measurements.
