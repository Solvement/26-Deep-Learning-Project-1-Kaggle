# TA exact paired replication: independent repair1

Issue7 / draftPR8. Recorded2026-11-02 after the first actual bounded pair failed.
This new entry/plan/schema supersedes v1 for future pairing/freezing; v1 source,
plan, tests and failed private receipt remain byte-for-byte unchanged. There
was no successful v1 pair or primary freeze, and no seed27 result is available.

## Evidence and limited diagnosis

The actual `ta_dualpair_20261102_v1` stopped after21.808958s with authenticated
worker exit1, verified OSJob cleanup and no paired result. The error from the
frozen `collect_correctness` merged two conditions: pixel fingerprint mismatch
or accuracy mismatch. It did not identify a comparator/case or print its actual
cuDNN environment. Therefore neither a particular failed case nor an actually
loaded wrong DLL is established by that log.

One implementation error is established statically: the new v1 worker imported
Torch before `runtime`, violating frozen `runtime.py`'s Windows requirement to
remove the NVIDIA pip cuDNN PATH entries before Torch first loads. The original
TA benchmark and old paired-incumbent worker follow runtime-first order. This
repair corrects that ordering; causation of the observed mismatch remains a
hypothesis until the new actual measurement.

## Minimal change and unchanged acceptance

`research/trivialaugment_paired_v2.py` has its own explicit supervisor/worker/
freeze schema, script/argv/nonce/source identity and PLAN. It imports runtime
first, then Torch, only after early OS/lease/budget/input authentication. It
checks the actual `runtime.environment()` against the source-bound primary's
environment: Python executable/version, platform, all recorded packages,
device, CUDA, cuDNN, GPU/memory, native AMP, cuDNN benchmarking and deterministic
flags must match. Only the nonnegative process-local removed-PATH-entry count
may differ. The frozen primary records Torch2.12.0+cu130/CUDA13/cuDNN92000,
RTX5060Laptop/nativeBF16, benchmarkTrue/deterministicFalse. No precision or
algorithm flag is changed to obtain a result.

The per-case collection uses the unchanged frozen stress/infer functions and
the same ordered5000 source IDs, B128, six cases, untouched BN, logits argmax
and original exact pixel/accuracy checks. A minimal independent collector adds
private diagnostics before failing: comparator, policy, case, expected/actual
pixel SHA256, integer correct counts and accuracies. It never writes flags,
source IDs, images or prediction rows. The original accuracy tolerance remains
1e-12; a differing pixel or correct count causes failure, no pair/freeze and no
repeat. Diagnostic aggregates do not substitute for qualification.

Both actual comparisons remain distinct: candidate weighted versus fixed
WRN300 weighted, and candidate weighted versus trusted original baseline flip.
Each uses10000 original-image bootstrap draws with the five stress views kept
together. Complete intervals/discordance are reported even if a CI crosses0;
the old robustness-driven legacy confidence condition remains unchanged. No
new positive-clean-CI condition is introduced.

## Execution, freeze and resources

```text
python research/trivialaugment_paired_v2.py pair --run UNIQUE_V2 \
  --candidate-checkpoint results/research/wrn34_trivialaugment_bf16_s2026/best.pt \
  --candidate-benchmark results/research/wrn34_trivialaugment_bf16_s2026/benchmark.json
python research/trivialaugment_paired_v2.py freeze --paired-dir results/research/UNIQUE_V2 \
  --decision PRIVATE_DECISION.json --output PRIVATE_FROZEN.json
```

The sole candidate still has checkpointdcfe2135...f0fba/epoch159EMA and benchmark
47cfe0eb...b205. Original complete typed selected/last/SGD/RNG/history validation,
all18 benchmark-pixel/source/config checks, old weighted plus predeclared
incumbent mean/per-case gates, same comparator identities and no confirmation
or hidden-image access all remain mandatory. The original scalar screening
was not rerun under a modified gate.

The actual three-model/dual-bootstrap job keeps one canonical lease,
concurrency1,180s cap,6h/UTCday,3h/job and Nov5 05:00UTC hard stop. Unknown cleanup
retains its lease. This is repair1; root launches at most one reviewed actual
v2 attempt after CI and verifies receipt/cleanup. A remaining mismatch gets
its concrete diagnostic investigated; no blind retry, tolerance relaxation,
new benchmark overwrite or longer reservation follows automatically.

New `research_trivialaugment_primary_freeze_v2` and stdlib-only
`validate_frozen(path)->dict` preserve the explicit primary26/planned27 fields
from the v1 plan. Freeze requires the real final v2 pair, inactive original
owner/child/watchdog identities and completed authenticated OS cleanup, actual
source/evidence hashes, exact seed27 raw config9686ddea...4981 and effective
config, and the complete *actual final* repeat source map. The map includes
this new v2 entry/plan, unchanged learning/benchmark/guard dependencies and the
new repeat entry/plan. No source hash is guessed; no hardcoded hash cycle is
introduced. Freeze precedes every seed27 preflight/formal attempt; later
validation permits the repeat folder to exist.

This entry has no GPU prediction, uploader or full50k path. Only a separately
validated fresh27 repeat and explicit TA delivery support may qualify the
already-frozen primary26 for later hidden prediction. The original uploader
allowlists, official-ID/quota/hash/duplicate/unknown-result protections remain
unchanged. No Kaggle-score improvement is claimed.
