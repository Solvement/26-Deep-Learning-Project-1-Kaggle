# One evidence repair: explicit CPU ranges in WRN300 diagnostic v2

Issue7 / PR8. The first diagnostic completed its23 proof-wrapped steps but
failed the exact-five-CPU-ranges assertion after profiler shutdown. Its log
does not contain the actual aggregate range identities, so the root cause is
not established. Preserve the v1 source, plan, negative receipt and log.
The v1 source SHA is
`2399ac392a30066409b993166f387321ee47b2aaea69bf5f8ccaf210df089878`.

This new entry owns its own schema, plan, source snapshot, script/argv and
actual native OS Job/session validation before runtime or Torch. Retain the
v1 learning body exactly: official40k, WRN34-3/300 primary2026, B128, native
BF16, FP32 model/EMA, disabled scaler1, same RNG, augmentation, MixUp,
schedule, SGD/EMA and complete gradient/model/EMA/optimizer finite and
parameter/EMA/31-BN proofs. All23 steps stay whole-step timed, with three
complete warmups outside capture and twenty complete steps inside capture.

The sole aggregate interpretation repair explicitly selects DeviceType.CPU
events before producing the five instrumented CPU range rows. No aggregation
across device types, invented counts, dropped CPU duplicate rows or relaxed
assertions is allowed: exactly the five named CPU ranges, each20 calls,
remain required even if device activities are unavailable. Save up to30 actual
range aggregate identities (device type, controlled name, count) separately,
plus total/truncated counts, so another failure is inspectable. This does not
prove that mixed device range annotations caused the first failure.

Before validating a complete result, save only a separately allowlisted scalar
diagnostic envelope with status unvalidated. If result validation fails, retain
that envelope as failed with exception type only; no complete summary is
written. Successful validation marks the envelope validated and permits the
complete diagnostic summary. No traces, tensors, shape/stack data, images,
labels, row IDs, predictions or model weights are exported.

Retain the original exclusive local lease, full180-second reservation, UTC-day
6h ledger, campaign hardstop2026-11-05T05:00:00Z, exact process creation times,
nonce, source/config/data binding and verified cleanup before lease release.
Unknown cleanup keeps the lease. No current artifact path is reused. Pure
contracts and a CPU-only real-profiler check precede root review and CI; at
most one new real GPU attempt is authorized for this evidence repair.

Every summary remains diagnostic_only=True and
formal_admission_evidence=False. Profiler timings never admit formal training.
Any subsequent proof-cost optimization requires new independent source,
equivalence/boundary checks and a fresh uninstrumented actual preflight.
