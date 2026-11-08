# Matched incumbent original-image bootstrap — Issue7 / PR8

This new diagnostic uses its own script/schema/nonce/argv identity. No existing
helper, uploader allowlist, source marker or result is modified. It creates
conditional paired development evidence only, never new delivery eligibility.

## Fixed comparison

Reference is frozen primary WRN34-3/300 checkpoint
`53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`,
at `results/research/colab_wrn34_3_300_fp16_s2026_v3/candidate/best.pt`.
Its complete **local** benchmark is
`results/research/wrn34_3_300_s2026_local_v3/benchmark.json`, raw SHA
`e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`.
Both models use `center80_shift20`, native CUDA BF16/Torch2.12.0+cu130,
batch128, fixed official40k/5000dev/split2026. No T4 FP16 benchmark mixing.

Existing pair runs measured32.22–48.72 seconds under180-second reservations.
The new run likewise requires the complete180 seconds within canonical local
concurrency1, UTC-day21600 seconds, job10800 seconds and hard stop
2026-11-05 05:00UTC. An existing lease or artifact stops launch. Unknown cleanup
retains the matching lease and requires reconciliation. No automatic retries.

Parent and early worker use only stdlib/psutil. Reuse frozen
`preflight_core_control.accounting`, `official_inputs`, atomic `claim_lease`,
`bounded_seconds`, `lease_can_release`; generic `process_tree_guard.run_guarded`
and actual `profile_augmix.verify_os_boundary`. Before Torch the new worker
authenticates parent/child PID creation times, live canonical lease, actual
Job/session membership, unique guard nonce, its own exact script/source/argv,
deadline, source/config/cache/checkpoint/benchmark/history/metrics byte hashes.
It never calls an old helper's script-specific worker validator.

## Actual inference and statistics

After authentication, reuse frozen `paired_compare_v2.load_experiment` for the
actual checkpoint/model/config/history and typed-method audit. Current typed
support remains core, shadow, SAM and ResizeMix; unknown trainers fail closed.
No KD/TrivialAugment delivery or training validator is inferred from aggregates.
Construct/load each actual single model, enforce total/trainable<=5M and finite
state, require actual native BF16, and call frozen `collect_correctness`.

Both models receive the same ordered5000 unique original dev IDs and six cases.
Every recomputed pixel hash and accuracy must match its full saved benchmark.
The two binary matrices have shape `[5000,6]` in fixed clean+five-stress order.
Call frozen `paired_bootstrap` for10000 draws, seed20261031, sampling original
image rows with all five stress views carried together. Never treat25000 views
as independent. Save hashes of ordered official IDs, development index order,
and uint8 matrix bytes; raw flags/IDs remain in memory and are not published.
Do not reconstruct pairing from marginal aggregate accuracies.

The result includes full same-policy scalar comparison, method/config/source/
checkpoint and actual input hashes, conditional95% clean/stress intervals and
discordant counts. Source/evidence stability is checked after inference and
CPU bootstrap and again by the parent. Any future cache protocol is separate.
Intervals do not measure independent confirmation, seed variance or hidden
accuracy. Original confirmation is never evaluated. Existing failed candidates
are not rescued; `automatic_promotion=false` and `delivery_authorized=false`.

Only the supervising root launches a GPU diagnostic after review and CI:

```text
python research/paired_incumbent.py --run NEW_UNIQUE_ID --candidate-checkpoint results/research/NEW_PRIMARY/best.pt --candidate-benchmark results/research/NEW_PRIMARY/benchmark.json
```
