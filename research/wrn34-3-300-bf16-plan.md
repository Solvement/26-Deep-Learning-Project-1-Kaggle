# Local native-BF16 WRN34-3/300 exploratory control

Issue 7 / PR 8. This is a planned, independent 40k training recipe, not the
already scored T4/FP16 primary or its full50k refit. No new GPU timing, benchmark,
checkpoint, prediction or score exists for this local BF16 combination.
Healthy NativeBN training continues first. Do not load confirmation or hidden
images for this experiment.

The explicitly named BF16 [primary configuration](configs/wrn34_3_300_bf16.json)
(rawSHA256 `cc55380bd03bd0ca2147c7e4a34c31de37345480d71a9721b65737717fc2f348`)
and [planned seed2027 repeat](configs/wrn34_3_300_bf16_seed2027.json)
(`4ce28d4d6f479addb1d471b21c38bc1ef2ae204256c04b1fbd2f2aed7c1b5363`)
have exactly the same parsed hyperparameters as their existing FP16 recipe
counterparts. They differ only in training seed. The old primary's frozen raw
CRLF bytes (`73b6f816a5a1c19ed9fa458c151133c6003e2bff40899df6e9848e0a3b943b85`)
remain unchanged; the new BF16 files use explicit LF for identical Git/Windows/
Linux bytes. This does not change any old checkpoint, schema or allowlist or use
an FP16 weight as BF16 evidence. The existing loader supplies unchanged
DEFAULT_CONFIG fields; raw and effective configurations must both be recorded.
Fix architecture34-3,4,166,298 total/trainable parameters, dropout0, epochs300,
sourcebatch128, official40k/5k splitseed2026 and the existing SGD/schedule/MixUp/
Cutout/EMA/selection rules. Use the unchanged shadow trainer with explicit
nativeCUDA BF16, no emulation. Its GradScaler is disabled (scale1); the actual
optimizer_step API takes that disabled object, not None. Nonfinite loss or
gradients are a failure. Exactly313 attempted updates per epoch (tail64) gives
**93,900 successful updates and zero skips**, with12,000,000 source examples.

The existing T4/FP16 primary scored0.878, but cannot stand in for this experiment
or its repeat. In addition to precision, its GPU/Torch environment and36 skipped
updates differ. Do not attribute a difference solely to precision or forecast
improvement. The prior local34-3/160 BF16 run is useful planning context;
doubling or scaling its timing does not admit this300-epoch recipe. The old
confirmation result and all original weights/receipts remain frozen.

Use the new `preflight_core_control.py` engineering entry with explicit recipe
`wrn34_3_300_bf16` and seed2026. This is still subject to source/CPU review and a
real complete official-image preflight; a controller or static compatibility
finding is not measurement. Require full40k training residency, nativeBF16
logits/FP32 master parameters,31 BN layers with one forward statistic update,
finite positive gradients and real SGD/EMA updates. Three warmup plus twenty
timed sourcebatch128 steps include the original augmentation, mixing, loss,
backward, optimizer, EMA and proof costs. No development/confirmation evaluation
or hidden inputs are allowed in preflight. Discard its state and reseed formal
training from scratch. The fixed admission guard is
`1.5 * median_step_seconds * 313 * 300 + 300`, with matching raw/effective config,
source, official data, physical GPU and runtime; it must fit actual remaining
daily budget, the3h job limit and campaign stop2026-11-05 05:00UTC.

After actual admission and a free verified lease, the existing entry may run a
fresh `wrn34_3_300_bf16_s2026` via `run_local.py --trainer shadow --precision bf16`.
Do not start it while another local GPU job is live. Preserve all original
source files and use an immutable new experiment directory. The existing
benchmark_shadow path requires300 full epochs, the precise data/context/source
chain and zero BF16 skips. Report all5000 development examples, clean/NLL, five
stress cases and all three policies. Weighted `center80_shift20` is fixed as the
primary policy; the campaign gate is unchanged.

Only a completed qualifying primary proceeds to paired-v2 original-image
cluster10,000-bootstrap evidence and a **new** freeze-v2 binding its checkpoint,
epoch/variant, weighted policy and exact planned seed2027 configuration/source.
The new repeat must start from scratch with the same local BF16 protocol and
complete the same fullv3 benchmark before native-repeat validation. An old T4
FP16 checkpoint/repeat is rejected by the precision marker and is not reused.
Only the newly frozen primary may then predict official hidden images through
predict-v2 and the existing strict uploader, subject to real quota, content/ref
deduplication, intent/unknown protection and the hard stop.

The report currently has a separate row for the old FP16 WRN300 result. Future
BF16 evidence needs its own explicit row/type and must not replace0.878, the old
primary's provenance, or any original study result. The existing FP16 full50k
refit contract does not automatically support this new BF16 parent. Any future
full-data extension requires its own valid parent/refit relation and zero
validation selection; no overlapping development accuracy is independent.
