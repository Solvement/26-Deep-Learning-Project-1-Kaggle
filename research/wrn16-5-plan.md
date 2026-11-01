# Parameter-matched shallow/wide control: WRN16-5

Issue 7 / PR 8. Predeclared future experiment, not a trained candidate. Current
healthy NativeBN training and the qualified WRN300 full-data route have priority.
No GPU time, independent development result, hidden prediction or score exists
for this architecture. No confirmation images may be evaluated.

The [original WRN paper, section 3 and Table 4](https://arxiv.org/pdf/1605.07146#page=7)
studies decreasing depth while increasing width. Its Table 4 does not report
WRN16-5, and its larger-model errors are not an expected project score.
[The author implementation](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua#L81-L95)
uses depth `6n+4` and channels `16,16k,32k,64k`, as does the frozen project model.

Static arithmetic from `model.py`, without constructing a Torch model, gives
**4,289,754 total and trainable parameters**: stem432, three stages186,112 /
820,320 / 3,279,040, final BN640 and classifier3,210. BN running statistics are
buffers. The first residual block from channels a to b contributes
`10*a*b + 9*b*b + 2*a + 2*b`; a subsequent same-width block contributes
`18*b*b + 4*b`. This is9,216 parameters (0.21438%) below WRN22-4's4,298,970,
and below the5M cap. Static arithmetic is not a native parameter/update test.

The primary [configuration](configs/wrn16_5.json) copies the existing WRN34-3/160
control and changes only depth16 and width5. Epochs160, batch128, SGD/schedule,
MixUp, Cutout, EMA, primaryseed2026 and official40k/5k splitseed2026 stay fixed.
For architecture attribution, use the same original core nativeBF16 training
protocol as the160-epoch WRN controls; do not silently substitute shadowFP16 or
native-BN AugMix. The300-epoch0.878 submission is a separate competition result,
not an equal-training-budget architecture control. Thirteen BN layers may alter
cost relative to nineteen/thirty-one, but wider activations can increase cost;
throughput is unmeasured.

Before any formal run, a new small supervised actual-image full-pipeline
preflight is still required. It must use the immutable original model/data/
augmentation/schedule primitives, official40k training inputs, native CUDA BF16
with emulation excluded, the exact configuration, finite positive gradients and
real SGD/EMA updates. Three warmup plus twenty timed successful full steps,
source/config/data/hardware/runtime hashes, parameter count and verified OS-tree
cleanup are required. Do not use the old synthetic45k benchmark as admission.
The fixed guard is `1.5 * median_step_seconds * 313 * 160 + 300`; training plus
complete benchmark must fit actual daily budget, the3h cap and campaign stop.
The actual-image preflight controller does not yet exist. A configuration file
does not authorize bypassing it, another local GPU job or the occupied lease.

Only a complete5,000-image nativev3 benchmark may qualify the primary. Preserve
clean/NLL, all five stress cases and the three existing policies, with weighted
`center80_shift20` fixed as the primary policy. Use the campaign's unchanged gate.
If it passes, conduct the existing original-image-cluster10,000-bootstrap paired
comparison; freeze primary checkpoint/variant/epoch, policy and the exact
[planned independent seed2027 configuration](configs/wrn16_5_seed2027.json)
before its repeat. A full same-context repeat must pass before only the frozen
primary may predict official hidden images. The old validation-only stability
checkpoint and all repeat weights remain excluded from hidden prediction.

Existing generic local core/shadow benchmark and delivery code supports legal
WRN depths/widths. The WRN34-3/300 cloud attestor is specific and cannot be reused
by renaming artifacts. Any unsupported new evidence type requires an explicit
audited adapter. Preserve all existing source, receipts, original study and
confirmation result; public outputs contain only source/config/aggregate evidence.
