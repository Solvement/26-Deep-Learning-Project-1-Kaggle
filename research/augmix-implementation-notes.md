# AugMix CPU transform and loss implementation

Status: **independent CPU mathematics only**. No official images, GPU preflight,
formal training, development benchmark, paired comparison, freeze, prediction
or submission was run by this implementation task. The existing registered
[AugMix plan](augmix-plan.md) remains the experiment specification.

## Verified primary sources

The algorithm is from [Hendrycks et al., AugMix](https://arxiv.org/abs/1912.02781).
Implementation mechanics were checked read-only at author repository commit
[`9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`](https://github.com/google-research/augmix/commit/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4):

- [cifar.py, transformation and loss](https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/cifar.py#L152-L236)
- [augmentations.py, operator definitions](https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/augmentations.py)
- [cifar.py, source preprocessing](https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/cifar.py#L289-L299)

The author source is Apache-2.0 licensed. This module independently expresses
the algorithm, validates new input contracts and implements a separate operator
dispatch; it does not copy or distribute upstream source, import its modules,
or execute its data entrypoints. The paper and author implementation retain
explicit attribution. No dataset was downloaded.

## Fixed image mechanics

`AugMixConfig` accepts only width 3, random depth 1–3 inclusive, severity 3,
alpha 1 and `all_ops=False`. Each branch starts from a copy of the same already
crop/flipped RGB32 base. Branch weights are Dirichlet(1,1,1), cast to float32;
the clean/augmented blending coefficient is Beta(1,1), cast to float32. Each
operation is selected uniformly from the default nine, with replacement.

For each magnitude-bearing operation, `u` is sampled uniformly from `[0.1,3)`.
Positive integer magnitudes use truncation toward zero. For geometry,
another independent uniform draw greater than 0.5 negates the magnitude.

| Operation | Exact parameter and pixel behavior |
| --- | --- |
| Autocontrast / equalize | PIL ImageOps; no magnitude draw. |
| Posterize | Retained bits `4 - int(u*4/10)`, hence 3 or 4; PIL bit masking. |
| Rotate | Signed `int(u*30/10)` degrees; 0–8 magnitude. |
| Solarize | Threshold `256 - int(u*256/10)`, hence 180–254; PIL threshold/inversion. |
| Shear x / y | Signed `u*0.3/10` inverse-affine coefficient on the selected axis. |
| Translate x / y | Signed `int(u*(32/3)/10)` inverse-affine offset, magnitude 0–3 pixels. |

Rotation and affine operators use Pillow bilinear resampling, retain the 32×32
canvas and explicitly use black RGB fill. These settings express the author's
PIL defaults explicitly. PIL supplies interpolation, clipping and uint8 pixel
rounding; no additional nearest-neighbor path, floating-image conversion or
custom rounding is introduced. Exact environment/Pillow versions must be
recorded by a future trainer.

`rgb_image` copies strict PIL RGB32 or NumPy uint8 NHWC32 input; it refuses
implicit mode, scale, dtype or size conversion. `normalize_image` returns CPU
float32 CHW and keeps this campaign's mean `(0.4914,0.4822,0.4465)` and standard
deviation `(0.2470,0.2435,0.2616)`. Branches are normalized before convex
mixing, with float32 output and no second uint8 quantization.

### Explicit deviations and later integration requirements

The author uses legacy global NumPy state. This implementation requires an
explicit, checkpointable `numpy.random.Generator`. `augmix_views` draws two
uint64 seeds from its parent stream first, then creates two independent PCG64
child streams. Parent `bit_generator.state` alone reproduces the entire pair.
Branch depths in the first view do not consume the second view's random stream.
Sampling distributions match the specified method; this is **not bitwise
reproduction of the author's legacy RNG sequence**. `AugMixTrace` exposes
weights, blend and sampled chains for correctness audits.

This module performs no crop/flip. The author applies flip then RandomCrop
with torchvision's default constant padding; the current campaign performs
crop/flip using reflection padding. A future controlled trainer must declare
and implement its common preprocessing once before the two views, retaining
the chosen campaign baseline behavior rather than silently changing padding
or applying independent crops to different views. The author's `[0.5]*3`
normalization also differs from the campaign normalization retained here.

## Three-view objective and numerical design

`jsd_three_view(clean_logits, aug1_logits, aug2_logits)` takes three matching
floating `N×K` tensors, `N>0`, `K>=2`, on one device. It computes probabilities
in float32 outside autocast. The mixture is the arithmetic mean of the three
probability vectors, clamped to `[1e-7,1]` and **not renormalized**. The result
is the mean of `KL(p_view || mixture)` across the three views, each summing
over classes and averaging over the source batch. All mixture/target paths
remain differentiable; no distribution is detached.

Internally, FP32 log-softmax and `kl_div(..., log_target=True)` express the
same KL direction and reduction while avoiding a probability-target
`0*log(0)` backward singularity when softmax saturates. Moderate-logit tests
compare against the author's probability-target formula. Saturated FP16 and
BF16 input tests check finite float32 loss and all three gradients. Finite
inputs outside representable FP32 log-probability range are rejected. Tiny
negative JSD from floating-point roundoff is not clipped to alter the method.

`augmix_loss(clean, aug1, aug2, targets)` returns clean-only mean cross entropy
plus exactly `12 * JSD`; targets are same-device int64 source-batch class IDs.
`augmix_no_jsd` and `no_jsd_loss` are explicit single-view/CE-only ablation
interfaces, not a memory fallback or an equivalent three-view execution.

The next admission test must measure the actual official-image CPU/PIL work,
host/device transfer, one concatenated `3B` model forward, full objective,
backward and optimizer update. No throughput estimate, result qualification or
new accuracy claim follows from these CPU transform/loss tests.

## Measured CPU hot-path correction

Five synthetic source batches of128 images,384 output views per batch, exposed
repeated tiny Torch normalization and validation dispatches. After8 warmup
images, median augmentation-plus-stack time was0.313966s. An internal NumPy
float32 path retains the same separate multiply/add rounding order and converts
to a CPU tensor at the view boundary; the public Torch mixture validator remains
available as an independent reference. The new median was0.148704s, an observed
2.11x CPU-only ratio in this context.

The original source bytes are preserved privately and reverified against their
SHA-256. For32 fixed synthetic images/seeds, every output element is bitwise
equal to the original, with identical sampled traces and final parent RNG
states. A new CPU regression independently reconstructs sampled traces with
Torch normalization and mixture operations. All21 transform/loss CPU checks
pass. Context, hashes, raw five-batch timings and scope are in
[augmix-engineering-results.json](../docs/augmix-engineering-results.json).

These timings exclude transfer, model, loss and optimizer, use no official
images and do not admit formal training. The forthcoming trainer is preparing
two CPU workers with explicit per-image seeds, preserving ordered source batches
and epoch-boundary replay. Its actual complete native pipeline must still be
measured under the fixed three-hour guard; batch128 and single3B remain fixed.
