# TrivialAugment fixed_standard CPU implementation — Issue7 / PR8

Material passport: primary paper/author code, exact pinned source, synthetic CPU
pixel/RNG tests. This implements an augmentation only. No trainer, data loader,
GPU preflight, training, dev accuracy or competition score is produced here.

## Primary source and exact space

Reviewed2026-11-02 against [Mueller & Hutter, ICCV2021 Algorithm1 and AppendixA](https://arxiv.org/html/2103.10158v2),
and the [author code at e6545d80affcc18b7024185a096e93e21010023e](https://github.com/automl/trivialaugment/blob/e6545d80affcc18b7024185a096e93e21010023e/aug_lib.py).
Actual downloaded source SHA-256:
`8f5f9febb671fd12be067e311dbdcb837a2a29c672861f20c2cf03732fdd2868`.
The [pinned README](https://github.com/automl/trivialaugment/blob/e6545d80affcc18b7024185a096e93e21010023e/README.md)
has SHA `16bc9ac3b424c96ffab0e1c211ee8a71ee5b1fc7fd47d7385cf192d7b9d9805b`.
The independent implementation attributes the algorithm and author reference;
upstream is [MIT, Copyright2021 Samuel Gabriel Mueller](https://github.com/automl/trivialaugment/blob/e6545d80affcc18b7024185a096e93e21010023e/LICENSE),
license file SHA `b97576d1345e463979111dd49bffea6cd5bbe010b7919fbfb514f3478e6c5835`.
The original copyright and permission text is preserved byte-for-byte in
`research/licenses/TrivialAugment-MIT.txt`.
No upstream module, training entrypoint or dataset-download code is executed.

One operation is sampled uniformly per image; its level is independently
uniform in integers0..30. The geometric sign is an extra independent coin.
All14 operations are fixed: identity, autocontrast, equalize, rotate, solarize,
color, posterize, contrast, brightness, sharpness, shear-x/y and translate-x/y.
There are no added noise, blur, vertical flip, invert or task-selected operators.

| Operation family | Parameter from level m | Endpoint range |
| --- | --- | --- |
| Shear | sign * float(m)*0.3/30 | -0.3..+0.3 |
| Translate | sign * int(m*10/30) | -10..+10 pixels |
| Rotate | sign * int(m*30/30) | -30..+30 degrees |
| Solarize | 256-int(m*256/30) | threshold256..0 |
| Posterize | 8-int(m*4/30) | 8..4 bits |
| Color/contrast/brightness/sharpness | float(m)*(1.9-0.1)/30+0.1 | 0.1..1.9 |
| Identity/autocontrast/equalize | level sampled but unused | no strength parameter |

Integer conversion truncates exactly as the author's helper. Fixed-standard
posterization never uses the old0-bit endpoint. Geometry explicitly uses PIL
nearest neighbor, unchanged32x32 canvas and black RGB fill, matching the
reference's defaults. There is no bilinear interpolation or mirrored enhancement
factor. This is **fixed_standard**, not the different `wide_standard` or
`torchvision.TrivialAugmentWide` ranges. The author recommends different spaces
for different models; we deliberately hold this space fixed for this experiment
and do not claim to reproduce their larger-model recipe or reported scores.

## API, RNG and pixel checks

`trivialaugment.augment(image, rng)` returns one RGB32 PIL image. Inputs must be
RGB32 PIL or uint8 HWC32x32x3 arrays, including read-only/noncontiguous arrays.
Input pixels are copied and never mutated; modes/sizes/dtypes are not silently
converted. `return_trace=True` also returns a frozen `Operation(name,level,sign,
space)`; `apply_operation` replays it without RNG access.

An explicit `numpy.random.Generator` or checkpointable `random.Random` is
required. Global NumPy/Python RNG modules and SystemRandom are rejected. A future
trainer must save the supplied per-image PCG64 state or derive stable seeds from
epoch/original-ID, without consuming the parent's crop/Mixup global streams.
No author seed-stream equivalence is claimed: the implementation omits the
reference wrapper's redundant probability-one draw and uses explicit local
integer sampling. Deterministic replay means same input/trace or saved local
RNG state and the same NumPy/Pillow versions, not cross-library bit equivalence.

CPU tests cover every operation at all31 levels, parameter quantization and
endpoints, both geometric signs, nearest-neighbor color preservation, explicit
black boundaries and coordinate direction, posterize/solarize extremes, RGB
channels/output bounds, input immutability, state replay, and global RNG isolation.
They use small synthetic pixels and do not assess accuracy or GPU throughput.

## Prospective integration; old hypothesis remains unchanged

The previous unexecuted `trivialaugment-plan.md` proposed a different WRN22
package with post-crop TA and removal of Mixup. Its historical bytes remain
unchanged. The current requested future recipe is independent: scratch
WRN34-3/160,4,166,298 parameters, sourceB128, official40k/5000dev, seed2026,
split2026, original SGD/EMA/WD/schedule and weighted inference. Apply TA to each
**original**32x32 RGB image, then pass it to the unchanged parent's reflect4
crop/horizontal flip, normalization, Cutout16 and existing Mixup0.2/prob0.5.
Only TA is added to that parent pipeline; one view, no JSD/SAM/SE and no test-time
augmentation change. This order follows the README's general recommendation to
place TA before custom augmentation. It is not the old proposed recipe or a
200-epoch paper reproduction. No source in the frozen core is changed.

Before formal use, a separately reviewed typed trainer/controller must prove
parent RNG isolation, order, actual parameter updates and full official-image
pipeline timing under existing source/lease/OS guard/day/job/hard-stop budgets.
This module alone creates no training or delivery qualification. A fresh full
5000 native-v3 benchmark must keep all18 cases, original gate and predeclared
same-policy incumbent criteria, including each stress regression. A candidate
must not be tailored to the observed noise failure or rescue an old rejection.
Only later qualified primary freeze, fresh27 repeat, explicit method delivery
support and strict actual Kaggle receipt can establish a competition gain.
