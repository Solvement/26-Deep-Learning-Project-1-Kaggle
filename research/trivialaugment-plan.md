# TrivialAugment fixed-standard: independent next candidate

Material passport: `mode=quick`; `sources=primary paper, pinned author implementation, versioned torchvision documentation`; `status=predeclared candidate, not implemented or trained`; reviewed 2026-10-31. Refs #7.

## Why test this separately

The [ICCV paper](https://arxiv.org/pdf/2103.10158), Algorithm 1, samples one operation and one of 31 strengths for each image. It does not require an augmentation-policy search. Its clean-image classification results do not establish corruption robustness or performance in this competition. Our working hypothesis is that a single-view augmentation package may preserve clean performance with lower measured cost than our three-view AugMix pipeline. Actual throughput and complete development results must decide.

The [author repository](https://github.com/automl/trivialaugment) recommends `fixed_standard` for inexpensive WRN setups. The exact reference is [aug_lib.py at e6545d80affcc18b7024185a096e93e21010023e](https://github.com/automl/trivialaugment/blob/e6545d80affcc18b7024185a096e93e21010023e/aug_lib.py), including `set_augmentation_space` and `TrivialAugment`. Do not invoke its dataset download or training entrypoints.

## Fixed target before any outcomes

Use the author's 14-operation `fixed_standard` space: identity, autocontrast, equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x/y and translate-x/y. Sample operation uniformly, strength uniformly from integers 0 through 30, and the geometric sign independently. Retain the exact integer truncation rules and default nearest-neighbor PIL geometry with black fill. Fixed ranges are shear magnitude 0–0.3, translation 0–10 pixels, rotation 0–30 degrees, solarization threshold 256 down to 0, posterization 8 down to 4 bits and enhancement factor 0.1–1.9. Test endpoints and quantization explicitly.

The [torchvision 0.27 TrivialAugmentWide implementation](https://docs.pytorch.org/vision/0.27/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide) uses wider ranges and different enhancement/posterization mechanics. It is not an implementation of this fixed-standard target. Do not silently substitute it, invert images, add vertical flips or choose operations using stress-case feedback.

Campaign adaptation: own scratch WRN22-4, 4,298,970 parameters, 160 epochs, source batch 128, fixed SGD/EMA schedule, weight decay 0.0005, official 40k/5k split and predeclared weighted inference. Apply our existing reflect4 crop/horizontal flip once, then one TA operation before normalization; retain the existing final Cutout16. Remove Mixup and use ordinary class-label cross entropy. This is an augmentation package comparison, not an isolated test of adding TA to Mixup, and not a reproduction of the paper's 200-epoch setup. No SAM, SE, JSD or multi-view batch is included. Use explicit per-image PCG64 state; different random libraries and preprocessing order prevent claiming author seed equivalence.

## Admission, evaluation and stopping

Current qualified submissions and SAM seed repeat retain priority. New method files must leave all existing bound sources untouched. First establish CPU pixel/RNG/serialization/update correctness and <=5M total/trainable parameters. A supervised native actual-training-image preflight must include CPU augmentation, queue, transfer, loss, backward, SGD/EMA and numerical checks. Keep fixed batch and recipe; formal training requires the existing 1.5 measured-time margin, 300-second finalization reserve, daily budget and three-hour/hard-stop limits. This plan contains no speed measurement.

Only a complete native v3 evaluation of all 5,000 development images can qualify the fixed weighted policy. Apply the existing clean/stress gate, original-image clustered paired comparison, primary freeze and independent fresh seed2027 repeat. Preserve case-level regressions and rejected results. No confirmation access, external data, pretrained weights or hidden-label use. A new typed proof/delivery route is required before primary-only prediction or strict upload; existing AugMix, SAM and ResizeMix receipts cannot stand in for TA. Actual Kaggle scoring alone establishes an improvement.
