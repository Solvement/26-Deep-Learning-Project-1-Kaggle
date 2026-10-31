# AugMix with native training BatchNorm: numerical protocol

Material passport: `mode=quick`; `sources=own measured CUDA profile and controlled synthetic diagnostic`; `status=registered before development accuracy or formal training`; registered 2026-10-31. Refs #7. This follows engineering results, not an accuracy result.

## Measured reason for the control

The unchanged actual-image AugMix preflight was numerically correct but required a 31,101.54-second budget. A separate instrumented run attributed 7,894.91ms to three concrete BatchNorm CUDA leaf kernels across 24 passes; current consumer queue waiting was small. See [profile evidence](../docs/augmix-profile-results.json). Do not add nested parent, leaf and host-wait timings.

The following synthetic two-arm diagnostic kept source batch128, one384-view pass, FP16, identical initial tensors/input/RNG, the original loss/SGD/EMA and GradScaler init256. Only BatchNorm forward used a local cuDNN-disabled context in the native arm. Both arms completed23 actual updates, verified all19 BN counters, finite positive gradients and parameter changes, with convolutions retaining cuDNN. Observed median step duration was0.39478015s versus0.08672155s. The raw aggregate SHA is `3b11a371655c6716c04c3115a125aff9285e56be0c60a32f9154da02dfc4d12c`.

This is not bitwise equivalence: first-step maximum output difference was0.000244140625, gradient difference0.0011942386627197266 and BN-buffer difference0.00003528594970703125. The gradient RMS difference was0.000012436666423004267. Reusing one synthetic batch, additional proof work and omission of the CPU pipeline limit the timing comparison. No training-time estimate or accuracy follows from it.

## Fixed new protocol

Retain the original AugMix recipe, raw configuration, official40k/5k split, scratch WRN22-4 with4,298,970 parameters,160 epochs, sourceB128/single3B384, nativeFP16/init256, optimizer/schedule/EMA, per-image RNG and two-worker pipeline. Do not change crop/flip, severity, mixture/JSD, batch size, model widths, normalization parameters or data.

Only training-mode BatchNorm uses native ATen via a forward-local cuDNN-disabled context. Evaluation-mode BN, convolutions and other operations keep the original backend behavior. Preserve state names, initial tensors, running-statistic parameters and exactly one counter increment per training forward, including an overflow-skipped optimizer step. Restore any in-process patch on all exit paths. Record actual backend nodes and check restoration, rather than merely labeling an intended backend.

This is an explicit numerical implementation adaptation. Its marker must include the new helper/entry hashes and fixed original engine hashes; it must not masquerade as the earlier AugMix or attribute all differences solely to augmentation. Old source, raw receipts, checkpoints and confirmation remain frozen.

## Actual pipeline admission and evaluation

Before formal training, the new supervised entry must run the entire official-training-image CPU/PIL/queue/transfer/3B/loss/backward/SGD/EMA pipeline, including the new backend checks: three warmups,20 timed updates and a separate overflow skip. Measure nativeFP16 on the actual target environment. Use the fixed `1.5 * median * 313 * 160 + 300` guard, daily6-hour/shared local lease,3-hour job cap and campaign hard stop. The synthetic0.0867s result cannot admit training. Neither old preflight receipts nor different environments substitute for the new source-bound proof.

Use a new typed full benchmark, retaining the complete5000-image native v3 suite, original baseline, all policies, clean/NLL and five stress cases. Only the fixed weighted policy may qualify under the unchanged clean/stress gate. A new validator must verify the numerical marker before reusing any original AugMix state/count/RNG checks; unknown fields or sources must fail. Preserve raw artifact hashes while normalizing metadata for a delegated validator, never weights or evidence values.

A qualifying primary still needs original-image clustered pairing, primary/configuration freeze and a fresh independently trained seed2027 repeat. Formal primary entry initially admits only2026. No confirmation reuse, external data, pretrained weights, hidden labels, prediction or upload is authorized by an engineering pass. Existing qualified deliveries and the SAM repeat retain priority. Actual Kaggle receipts alone establish score improvement.
