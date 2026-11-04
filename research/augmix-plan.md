# AugMix: controlled robustness candidate

Material passport: `mode=quick`; `sources=original paper and author code`; `status=predeclared concept, no implementation or GPU result`; reviewed 2026-10-31. Refs #7.

## Evidence and hypothesis

The [original AugMix paper](https://arxiv.org/pdf/1912.02781), Table 1 and Figure 12, reports CIFAR-10-C mean error of 26.9% for its standard WRN40-2 versus 11.2% for AugMix. The architecture, data protocol and corruptions differ from this campaign. These numbers are literature evidence, not predictions of Kaggle score or of the five local synthetic stress cases. In particular, our `resize24` view has no direct equivalent demonstrated by that comparison. No hidden corruption is assumed.

The working hypothesis is that an independent augmentation-and-consistency package can improve robustness to some transformations while preserving clean development performance. Replace the current Mixup/Cutout package; retain our scratch WRN22-4, fixed official 40k/5k split, 160 epochs, SGD schedule, EMA and predeclared weighted inference policy. Because both augmentation and loss change as a package, do not attribute any difference solely to one component.

Section 4.3 provides a counterexample to combining methods by default: AugMix plus Mixup has 13.3% error in that ablation versus 11.2% for AugMix. Table 4 reports 13.1% without JSD versus 11.2% with JSD. These are matched paper ablations, not results from this competition. Do not initially stack Mixup, Cutout, SAM or SE onto AugMix.

## Fixed implementation target

Use the [author CIFAR implementation](https://github.com/google-research/augmix/blob/master/cifar.py) and [operation definitions](https://github.com/google-research/augmix/blob/master/augmentations.py) to verify exact mechanics before implementation. The controlled target is:

- Mixture width 3; independently sampled chain depths 1 through 3; severity 3; Dirichlet and Beta concentration 1.
- Default nine operations: autocontrast, equalize, posterize, rotate, solarize, shear-x/y and translate-x/y. `all_ops` remains disabled.
- Apply the fixed crop/flip preprocessing once to an original image. Produce two independently sampled AugMix views from that same preprocessed image. Concatenate clean and the two views into a single `3B` model pass.
- Compute cross entropy only on the clean logits. Add coefficient 12 times the mean KL of the three predictive distributions to their mixture, clamped at probability 1e-7 as in the author implementation. Verify KL direction and reduction against the cited source; do not silently substitute pairwise KL.
- The separate no-JSD ablation uses one AugMix view and cross entropy. It is not an equivalent lower-memory execution of the three-view recipe and is not a fallback after observing poor training results.

The author data entrypoints download CIFAR and CIFAR-C; **do not run those entrypoints**. Transform only the already verified official competition training images. The same own residual CNN remains 4,298,970 total/trainable parameters, with no external images, pretrained weights, confirmation access or hidden labels.

## Admission checks and budget

Complete current frozen independent repeats first. A future independent trainer, benchmark and supervisor must preserve all existing source-bound files. No existing typed delivery path currently supports AugMix; it must not be relabeled as another method.

Before formal training, run a supervised native-precision preflight using the actual official training-image path, PIL/CPU operations, transfers, concatenated `3B`, full loss, backward and optimizer update. Synthetic SGD speed from earlier methods cannot estimate this pipeline. Check finite probabilities/loss/gradients, actual parameter changes, distinct random views, operation boundaries and near-zero JSD for identical distributions. Persist exact source/config/data/RNG/environment hashes and real timings.

Keep source batch size 128 and the single concatenated pass. If it cannot fit memory or the conservative three-hour and remaining daily budget, pause this recipe. Do not lower batch size or split views into separate passes during training: that changes BatchNorm conditions. Any alternative is a separately registered experiment before outcomes are observed. The preflight must include realistic steady-state CPU augmentation and model work, plus benchmark/transfer/cleanup reserves and the remaining campaign hard stop.

## Evaluation and rejection

Use the complete fixed native v3 5,000-image development suite, all three policies, clean accuracy/NLL and all five stress cases, with data/pixel/context fingerprints. Retain the original gate: clean improvement at least +0.10pp with stress no worse than -0.20pp, or stress improvement at least +0.50pp with clean no worse than -0.20pp. Only the predeclared weighted policy can be promoted.

A failed clean gate, failed numerical/geometry check, unaffordable measured pipeline or incomplete run is not a qualified candidate. Preserve negative results; do not use the confirmation set, hidden labels or leaderboard feedback to tune corruption types or severity. No per-case improvement is asserted merely from a higher mean stress score.

After a complete qualifying primary, run the established original-image cluster paired comparison and freeze primary weights/policy/config before a fresh independent seed2027 repeat. Conditional intervals reuse selected development and do not constitute fresh confirmation. New method-specific proof and delivery validation must exist before primary-only hidden prediction or strict upload. Actual Kaggle receipts alone establish whether the score improves.
