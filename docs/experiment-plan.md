# Experiment plan and decision log

Owner: Kevin Yang (`ky2738`). Kaggle team: **Saber**. Seed: `2026`.

This file separates the proposed protocol from measured outcomes. Actual scores,
durations, checkpoints, configuration and code provenance are written by the
training scripts under ignored `results/` and `outputs/` paths.

## Questions

1. Does increasing residual-network capacity improve held-out accuracy within
   the 5,000,000-parameter budget?
2. Can a single WRN-22-4 trained from scratch generalize well enough for the
   hidden competition evaluation set?

## Fixed evaluation protocol

- Use only the 50,000 official competition images and `train_labels.csv`.
  The data preparation script packages the official Kaggle mount, without
  accessing any external CIFAR-10 distribution or hidden labels.
- Keep a deterministic stratified split of 40,000 training images, 5,000
  development images and 5,000 confirmation images, with 500 images per class
  in each holdout.
- Use the same split for all comparisons. Clean development images select the
  checkpoint/recipe; horizontal-flip averaging is also selected on development.
- Evaluate the selected recipe exactly once on the independent confirmation
  split. Do not use confirmation to revise the recipe or select a different run.
- Retrain a fresh model on all 50,000 official training images using the frozen
  recipe, epoch budget, raw/EMA variant and inference rule. Do not report an overlapping
  development score for this full-data model.
- Submit predictions for the 2,000 competition images through Kaggle. The
  public and private leaderboard each use 50% of those images. Record a public
  score as **public leaderboard test accuracy**, never as the unreleased private
  test score.

## Planned experiments

| Run | Purpose | Architecture | Planned comparison |
| --- | --- | --- | --- |
| Baseline | Small residual reference | WRN-16-2, crop/flip | 160 epochs |
| Architecture | Increase residual capacity | WRN-22-4, crop/flip | Same split, seed, schedule and optimizer as baseline |
| Regularized | Test a regularization bundle | WRN-22-4, crop/flip + Cutout + MixUp | Same schedule and optimizer as architecture run |
| Final full-data | Refit the selected fixed recipe | Fresh scratch model on all 50,000 images | No development or confirmation evaluation |
| Seed stability | Repeat the regularized recipe with seed 2027 | Same recipe and fixed split seed 2026 | Excluded from recipe selection and confirmation |

The planned aligned schedule uses 160 epochs, SGD (learning rate 0.1, Nesterov
momentum 0.9, weight decay 0.0005), five warmup epochs and cosine decay. The
regularized candidate adds Cutout size 16 and MixUp alpha 0.2 with probability
0.5. EMA is measured for every recipe. Saved configurations remain the source
of truth for the actual values, deviations, and measured epoch budgets.

The architecture comparison changes depth and width together; it measures the
capacity configuration rather than isolated depth or width effects. The
regularization comparison changes Cutout and MixUp together; it cannot isolate
their individual contributions. A single seed/split does not establish
statistical significance or seed stability.

## User-directed official submission comparisons

The user requested full use of the daily five-entry allowance. Submit the fixed
`baseline`, `architecture`, `regularized`, `final_full`, and seed-2027 regularized
models as five official benchmarks, subject to the actual remaining Kaggle quota.
Use the existing frozen checkpoints and development-selected inference rules;
do not use public feedback to alter the training recipe or repeat confirmation.

Keep every receipt in `results/kaggle_submissions.json`, including the exact run,
training sample count, parameters, score, submission reference, timestamp and
prediction-file SHA-256. The independent study's primary full-data receipt stays
in `results/kaggle_submission.json`. Public scores may select the official Kaggle
entry, but the report must distinguish that choice from the already frozen
development/confirmation protocol. Report the full-data result and best public
benchmark separately when they differ. Private accuracy remains unknown, and
selecting the largest public score among five can favor sampling noise.

## Records and delivery

Each real run records configuration, random seed, code revision, Python/library
versions, GPU, data/split fingerprints, training/validation history, runtime,
checkpoint selection, and artifact SHA-256 checksums. Generated weights, images,
prediction CSVs and full notebook outputs stay outside Git.

The report builder uses actual saved artifacts. Missing scores remain pending;
the builder must not substitute expected or literature-reported accuracy. Public
score feedback is limited to five submissions per team per day.

## Local runtime finding

The local RTX 5060 Laptop GPU passed real convolution forward/backward and an
SGD step after removing a conflicting standalone cuDNN directory from the
process-local `PATH` before importing PyTorch. This does not change the global
environment. `runtime.py` records the environment used by each run.
