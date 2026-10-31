# Project 1 delivery checklist

Owner: Kevin Yang (`ky2738`), team **Saber**. Sources:
the local three-page assignment PDF and [verified requirements](competition.md).

## Model and data

- [x] Implement a convolutional residual network initialized from scratch.
- [x] Enforce the 5,000,000 total/trainable-parameter budget in the constructor.
- [x] WRN-22-4 has exactly 4,298,970 total and trainable parameters.
- [ ] Complete and record a real training run and selected checkpoint.
- [ ] Preserve official data fingerprints, deterministic split, logs and weights.
- [x] Use official Kaggle `train_images/` and `train_labels.csv` only.
- [x] No external datasets, pretrained weights, hidden-label lookups or manual labels.

## Code, report and notebook

- [ ] Push reproducible source, configuration and tests through a feature-branch PR.
- [x] Confirm the GitHub repository is publicly accessible; implemented source is
  on `feat/cifar10-resnet-submission` / PR 6, awaiting human review before merge.
- [x] Share the code in the verified [competition discussion](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/discussion/754332),
  as required by the Kaggle foundational public-code-sharing rule.
- [ ] Generate and visually verify a PDF of at most 8 pages, including references.
- [ ] Put the public GitHub link, Kevin Yang / ky2738 and team Saber on page one.
- [ ] Include architecture, parameter count, actual experiments, design decisions,
  public test accuracy, limitations and citations.
- [ ] Generate an executed notebook with plots and printed test accuracy and
  parameter count. Full outputs stay in ignored `outputs/`; only the audited,
  non-sensitive final results notebook is copied into tracked `notebooks/`.

## Kaggle and course submission

- [ ] Validate `submission.csv`: exactly `id,label`, every sample ID once, correct
  sample ordering, and integer labels in `[0,9]`.
- [ ] Upload the selected predictions and confirm Kaggle completed scoring.
- [ ] Use the available five-entry daily allowance for the fixed baseline,
  architecture, regularized, full-data and seed-repeat benchmark entries; retain
  every scored receipt in `results/kaggle_submissions.json`.
- [ ] Record public score, submission reference, code revision and file checksum.
- [ ] Distinguish primary full-data public accuracy from the best public benchmark;
  preserve the frozen recipe/confirmation study and leave private accuracy unknown.
- [ ] **Human action:** confirm the course Google Form registration uses team Saber.
- [ ] **Human action:** add the future teammate through official course/Kaggle
  procedures before the applicable team deadline.
- [ ] **Human action:** upload the final PDF to Brightspace by the course deadline.

The live Kaggle header confirms **2026-11-06 23:59:59 EST (UTC-05:00)**, or
**2026-11-07 04:59:59 UTC**. The assignment's report deadline is
**2026-11-09 23:59**, with its timezone still unspecified. Confirm that report
timezone in the course channels. Private leaderboard accuracy is unavailable
until released by Kaggle. Team Saber currently has one member; enforce the
written three-person maximum even though the platform UI displays capacity five.
