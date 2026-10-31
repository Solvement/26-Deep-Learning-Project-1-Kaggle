# Project 1 delivery checklist

Owner: Kevin Yang (`ky2738`), team **Saber**. Sources: the three-page course PDF
and [verified requirements](competition.md). All five distinct official entries
are scored; Kaggle reported five submissions and zero remaining for
**2026-10-31 UTC**. The
exploratory T4 run completed its full 300-epoch budget.

## Model and official data

- [x] Scratch-initialized residual CNN; enforce at most 5,000,000 total/trainable
  parameters. WRN-22-4 has exactly 4,298,970.
- [x] Use official Kaggle images/labels only; preserve source/CSV fingerprints and
  the deterministic disjoint 40,000/5,000/5,000 split (seed 2026).
- [x] Exclude external datasets, pretrained weights, hidden-label lookups and
  manual labels. Full-data refits use only the 50,000 permitted training images.
- [x] Preserve completed primary-run configurations, code hashes, environment,
  histories, checkpoint hashes, raw/EMA choice and inference rules locally.
- [x] Keep raw data, model bytes, prediction rows and credentials outside Git.

## Measured experiments and Kaggle

- [x] Complete the aligned 160-epoch baseline/architecture/regularized comparisons
  and frozen primary full-data refit; full-data models have no overlapping dev score.
- [x] Evaluate confirmation once: 96.64% for the selected 40,000-image regularized
  candidate, without attributing that result to either full-data model.
- [x] Complete seed-2027 stability: 96.36% development versus 96.30% for seed 2026;
  exclude it from recipe selection, confirmation and hidden-test prediction.
- [x] Validate and score the first four official CSVs: baseline 83.50%, architecture
  85.40%, regularized 87.40%, primary full-data 87.10%; retain actual refs/checksums.
- [x] Complete the exploratory 300-epoch robust_full job and fixed final EMA/BN/
  ten-view output: 86.20%, reference 57827678. It did not improve the best 87.40%
  public score; no individual cause is inferred and the primary study is unchanged.

## Code and public evidence

- [x] Publish the reproducible source/configuration/tests on the public
  `feat/cifar10-resnet-submission` branch and PR 6; main is not merged.
- [x] Share the source in the verified [competition discussion](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/discussion/754332).
- [x] Use scripts/submit_verified.py for local completed-run/CSV/checkpoint checks,
  quota-aware upload intent/reference handling and duplicate protection.
- [x] Generate sanitized docs/results.json with BOTH strict flags:
  `--require-complete --require-five-submissions`; do not publish pending evidence
  as a completed delivery. Final status is complete with no pending fields and
  five distinct completed official receipts.
- [x] Push the final results/documentation/artifact references in the same feature
  branch after the fifth score and final audits are complete.

## Report and executed notebook

- [x] Verify the final five-page report: Kevin Yang / ky2738, Saber and
  public implementation-branch link on page one; architecture, parameters,
  methods, actual findings, limits and citations are included.
- [x] Execute the final results notebook: six of six code cells ran without errors,
  with actual metrics/plots and printed parameter count/public accuracy; private
  accuracy remains unknown.
- [x] Refresh the final PDF after all five scores; render every page and confirm
  at most eight pages including figures/tables/references and no unsupported
  causal explanation for the development/public gap. All five pages were checked.
- [x] Refresh and independently audit the final executed notebook; copy only
  non-sensitive scalar/config/aggregate-plot outputs into
  notebooks/Project1_Results_Executed.ipynb and verify its SHA matches the audited source.
- [x] Verify final report/notebook checksums; clearly distinguish primary full-data
  public accuracy, highest official public benchmark and private score still unknown.

## Human and course actions

- [ ] Independently confirm the course Google Form registration uses team Saber.
- [ ] Upload the final PDF to Brightspace for each registered member.
- [ ] Obtain human maintainer PR review and explicit merge authorization before
  merging; an agent review does not count as human approval.
- [ ] Only if adding the future teammate: complete the official course/Kaggle
  membership procedures and retain the written three-person team limit.
- [ ] Confirm the report deadline's timezone in the course announcements.

The live Kaggle deadline is **2026-11-06 23:59:59 EST (UTC-05:00)**, equivalent to
**2026-11-07 04:59:59 UTC**. The assignment's report deadline is
**2026-11-09 23:59**, with its timezone unspecified. Team Saber currently has one
member. The platform UI's capacity of five does not override the written maximum
of three. Private leaderboard accuracy remains unknown until officially released.
