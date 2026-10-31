# Saber: Efficient ResNet for CIFAR-10

Kevin Yang (`ky2738`) | Team **Saber** | NYU Deep Learning Project 1

[Competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl) · [Verified requirements](docs/competition.md) · [Experiment protocol](docs/experiment-plan.md) · [Delivery checklist](docs/requirements-checklist.md)

The repository is public. The implemented pipeline is on
[`feat/cifar10-resnet-submission`](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/tree/feat/cifar10-resnet-submission)
and [PR 6](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/pull/6),
pending human review; `main` still contains the earlier environment harness.

The main model is a scratch-initialized **WideResNet-22-4 with 4,298,970 total and
trainable parameters**. Its constructor rejects models above the 5,000,000
parameter limit. Training uses only the official competition images and labels;
no external data or pretrained weights are used.

## Measured results

All **five official submissions are scored**, using the full daily allowance
for **2026-10-31 UTC** (Kaggle reported 5 submitted and 0 remaining on that date).
The exploratory T4 run completed
300 epochs plus training-only BN recalibration and fixed ten-view prediction.

| Run | Training images | Parameters | Development accuracy | Kaggle public accuracy | Submission reference |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 40,000 | 691,674 | 94.86% | 83.50% | 57825539 |
| `architecture` | 40,000 | 4,298,970 | 95.72% | 85.40% | 57825558 |
| `regularized` | 40,000 | 4,298,970 | 96.30% | **87.40%** | 57825566 |
| `final_full` (primary refit) | 50,000 | 4,298,970 | Not evaluated | 87.10% | 57825964 |
| `robust_full` (exploratory) | 50,000 | 4,298,970 | Not evaluated | 86.20% | 57827678 |

The independent one-time confirmation result is **96.64%** for the selected
40,000-image `regularized` candidate. It is not a confirmation score for either
full-data model. The current highest completed public score is 87.40%; the
primary 160-epoch full-data result is separately reported as 87.10%. The
exploratory 86.20% result did not improve the best observed public score. The
primary study and its one-time confirmation remain unchanged; no individual
cause is inferred from this bundled extension. Private leaderboard scores are
not yet available. The cause of the development/public gap has not been established.
Verified aggregate records are available in [docs/results.json](docs/results.json)
and the [audited executed results notebook](notebooks/Project1_Results_Executed.ipynb).

The independent stability repeat keeps the same regularized recipe and split:

| Training seed | Selected checkpoint | Plain development | Flip-average development | Used for recipe selection |
| --- | --- | --- | --- | --- |
| 2026 | epoch 151, raw weights | 96.08% | 96.30% | Yes |
| 2027 | epoch 148, EMA weights | 96.14% | 96.36% | No |

The 0.06 percentage-point difference is a two-run observation, not a confidence
interval or significance result. The repeat generated no hidden-test predictions
and consumed no Kaggle submission.

## Reproduce a run

Use Python with a compatible PyTorch installation for your chosen CPU/GPU. Keep
an already working Kaggle/Colab CUDA installation. Local execution was verified
with Python 3.14, PyTorch 2.12.0+cu130, torchvision 0.27.0+cu130 and an RTX 5060
Laptop GPU. Library/hardware details for each actual run are saved in its
`environment.json`. The older CPU development environment remains documented in
[cloud development](docs/cloud-development.md).

1. **Prepare official data.** Obtain the competition's `train_labels.csv`,
   `train_images/`, `test.csv`, `test_images/` and `sample_submission.csv` using
   your own authorized Kaggle identity. Alternatively, run
   `scripts/export_official_data.py` privately in a Kaggle CPU notebook with this
   competition attached, then download its `official_train.npz`,
   `official_source_manifest.json` and `train_labels.csv` into `data/`. The private
   `scripts/export_official_test.py` helper similarly exports `official_test.npz`,
   `official_test_source_manifest.json`, `test.csv` and `sample_submission.csv`.
   Keep both official exports private. Prepare the fixed stratified split:

   ```bash
   python prepare_data.py --data-dir data --seed 2026
   ```

   `data.py` checks official CSV fingerprints, image shapes, IDs, class counts,
   source archive fingerprint and split disjointness. An original Toronto archive
   downloaded during setup is unused; this pipeline never reads it. If bulk
   competition download returns 404, individual test downloads are supported:

   ```bash
   uv run --no-project --with kaggle==2.2.5 python download_kaggle.py --data-dir data --workers 4
   ```

   This helper requires the Kaggle package in the selected Python environment;
   it reads credentials from supported Kaggle configuration, never from code.

2. **Check the model and benchmark the device.** These checks are not leaderboard
   scores. CUDA requests fail clearly if a GPU is unavailable.

   ```bash
   python model.py --smoke-test
   python train.py --config configs/wrn_final.json --output-dir results/benchmark --device cuda --benchmark-steps 20
   ```

3. **Run the comparison, confirmation and full-data refit.** The campaign has a
   three-hour local wall-time cap, preserves progress, and uses the fixed JSON
   configurations. Actual duration is measured in the logs.

   ```bash
   python run_campaign.py --max-seconds 10800
   ```

   The campaign trains `baseline`, `architecture` and `regularized` candidates
   with aligned 160-epoch schedules, chooses raw/EMA and flip averaging using
   development only, freezes `results/selection.json`, evaluates the selected
   recipe once on confirmation, and retrains from scratch on all 50,000 images.
   The final recipe is `results/final_full_config.json`; the final model has no
   overlapping validation evaluation. Re-running the campaign resumes existing
   stages. Individual training commands also accept `--resume`, requiring matching
   config, source hashes and official data/split fingerprints. Optimizer, raw/EMA
   weights and RNG states are retained in `last.pt`.

4. **Inspect the final prediction and submit.** The campaign already runs final
   inference with the frozen development-selected rule. It can be rerun separately:

   ```bash
   python predict.py --checkpoint results/final_full/best.pt --output-dir results/final_full --device cuda
   ```

   The command writes `evaluation.json` and `submission.csv`, and checks exactly
   2,000 unique official IDs in sample order, columns `id,label`, and integer
   labels 0 through 9. Inspect the saved development/confirmation results and
   submission checksum before uploading:

   ```bash
   python scripts/submit_verified.py --run final_full --csv results/final_full/submission.csv
   uv run --no-project --with kaggle==2.2.5 python scripts/submit_verified.py --run final_full --csv results/final_full/submission.csv --execute
   ```

   The first command validates locally without uploading. The second uses an
   isolated official Kaggle client and existing secure authentication. The helper
   validates the completed training budget, official CSV and checkpoint hashes,
   records an upload intent/reference, and recovers a known submission instead of
   uploading duplicate bytes. It refuses an ambiguous prior attempt rather than
   silently spending another slot. The recorded `final_full` submission is already
   complete, so repeating this command returns its existing receipt.

   To regenerate the first three official benchmark predictions from their
   frozen checkpoints, use:

   ```bash
   python scripts/make_benchmark_submissions.py --runs baseline architecture regularized --device cuda
   ```

   These CSVs are under `results/submission_<run>/`; validate/upload each with the
   same helper and its corresponding `--run`. `robust_full` generates its own
   `results/robust_full/submission.csv` after its fixed training and inference;
   this delivery's robust entry is already scored as well.

   The team limit is five submissions per day; check the actual remaining quota.
   The five-entry campaign benchmarks `baseline`, `architecture`,
   `regularized`, `final_full`, and a separately labeled exploratory `robust_full`.
   The first three train on 40,000 images; both full-data variants use all 50,000.
   The seed-2027 regularized repeat remains a local validation-only stability
   check and is not the fifth upload. Store all scored receipts in
   `results/kaggle_submissions.json`, including run, training size, parameter
   count, reference, public score, file SHA-256 and timestamp. Keep the primary
   full-data receipt separately in `results/kaggle_submission.json`.

   Public feedback motivated the separate robust exploration and may guide
   official Kaggle entry selection. It does not revise the completed primary
   160-epoch recipe or repeat its confirmation study. Report both
   the primary full-data score and the highest observed public benchmark if
   they differ; the best of five on 1,000 public images can reflect sampling
   noise. Private leaderboard accuracy remains unknown until Kaggle releases it.

5. **Build the report and executed notebook from real evidence.**

   ```bash
   python build_report.py --results-dir results --final-run final_full --output-dir outputs
   python scripts/build_public_summary.py --output docs/results.json --require-complete --require-five-submissions
   ```

   The builder creates `outputs/ky2738_Project1_Report.pdf`,
   `outputs/Project1_Results_Executed.ipynb`, plots and a checksum manifest. It
   executes the results notebook, prints parameter count and public test accuracy,
   and enforces the eight-page PDF limit. Missing results stay visibly pending.
   The generated notebook reads saved artifacts; the complete training code is
   the separate `train.py` entrypoint. The strict summary command requires both
   `--require-complete` and `--require-five-submissions`; it verifies actual
   evidence/checkpoint bytes and all five official scored receipts before writing
   `docs/results.json`. It does not accept a pending fifth entry as a completed
   delivery. Its output contains only allowlisted configurations, aggregate
   results and hashes, excluding local paths, raw images, weights, prediction
   rows and credentials. Install report dependencies from `requirements/training.txt`
   if they are missing. Inspect every final notebook cell/output before copying
   only `outputs/Project1_Results_Executed.ipynb` to `notebooks/` for Git.

## Training and selection

The research split contains 40,000 training, 5,000 development and 5,000
confirmation images, stratified with 500 images per class in each holdout (seed
2026). Every candidate shares this split. The smaller WRN-16-2 baseline and
WRN-22-4 architecture comparison use crop/flip; the regularized WRN-22-4 adds
Cutout and MixUp together. All use aligned schedules, Nesterov SGD and EMA
evaluation. Actual configurations and histories remain the source of truth.

Training history records raw and EMA validation results. Checkpoint selection
uses development only; inference averaging is also chosen on development. The
selected recipe is checked once on confirmation, then a new model is trained on
all 50,000 images with fixed duration, selected raw/EMA variant and inference
rule. The full-data model has no validation score. Augmented/MixUp training
accuracy is not clean-image test accuracy. The regularization bundle does not
isolate the effects of its two components. A single split/seed does not establish
statistical significance. The public leaderboard is a separate partial hidden
test and does not establish the final private ranking.

The additional **exploratory `robust_full`** recipe was specified after official
public feedback: scratch WRN-22-4, all 50,000 official images, 300 SGD/cosine
epochs, RandAugment N=2/M=9 plus crop/flip, Cutout and MixUp, EMA decay 0.999,
clean-training-only batch-normalization recalibration, and fixed reflect-pad-four
five-crop-plus-flip logit averaging. The private free-quota T4 job completed all
300 epochs using FP16 with GradScaler, followed by train-only BN recalibration
and ten-view inference. Recorded training/calibration/prediction time was
**4,838.19 seconds (80.64 minutes)**. Its official public score was **86.20%**,
which did not improve the current best 87.40%. This bundled exploration has no development score and no
new confirmation evaluation; the earlier confirmation result belongs only to
the selected 40,000-image primary-study candidate.

The robust source/config are frozen at commit
`4c2d5fdcd109e3772a88ceaedf39bee9fdd48f8a`. To reproduce the same fixed recipe on
an authorized CUDA machine with official data, run it in a fresh output directory:

```bash
python train_robust.py --config configs/robust_full.json --data-dir data --output-dir results/robust_full --device cuda --max-seconds 10500
```

The job uses the fixed final-epoch EMA, recalibrates BN only on clean permitted
training images, and applies the prescribed ten views without selecting a
checkpoint or changing its TTA after observing its own public score.
`scripts/launch_robust_kaggle.py` packages the committed
source for a private, free-quota-checked cloud run; its existing launch-attempt
receipt prevents accidentally starting this delivery's job twice.

## Evidence and private artifacts

| Path | Contents | Git policy |
| --- | --- | --- |
| `model.py`, `train.py`, `predict.py`, `data.py`, `configs/` | Reproducible implementation and configuration | Source is tracked |
| `results/<run>/` | Actual metrics, configuration, environment, provenance, history and checkpoints | Ignored, retained locally |
| `data/` | Official competition files, manifests and split caches | Ignored, kept private |
| `outputs/` | Generated PDF, executed notebook, plots and checksums | Ignored; submit through course channels |
| `notebooks/Project1_Results_Executed.ipynb` | Audited final executed results, scalar metrics and plots only | Tracked only after the final output audit |
| `results/kaggle_submission.json` | Actual submission reference/public score/checksum | Ignored, retained locally |
| `results/kaggle_submissions.json` | Five fixed-model official benchmark receipts | Ignored; summarized in report/notebook |
| `docs/results.json` | Strictly verified, sanitized aggregate evidence and official receipts | Tracked after all five entries complete |

No model weights, prediction CSVs, credentials, raw competition images or full
unreviewed runtime outputs are committed. The final curated results notebook
may be tracked after verifying that its outputs contain only scalar metrics,
configurations, architecture/learning-curve plots and aggregate confusion counts.
Original code is licensed under [MIT](LICENSE).

The source has been shared with all participants in the verified
[competition discussion](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/discussion/754332),
satisfying the associated public-code-sharing step. The discussion links the
actual implementation branch and source revision. No competition data, weights
or prediction files are included. The optional `scripts/build_public_notebook.py`
can package a CPU-only source notebook; publishing such a notebook is unnecessary
for this delivery because the competition-associated discussion is already live.

## Course delivery and collaboration

The report names **Kevin Yang (ky2738)** and **Saber**; a future teammate must be
added only through official course/Kaggle procedures. Confirm the mandatory
course Google Form registration independently. The PDF must have a public GitHub
link on page one and be uploaded to Brightspace by each registered member.

The verified Kaggle deadline is **6 November 2026, 23:59:59 EST (UTC-05:00)**,
equivalent to **7 November 2026, 04:59:59 UTC**. The assignment gives
**9 November 2026, 23:59** for the report without a timezone; confirm that report
timezone in the course channels. Written project rules cap teams at three,
even though the Kaggle team interface displays capacity five.

Changes use one feature branch and a PR linked to its Issue. Human maintainer
review and explicit merge authorization remain required; an agent's review does
not supply human approval. Existing team conventions remain in
[team workflow](docs/team-workflow.md), [collaboration](docs/collaboration.md),
[agent instructions](AGENTS.md), and [API access](docs/api-access.md).

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Zagoruyko and Komodakis, [Wide Residual Networks](https://arxiv.org/abs/1605.07146).
- Zhang et al., [mixup](https://arxiv.org/abs/1710.09412).
- DeVries and Taylor, [Cutout](https://arxiv.org/abs/1708.04552).
- Loshchilov and Hutter, [SGDR](https://arxiv.org/abs/1608.03983), cosine-schedule reference; this project need not use restarts.

Exploratory augmentation reference: Cubuk et al.,
[RandAugment](https://arxiv.org/abs/1909.13719).
