# Verified competition and assignment requirements

Verified on **2026-10-30** from the authenticated Kaggle API, official manifests,
the local three-page `project1.pdf`, and prior verified research on
`origin/docs/verified-competition-rules`. Full fetched page text remains local
under ignored `data/competition_pages.json`. No credentials are stored in Git.

Competition: [CIFAR-10 ResNet Architecture Challenge](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).

## Model and permitted data

| Requirement | Evidence | Implementation consequence |
| --- | --- | --- |
| CNN with residual/skip connections | Description; rules sections 1 and 4; PDF pp. 1-2 | CIFAR WideResNet with projection shortcuts |
| At most 5,000,000 trainable parameters | Description; rules section 4; PDF pp. 1-2 | Also enforce total parameters; WRN-22-4 = 4,298,970 |
| Train every model from scratch | Rules section 5; PDF pp. 1-2 | Random initialization, no external checkpoints |
| Only official permitted training data | Data description; rules section 3 | Load official Kaggle train images and labels |
| No external datasets or pretrained weights | Rules sections 3 and 5 | No ImageNet, CIFAR-100, pretrained encoders or external teachers |
| No hidden-label reconstruction or manual labeling | Data description; rules section 12 | Hidden evaluation images used only for inference |

The official competition has **50,000** `train_images/` images and
`train_labels.csv` (`id,filename,label`), **2,000** `test_images/` images,
`test.csv` (`id,filename`), and `sample_submission.csv` (`id,label`). The prior
verified file inventory contains 52,003 unique files. Each training class has
5,000 images. Training inputs are 32 by 32 RGB.

Use the official competition mount or a fingerprinted export of those exact
files. An original Toronto CIFAR-10 archive downloaded during setup is unused;
it is not a training source. No official CIFAR test labels are loaded.

Label indices: `0 airplane`, `1 automobile`, `2 bird`, `3 cat`, `4 deer`,
`5 dog`, `6 frog`, `7 horse`, `8 ship`, `9 truck`.

## Evaluation and submission

| Item | Verified value | Source |
| --- | --- | --- |
| Metric | Classification accuracy, higher is better | Description; rules section 10 |
| Public/private split | 50% / 50%; private score determines final ranking | Data description; rules section 11 |
| Evaluation rows | 2,000 | Official test and sample manifests |
| Submission | `submission.csv`, columns exactly `id,label` | Data description; rules section 7 |
| Label domain | Integers 0 through 9 | Rules section 7 |
| ID/order | One prediction per image, same IDs and order as sample | Rules section 7 |
| Daily limit | 5 submissions per entire team | Rules section 8; submission-limits API |

The bulk competition ZIP endpoint returned HTTP 404. Individual downloads and
the official Kaggle competition mount work. `download_kaggle.py` uses bounded
individual downloads. The export helper can package the official mounted
training files into a private, fingerprinted artifact for local GPU training.
This observed platform behavior does not change the data restrictions.

## Team and public code

- Maximum team size is three. Each participant may belong to one team.
- The Kaggle team UI displays a platform capacity of five, but the written
  competition rules and assignment cap this project at **three**; enforce three.
- Course Google Form registration is mandatory; the registered team name must
  match Kaggle. The browser verified team **Saber**, with `Kevin_1220` as its sole
  current leader/member. The report identifies Kevin Yang (`ky2738`). Google
  Form registration is a separate course action and has not been verified.
- Sharing trained weights, predictions or private solutions across teams is forbidden.
- Public reference code is permitted with attribution. Kaggle foundational
  rules section 6 also requires publicly shared code to be shared in a
  competition-associated Kaggle discussion or notebook under an OSI-approved
  license. Original code in this repository uses MIT.
- GitHub permissions and Kaggle team membership are separate. A later teammate
  must complete the official registration and team-join process.

## Required deliverables

The PDF awards ten points for the report and ten for the GitHub code repository.
The report must be at most **eight pages including references**, name every
member, describe design choices and experiments, report architecture, parameter
count and test accuracy, and include citations. A publicly accessible GitHub
link must appear on page one. The repository must support reproduction. The PDF
also asks for notebooks with visible execution results, clear plots, and printed
final test accuracy and parameter count.

Course staff may request checkpoints, logs and reproduction. Preserve these in
controlled local artifacts; do not publish them across teams. Full generated
notebooks and reports are built under ignored `outputs/`, while their generator,
source code and summarized evidence are reviewable in Git.

## Deadlines and course actions

The authenticated Kaggle header tooltip was read in the browser and states
**Friday 2026-11-06 23:59:59 GMT-0500 (Eastern Standard Time)**. This is
**2026-11-07 04:59:59 UTC**. The PDF's competition date is consistent with that
deadline. The assignment gives **2026-11-09 at 23:59** for the report, without a
timezone; confirm the report timezone through the course announcements.
Competition rules section 16 still contains TBA text, so distinguish that stale
written placeholder from the verified live Kaggle deadline.

The GitHub repository is now publicly accessible. The implemented pipeline is
on [the feature branch](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/tree/feat/cifar10-resnet-submission)
and PR 6; `main` remains the earlier harness pending human review. Report links
therefore point directly to the code branch rather than implying it was merged.

The PDF itself states no AI-use prohibition. Kaggle rules require university
academic-integrity compliance; this file does not infer unseen course policies.
Brightspace upload, Google Form registration, team membership and human PR
review remain human/course actions unless separately completed and recorded.

## Sources

- [Competition overview](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview)
- [Competition rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules)
- [Competition data](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data)
- Local `project1.pdf`, pages 1-3 (course assignment, intentionally not republished)
- API pages `Description`, `data-description`, `rules`, `abstract`,
  `foundational-rules`; official CSV manifests and submission limits
- Earlier verified inventory: branch `docs/verified-competition-rules`,
  `docs/data-audit.json`; its separate T4 training results are not local WRN results
