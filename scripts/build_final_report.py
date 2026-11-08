"""Build the final course report PDF from committed scalar evidence in docs/.

Writes docs/figures/final_*.png, reports/final/report.html and
reports/final/ky2738_Saber_Project1_Report.pdf (headless Edge/Chrome print).
Usage: python scripts/build_final_report.py
"""
from pathlib import Path
import base64
import json
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/final"
FIG = ROOT / "docs/figures"
REPO = "https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle"
scores = json.loads((ROOT / "docs/final-kaggle-scores.json").read_text())["submissions"]
curves = json.loads((ROOT / "docs/final-curves.json").read_text())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_kd():
    r, k = curves["no_kd_regularized_40k"], curves["kd_4teachers_40k"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(r["epoch"], [100*v for v in r["dev"]], lw=1, label="Same recipe, no KD")
    ax.plot(k["epoch"], [100*v for v in k["dev"]], lw=1, label="KD from 4 scratch teachers")
    ax.axhline(100*k["teacher_ensemble_dev"], ls="--", c="gray", lw=1, label="Teacher ensemble (97.08%)")
    ax.set(ylim=(86, 98.5), xlabel="Epoch", ylabel="Dev accuracy (%)")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "final_kd_curves.png")


def fig_scatter():
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    groups = {"Original study": range(0, 5), "Architecture / regularizer variants": range(5, 13), "Knowledge distillation": range(13, 16)}
    for name, idx in groups.items():
        ax.scatter([100*scores[i]["public"] for i in idx], [100*scores[i]["private"] for i in idx], s=22, label=name)
    for s in scores:
        if s["label"] in ("baseline", "final_full (pre-declared primary)", "TrivialAugment full-data refit", "KD 8 teachers"):
            ax.annotate(s["label"].split(" (")[0], (100*s["public"], 100*s["private"]), fontsize=7, xytext=(4, -3), textcoords="offset points")
    ax.plot([80, 90], [80, 90], ls=":", c="gray", lw=1)
    ax.set(xlim=(83, 89.5), ylim=(80, 88), xlabel="Public accuracy (%)", ylabel="Private accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return save(fig, "final_public_private.png")


def img(path, width):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width}">'


def score_rows():
    rows = []
    for s in scores:
        bold = s["label"] in ("KD 8 teachers",)
        cells = [s["label"], s["arch"], f'{s["params"]/1e6:.2f}M', f'{s["train_images"]//1000}k', s["epochs"], s["method"],
                 f'{100*s["public"]:.1f}', f'{100*s["private"]:.1f}']
        tag = ' class="hl"' if bold else ""
        rows.append(f"<tr{tag}>" + "".join(f'<td{"" if i in (0, 5) else " class=n"}>{c}</td>' for i, c in enumerate(cells)) + "</tr>")
    return "\n".join(rows)


def html():
    pub = np.array([s["public"] for s in scores]); priv = np.array([s["private"] for s in scores])
    corr_late = np.corrcoef(pub[3:], priv[3:])[0, 1]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Saber Project 1 Report</title><style>
@page {{ size: Letter; margin: 0.7in 0.75in; }}
body {{ font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 4px; }} h2 {{ font-size: 12.5pt; margin: 16px 0 5px; border-bottom: 1px solid #999; }}
h3 {{ font-size: 11pt; margin: 10px 0 3px; }} p {{ margin: 4px 0 7px; text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 6px 0; }}
th, td {{ border-bottom: 1px solid #ccc; padding: 2.5px 5px; text-align: left; }} th {{ background: #eee; }}
tr {{ break-inside: avoid; }} td.n {{ white-space: nowrap; }}
tr.hl td {{ font-weight: bold; background: #f3f3f3; }} .meta {{ font-size: 10pt; margin-bottom: 8px; }}
.cap {{ font-size: 8.8pt; color: #333; margin: 2px 0 10px; }} .fig {{ text-align: center; }} ul {{ margin: 3px 0 7px 18px; padding: 0; }}
li {{ margin: 2px 0; }} code {{ font-size: 9pt; }} .ref p {{ font-size: 9pt; margin: 2px 0; text-align: left; }} .pb {{ break-before: page; }}
</style></head><body>
<h1>Distilling an Ensemble into a 4.3M-Parameter Wide ResNet for CIFAR-10</h1>
<div class="meta"><b>Team Saber</b> &mdash; Kevin Yang (ky2738), New York University, Deep Learning Fall 2026, Project 1<br>
<b>Code (public GitHub):</b> <a href="{REPO}">{REPO}</a></div>

<h2>1. Overview</h2>
<p>We designed residual CNNs for the course Kaggle challenge: classify a held-out 2,000-image CIFAR-10-style test set
with at most 5 million parameters, training from scratch on the 50,000 official labeled images only. Our final model is a
pre-activation <b>WideResNet-22-4 with 4,298,970 parameters</b>, trained on all 50,000 images by
<b>knowledge distillation (KD) from an ensemble of eight scratch-trained residual networks</b>.</p>
<table><tr><th>Final model</th><th>Parameters</th><th>Public test acc.</th><th>Private test acc.</th><th>Dev acc. (40k-train KD run)</th></tr>
<tr class="hl"><td>WRN-22-4 student, KD from 8 teachers</td><td>4,298,970</td><td>88.8%</td><td>85.4%</td><td>96.96% (4 teachers)</td></tr></table>
<p><b>Main findings.</b></p>
<ul>
<li>Going from WRN-16-2 to WRN-22-4 and adding Cutout + MixUp raised development accuracy from 94.86% to 96.30%. A 4-teacher distillation then took the same student to 96.96%, almost the teacher ensemble's 97.08%.</li>
<li>The hidden test set is much harder than an in-distribution split: every model scores 8&ndash;14 points lower than on the development split. Test images are brighter and higher-contrast than the training images, which suggests a natural distribution shift.</li>
<li>Synthetic-corruption robustness (TrivialAugment, AugMix, RandAugment) improved synthetic stress tests by up to 42 points but did not improve public accuracy.</li>
<li>Across our 16 official submissions, public and private scores were barely correlated once models were reasonably strong (r = {corr_late:.2f}). Choosing models by public score fit noise: our best-public model scored 85.4% private, while the pre-declared primary model scored 86.5% and the best private entry 86.8%.</li>
</ul>

<h2>2. Methodology</h2>
<h3>2.1 Data and evaluation protocol</h3>
<p>We use only the official <code>train_images/</code> and <code>train_labels.csv</code>. A deterministic stratified split
(seed 2026) gives 40,000 optimization images, a 5,000-image <b>development</b> set used for all design decisions, and a 5,000-image
<b>confirmation</b> set evaluated exactly once. Final models are retrained from random initialization on all 50,000 images with
the recipe frozen beforehand. No pretrained weights, external data or test labels are used. The 2,000 test images are only read
for inference.</p>

<h3>2.2 Architecture design</h3>
<p>Every network follows the residual form <i>y</i> = <i>S</i>(<i>x</i>) + <i>F</i>(<i>x</i>) [1]. We use the pre-activation
ordering (BN&rarr;ReLU&rarr;conv) [2] because it keeps an identity path through each block. On 32&times;32 inputs we keep a
stride-1 3&times;3 stem and three stages at 32/16/8 resolution, followed by global average pooling (P = 8) and a linear
classifier. Under a fixed budget, the main design choice is how to split parameters between <b>depth</b> (blocks
<i>B<sub>i</sub></i>) and <b>width</b> (channels <i>C<sub>i</sub></i>):</p>
<table style="break-inside: avoid"><tr><th>Network</th><th>Blocks / stage</th><th>Channels C<sub>i</sub></th><th>F<sub>i</sub>/K<sub>i</sub></th><th>Params</th><th>Dev acc.</th><th>Pros / cons</th></tr>
<tr><td>WRN-16-2 (baseline)</td><td>2-2-2</td><td>32-64-128</td><td>3 / 1</td><td>0.69M</td><td>94.86%</td><td>Fast; underfits</td></tr>
<tr class="hl"><td>WRN-22-4 (final)</td><td>3-3-3</td><td>64-128-256</td><td>3 / 1</td><td>4.30M</td><td>95.72%*</td><td>Best accuracy per training hour</td></tr>
<tr><td>WRN-34-3</td><td>5-5-5</td><td>48-96-192</td><td>3 / 1</td><td>4.17M</td><td>95.96%&Dagger;</td><td>Deeper and thinner; ~10% slower; no gain</td></tr>
<tr><td>PreAct ResNet-18 (4 stages)</td><td>2-2-2-2</td><td>64-128-256-512</td><td>3 / 1</td><td>11.2M</td><td>96.0%&dagger;</td><td>Over budget; teacher candidate only</td></tr></table>
<p class="cap">*Same recipe as the baseline (crop/flip only); 96.30% after adding Cutout + MixUp. &dagger;With the full regularized recipe, flip averaging. &Dagger;Regularized recipe, single view (WRN-22-4: 96.08%). Other dev values use horizontal-flip averaging.</p>
<p>Most of the WRN-22-4 parameters (3.28M of 4.30M) are in the 256-channel last stage, where features are most class-specific.
Wider stages train faster on a GPU than deeper, thinner ones with the same parameter count [3]. We kept 3&times;3 convolutions,
1&times;1 projection shortcuts (only where shape changes) and 8&times;8 pooling: larger kernels cost parameters quadratically with
little gain at 32&times;32. A four-stage ResNet-18 downsamples to 4&times;4 and must be narrowed to about base width 40 to fit
5M parameters. At full width (11.2M) it was still below WRN-22-4 and did not add useful ensemble diversity (Section 3.3).</p>

<h3>2.3 Training recipe</h3>
<p>SGD with Nesterov momentum 0.9, batch 128, peak learning rate 0.1 with 5 warm-up epochs then cosine decay [4], weight
decay 5&times;10<sup>-4</sup>, 160 epochs. Augmentation: random crop with 4-pixel padding, horizontal flip, Cutout 16&times;16 [5],
and MixUp (&alpha; = 0.2, applied to half the batches) [6]. Training uses BF16 autocast and channels-last tensors on one RTX 5060
laptop GPU (about 11 s per epoch; 160 epochs take 30 minutes). Inference averages logits over the image and its mirror (weight 0.8) and over
four 2-pixel reflect-padded shifts with mirrors (weight 0.2).</p>
<p>We also tested, each on the fixed development split: SAM [7] (&rho; = 0.05), TrivialAugment [8], ResizeMix [9],
RandAugment [10] with EMA weights, 300-epoch schedules and full-data refits. The best single 40k models reached
96.7&ndash;96.9% dev; none was clearly better than the others on the hidden test set (Table 2).</p>

<h3>2.4 Ensemble knowledge distillation</h3>
<p>An ensemble of our models was clearly stronger than any single one: four WRNs reached about 97.1% dev and eight reached 97.40%,
versus at most 96.9% for one model. But an ensemble breaks the 5M limit, and the rules allow teacher&ndash;student
distillation. We therefore train one WRN-22-4 student to match the averaged softened predictions of the teachers [11]:</p>
<p style="text-align:center"><i>L</i> = 0.1 &middot; CE(<i>y</i><sub>mix</sub>, <i>z</i><sub>s</sub>) + 0.9 &middot; <i>T</i><sup>2</sup> &middot;
KL( mean<sub>k</sub> softmax(<i>z</i><sub>k</sub>/<i>T</i>) &Vert; softmax(<i>z</i><sub>s</sub>/<i>T</i>) ),&nbsp;&nbsp; <i>T</i> = 4.</p>
<p>Teachers are computed online on the <i>same</i> augmented and MixUp-ed batch the student sees ("consistent teaching" [12]),
so the soft targets match the augmented input. Every teacher was itself trained from scratch on the official training images
(WRN-22-4 with SAM, WRN-34-3 with TrivialAugment, ResizeMix and plain-recipe models, including full-data refits). The student
has the same 4,298,970 parameters as before. Only its targets change. KD costs about 40 s per epoch with eight teachers
(1.8 h for 160 epochs, versus 0.5 h without distillation).</p>

<h2 class="pb">3. Results</h2>
<h3>3.1 Development ablations</h3>
<table><tr><th>Step (40k training images, 160 epochs, seed 2026)</th><th>Params</th><th>Dev acc.</th><th>&Delta;</th></tr>
<tr><td>Baseline WRN-16-2, crop/flip</td><td>0.69M</td><td>94.86%</td><td>&ndash;</td></tr>
<tr><td>+ wider/deeper WRN-22-4</td><td>4.30M</td><td>95.72%</td><td>+0.86</td></tr>
<tr><td>+ Cutout + MixUp (regularized)</td><td>4.30M</td><td>96.30%</td><td>+0.58</td></tr>
<tr><td>&nbsp;&nbsp;seed repeat (2027)</td><td>4.30M</td><td>96.36%</td><td>(+0.06)</td></tr>
<tr><td>&nbsp;&nbsp;one-time confirmation split</td><td>4.30M</td><td>96.64%</td><td>&ndash;</td></tr>
<tr class="hl"><td>+ KD from 4 scratch teachers (final epoch)</td><td>4.30M</td><td>96.96%</td><td>+0.66</td></tr>
<tr><td>Teacher ensemble (not submittable: 4 models)</td><td>16.9M</td><td>97.08%</td><td>&ndash;</td></tr></table>
<p class="cap">Table 1. Development accuracy with flip test-time averaging. Each step changes the listed factor jointly, so architecture
and regularization effects are not separated further. One seed per row except the repeat.</p>
<div class="fig">{img(FIG / "final_kd_curves.png", "86%")}</div>
<p class="cap">Figure 1. Development accuracy (single view) of the same WRN-22-4 recipe with and without distillation. The KD student ends
at 96.64% versus 95.80% and nearly reaches its teacher ensemble.</p>

<h3>3.2 Hidden test set (Kaggle)</h3>
<table><tr><th>Entry</th><th>Arch.</th><th>Params</th><th>Train</th><th>Ep.</th><th>Method</th><th>Public %</th><th>Private %</th></tr>
{score_rows()}</table>
<p class="cap">Table 2. All 16 official submissions. Public and private halves have 1,000 images each, so one score has a binomial standard error
of about 1.1 points. The highlighted row is our final model (best public score).</p>
<div class="fig">{img(FIG / "final_public_private.png", "52%")}</div>
<p class="cap">Figure 2. Public versus private accuracy of each submission. After the first three entries, public score predicts private
score only weakly (r = {corr_late:.2f}).</p>

<h3>3.3 Analysis</h3>
<p><b>Distribution shift.</b> All models lose 8&ndash;14 points from development to test. Unlabeled pixel statistics show the test
images are brighter (mean RGB 0.513/0.498/0.461 vs 0.491/0.482/0.447) and higher-contrast, with no blur or noise. This pattern
fits a natural re-collection shift like CIFAR-10.1 [13], where accuracy drops sharply but model rankings mostly persist. Our
four teachers agree on 95.3% of development images but only 88.4% of test images. Synthetic-corruption augmentation improved
blur and noise stress accuracy by up to 42 points but did not raise test accuracy. More real training data was the most reliable lever: refitting on 50k instead of 40k
images raised private accuracy in three of four matched pairs (mean +0.85 points).</p>
<p><b>Why distillation helped only a little on the test set.</b> The student almost matches its teachers (96.96% vs 97.08% dev),
so its ceiling is the ensemble. On the shifted test set, a large share of errors are made by <i>all</i> teachers at once:
43% of the ensemble's development errors already have unanimous wrong votes. KD cannot correct those. A 4-stage ResNet-18
teacher did not help: its test disagreement with the WRNs (7.8%) was barely above WRN&ndash;WRN disagreement (7.4%). KD raised
public accuracy from 88.1% to 88.8% but private stayed within noise (85.4&ndash;85.8%).</p>
<p><b>Leaderboard overfitting.</b> With 1,000 public images, two models 1 point apart are statistically indistinguishable. We
selected later models by public score and gradually overfit it. The model we pre-declared before seeing any test feedback
(<code>final_full</code>) scored 86.5% private, better than our best-public model (85.4%). The best private entry was the
full-data WRN-34-3 with TrivialAugment (86.8%), which ranked only mid-table on public. Longer training (300 epochs) changed
private accuracy by at most +0.4 points (WRN-34-3: 84.4&rarr;84.7; KD: 85.4&rarr;85.8).</p>

<h3>3.4 Lessons learned</h3>
<ul>
<li>At this budget, a wide 22-layer network matched or beat a deeper, thinner 34-layer one and trained faster.</li>
<li>Regularization (Cutout + MixUp) and full-data refits gave reliable gains. Corruption-style augmentation did not transfer to a natural shift.</li>
<li>Ensemble distillation is a legal way to compress several models into the budget, but the student cannot exceed its teachers, and teacher <i>diversity</i> matters more than teacher count.</li>
<li>Fix the selection rule before looking at the leaderboard and report uncertainty: public differences under about 2 points were noise here.</li>
</ul>

<h2>4. Reproducibility</h2>
<p>The repository contains the model (<code>model.py</code>), data split (<code>data.py</code>, <code>prepare_data.py</code>),
training (<code>train.py</code>, <code>research/train_kd_ensemble.py</code>), inference (<code>research/predict_kd.py</code>),
tests and executed notebooks (<code>notebooks/Project1_Final_Results.ipynb</code> prints the parameter count and final accuracies).
Each run stores its configuration, data fingerprints, environment, per-epoch history and checkpoint hashes. Data, weights and
prediction files are kept out of Git.</p>

<h2>References</h2><div class="ref">
<p>[1] K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.</p>
<p>[2] K. He, X. Zhang, S. Ren, J. Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.</p>
<p>[3] S. Zagoruyko, N. Komodakis. Wide Residual Networks. BMVC 2016.</p>
<p>[4] I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.</p>
<p>[5] T. DeVries, G. Taylor. Improved Regularization of CNNs with Cutout. arXiv:1708.04552, 2017.</p>
<p>[6] H. Zhang, M. Cisse, Y. Dauphin, D. Lopez-Paz. mixup: Beyond Empirical Risk Minimization. ICLR 2018.</p>
<p>[7] P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization. ICLR 2021.</p>
<p>[8] S. M&uuml;ller, F. Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV 2021.</p>
<p>[9] J. Qin et al. ResizeMix: Mixing Data with Preserved Object Information and True Labels. arXiv:2012.11101, 2020.</p>
<p>[10] E. Cubuk, B. Zoph, J. Shlens, Q. Le. RandAugment. CVPR Workshops 2020.</p>
<p>[11] G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.</p>
<p>[12] L. Beyer et al. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022.</p>
<p>[13] B. Recht, R. Roelofs, L. Schmidt, V. Shankar. Do CIFAR-10 Classifiers Generalize to CIFAR-10? arXiv:1806.00451, 2018.</p>
</div></body></html>"""


def browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).is_file():
            return p
    return shutil.which("chromium") or shutil.which("google-chrome")


def main():
    FIG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    fig_kd(); fig_scatter()
    page = OUT / "report.html"
    page.write_text(html(), encoding="utf-8")
    pdf = OUT / "ky2738_Saber_Project1_Report.pdf"
    # Print into a fresh temp dir (own browser profile), then copy: a PDF open in a viewer stays locked on Windows,
    # and the browser exits 0 without writing, which would silently keep a stale PDF.
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / pdf.name
        subprocess.run([browser(), "--headless", "--disable-gpu", f"--user-data-dir={tmp}/profile", "--no-pdf-header-footer",
                        f"--print-to-pdf={fresh}", page.as_uri()], check=True, capture_output=True, timeout=120)
        assert fresh.is_file(), "browser did not write the PDF"
        try:
            shutil.copyfile(fresh, pdf)
        except PermissionError:
            raise SystemExit(f"{pdf} is open in another program; close it and rerun") from None
    from pypdf import PdfReader
    pages = len(PdfReader(pdf).pages)
    assert pages <= 8, f"report has {pages} pages (limit 8)"
    print(f"wrote {pdf} ({pages} pages)")


if __name__ == "__main__":
    main()
