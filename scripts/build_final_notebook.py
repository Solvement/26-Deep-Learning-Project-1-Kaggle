"""Build and execute notebooks/Project1_Final_Results.ipynb from sanitized docs/ evidence.

Inputs are scalar-only files committed in docs/ (scores, learning curves); no data,
weights or prediction rows are read, so the notebook re-runs on a fresh clone (CPU).
Usage: python scripts/build_final_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/Project1_Final_Results.ipynb"
REPO = "https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle"

md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = [
    md(f"# Project 1: final results\n\n**Team Saber: Kevin Yang (ky2738)**  \nCode: [{REPO}]({REPO})\n\n"
       "This notebook prints the final model's parameter count and test accuracy and plots the measured "
       "learning curves and Kaggle scores. It reads only scalar evidence committed under `docs/`."),
    code("from pathlib import Path\nimport json, sys\n"
         "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'model.py').is_file())\n"
         "sys.path.insert(0, str(ROOT))\n"
         "scores = json.loads((ROOT / 'docs/final-kaggle-scores.json').read_text())['submissions']\n"
         "curves = json.loads((ROOT / 'docs/final-curves.json').read_text())\n"
         "print(len(scores), 'official Kaggle submissions loaded')"),
    md("## 1. Final architecture and parameter budget\n"
       "The final student is a pre-activation WideResNet-22-4: 3x3 stem (16 ch), three stages of three "
       "residual blocks (64/128/256 channels, stride 1/2/2), 1x1 projection shortcuts, global average pooling, linear head."),
    code("import torch\nfrom model import build_model, count_parameters\n"
         "model = build_model(depth=22, widen_factor=4, dropout=0.0, num_classes=10)\n"
         "n_total, n_train = count_parameters(model), count_parameters(model, trainable_only=True)\n"
         "assert n_total <= 5_000_000 and n_train == n_total\n"
         "print(f'Trainable parameters: {n_train:,}  (limit 5,000,000)')\n"
         "print('Output shape:', tuple(model(torch.zeros(2, 3, 32, 32)).shape))\n"
         "for name, child in model.named_children():\n"
         "    print(f'  {name:10s} {count_parameters(child):>10,}')"),
    md("## 2. Knowledge distillation vs. the same recipe without it\n"
       "Both runs: WRN-22-4, 40,000 training images, 160 epochs, seed 2026, SGD + cosine, Cutout + MixUp. "
       "The KD run adds the soft targets of four scratch-trained teachers (T=4, alpha=0.9). "
       "Accuracy is on the 5,000-image development split."),
    code("import matplotlib.pyplot as plt\n"
         "r, k = curves['no_kd_regularized_40k'], curves['kd_4teachers_40k']\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))\n"
         "ax[0].plot(r['epoch'], [100*v for v in r['dev']], label='no KD (regularized)')\n"
         "ax[0].plot(k['epoch'], [100*v for v in k['dev']], label='KD, 4 teachers')\n"
         "ax[0].axhline(100*k['teacher_ensemble_dev'], ls='--', c='gray', label='teacher ensemble')\n"
         "ax[0].set(ylim=(85, 98), xlabel='epoch', ylabel='dev accuracy (%)', title='Development accuracy'); ax[0].legend()\n"
         "k8 = curves['kd_8teachers_50k']\n"
         "ax[1].plot(k8['epoch'], k8['loss']); ax[1].set(xlabel='epoch', ylabel='KD loss', title='Final model (8 teachers, 50k) training loss')\n"
         "plt.tight_layout(); plt.show()\n"
         "print(f\"No KD : final dev {100*r['dev'][-1]:.2f}%  (best {100*max(r['dev']):.2f}%)\")\n"
         "print(f\"KD    : final dev {100*k['dev'][-1]:.2f}%  flip-TTA {100*k['dev_flip'][-1]:.2f}%\")\n"
         "print(f\"Teacher ensemble dev: {100*k['teacher_ensemble_dev']:.2f}%\")"),
    md("## 3. All official Kaggle scores (public and private halves, 1,000 images each)"),
    code("import numpy as np\n"
         "print(f\"{'entry':34s}{'arch':10s}{'train':>7s}{'ep':>5s}{'public':>8s}{'private':>9s}\")\n"
         "for s in scores:\n"
         "    print(f\"{s['label']:34s}{s['arch']:10s}{s['train_images']:>7d}{s['epochs']:>5d}{100*s['public']:>7.1f}%{100*s['private']:>8.1f}%\")\n"
         "pub, priv = np.array([s['public'] for s in scores]), np.array([s['private'] for s in scores])\n"
         "fig, ax = plt.subplots(figsize=(5.5, 4.5))\n"
         "ax.scatter(100*pub, 100*priv)\n"
         "for s in scores: ax.annotate(s['label'].split(' (')[0], (100*s['public'], 100*s['private']), fontsize=6)\n"
         "ax.plot([80, 90], [80, 90], ls=':', c='gray'); ax.set(xlabel='public (%)', ylabel='private (%)', title='Public vs private')\n"
         "plt.show()\n"
         "late = slice(3, None)\n"
         "print(f'corr(public, private) all={np.corrcoef(pub, priv)[0,1]:.2f}; after the first three entries={np.corrcoef(pub[late], priv[late])[0,1]:.2f}')\n"
         "print(f'binomial SE of one 1,000-image half at 86%: {100*np.sqrt(.86*.14/1000):.1f} pp')"),
    md("## 4. Final test accuracy"),
    code("final = next(s for s in scores if s['label'] == 'KD 8 teachers')\n"
         "best_priv = max(scores, key=lambda s: s['private'])\n"
         "print(f\"FINAL MODEL: {final['arch']} student, {final['method']}, {final['train_images']:,} images\")\n"
         "print(f\"Number of parameters: {final['params']:,}\")\n"
         "print(f\"Final test accuracy: public {100*final['public']:.1f}%, private {100*final['private']:.1f}%  (Kaggle ref {final['ref']})\")\n"
         "print(f\"Best private score among all entries: {100*best_priv['private']:.1f}% ({best_priv['label']}, {best_priv['params']:,} params)\")"),
    md("## Reproduction\n"
       "```\npython prepare_data.py            # official Kaggle files only\n"
       "python train.py --config configs/wrn_final.json --output-dir results/final_full   # one teacher / no-KD baseline\n"
       "python research/train_kd_ensemble.py --teachers full50k_8 --output-dir results/research/kd_full50k8_s2026_v1\n"
       "python research/predict_kd.py --checkpoint results/research/kd_full50k8_s2026_v1/final_model.pt --output-dir out/\n```\n"
       "See README for the teacher runs. Data, checkpoints and prediction files are not stored in Git."),
]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbformat.write(nb, OUT)
print("wrote", OUT)
