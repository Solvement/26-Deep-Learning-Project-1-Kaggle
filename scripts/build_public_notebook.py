"""Package an explicit committed-source allowlist as a public CPU code notebook.

This builder never reads competition data, checkpoints, predictions or secrets.
It does not publish anything. The generated Kaggle notebook only verifies the
archive and inspects a randomly initialized CPU model; training is opt-in.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
KERNEL_ID = "kevin1220/saber-efficient-wrn-reproducible-code"
TITLE = "Saber Efficient WRN Reproducible Code"
NOTEBOOK_NAME = "saber_reproducible_code.ipynb"
SOURCE_ALLOWLIST = (
    "model.py", "train.py", "data.py", "runtime.py", "prepare_data.py",
    "predict.py", "confirm.py", "run_campaign.py", "build_report.py",
    "configs/wrn_baseline.json", "configs/wrn_architecture.json", "configs/wrn_final.json",
    "scripts/export_official_data.py", "scripts/export_official_test.py",
    "requirements/training.txt", "LICENSE", "README.md",
)


def git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, stderr=subprocess.PIPE)


def source_archive(revision: str) -> tuple[bytes, dict]:
    """Read only named Git blobs from one immutable commit, never the worktree."""
    commit = git_bytes("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    buffer = io.BytesIO()
    files = {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in SOURCE_ALLOWLIST:
            payload = git_bytes("show", f"{commit}:{name}")
            # Guard against accidentally committing a credential in an allowed
            # source file. The check reports only the filename, never its data.
            if b"KGAT_" in payload or b"sk-proj-" in payload or b"github_pat_" in payload:
                raise RuntimeError(f"Credential-like literal detected in allowlisted source: {name}")
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, payload)
            files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    payload = buffer.getvalue()
    return payload, {"source_commit": commit, "license": "MIT",
                     "archive_sha256": hashlib.sha256(payload).hexdigest(),
                     "files": files, "contains_competition_data": False,
                     "contains_trained_weights": False, "contains_predictions": False,
                     "automatic_training": False, "automatic_submission": False}


def build_notebook(payload: bytes, manifest: dict):
    import nbformat

    encoded = base64.b64encode(payload).decode("ascii")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    nb.metadata["saber_source"] = {"commit": manifest["source_commit"],
                                    "archive_sha256": manifest["archive_sha256"],
                                    "license": "MIT", "training_enabled": False}
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Saber: efficient WideResNet reproducible source\n\n"
            "**Kevin Yang (ky2738), team Saber**\n\n"
            f"[Fixed training source](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/tree/{manifest['source_commit']}) · "
            f"[Associated competition](https://www.kaggle.com/competitions/{COMPETITION})\n\n"
            "This competition-associated notebook shares the original MIT-licensed source with all "
            "participants. It contains **source code only**: no competition images, trained weights, "
            "prediction files, hidden labels or credentials.\n\n"
            "The default notebook uses CPU with internet disabled. It unpacks and verifies the source "
            "and prints a fresh WRN-22-4's parameter count. **It does not train a model or make a submission.**\n\n"
            f"Embedded source commit: `{manifest['source_commit']}`. "
            f"Archive SHA-256: `{manifest['archive_sha256']}`."),
        nbformat.v4.new_code_cell(
            "# Force CPU inspection even when this notebook is run locally on a GPU machine.\n"
            "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
            "os.environ['OMP_NUM_THREADS'] = '2'\nos.environ['MKL_NUM_THREADS'] = '2'\n"
            "from pathlib import Path, PurePosixPath\n"
            "import base64, hashlib, io, json, zipfile\n"
            f"SOURCE_MANIFEST = json.loads({json.dumps(json.dumps(manifest, sort_keys=True))})\n"
            f"SOURCE_ZIP_BASE64 = {encoded!r}\n"
            "archive_bytes = base64.b64decode(SOURCE_ZIP_BASE64, validate=True)\n"
            "assert hashlib.sha256(archive_bytes).hexdigest() == SOURCE_MANIFEST['archive_sha256']\n"
            "source_dir = Path.cwd() / 'saber_code'\nsource_dir.mkdir(exist_ok=True)\n"
            "with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:\n"
            "    assert set(archive.namelist()) == set(SOURCE_MANIFEST['files'])\n"
            "    for name in archive.namelist():\n"
            "        relative = PurePosixPath(name)\n"
            "        assert not relative.is_absolute() and '..' not in relative.parts\n"
            "        target = source_dir.joinpath(*relative.parts)\n"
            "        assert target.resolve().is_relative_to(source_dir.resolve())\n"
            "        data = archive.read(name)\n"
            "        assert hashlib.sha256(data).hexdigest() == SOURCE_MANIFEST['files'][name]['sha256']\n"
            "        target.parent.mkdir(parents=True, exist_ok=True)\n"
            "        target.write_bytes(data)\n"
            "print('Verified source commit:', SOURCE_MANIFEST['source_commit'])\n"
            "print('Verified source files:', len(SOURCE_MANIFEST['files']))\n"
            "print('Extracted source directory:', source_dir.name)\n"
            "print('Data, weights and predictions included: False')"),
        nbformat.v4.new_markdown_cell(
            "## Scratch model and budget check\n\n"
            "WRN-22-4 uses a 3×3 CIFAR stem, three pre-activation residual stages and global average "
            "pooling. All weights below are initialized randomly. The constructor enforces no more "
            "than 5,000,000 total parameters, including the classification head and normalization."),
        nbformat.v4.new_code_cell(
            "import sys\nsys.path.insert(0, str(source_dir))\n"
            "from runtime import configure_runtime\nconfigure_runtime()\n"
            "import torch\ntorch.set_num_threads(2)\ntorch.manual_seed(2026)\n"
            "from model import WideResNet, count_parameters\n"
            "model = WideResNet(depth=22, widen_factor=4).cpu().eval()\n"
            "total = count_parameters(model)\n"
            "trainable = count_parameters(model, trainable_only=True)\n"
            "assert total == trainable == 4_298_970\n"
            "assert total <= 5_000_000\n"
            "with torch.inference_mode():\n"
            "    logits = model(torch.randn(2, 3, 32, 32))\n"
            "assert tuple(logits.shape) == (2, 10) and torch.isfinite(logits).all()\n"
            "print(f'Total parameters: {total:,}')\n"
            "print(f'Trainable parameters: {trainable:,}')\n"
            "print('Initialized from scratch: True')\n"
            "print('CPU forward shape:', tuple(logits.shape))\n"
            "print('Training executed: False | Kaggle submission executed: False')\n"
            "del model, logits"),
        nbformat.v4.new_markdown_cell(
            "## Reproduction protocol\n\n"
            "Use only the official competition `train_images/` and `train_labels.csv`. "
            "The supplied private export helpers can package official attached inputs for local training. "
            "Run exports and real training in a **private copy**, keeping competition images, weights "
            "and prediction outputs private. This public source notebook does not execute those helpers.\n\n"
            "The fixed stratified research partition is **40,000 train / 5,000 development / 5,000 "
            "one-time confirmation**, seed 2026. Three candidates compare a smaller WRN-16-2, "
            "WRN-22-4 with crop/flip, and WRN-22-4 with a Cutout-plus-MixUp bundle. They use the "
            "same 160-epoch schedule. Development chooses the recipe, checkpoint variant and flip "
            "averaging; confirmation is checked once after selection. A new full-data model is "
            "then trained on all 50,000 official training images with the frozen recipe. "
            "The full-data model has no overlapping validation score.\n\n"
            "After authorized official inputs have been prepared in the extracted source's `data/` "
            "directory, the following commands reproduce the pipeline. They are shown as documentation "
            "and are **not executed** by this notebook:\n\n"
            "```bash\ncd /kaggle/working/saber_code\n"
            "python prepare_data.py --data-dir data --seed 2026\n"
            "python run_campaign.py --max-seconds 10800\n"
            "python predict.py --checkpoint results/final_full/best.pt --output-dir results/final_full --device cuda\n"
            "python build_report.py --results-dir results --final-run final_full --output-dir outputs\n```\n\n"
            "Review `README.md`, `requirements/training.txt` and the actual configuration files. "
            "GPU availability and throughput depend on the runtime. Running these commands requires "
            "opting into a suitable GPU runtime and its resource budget; no GPU is requested here. "
            "Kaggle upload is a separate action after CSV validation. The team limit is five "
            "submissions per day. No hidden labels, external datasets or pretrained weights may be used."),
        nbformat.v4.new_markdown_cell(
            "## License and references\n\n"
            "Original implementation: **MIT License**, copyright 2026 Kevin Yang. "
            "The complete license is included in the source archive. Architecture and method references:\n\n"
            "- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).\n"
            "- Zagoruyko and Komodakis, [Wide Residual Networks](https://arxiv.org/abs/1605.07146).\n"
            "- Zhang et al., [mixup](https://arxiv.org/abs/1710.09412).\n"
            "- DeVries and Taylor, [Cutout](https://arxiv.org/abs/1708.04552).\n"
            "- Loshchilov and Hutter, [SGDR](https://arxiv.org/abs/1608.03983), cosine-schedule reference; "
            "this implementation uses no warm restarts.\n\n"
            "A single split and seed do not establish statistical significance. The regularization "
            "comparison changes a bundle and cannot isolate the individual Cutout and MixUp effects. "
            "This notebook reports parameter and execution checks, **not competition accuracy**."),
    ]
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD", help="Immutable Git source revision to embed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp/public-code")
    parser.add_argument("--execute", action="store_true", help="Execute CPU-only notebook locally as a smoke check")
    parser.add_argument("--standalone", action="store_true",
                        help="Share code without attaching restricted competition inputs; retains the explicit competition link")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, manifest = source_archive(args.revision)
    nb = build_notebook(payload, manifest)
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(args.output_dir.resolve())}}).execute()
    import nbformat
    target = args.output_dir / NOTEBOOK_NAME
    nbformat.write(nb, target)
    metadata = {"id": KERNEL_ID, "title": TITLE, "code_file": NOTEBOOK_NAME,
                "language": "python", "kernel_type": "notebook", "is_private": False,
                "enable_gpu": False, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [] if args.standalone else [COMPETITION],
                "kernel_sources": [], "model_sources": []}
    (args.output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(target.resolve()),
                      "metadata": str((args.output_dir / "kernel-metadata.json").resolve()),
                      "source_manifest": str((args.output_dir / "source_manifest.json").resolve()),
                      "source_commit": manifest["source_commit"],
                      "source_files": len(manifest["files"]), "archive_bytes": len(payload),
                      "notebook_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "executed_locally_on_cpu": args.execute,
                      "competition_data_attached": not args.standalone,
                      "published": False}, indent=2))


if __name__ == "__main__":
    main()
