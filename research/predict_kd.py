"""Predict the official 2,000 test rows with a fixed-final KD student checkpoint.

Inference only, policy center80_shift20 (predeclared, same as earlier deliveries).
Writes submission.csv plus a scalar-only manifest (checkpoint/CSV hashes, row checks).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import write_json  # noqa: E402  (isolates cuDNN before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from data import load_kaggle_test  # noqa: E402
from model import count_parameters  # noqa: E402
from research.train_kd_ensemble import make_model  # noqa: E402
from research.benchmark import infer  # noqa: E402

POLICY = "center80_shift20"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def predict(checkpoint: Path, out_dir: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    params = count_parameters(model, trainable_only=True)
    if params > 5_000_000 or count_parameters(model) > 5_000_000:
        raise ValueError("Model exceeds 5M parameters")
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    ids, pixels = load_kaggle_test(ROOT/"data")
    images = torch.as_tensor(pixels).cuda()
    labels = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(images), 250):
            labels.append(infer(model, images[start:start+250], POLICY).float().argmax(1).cpu())
    labels = torch.cat(labels).numpy()
    sample_ids = [row["id"] for row in csv.DictReader(open(ROOT/"data/sample_submission.csv", encoding="utf-8"))]
    if ids != sample_ids or len(labels) != 2000 or labels.min() < 0 or labels.max() > 9:
        raise ValueError("Prediction rows do not match the official sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir/"submission.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, labels.tolist()))
    manifest = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
                "epoch": ckpt["epoch"], "variant": ckpt["variant"], "parameter_count": params, "policy": POLICY,
                "rows": len(labels), "submission_sha256": sha256(path),
                "predicted_class_counts": np.bincount(labels, minlength=10).tolist()}
    write_json(out_dir/"prediction_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(predict(a.checkpoint.resolve(), a.output_dir))
