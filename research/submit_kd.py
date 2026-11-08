"""Upload a predict_kd.py submission once, with row/hash/duplicate/quota checks, and record the receipt.

Usage: python research/submit_kd.py --prediction-dir DIR --message TEXT
Authorized by the owner in chat on 2026-11-04/05 for the KD route past the research stop time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.submit_candidate import KaggleClient  # noqa: E402

COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
KAGGLE = str(Path.home()/".local/bin/kaggle.exe")
RECEIPTS = ROOT/"results/research/kd_submission_receipts.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kaggle(*args: str) -> str:
    return subprocess.run([KAGGLE, "competitions", *args], capture_output=True, text=True, check=True, timeout=180).stdout


def latest_submissions() -> list[dict]:
    text = kaggle("submissions", COMPETITION, "--csv", "--page-size", "20")
    return list(csv.DictReader(io.StringIO(text[text.index("ref,fileName"):])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--message", required=True)
    a = p.parse_args()
    manifest = json.loads((a.prediction_dir/"prediction_manifest.json").read_text())
    path = a.prediction_dir/"submission.csv"
    digest = sha256(path)
    rows = list(csv.reader(open(path, encoding="utf-8")))
    sample = [r["id"] for r in csv.DictReader(open(ROOT/"data/sample_submission.csv", encoding="utf-8"))]
    if digest != manifest["submission_sha256"] or rows[0] != ["id", "label"] or [r[0] for r in rows[1:]] != sample \
            or not all(r[1] in tuple("0123456789") for r in rows[1:]) or manifest["parameter_count"] > 5_000_000:
        raise SystemExit("Submission failed row/hash/parameter checks")
    receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.exists() else []
    scored = json.loads((ROOT/"docs/research-official-scores.json").read_text())["scored_candidates"]
    if digest in {r["submission_sha256"] for r in receipts + scored}:
        raise SystemExit("Duplicate CSV already uploaded")
    limits = KaggleClient().limits()
    print("quota", limits)
    if limits["num_allowed_now"] < 1:
        raise SystemExit("No team submission left today")
    receipt = {"message": a.message, "submission_sha256": digest, "manifest": manifest, "quota_before": limits,
               "uploaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "upload_attempt_started"}
    receipts.append(receipt); RECEIPTS.write_text(json.dumps(receipts, indent=2))
    print(kaggle("submit", COMPETITION, "-f", str(path), "-m", a.message).strip())
    for _ in range(40):
        time.sleep(15)
        row = next((r for r in latest_submissions() if r.get("description") == a.message), None)
        if row and row.get("status", "").lower().endswith("complete"):
            receipt.update(status="complete", public_score=row.get("publicScore"), ref=row.get("ref"))
            break
    RECEIPTS.write_text(json.dumps(receipts, indent=2))
    print(json.dumps({k: receipt.get(k) for k in ("status", "public_score", "ref")}))


if __name__ == "__main__":
    main()
