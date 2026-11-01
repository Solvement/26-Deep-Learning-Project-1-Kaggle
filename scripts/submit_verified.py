"""Validate one completed run; upload once only with --execute.

Uses only the standard library plus the official Kaggle client when executing.
An upload intent and immediate reference receipt prevent ambiguous retries.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
SAMPLE_SHA = "83aa7ddd58d46026c3b3f870f507c33b178ebfbdf24e0a69d9f7a9d6d8cfff59"
ALLOWED_RUNS = ("baseline", "architecture", "regularized", "final_full", "robust_full")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_csv(path, sample, expected_sample_sha=SAMPLE_SHA):
    if sha256(sample) != expected_sample_sha:
        raise ValueError("Official sample submission SHA-256 mismatch")
    def rows(csv_path):
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "label"]:
                raise ValueError("CSV columns must be exactly id,label")
            result = list(reader)
        if any(set(row) != {"id", "label"} for row in result):
            raise ValueError("Malformed CSV rows")
        return result
    expected, actual = rows(sample), rows(path)
    ids = [row["id"] for row in actual]
    if len(actual) != 2000 or len(expected) != 2000 or len(set(ids)) != 2000:
        raise ValueError("Submission must have exactly 2,000 unique IDs")
    if ids != [row["id"] for row in expected]:
        raise ValueError("Submission ID order differs from the official sample")
    if any(row["label"] not in {str(i) for i in range(10)} for row in actual):
        raise ValueError("Every label must be an integer 0..9")
    return sha256(path)


def validate_run(run, csv_path):
    if run not in ALLOWED_RUNS:
        raise ValueError("Run is not authorized for submission")
    folder = RESULTS / run
    metrics = read_json(folder / "metrics.json")
    if not metrics or metrics.get("status") != "complete":
        raise ValueError("Training run is not complete")
    full_data_run = run in {"final_full", "robust_full"}
    examples = 50000 if full_data_run else 40000
    count = metrics.get("parameter_count", 0)
    if metrics.get("training_examples") != examples or not isinstance(count, int) or not 0 < count <= 5_000_000:
        raise ValueError("Training example count or parameter budget is invalid")
    if metrics.get("epochs_completed") != metrics["config"]["epochs"]:
        raise ValueError("Training did not finish its configured epoch budget")
    if full_data_run and (metrics["config"].get("train_all") is not True or
                          metrics["config"]["epochs"] != (300 if run == "robust_full" else 160)):
        raise ValueError("Full-data run requires train_all=true and its fixed epoch budget")
    digest = validate_csv(csv_path, ROOT / "data" / "sample_submission.csv")
    checkpoint_sha = sha256(folder / "best.pt")
    meta = read_json(csv_path.parent / "metadata.json")
    if meta is not None:
        if meta.get("run") != run or meta.get("competition") != COMPETITION:
            raise ValueError("Prediction metadata belongs to another run/competition")
        if meta.get("hidden_test_labels_loaded") is not False:
            raise ValueError("Prediction metadata must exclude hidden test labels")
        if meta.get("original_evaluation_sha256") != sha256(folder / "evaluation.json"):
            raise ValueError("Original development evaluation changed")
    elif run == "final_full" and csv_path.resolve() == (folder / "submission.csv").resolve():
        evaluation = read_json(folder / "evaluation.json")
        if evaluation.get("validation_accuracy") is not None or not metrics["config"].get("train_all"):
            raise ValueError("Final full-data model must have no overlapping validation score")
        meta = {
            "submission_sha256": evaluation["submission"]["sha256"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "parameter_count": evaluation["parameter_count"], "training_examples": examples,
            "flip_tta": evaluation["flip_tta"],
            "source_git_commit": evaluation["provenance"]["git_commit"],
        }
    else:
        raise ValueError("Verified prediction metadata.json is missing")
    if meta.get("submission_sha256") != digest or meta.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Prediction CSV or checkpoint SHA-256 differs from metadata")
    if meta.get("parameter_count") != count or meta.get("training_examples") != examples:
        raise ValueError("Prediction metadata disagrees with completed training")
    if not isinstance(meta.get("flip_tta"), bool) or not meta.get("source_git_commit"):
        raise ValueError("Prediction inference rule or source revision is missing")
    details = {"run": run, "submission_sha256": digest, "training_examples": examples,
               "parameter_count": count, "flip_tta": meta["flip_tta"],
               "source_git_commit": meta["source_git_commit"]}
    if "tta_mode" in meta:
        if not isinstance(meta["tta_mode"], str) or not meta["tta_mode"]:
            raise ValueError("Prediction tta_mode must be a nonempty string")
        details["tta_mode"] = meta["tta_mode"]
    return details


def duplicate_receipt(receipts, digest):
    if not isinstance(receipts, list):
        raise ValueError("kaggle_submissions.json must contain a list")
    return next((item for item in receipts if item.get("submission_sha256") == digest), None)


def recover_recent(submissions, marker):
    matches = [item for item in submissions or [] if item is not None and marker in (item.description or "")]
    if len(matches) > 1:
        raise RuntimeError("Multiple official submissions match the hash marker; manual reconciliation required")
    return matches[0] if matches else None


def store_receipt(receipt, results=RESULTS):
    path = results / "kaggle_submissions.json"
    receipts = read_json(path, [])
    prior = duplicate_receipt(receipts, receipt["submission_sha256"])
    if prior is not None:
        if prior["submission_ref"] != receipt["submission_ref"]:
            raise RuntimeError("CSV hash already has a different submission reference")
        return prior
    receipts.append(receipt)
    atomic_json(path, receipts)
    if receipt["run"] == "final_full":
        atomic_json(results / "kaggle_submission.json", receipt)
    return receipt


def submit_once(api, details, csv_path, results=RESULTS, timeout=180, poll_seconds=7):
    digest = details["submission_sha256"]
    prior = duplicate_receipt(read_json(results / "kaggle_submissions.json", []), digest)
    if prior is not None:
        return prior
    marker = f"sha256:{digest[:16]}"
    description = f"Saber {details['run']} {marker}"
    intents = results / "submission_intents"
    intents.mkdir(parents=True, exist_ok=True)
    intent_path = intents / f"{digest}.json"
    intent = read_json(intent_path)
    recent = recover_recent(api.competition_submissions(COMPETITION, page_size=100), marker)
    if recent is not None:
        ref = int(recent.ref)
        if intent is not None and intent.get("submission_ref") not in (None, ref):
            raise RuntimeError("Local intent and official recovery references disagree")
        intent = {**(intent or {}), **details, "description": description,
                  "submission_ref": ref, "status": "recovered", "submitted_at": utc_now()}
        atomic_json(intent_path, intent)
    elif intent is not None:
        if not intent.get("submission_ref"):
            raise RuntimeError("Existing upload attempt has unknown outcome; no automatic re-upload allowed")
        ref = int(intent["submission_ref"])
    else:
        limits = api.competition_get_submission_limits(COMPETITION)
        if int(limits.num_allowed_now) < 1:
            raise RuntimeError("Kaggle reports no remaining team submission quota now")
        intent = {**details, "description": description, "status": "upload_attempt_started",
                  "submitted_at": utc_now(), "submission_ref": None,
                  "quota_num_today": int(limits.num_today), "quota_remaining_before": int(limits.num_allowed_now)}
        # Exclusive creation is deliberate: an unknown previous attempt must
        # never be silently overwritten, even by another simultaneous process.
        with intent_path.open("x", encoding="utf-8") as stream:
            json.dump(intent, stream, indent=2)
        try:
            response = api.competition_submit(file_name=str(csv_path), message=description,
                                              competition=COMPETITION, quiet=True)
            ref = int(response.ref)
            if ref <= 0:
                raise RuntimeError("Kaggle did not return a valid submission reference")
            intent.update(submission_ref=ref, status="submitted")
            atomic_json(intent_path, intent)  # Persist before polling or other API work.
        except BaseException as error:
            intent.update(status="unknown", error_type=type(error).__name__)
            atomic_json(intent_path, intent)
            raise
    deadline = time.monotonic() + timeout
    while True:
        try:
            submission = api.get_submission(ref)
        except Exception as error:
            # A newly created reference can briefly return 404. Retry this read,
            # never the upload, while retaining the already-persisted reference.
            code = getattr(getattr(error, "response", None), "status_code", None)
            code = code if code is not None else getattr(error, "status", None)
            if code != 404 or time.monotonic() >= deadline:
                raise
            time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))
            continue
        status = getattr(submission.status, "name", str(submission.status)).upper()
        if status.endswith("ERROR"):
            intent.update(status="scoring_error")
            atomic_json(intent_path, intent)
            raise RuntimeError(f"Kaggle marked submission {ref} as ERROR; no re-upload attempted")
        if status.endswith("COMPLETE") and submission.public_score not in (None, ""):
            score = float(submission.public_score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("Official public score is invalid")
            if getattr(submission, "team_name", "") not in (None, "", "Saber"):
                raise RuntimeError("Official submission team is not Saber")
            submitted_at = getattr(submission, "date", None)
            if isinstance(submitted_at, datetime):
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=timezone.utc)
                submitted_at = submitted_at.isoformat()
            else:
                submitted_at = intent["submitted_at"]
            receipt = {
                "run": details["run"], "public_score": score, "private_score": None,
                "submission_ref": ref, "submission_sha256": digest, "submitted_at": submitted_at,
                "competition": COMPETITION, "team": "Saber", "status": "complete",
                "training_examples": details["training_examples"], "parameter_count": details["parameter_count"],
                "flip_tta": details["flip_tta"], "source_git_commit": details["source_git_commit"],
            }
            if "tta_mode" in details:
                receipt["tta_mode"] = details["tta_mode"]
            receipt = store_receipt(receipt, results)
            intent.update(status="complete", public_score=score)
            atomic_json(intent_path, intent)
            return receipt
        if time.monotonic() >= deadline:
            intent.update(status="scoring_pending")
            atomic_json(intent_path, intent)
            raise TimeoutError(f"Submission {ref} is still scoring; rerun to poll its saved ref without uploading")
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def self_test():
    import tempfile
    import unittest
    from types import SimpleNamespace
    from unittest.mock import Mock

    class Checks(unittest.TestCase):
        def test_csv_contract(self):
            with tempfile.TemporaryDirectory() as directory:
                sample, predicted = Path(directory) / "sample.csv", Path(directory) / "submission.csv"
                sample.write_text("id,label\n" + "".join(f"{i},0\n" for i in range(2000)))
                predicted.write_bytes(sample.read_bytes())
                self.assertEqual(validate_csv(predicted, sample, sha256(sample)), sha256(predicted))
                predicted.write_text(predicted.read_text().replace("0,0\n", "0,10\n", 1))
                with self.assertRaisesRegex(ValueError, "integer 0..9"):
                    validate_csv(predicted, sample, sha256(sample))
                predicted.write_text("id,label\n" + "".join(f"{i},0\n" for i in reversed(range(2000))))
                with self.assertRaisesRegex(ValueError, "ID order"):
                    validate_csv(predicted, sample, sha256(sample))
                with self.assertRaisesRegex(ValueError, "sample submission SHA"):
                    validate_csv(predicted, sample, "0" * 64)

        def test_duplicate_and_ambiguous_recovery(self):
            receipt = {"submission_sha256": "a" * 64}
            self.assertIs(duplicate_receipt([receipt], "a" * 64), receipt)
            self.assertIsNone(duplicate_receipt([receipt], "b" * 64))
            matches = [SimpleNamespace(description="Saber sha256:abc", ref=1)]
            self.assertEqual(recover_recent(matches, "sha256:abc").ref, 1)
            with self.assertRaisesRegex(RuntimeError, "Multiple official"):
                recover_recent(matches * 2, "sha256:abc")

        def test_upload_called_once_and_receipt_prevents_repeat(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                details = {"run": "baseline", "submission_sha256": "a" * 64,
                           "training_examples": 40000, "parameter_count": 691674,
                           "flip_tta": True, "source_git_commit": "fixture"}
                api = Mock()
                api.competition_submissions.return_value = []
                api.competition_get_submission_limits.return_value = SimpleNamespace(num_allowed_now=5, num_today=0)
                api.competition_submit.return_value = SimpleNamespace(ref=123)
                api.get_submission.return_value = SimpleNamespace(status=SimpleNamespace(name="COMPLETE"),
                    public_score="0.95", team_name="Saber", date=None)
                receipt = submit_once(api, details, folder / "fixture.csv", folder)
                self.assertEqual(receipt["public_score"], 0.95)
                self.assertIsNone(receipt["private_score"])
                self.assertEqual(submit_once(api, details, folder / "fixture.csv", folder), receipt)
                api.competition_submit.assert_called_once()

        def test_unknown_upload_outcome_never_retries(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                details = {"run": "baseline", "submission_sha256": "a" * 64}
                atomic_json(folder / "submission_intents" / ("a" * 64 + ".json"), {"status": "unknown"})
                api = Mock()
                api.competition_submissions.return_value = []
                with self.assertRaisesRegex(RuntimeError, "unknown outcome"):
                    submit_once(api, details, folder / "fixture.csv", folder)
                api.competition_submit.assert_not_called()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=ALLOWED_RUNS)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.run is None or args.csv is None:
        parser.error("--run and --csv are required")
    csv_path = args.csv.resolve()
    details = validate_run(args.run, csv_path)
    prior = duplicate_receipt(read_json(RESULTS / "kaggle_submissions.json", []), details["submission_sha256"])
    if not args.execute:
        print(json.dumps({"dry_run": True, "validated": True, **details,
                          "already_recorded": prior is not None}, indent=2))
        return
    if prior is not None:
        print(json.dumps(prior, indent=2))
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    # Local files are checked again immediately before the API mutation path.
    if validate_run(args.run, csv_path) != details:
        raise RuntimeError("Prediction inputs changed during preflight")
    print(json.dumps(submit_once(api, details, csv_path), indent=2))


if __name__ == "__main__":
    main()
