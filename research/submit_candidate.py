"""Fail-closed research uploads; offline preparation never creates upload intent.

Delivery validation is rerun before mutation, including in the isolated Kaggle
transport through the project Python. Unknown upload outcomes are never retried.
Only known references or an unambiguous competition hash marker may be polled.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.submit_verified import COMPETITION, SAMPLE_SHA, sha256, validate_csv

KAGGLE_VERSION = "2.2.5"
SHA = re.compile(r"[0-9a-f]{64}")


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def durable_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive: target.replace(path)


def submission_window(root):
    policy = read_json(root/"research/campaign.json")
    deadline = datetime.fromisoformat(policy["stops_at"])
    if deadline.tzinfo is None or now() >= deadline:
        raise RuntimeError("Research submission window has ended")
    if policy.get("team") != "Saber" or policy.get("submission_limit_per_team_utc_day") != 5:
        raise ValueError("Expected the verified Saber five-submission policy")
    return policy


def validate_manifest(path, root=ROOT):
    # Lazy import: the isolated Kaggle environment does not need torch.
    schema = read_json(path).get("schema")
    if schema == "research_prediction_delivery_v1":
        from research.delivery import validate_delivery
        validator, examples, source = validate_delivery, 40000, root/"research/delivery.py"
    elif schema == "research_refit_delivery_v1":
        from research.refit_delivery import validate_refit_delivery
        validator, examples, source = validate_refit_delivery, 50000, root/"research/refit_delivery.py"
    else:
        raise ValueError("Unsupported research delivery schema")
    result = dict(validator(Path(path), require_prediction=True))
    if (result.get("competition") != COMPETITION or result.get("training_examples") != examples or
            result.get("seed") != 2026 or result.get("policy") not in ("plain", "flip", "center80_shift20") or
            type(result.get("parameter_count")) is not int or not 0 < result["parameter_count"] <= 5000000 or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,70}", result.get("run", "")) or
            result["run"].startswith("stability")):
        raise ValueError("Delivery is not an audited supported research candidate")
    if examples == 50000 and (result.get("validation_examples") != 0 or result.get("development_accuracy") is not None or
            result.get("independent_development_evaluated") is not False or
            not isinstance(result.get("parent_delivery_sha256"), str) or not SHA.fullmatch(result["parent_delivery_sha256"])):
        raise ValueError("Full refit requires its qualified parent and zero independent development examples")
    for key in ("submission_sha256", "checkpoint_sha256", "manifest_sha256", "frozen_manifest_sha256"):
        if not isinstance(result.get(key), str) or not SHA.fullmatch(result[key]):
            raise ValueError("Missing validated delivery fingerprint")
    csv_path = Path(result["csv_path"]).resolve()
    if (validate_csv(csv_path, root/"data/sample_submission.csv", SAMPLE_SHA) != result["submission_sha256"] or
            sha256(path) != result["manifest_sha256"]):
        raise ValueError("Actual prediction CSV or delivery manifest changed")
    result["csv_path"] = str(csv_path)
    result["delivery_schema"] = schema
    result["uploader_source_sha256"] = sha256(Path(__file__))
    result["delivery_validator_source_sha256"] = sha256(source)
    return result


def ledger_records(root):
    """Original receipts stay frozen; research receipts and intents are separate."""
    legacy = root/"results/kaggle_submissions.json"
    records = read_json(legacy)  # Missing legacy evidence must fail closed.
    if not isinstance(records, list): raise ValueError("Legacy submission ledger must be a list")
    records = list(records)
    single = root/"results/kaggle_submission.json"
    if single.exists(): records.append(read_json(single))
    for directory in (root/"results/submission_intents", root/"results/research/submission_intents",
                      root/"results/research/submission_receipts"):
        records.extend(read_json(path) for path in sorted(directory.glob("*.json")))
    hashes, refs = {}, {}
    for record in records:
        digest, ref = record.get("submission_sha256"), record.get("submission_ref")
        if not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise ValueError("Submission ledger contains a missing/invalid CSV hash")
        if record.get("status") == "complete" and ref is None:
            raise ValueError("A complete submission record requires its official reference")
        if ref is not None:
            if type(ref) is not int or ref <= 0: raise ValueError("Submission reference is invalid")
            if hashes.get(digest, ref) != ref or refs.get(ref, digest) != digest:
                raise RuntimeError("Existing CSV/reference records conflict; reconciliation required")
            hashes[digest], refs[ref] = ref, digest
    return records


def matching_recent(records, digest):
    marker = re.compile("sha256:"+digest+r"(?![0-9a-f])")
    matches = [row for row in records if marker.search(row.get("description") or "")]
    if len(matches) > 1: raise RuntimeError("Multiple official submissions match this CSV marker")
    if matches and (type(matches[0].get("ref")) is not int or matches[0]["ref"] <= 0 or
                    matches[0].get("team_name") not in (None, "", "Saber")):
        raise RuntimeError("Official recovery reference/team is invalid")
    return matches[0] if matches else None


def validate_quota(limits):
    today, remaining = limits.get("num_today"), limits.get("num_allowed_now")
    if (type(today) is not int or type(remaining) is not int or not 0 <= today <= 5 or
            not 0 <= remaining <= 5 or today+remaining > 5):
        raise ValueError("Actual whole-team quota response is inconsistent with the five/day limit")
    if today >= 5 or remaining < 1:
        raise RuntimeError("Kaggle reports no remaining whole-team UTC-day quota")


def poll_known(api, intent, intent_path, root, timeout=180, poll_seconds=7):
    ref, digest = intent["submission_ref"], intent["submission_sha256"]
    deadline = time.monotonic()+timeout
    while True:
        try: response = api.get(ref)
        except Exception as error:
            if getattr(error, "status", None) != 404 or time.monotonic() >= deadline: raise
            time.sleep(min(poll_seconds, max(.01, deadline-time.monotonic())))
            continue
        if (response.get("ref") != ref or response.get("team_name") not in (None, "", "Saber") or
                (response.get("description") and not re.search("sha256:"+digest+r"(?![0-9a-f])", response["description"]))):
            raise RuntimeError("Official score reference/team/hash marker differs from saved intent")
        status = str(response.get("status", "")).upper()
        if status.endswith("ERROR"):
            intent.update(status="scoring_error")
            durable_json(intent_path, intent)
            raise RuntimeError("Official scoring failed; no upload retry is permitted")
        if status.endswith("COMPLETE") and response.get("public_score") not in (None, ""):
            score = float(response["public_score"])
            if not math.isfinite(score) or not 0 <= score <= 1: raise ValueError("Official public score is invalid")
            receipt = {key: intent[key] for key in ("run", "submission_sha256", "checkpoint_sha256", "parameter_count",
                        "policy", "seed", "training_examples", "manifest_sha256", "frozen_manifest_sha256")}
            receipt.update(competition=COMPETITION, team="Saber", status="complete", submission_ref=ref,
                           public_score=score, private_score=None, submitted_at=intent["submitted_at"], checked_at=now().isoformat())
            for key in ("delivery_schema", "validation_examples", "development_accuracy", "independent_development_evaluated", "parent_delivery_sha256"):
                if key in intent: receipt[key] = intent[key]
            receipt_path = root/"results/research/submission_receipts"/f"{ref}.json"
            if receipt_path.exists():
                old = read_json(receipt_path)
                if old.get("submission_sha256") != digest or old.get("submission_ref") != ref:
                    raise RuntimeError("Existing research score receipt conflicts")
                if old.get("public_score") != score:
                    history = root/"results/research/submission_receipt_history"/f"{ref}_{time.time_ns()}.json"
                    durable_json(history, old, exclusive=True)
            durable_json(receipt_path, receipt)
            intent.update(status="complete", public_score=score)
            durable_json(intent_path, intent)
            return receipt
        if time.monotonic() >= deadline:
            intent.update(status="scoring_pending")
            durable_json(intent_path, intent)
            return {"status": "scoring_pending", "submission_ref": ref, "submission_sha256": digest}
        time.sleep(min(poll_seconds, max(.01, deadline-time.monotonic())))


def submit_once(manifest_path, api, root=ROOT, validator=None, timeout=180, poll_seconds=7):
    validator = validator or (lambda path: validate_manifest(path, root))
    details = validator(Path(manifest_path))  # Before quota/auth/API access.
    submission_window(root)
    digest = details["submission_sha256"]
    records = ledger_records(root)
    complete = next((row for row in records if row["submission_sha256"] == digest and row.get("status") == "complete"), None)
    if complete is not None: return complete
    research = root/"results/research"
    lock = research/"submission_upload.lock"
    lock_token = uuid.uuid4().hex
    durable_json(lock, {"token": lock_token, "pid": os.getpid(), "created_at": now().isoformat()}, exclusive=True)
    try:
        records = ledger_records(root)
        own_path = research/"submission_intents"/f"{digest}.json"
        existing = read_json(own_path) if own_path.exists() else None
        recovered = matching_recent(api.recent(), digest)
        known = {row["submission_ref"] for row in records if row["submission_sha256"] == digest and row.get("submission_ref")}
        if recovered is not None: known.add(recovered["ref"])
        if len(known) > 1: raise RuntimeError("Local and official submission references conflict")
        if known:
            ref = known.pop()
            intent = existing or {**details, "submitted_at": now().isoformat(),
                "description": f"Saber research {details['run']} sha256:{digest}"}
            intent.update(submission_ref=ref, status="recovered")
            durable_json(own_path, intent)
            return poll_known(api, intent, own_path, root, timeout, poll_seconds)
        if existing is not None or any(row["submission_sha256"] == digest for row in records):
            raise RuntimeError("Previous upload outcome is unknown; never retry this CSV")
        if any(row.get("submission_ref") is None for row in records):
            raise RuntimeError("An unresolved upload intent requires reconciliation before new uploads")
        limits = api.limits()
        validate_quota(limits)
        if validator(Path(manifest_path)) != details:
            raise RuntimeError("Validated delivery changed immediately before upload")
        submission_window(root)
        staged = research/"upload_files"/digest/"submission.csv"
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists():
            with Path(details["csv_path"]).open("rb") as source, staged.open("xb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        if sha256(staged) != digest: raise ValueError("Staged upload CSV differs from validated predictions")
        intent = {**details, "competition": COMPETITION, "team": "Saber", "submission_ref": None,
                  "status": "upload_attempt_started", "submitted_at": now().isoformat(), "quota_utc_day": now().date().isoformat(),
                  "quota_before": limits, "manifest_path": str(Path(manifest_path).resolve()),
                  "validated": details, "staged_csv": str(staged.resolve()), "validation_python": sys.executable,
                  "lock_token": lock_token, "description": f"Saber research {details['run']} sha256:{digest}"}
        durable_json(own_path, intent, exclusive=True)  # Durable BEFORE mutation; never produced by --prepare.
        try:
            submission_window(root)
            response = api.submit(own_path)
            ref = response.get("ref")
            if type(ref) is not int or ref <= 0: raise RuntimeError("Upload returned no unambiguous official reference")
            if any(row.get("submission_ref") == ref and row["submission_sha256"] != digest for row in records):
                raise RuntimeError("Upload response conflicts with an existing reference")
            intent.update(submission_ref=ref, status="submitted")
            durable_json(own_path, intent)  # Persist reference before any polling.
        except BaseException as error:
            intent.update(status="unknown", error_type=type(error).__name__)
            durable_json(own_path, intent)
            raise
        return poll_known(api, intent, own_path, root, timeout, poll_seconds)
    finally:
        if lock.exists() and read_json(lock).get("token") == lock_token: lock.unlink()


def check_known(ref, api, root=ROOT, timeout=180, poll_seconds=7):
    if type(ref) is not int or ref <= 0: raise ValueError("A positive existing reference is required")
    ledger_records(root)  # Reject conflicting evidence before any API call.
    matches = [(path, read_json(path)) for path in (root/"results/research/submission_intents").glob("*.json")]
    matches = [(path, intent) for path, intent in matches if intent.get("submission_ref") == ref]
    if len(matches) != 1: raise ValueError("Check requires exactly one locally known research reference")
    path, intent = matches[0]
    return poll_known(api, intent, path, root, timeout, poll_seconds)


class KaggleError(RuntimeError):
    def __init__(self, status=None):
        super().__init__("Official Kaggle operation failed; no upload retry was attempted")
        self.status = status


class KaggleClient:
    def __init__(self, python=None):
        candidate = Path(os.environ.get("APPDATA", ""))/"uv/tools/kaggle/Scripts/python.exe"
        self.python = str(python or (candidate if candidate.is_file() else sys.executable))

    def call(self, action, payload=None):
        response = subprocess.run([self.python, str(Path(__file__).resolve()), "--_transport", action],
            input=json.dumps(payload or {}), capture_output=True, encoding="utf-8", errors="replace", timeout=180)
        try: result = json.loads(response.stdout)
        except (ValueError, TypeError): raise KaggleError() from None
        if response.returncode or result.get("ok") is not True: raise KaggleError(result.get("status_code"))
        return result["result"]

    def recent(self): return self.call("recent")
    def limits(self): return self.call("limits")
    def get(self, ref): return self.call("get", {"ref": ref})
    def submit(self, intent_path): return self.call("submit", {"intent_path": str(intent_path), "intent_sha256": sha256(intent_path)})


def transport(action, payload):
    """Official Kaggle-only interpreter; no torch import or credential output."""
    if action == "submit":
        intent_path = Path(payload["intent_path"])
        if sha256(intent_path) != payload["intent_sha256"]: raise ValueError("Upload intent changed")
        intent = read_json(intent_path)
        if intent.get("status") != "upload_attempt_started" or intent.get("submission_ref") is not None:
            raise ValueError("Only a fresh durable upload intent permits mutation")
        lock = read_json(ROOT/"results/research/submission_upload.lock")
        if lock.get("token") != intent.get("lock_token"): raise ValueError("Submission owner lock differs")
        # Rerun the actual artifact validator NOW using the torch-enabled project
        # interpreter, rather than trusting an intent/manifest JSON assertion.
        checked = subprocess.run([intent["validation_python"], str(Path(__file__).resolve()), "--_validate", intent["manifest_path"]],
                                  capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        if checked.returncode: raise ValueError("Final artifact validation failed")
        if json.loads(checked.stdout) != intent["validated"] or sha256(intent["staged_csv"]) != intent["submission_sha256"]:
            raise ValueError("Final artifact/CSV source changed")
        submission_window(ROOT)
    if importlib.metadata.version("kaggle") != KAGGLE_VERSION:
        raise RuntimeError("Use the verified official Kaggle 2.2.5 environment")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if action == "limits":
        result = api.competition_get_submission_limits(COMPETITION)
        return {"num_today": int(result.num_today), "num_allowed_now": int(result.num_allowed_now)}
    if action == "recent":
        return [{"ref": int(row.ref), "description": row.description, "team_name": getattr(row, "team_name", None)}
                for row in api.competition_submissions(COMPETITION, page_size=100)]
    if action == "submit":
        validate_quota({"num_today": int((limits := api.competition_get_submission_limits(COMPETITION)).num_today),
                        "num_allowed_now": int(limits.num_allowed_now)})
        submission_window(ROOT)
        result = api.competition_submit(file_name=intent["staged_csv"], message=intent["description"], competition=COMPETITION, quiet=True)
        return {"ref": int(result.ref)}
    if action == "get":
        result = api.get_submission(int(payload["ref"]))
        return {"ref": int(getattr(result, "ref", None) or payload["ref"]),
                "description": getattr(result, "description", None), "team_name": getattr(result, "team_name", None),
                "status": getattr(result.status, "name", str(result.status)), "public_score": result.public_score}
    raise ValueError("Unknown official transport action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--check", type=int, metavar="REF")
    parser.add_argument("--kaggle-python", type=Path)
    parser.add_argument("--_transport", choices=("recent", "limits", "submit", "get"), help=argparse.SUPPRESS)
    parser.add_argument("--_validate", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._transport:
        try:
            payload = json.load(sys.stdin)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = transport(args._transport, payload)
            print(json.dumps({"ok": True, "result": result}))
        except BaseException as error:
            status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status", None)
            print(json.dumps({"ok": False, "error_type": type(error).__name__, "status_code": status}))
            raise SystemExit(1)
        return
    if args._validate:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = validate_manifest(args._validate)
        print(json.dumps(result))
        return
    if args.check is not None:
        result = check_known(args.check, KaggleClient(args.kaggle_python))
    else:
        if args.manifest is None: parser.error("--manifest is required for validation or submission")
        if args.submit: result = submit_once(args.manifest, KaggleClient(args.kaggle_python))
        else: result = {"validated": True, "uploaded": False, "upload_intent_created": False, **validate_manifest(args.manifest)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
