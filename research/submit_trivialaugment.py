"""Submit only the frozen TrivialAugment primary after its typed repeat delivery.

The existing once-only ledger, team quota, duplicate and unknown-result guards
are reused unchanged. This entry owns its schema and final mutation transport;
preparation validates local evidence without creating an intent or contacting
Kaggle. A known reference can be checked without uploading again.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import submit_candidate as shared

SCHEMA = "research_trivialaugment_prediction_delivery_v1"
REPEAT_SCHEMA = "research_trivialaugment_repeat_supervisor_v1"
PRIMARY_RUN = "wrn34_trivialaugment_bf16_s2026"
PRIMARY_CHECKPOINT = "dcfe213533481aa6c90d2fdd9ac853dc291c124098c9fe0ab8866337ab3f0fba"
PRIMARY_FREEZE = "88682c6f719ccc4791adaaa26086bd0fa3e0778de526d463da89da1721099e9a"


def validate_manifest(path, root=ROOT):
    """Use the new typed delivery audit; never dispatch through old allowlists."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if shared.read_json(path).get("schema") != SCHEMA:
        raise ValueError("Use the explicit TrivialAugment prediction delivery schema")
    from research.trivialaugment_delivery import validate_delivery
    result = dict(validate_delivery(path, require_prediction=True))
    expected = {"competition": shared.COMPETITION, "run": PRIMARY_RUN,
        "training_examples": 40000, "seed": 2026, "parameter_count": 4166298,
        "policy": "center80_shift20", "checkpoint_sha256": PRIMARY_CHECKPOINT,
        "frozen_manifest_sha256": PRIMARY_FREEZE, "repeat_receipt_schema": REPEAT_SCHEMA}
    if any(type(result.get(key)) is not type(value) or result[key] != value
           for key, value in expected.items()):
        raise ValueError("Only the frozen primary2026 weights and completed typed repeat permit delivery")
    for key in ("submission_sha256", "checkpoint_sha256", "manifest_sha256",
                "frozen_manifest_sha256", "repeat_receipt_sha256", "repeat_validator_source_sha256"):
        if not isinstance(result.get(key), str) or not shared.SHA.fullmatch(result[key]):
            raise ValueError("Missing actual TrivialAugment delivery fingerprint")
    if result["repeat_validator_source_sha256"] != shared.sha256(root/"research/run_trivialaugment_repeat.py"):
        raise ValueError("Typed repeat validator source changed")
    csv_path = Path(result["csv_path"]).resolve()
    if (shared.validate_csv(csv_path, root/"data/sample_submission.csv", shared.SAMPLE_SHA) != result["submission_sha256"] or
            shared.sha256(path) != result["manifest_sha256"]):
        raise ValueError("Actual official TrivialAugment CSV or manifest changed")
    result.update(csv_path=str(csv_path), delivery_schema=SCHEMA,
        uploader_source_sha256=shared.sha256(Path(__file__)),
        delivery_validator_source_sha256=shared.sha256(root/"research/trivialaugment_delivery.py"))
    return result


def submit_once(manifest_path, api, root=ROOT, validator=None, timeout=180, poll_seconds=7):
    """Keep the original strict durable submission lifecycle unchanged."""
    return shared.submit_once(manifest_path, api, root,
        validator or (lambda path: validate_manifest(path, root)), timeout, poll_seconds)


class KaggleClient(shared.KaggleClient):
    """All calls use this explicit route, including its final typed validator."""
    def call(self, action, payload=None):
        response = subprocess.run([self.python, str(Path(__file__).resolve()), "--_transport", action],
            input=json.dumps(payload or {}), capture_output=True, encoding="utf-8", errors="replace", timeout=180)
        try:
            result = json.loads(response.stdout)
            if not isinstance(result, dict):
                raise ValueError("Invalid transport response")
        except (ValueError, TypeError):
            raise shared.KaggleError() from None
        if response.returncode or result.get("ok") is not True or "result" not in result:
            raise shared.KaggleError(result.get("status_code"))
        return result["result"]


def checked_intent(payload, root=ROOT):
    """Recheck the new schema and immutable staged CSV before authentication."""
    root = Path(root).resolve()
    intent_path = Path(payload["intent_path"]).resolve()
    if shared.sha256(intent_path) != payload["intent_sha256"]:
        raise ValueError("TrivialAugment upload intent changed")
    intent = shared.read_json(intent_path)
    digest = intent.get("submission_sha256")
    if (not isinstance(digest, str) or not shared.SHA.fullmatch(digest) or
            intent_path != root/"results/research/submission_intents"/(digest+".json") or
            intent.get("status") != "upload_attempt_started" or intent.get("submission_ref") is not None or
            intent.get("delivery_schema") != SCHEMA or
            intent.get("uploader_source_sha256") != shared.sha256(Path(__file__))):
        raise ValueError("Only a fresh canonical TrivialAugment upload intent permits mutation")
    validated = intent.get("validated")
    if (not isinstance(validated, dict) or validated.get("delivery_schema") != SCHEMA or
            any(type(intent.get(key)) is not type(value) or intent[key] != value for key, value in validated.items()) or
            intent.get("competition") != shared.COMPETITION or intent.get("team") != "Saber"):
        raise ValueError("TrivialAugment intent differs from its validated artifact chain")
    lock = shared.read_json(root/"results/research/submission_upload.lock")
    if not intent.get("lock_token") or lock.get("token") != intent["lock_token"]:
        raise ValueError("TrivialAugment submission owner lock differs")
    staged = Path(intent["staged_csv"]).resolve()
    if staged != root/"results/research/upload_files"/digest/"submission.csv":
        raise ValueError("Upload must use the canonical immutable staged CSV")
    checked = subprocess.run([intent["validation_python"], str(Path(__file__).resolve()), "--_validate", intent["manifest_path"]],
        capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    if checked.returncode:
        raise ValueError("Final TrivialAugment artifact validation failed")
    if json.loads(checked.stdout) != validated or shared.sha256(staged) != digest:
        raise ValueError("Final TrivialAugment artifact or staged CSV changed")
    shared.submission_window(root)
    return intent


def transport(action, payload, root=ROOT):
    """Official SDK only; actual network calls occur exclusively on invocation."""
    if action not in ("recent", "limits", "get", "submit"):
        raise ValueError("Unknown explicit TrivialAugment transport action")
    if action == "get" and (type(payload.get("ref")) is not int or payload["ref"] <= 0):
        raise ValueError("A positive known reference is required")
    intent = checked_intent(payload, root) if action == "submit" else None
    if importlib.metadata.version("kaggle") != shared.KAGGLE_VERSION:
        raise RuntimeError("Use the verified official Kaggle 2.2.5 environment")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if action == "limits":
        result = api.competition_get_submission_limits(shared.COMPETITION)
        return {"num_today": int(result.num_today), "num_allowed_now": int(result.num_allowed_now)}
    if action == "recent":
        return [{"ref": int(row.ref), "description": row.description, "team_name": getattr(row, "team_name", None)}
                for row in api.competition_submissions(shared.COMPETITION, page_size=100)]
    if action == "get":
        result = api.get_submission(payload["ref"])
        return {"ref": int(getattr(result, "ref", None) or payload["ref"]),
            "description": getattr(result, "description", None), "team_name": getattr(result, "team_name", None),
            "status": getattr(result.status, "name", str(result.status)), "public_score": result.public_score}
    limits = api.competition_get_submission_limits(shared.COMPETITION)
    shared.validate_quota({"num_today": int(limits.num_today), "num_allowed_now": int(limits.num_allowed_now)})
    shared.submission_window(Path(root))
    if shared.sha256(intent["staged_csv"]) != intent["submission_sha256"]:
        raise ValueError("TrivialAugment staged CSV changed after final quota check")
    result = api.competition_submit(file_name=intent["staged_csv"], message=intent["description"],
        competition=shared.COMPETITION, quiet=True)
    return {"ref": int(result.ref)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--check-ref", "--check", dest="check_ref", type=int, metavar="REF")
    parser.add_argument("--kaggle-python", type=Path)
    parser.add_argument("--_transport", choices=("recent", "limits", "get", "submit"), help=argparse.SUPPRESS)
    parser.add_argument("--_validate", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
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
    if args.check_ref is not None:
        result = shared.check_known(args.check_ref, KaggleClient(args.kaggle_python))
    elif args.manifest is None:
        parser.error("--manifest is required for preparation or submission")
    elif args.submit:
        result = submit_once(args.manifest, KaggleClient(args.kaggle_python))
    else:
        result = {"validated": True, "uploaded": False, "upload_intent_created": False,
            **validate_manifest(args.manifest)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
