"""Explicit Nov 4 SAM full-refit uploads through the existing once-only ledger.

The original uploader and its supported schemas remain frozen. Only the final
mutation transport differs: it reruns this new typed validator in the project
interpreter before one official upload. Unknown outcomes are never retried.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import submit_candidate as shared

SCHEMA = "research_sam_full_refit_delivery_nov04_v1"


def validate_manifest(path, root=ROOT):
    """Audit only the new 50k/160 recipe and its actual official prediction CSV."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if shared.read_json(path).get("schema") != SCHEMA:
        raise ValueError("Use the explicit Nov 4 SAM full-refit delivery schema")
    from research.sam_full_refit_nov04_delivery import validate_refit_delivery
    result = dict(validate_refit_delivery(path, root=root))
    expected = {"competition": shared.COMPETITION, "training_examples": 50000,
        "validation_examples": 0, "development_accuracy": None,
        "independent_development_evaluated": False, "seed": 2026,
        "parameter_count": 4298970, "policy": "center80_shift20",
        "epochs_completed": 160, "checkpoint_epoch": 160, "weight_variant": "model",
        "training_precision": "bf16"}
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in expected.items()):
        raise ValueError("SAM refit requires fixed epoch-160 online model, 50k/zero-dev and BF16 training")
    if not re.fullmatch(r"sam_full_refit_nov04_[A-Za-z0-9][A-Za-z0-9_]{0,48}", result.get("run", "")):
        raise ValueError("Use a distinct exploratory SAM refit run")
    for key in ("submission_sha256", "checkpoint_sha256", "manifest_sha256",
                "frozen_manifest_sha256", "parent_delivery_sha256"):
        if not isinstance(result.get(key), str) or not shared.SHA.fullmatch(result[key]):
            raise ValueError("Missing actual SAM delivery fingerprint")
    csv_path = Path(result["csv_path"]).resolve()
    if (shared.validate_csv(csv_path, root/"data/sample_submission.csv", shared.SAMPLE_SHA) != result["submission_sha256"] or
            shared.sha256(path) != result["manifest_sha256"]):
        raise ValueError("Actual official SAM CSV or manifest changed")
    result.update(csv_path=str(csv_path), delivery_schema=SCHEMA,
        uploader_source_sha256=shared.sha256(Path(__file__)),
        delivery_validator_source_sha256=shared.sha256(root/"research/sam_full_refit_nov04_delivery.py"))
    # The shared ledger owns status, timestamps, and submission reference.
    # Pass only validated artifact fields that can remain equal in the intent.
    fingerprints = ("submission_sha256", "checkpoint_sha256", "manifest_sha256",
                    "frozen_manifest_sha256", "parent_delivery_sha256")
    fields = (*expected, *fingerprints, "run", "csv_path", "delivery_schema",
              "uploader_source_sha256", "delivery_validator_source_sha256")
    return {key: result[key] for key in fields}


def submit_once(manifest_path, api, root=ROOT, validator=None, timeout=180, poll_seconds=7):
    """Reuse shared locking, full-team quota, intents, duplicate and ref guards."""
    return shared.submit_once(manifest_path, api, root,
        validator or (lambda path: validate_manifest(path, root)), timeout, poll_seconds)


class KaggleClient(shared.KaggleClient):
    """Keep existing read-only calls; route this schema's mutation explicitly."""
    def submit(self, intent_path):
        response = subprocess.run([self.python, str(Path(__file__).resolve()), "--_transport"],
            input=json.dumps({"intent_path": str(intent_path), "intent_sha256": shared.sha256(intent_path)}),
            capture_output=True, encoding="utf-8", errors="replace", timeout=180)
        try:
            result = json.loads(response.stdout)
        except (ValueError, TypeError):
            raise shared.KaggleError() from None
        if response.returncode or result.get("ok") is not True:
            raise shared.KaggleError(result.get("status_code"))
        return result["result"]


def transport_submit(payload, root=ROOT):
    """Small official-only adapter; no Torch import or credential output here."""
    root = Path(root).resolve()
    intent_path = Path(payload["intent_path"]).resolve()
    if shared.sha256(intent_path) != payload["intent_sha256"]:
        raise ValueError("SAM upload intent changed")
    intent = shared.read_json(intent_path)
    digest = intent.get("submission_sha256")
    if (not isinstance(digest, str) or not shared.SHA.fullmatch(digest) or
            intent_path != root/"results/research/submission_intents"/(digest+".json") or
            intent.get("status") != "upload_attempt_started" or intent.get("submission_ref") is not None or
            intent.get("delivery_schema") != SCHEMA or
            intent.get("uploader_source_sha256") != shared.sha256(Path(__file__))):
        raise ValueError("Only a fresh canonical SAM upload intent permits mutation")
    validated = intent.get("validated")
    if (not isinstance(validated, dict) or validated.get("delivery_schema") != SCHEMA or
            any(type(intent.get(key)) is not type(value) or intent[key] != value for key, value in validated.items()) or
            intent.get("competition") != shared.COMPETITION or intent.get("team") != "Saber"):
        raise ValueError("SAM intent metadata differs from its validated artifact chain")
    lock = shared.read_json(root/"results/research/submission_upload.lock")
    if not intent.get("lock_token") or lock.get("token") != intent["lock_token"]:
        raise ValueError("SAM submission owner lock differs")
    staged = Path(intent["staged_csv"]).resolve()
    if staged != root/"results/research/upload_files"/digest/"submission.csv":
        raise ValueError("SAM upload must use the canonical immutable staged CSV")
    checked = subprocess.run([intent["validation_python"], str(Path(__file__).resolve()), "--_validate", intent["manifest_path"]],
        capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    if checked.returncode:
        raise ValueError("Final SAM artifact validation failed")
    if json.loads(checked.stdout) != intent["validated"] or shared.sha256(staged) != digest:
        raise ValueError("Final SAM artifact or staged CSV changed")
    shared.submission_window(root)
    if importlib.metadata.version("kaggle") != shared.KAGGLE_VERSION:
        raise RuntimeError("Use the verified official Kaggle 2.2.5 environment")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    limits = api.competition_get_submission_limits(shared.COMPETITION)
    shared.validate_quota({"num_today": int(limits.num_today), "num_allowed_now": int(limits.num_allowed_now)})
    shared.submission_window(root)
    if shared.sha256(staged) != digest:
        raise ValueError("SAM staged CSV changed after final quota check")
    response = api.competition_submit(file_name=str(staged), message=intent["description"],
        competition=shared.COMPETITION, quiet=True)
    return {"ref": int(response.ref)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--check", type=int, metavar="REF")
    parser.add_argument("--kaggle-python", type=Path)
    parser.add_argument("--_transport", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_validate", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._transport:
        if argv != ["--_transport"]:
            parser.error("Exact Nov 4 SAM transport arguments required")
        try:
            payload = json.load(sys.stdin)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = transport_submit(payload)
            print(json.dumps({"ok": True, "result": result}))
        except BaseException as error:
            status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status", None)
            print(json.dumps({"ok": False, "error_type": type(error).__name__, "status_code": status}))
            raise SystemExit(1)
        return
    if args._validate:
        if len(argv) != 2 or argv[0] != "--_validate":
            parser.error("Exact Nov 4 SAM validation arguments required")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = validate_manifest(args._validate)
        print(json.dumps(result))
        return
    if args.check is not None:
        intents = (ROOT / "results/research/submission_intents").glob("*.json")
        own_matches = [record for path in intents if (record := shared.read_json(path)).get("submission_ref") == args.check
                       and record.get("delivery_schema") == SCHEMA]
        if len(own_matches) != 1:
            raise ValueError("Check requires a locally known Nov 4 SAM reference")
        result = shared.check_known(args.check, KaggleClient(args.kaggle_python))
    else:
        if args.manifest is None:
            parser.error("--manifest is required for validation or submission")
        if args.submit:
            result = submit_once(args.manifest, KaggleClient(args.kaggle_python))
        else:
            result = {"validated": True, "uploaded": False, "upload_intent_created": False, **validate_manifest(args.manifest)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
