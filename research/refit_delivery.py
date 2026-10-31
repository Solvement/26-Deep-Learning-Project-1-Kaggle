"""Separate delivery for a scratch full-50k refit of a qualified frozen WRN.

The parent is the audited 40k model, paired benchmark and independent seed
repeat. The refit has ZERO held-out development examples and no new accuracy
claim. Final epoch, weight variant, policy and backend are fixed by its parent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from runtime import seed_all
import torch
from data import COMPETITION, CSV_SHA256, load_kaggle_test
from model import build_model, count_parameters
from train import code_provenance
from research import benchmark as suite
from research import delivery as primary

SCHEMA = "research_refit_delivery_v1"


def validate_refit_run(plan_path):
    """Deep-audit the completed run and revalidate its original 40k parent NOW."""
    from research.run_refit import derive_refit_config, validate_refit
    plan_path = Path(plan_path)
    info = dict(validate_refit(plan_path))
    parent_path = Path(info["parent_manifest_path"])
    parent = primary.validate_delivery(parent_path, require_prediction=True)
    parent_document = primary.read(parent_path)
    frozen_path = Path(parent_document["frozen_manifest_path"])
    primary.require(suite.sha256(parent_path) == info["parent_manifest_sha256"] == parent["manifest_sha256"] and
                    suite.sha256(frozen_path) == parent["frozen_manifest_sha256"], "Qualified parent delivery/freeze changed")
    frozen = primary.read(frozen_path)
    config = derive_refit_config(frozen)
    primary.require(info.get("training_examples") == 50000 and info.get("validation_examples") == 0 and
                    parent.get("training_examples") == 40000 and info.get("config") == config and
                    config.get("train_all") is True and type(config.get("epochs")) is int and config["epochs"] > 0 and config.get("seed") == 2026 and
                    config.get("final_variant") == frozen["weight_variant"] and
                    info.get("policy") == parent["policy"] == frozen["policy"] and
                    info.get("parameter_count") == parent["parameter_count"] == frozen["parameter_count"] and
                    info.get("data_signature") == frozen["data_signature"] and
                    info.get("source_sha256") == code_provenance()["source_sha256"] and
                    info.get("plan_sha256") == suite.sha256(plan_path),
                    "Refit counts/config/final variant/policy/source differ from its fixed parent")
    # Initial refit path supports the original frozen-core WRN only.
    checkpoint = torch.load(frozen["checkpoint_path"], map_location="cpu", weights_only=False)
    primary.require(not checkpoint.get("provenance", {}).get("research_training"),
                    "SAM/CutMix/ResizeMix and other shadow-trainer refits are unsupported")
    checkpoint_path = Path(info["checkpoint_path"])
    paths = [plan_path, parent_path, frozen_path, checkpoint_path,
             *(checkpoint_path.parent/name for name in ("last.pt", "input_config.json", "config.json", "metrics.json",
                "history.json", "provenance.json", "environment.json", "supervisor.json")),
             ROOT/"research/run_refit.py", Path(__file__), *(ROOT/name for name in info["source_sha256"])]
    return info, parent, frozen, paths


def validate_test_manifest(data_dir, path):
    manifest = primary.read(path)
    primary.require(manifest.get("competition") == COMPETITION and manifest.get("count") == 2000 and
        manifest.get("labels_loaded") is False and manifest.get("test_csv_sha256") == CSV_SHA256["test.csv"] and
        manifest.get("sample_submission_sha256") == CSV_SHA256["sample_submission.csv"] and
        manifest.get("source_provenance", {}).get("archive_sha256") == suite.sha256(data_dir/"official_test.npz") and
        manifest.get("source_provenance", {}).get("source_manifest_sha256") == suite.sha256(data_dir/"official_test_source_manifest.json"),
        "Official unlabeled test manifest differs")


def validate_refit_delivery(manifest_path, require_prediction=True):
    """Return an uploader summary only after actual parent/refit/CSV audits."""
    manifest_path = Path(manifest_path)
    document = primary.read(manifest_path)
    primary.require(document.get("schema") == SCHEMA and document.get("status") == "complete",
                    "A complete separate refit prediction delivery is required")
    plan_path = Path(document["refit_plan_path"])
    info, parent, frozen, paths = validate_refit_run(plan_path)
    expected = {"run": info["run"], "competition": COMPETITION, "policy": parent["policy"], "seed": 2026,
        "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
        "independent_development_evaluated": False, "checkpoint_sha256": info["checkpoint_sha256"],
        "parameter_count": info["parameter_count"], "refit_plan_sha256": info["plan_sha256"],
        "config": info["config"], "model_config": frozen["model_config"], "data_signature": info["data_signature"],
        "training_source_sha256": info["source_sha256"], "weight_variant": info["config"]["final_variant"],
        "epochs_completed": info["config"]["epochs"], "checkpoint_epoch": info["config"]["epochs"], "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "parent_delivery_path": str(Path(info["parent_manifest_path"]).resolve()),
        "parent_delivery_sha256": parent["manifest_sha256"], "parent_frozen_manifest_sha256": parent["frozen_manifest_sha256"],
        "evaluation_context": frozen["evaluation_context"], "generator_source_sha256": suite.sha256(Path(__file__)),
        "confirmation_evaluated": False, "hidden_test_labels_loaded": False, "external_training_data_loaded": False}
    primary.require(all(document.get(key) == value for key, value in expected.items()),
                    "Refit delivery metadata/source or no-development claim differs")
    primary.verify_snapshots(document.get("refit_evidence_sha256"))
    primary.require(document["refit_evidence_sha256"] == primary.snapshots(paths), "Refit evidence fingerprint coverage differs")
    data_dir = Path(frozen["data_dir"])
    test_manifest_path = Path(document["test_manifest_path"])
    test_paths = [*primary.validate_test_files(data_dir), test_manifest_path]
    primary.verify_snapshots(document.get("test_files_sha256"))
    primary.require(document["test_files_sha256"] == primary.snapshots(test_paths), "Refit test fingerprint coverage differs")
    validate_test_manifest(data_dir, test_manifest_path)
    digest = primary.validate_csv(Path(document["csv_path"]), data_dir/"sample_submission.csv")
    primary.require(digest == document.get("submission_sha256") and document.get("rows") == 2000 and
        document.get("columns") == ["id", "label"] and document.get("ids_match_official_order") is True,
        "Refit prediction rows/IDs/labels/checksum differ")
    return {"run": info["run"], "competition": COMPETITION, "csv_path": str(Path(document["csv_path"]).resolve()),
        "submission_sha256": digest, "checkpoint_sha256": info["checkpoint_sha256"], "parameter_count": info["parameter_count"],
        "policy": parent["policy"], "seed": 2026, "training_examples": 50000, "validation_examples": 0,
        "development_accuracy": None, "independent_development_evaluated": False,
        "manifest_sha256": suite.sha256(manifest_path), "frozen_manifest_sha256": info["plan_sha256"],
        "parent_delivery_sha256": parent["manifest_sha256"]}


def predict(plan_path, output_dir, device="cuda"):
    """Fixed final inference only; invoke through a budgeted supervisor."""
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    info, parent, frozen, paths = validate_refit_run(plan_path)
    hashes = primary.snapshots(paths)
    csv_path, manifest_path = output_dir/"submission.csv", output_dir/"delivery.json"
    primary.require(not csv_path.exists() and not manifest_path.exists(), "Refit prediction evidence is immutable")
    device = torch.device(device)
    primary.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    context = {"device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)),
               "torch_version": torch.__version__, "training_core_sha256": code_provenance()["source_sha256"]}
    primary.require(context == frozen["evaluation_context"], "Refit prediction must use the parent's fixed backend/precision")
    data_dir = Path(frozen["data_dir"])
    test_paths = primary.validate_test_files(data_dir)
    seed_all(2026)
    checkpoint = torch.load(info["checkpoint_path"], map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    primary.require(count_parameters(model) == info["parameter_count"], "Refit architecture parameter count differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    ids, images = load_kaggle_test(data_dir)
    test_manifest_path = data_dir/"kaggle_test_manifest.json"
    validate_test_manifest(data_dir, test_manifest_path)
    primary.require(len(ids) == 2000 and len(set(ids)) == 2000, "Official refit test IDs are invalid")
    test_hashes = primary.snapshots([*test_paths, test_manifest_path])
    predictions = []
    batch_size = primary.read(frozen["benchmark_path"])["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.as_tensor(images[start:start+batch_size], device=device)
            with torch.autocast(device.type, dtype=suite.benchmark_dtype(device), enabled=device.type == "cuda"):
                logits = suite.infer(model, batch, frozen["policy"])
            primary.require(torch.isfinite(logits).all().item(), "Nonfinite refit prediction logits")
            predictions.extend(logits.argmax(1).cpu().tolist())
    primary.verify_snapshots(hashes)
    primary.verify_snapshots(test_hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, predictions))
    digest = primary.validate_csv(csv_path, data_dir/"sample_submission.csv")
    document = {"schema": SCHEMA, "status": "complete", "created_at": primary.utc_now().isoformat(),
        "run": info["run"], "competition": COMPETITION, "refit_plan_path": str(plan_path.resolve()),
        "refit_plan_sha256": info["plan_sha256"], "parent_delivery_sha256": parent["manifest_sha256"],
        "parent_delivery_path": str(Path(info["parent_manifest_path"]).resolve()),
        "config": info["config"], "model_config": frozen["model_config"], "data_signature": info["data_signature"],
        "training_source_sha256": info["source_sha256"], "weight_variant": info["config"]["final_variant"],
        "epochs_completed": info["config"]["epochs"], "checkpoint_epoch": info["config"]["epochs"], "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "parent_frozen_manifest_sha256": parent["frozen_manifest_sha256"], "policy": parent["policy"], "seed": 2026,
        "checkpoint_sha256": info["checkpoint_sha256"], "parameter_count": info["parameter_count"],
        "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
        "independent_development_evaluated": False, "refit_evidence_sha256": hashes,
        "evaluation_context": context, "generator_source_sha256": suite.sha256(Path(__file__)),
        "csv_path": str(csv_path.resolve()), "submission_sha256": digest, "rows": 2000, "columns": ["id", "label"],
        "ids_match_official_order": True, "test_manifest_path": str(test_manifest_path), "test_files_sha256": test_hashes,
        "confirmation_evaluated": False, "hidden_test_labels_loaded": False, "external_training_data_loaded": False,
        "interpretation": "Full-50k scratch refit of a qualified 40k parent; no independent development evaluation on the refit."}
    primary.exclusive_json(manifest_path, document)
    return validate_refit_delivery(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--predict", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.predict:
        if not args.plan or not args.output_dir: parser.error("--predict requires --plan and --output-dir")
        result = predict(args.plan, args.output_dir, args.device)
    else:
        if not args.manifest: parser.error("--validate requires --manifest")
        result = validate_refit_delivery(args.manifest)
    print(json.dumps(result))


if __name__ == "__main__": main()
