"""Typed final-epoch delivery for one scratch WRN34-3/300 full-50k refit.

The qualified parent retains its independent 40k development evidence. The
new final300 model has zero held-out examples and makes no new development
accuracy claim. Prediction runs only inside the separate budgeted OS guard.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import submit_candidate as shared

SCHEMA = "research_wrn300_full_refit_delivery_v1"
PARENT_DELIVERY_SHA = "f2fc4b3f1577b11b7488f3fca713ac3086be572f7042681e83f7bac7c1625fab"
PARENT_FREEZE_SHA = "b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3"
PARENT_CHECKPOINT_SHA = "53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
EXCLUDED_CHECKPOINTS = {PARENT_CHECKPOINT_SHA,
    "b54934328880ff6930028cb274a6df5d00112a2da4a5129aa01a86a80d7a537a",
    "258c4f4714f1382ace8864141a817a14e629d15b01628ef0c94497247cacd049"}
ARTIFACT_NAMES = ("best.pt", "last.pt", "input_config.json", "config.json", "metrics.json",
                  "history.json", "provenance.json", "environment.json", "supervisor.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return shared.read_json(path)


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "WRN300 evidence timestamps require a timezone")
    return result


def snapshots(paths):
    return {str(path): shared.sha256(path) for path in sorted({Path(p).resolve() for p in paths})}


def same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(same_json(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_refit_run(plan_path):
    """Reaudit the new training proof and the unchanged qualified 40k parent."""
    from research.run_wrn300_refit import validate_refit
    plan_path = Path(plan_path).resolve()
    info = dict(validate_refit(plan_path))
    parent_path = Path(info["parent_delivery_path"]).resolve()
    parent = shared.validate_manifest(parent_path)
    parent_document = read(parent_path)
    frozen_path = Path(parent_document["frozen_manifest_path"]).resolve()
    frozen = read(frozen_path)
    require(shared.sha256(parent_path) == info["parent_delivery_sha256"] == parent["manifest_sha256"] == PARENT_DELIVERY_SHA and
        shared.sha256(frozen_path) == parent["frozen_manifest_sha256"] == PARENT_FREEZE_SHA and
        parent["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA and
        parent.get("repeat_receipt_schema") == "research_shadow_cloud_repeat_attestation_v1",
        "WRN300 refit requires its exact frozen primary, pairing and audited shadow repeat")
    expected_config = {**frozen["config"], "train_all": True, "final_variant": "model"}
    expected = {"training_examples": 50000, "validation_examples": 0,
        "epochs_completed": 300, "training_precision": "fp16", "weight_variant": "model",
        "initial_grad_scaler_scale": 65536.0, "attempted_updates": 117300, "source_examples_seen": 15000000,
        "parameter_count": 4166298, "policy": "center80_shift20", "config": expected_config,
        "data_signature": frozen["data_signature"], "evaluation_context": frozen["evaluation_context"]}
    require(all(same_json(info.get(k), v) for k, v in expected.items()),
        "WRN300 refit counts, FP16/final300/model or frozen recipe differ")
    require(parent.get("training_examples") == 40000 and parent.get("seed") == 2026 and
        parent.get("policy") == frozen.get("policy") == "center80_shift20" and
        frozen.get("weight_variant") == "model" and frozen["config"].get("epochs") == 300 and
        frozen.get("model_config") == {"depth": 34, "widen_factor": 3, "dropout": 0.0, "num_classes": 10},
        "WRN300 parent architecture or selection differs")
    checkpoint = Path(info["checkpoint_path"]).resolve()
    require(checkpoint == plan_path.parent/"best.pt" and shared.sha256(checkpoint) == info["checkpoint_sha256"] and
        info["checkpoint_sha256"] not in EXCLUDED_CHECKPOINTS and
        info["plan_sha256"] == shared.sha256(plan_path) and
        not any(part.startswith("stability") for part in checkpoint.parts),
        "Only the actual new full50k final checkpoint may enter hidden prediction")
    require(bool(info.get("source_sha256")) and all(shared.sha256(ROOT/name) == digest for name, digest in info["source_sha256"].items()),
        "WRN300 refit training source changed")
    require(isinstance(info.get("source_git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", info["source_git_commit"]),
        "WRN300 refit requires its recorded committed source revision")
    paths = [Path(p).resolve() for p in info["evidence_paths"]]
    required = {plan_path, parent_path, frozen_path, *(plan_path.parent/name for name in ARTIFACT_NAMES)}
    require(required.issubset(set(paths)) and len(paths) == len(set(paths)) and all(path.is_file() for path in paths),
        "WRN300 complete training/parent/guard evidence coverage is missing")
    paths += [Path(__file__).resolve(), (ROOT/"research/submit_wrn300_refit.py").resolve()]
    return info, parent, frozen, sorted(set(paths))


def test_inputs(data_dir):
    """Use the unchanged audited official NPZ export, with no hidden labels."""
    from research.delivery_v2 import validate_test_files
    return list(map(Path, validate_test_files(Path(data_dir))))


def validate_test_manifest(data_dir, path):
    from data import CSV_SHA256
    data_dir, path = Path(data_dir), Path(path).resolve()
    require(path == (data_dir/"kaggle_test_manifest.json").resolve(), "Use the canonical official test manifest")
    document = read(path)
    require(document.get("competition") == shared.COMPETITION and document.get("count") == 2000 and
        document.get("shape") == [2000, 32, 32, 3] and document.get("dtype") == "uint8" and
        document.get("layout") == "NHWC" and document.get("order") == "sample_submission.csv ID order" and
        document.get("labels_loaded") is False and document.get("test_csv_sha256") == CSV_SHA256["test.csv"] and
        document.get("sample_submission_sha256") == shared.SAMPLE_SHA and
        document.get("source_provenance", {}).get("archive_sha256") == shared.sha256(data_dir/"official_test.npz") and
        document.get("source_provenance", {}).get("source_manifest_sha256") == shared.sha256(data_dir/"official_test_source_manifest.json"),
        "Official WRN300 unlabeled test manifest differs")


def metadata(plan_path, info, parent, frozen):
    return {"run": info["run"], "competition": shared.COMPETITION,
        "refit_plan_path": str(Path(plan_path).resolve()), "refit_plan_sha256": info["plan_sha256"],
        "parent_delivery_path": str(Path(info["parent_delivery_path"]).resolve()),
        "parent_delivery_sha256": parent["manifest_sha256"], "parent_frozen_manifest_sha256": parent["frozen_manifest_sha256"],
        "checkpoint_path": str(Path(info["checkpoint_path"]).resolve()), "checkpoint_sha256": info["checkpoint_sha256"],
        "parameter_count": 4166298, "config": info["config"], "model_config": frozen["model_config"],
        "data_signature": info["data_signature"], "training_source_sha256": info["source_sha256"],
        "source_git_commit": info["source_git_commit"],
        "training_precision": "fp16", "initial_grad_scaler_scale": 65536.0,
        "weight_variant": "model", "checkpoint_epoch": 300, "epochs_completed": 300,
        "attempted_updates": 117300, "source_examples_seen": 15000000,
        "policy": "center80_shift20", "seed": 2026, "initialization": "scratch_seeded", "parent_weights_loaded": False,
        "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
        "independent_development_evaluated": False, "bn_recalibrated": False,
        "evaluation_context": frozen["evaluation_context"], "generator_source_sha256": shared.sha256(Path(__file__)),
        "confirmation_evaluated": False, "hidden_test_labels_loaded": False, "external_training_data_loaded": False}


def validate_prediction_receipt(manifest_path, document, info, evidence, official_test_paths):
    """Bind final inference to its completed real OS boundary and resource job."""
    from research.run_wrn300_refit import SUPERVISOR_SCHEMA
    manifest_path = Path(manifest_path).resolve()
    folder = manifest_path.parent
    receipt = read(folder/"supervisor.json")
    inputs_path = folder/"refit_inputs.json"
    inputs = read(inputs_path)
    sources = info["source_sha256"]
    require(receipt.get("schema") == SUPERVISOR_SCHEMA and receipt.get("status") == "complete" and
        receipt.get("backend") == "local" and receipt.get("mode") == "predict" and receipt.get("run") == folder.name and
        receipt.get("trainer") == "wrn300_full_refit" and receipt.get("precision") == "parent_frozen_inference" and
        receipt.get("plan_sha256") == info["plan_sha256"] and receipt.get("submitted") is False and
        receipt.get("confirmation_evaluated") is False and receipt.get("process_tree_cleanup_verified") is True and
        receipt.get("artifact") == manifest_path.name and receipt.get("artifact_sha256") == shared.sha256(manifest_path) and
        receipt.get("input_manifest_sha256") == document.get("prediction_input_manifest_sha256") == shared.sha256(inputs_path) and
        receipt.get("source_sha256") == inputs.get("source_sha256") == sources and
        inputs.get("mode") == "predict" and inputs.get("plan_sha256") == info["plan_sha256"] and
        Path(inputs["plan_path"]).resolve() == Path(document["refit_plan_path"]).resolve(),
        "WRN300 prediction requires a completed source-bound supervised receipt")
    binding = inputs.get("input_sha256")
    required_binding = snapshots([*evidence, *official_test_paths])
    require(isinstance(binding, dict) and binding == receipt.get("input_sha256") and
        all(binding.get(path) == digest for path, digest in required_binding.items()) and
        all(shared.sha256(path) == digest for path, digest in binding.items()),
        "WRN300 supervised prediction input coverage changed")
    began, ended = stamp(receipt["started_at"]), stamp(receipt["finished_at"])
    created = stamp(document["created_at"])
    train_receipt = read(Path(info["checkpoint_path"]).parent/"supervisor.json")
    hard_stop = stamp(read(ROOT/"research/campaign.json")["stops_at"])
    require(stamp(train_receipt["finished_at"]) <= began <= created <= ended <= hard_stop and
        type(receipt.get("elapsed_seconds")) in (int, float) and math.isfinite(receipt["elapsed_seconds"]) and
        type(receipt.get("reserved_seconds")) in (int, float) and math.isfinite(receipt["reserved_seconds"]) and
        0 < receipt["elapsed_seconds"] <= receipt["reserved_seconds"] <= 300 and
        abs((ended-began).total_seconds()-receipt["elapsed_seconds"]) <= 5 and
        type(receipt.get("deadline_unix")) in (int, float) and ended.timestamp() <= receipt["deadline_unix"]+1,
        "WRN300 prediction timing, hard stop or job budget differs")
    proofs = receipt.get("process_tree_guard_results")
    require(isinstance(proofs, list) and len(proofs) == 1, "Exactly one completed prediction OS guard is required")
    proof = proofs[0]
    guard = Path(proof["guard_directory"]).resolve()
    require(guard.parent == folder and re.fullmatch(r"process_tree_[0-9a-f]{32}", guard.name), "Prediction OS guard directory differs")
    spec, ready, completion, host = [read(guard/name) for name in ("spec.json", "ready.json", "completion.json", "host_result.json")]
    nonce = guard.name.removeprefix("process_tree_")
    script = (ROOT/"research/run_wrn300_refit.py").resolve()
    guard_source = shared.sha256(ROOT/"research/process_tree_guard.py")
    require(same_json(host, proof) and proof.get("schema") == spec.get("schema") == ready.get("schema") == completion.get("schema") == "owned_python_process_tree_v1" and
        proof.get("nonce") == spec.get("nonce") == ready.get("nonce") == completion.get("nonce") == nonce and
        proof.get("cleanup_verified") is True and proof.get("timed_out") is False and
        type(proof.get("child_returncode")) is int and proof["child_returncode"] == 0 and
        proof.get("completion_authenticated") is True and type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"] == 0 and
        type(completion.get("exit_code")) is int and completion["exit_code"] == 0 and
        proof.get("method") == ready.get("method") == completion.get("method") and
        proof.get("method") in {"windows_job_v1", "posix_session_v1"} and
        proof.get("child_pid") == ready.get("pid") == completion.get("pid") == receipt.get("child_pid") and
        proof.get("child_created_at") == ready.get("created_at") == completion.get("created_at") == receipt.get("child_created_at") and
        spec.get("host_pid") == receipt.get("pid") and spec.get("host_created_at") == receipt.get("process_created_at") and
        Path(spec["cwd"]).resolve() == ROOT.resolve() and Path(spec["script"]).resolve() == script and
        spec.get("script_sha256") == ready.get("script_sha256") == completion.get("script_sha256") == shared.sha256(script) and
        proof.get("guard_source_sha256") == spec.get("guard_source_sha256") == guard_source and
        spec.get("arguments") == ["--_worker", str(folder), "--_manifest-sha", shared.sha256(inputs_path)] and
        spec.get("job_name") == ("Local\\SaberAugMix_"+nonce if proof["method"] == "windows_job_v1" else None) and
        ended.timestamp()-1 <= spec.get("deadline_unix", 0) <= receipt["deadline_unix"]+1,
        "Actual prediction OS completion/source/identity proof differs")
    require(type(receipt.get("pid")) is int and receipt["pid"] > 0 and type(receipt.get("child_pid")) is int and receipt["child_pid"] > 0 and
        all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] > 0 for k in ("process_created_at", "child_created_at")),
        "Prediction process creation identity is missing")


def validate_refit_delivery(manifest_path, require_prediction=True):
    manifest_path = Path(manifest_path).resolve()
    document = read(manifest_path)
    require(require_prediction is True and document.get("schema") == SCHEMA and document.get("status") == "complete",
        "A completed explicit WRN300 refit prediction delivery is required")
    plan = Path(document["refit_plan_path"]).resolve()
    info, parent, frozen, paths = validate_refit_run(plan)
    expected = metadata(plan, info, parent, frozen)
    require(all(same_json(document.get(k), v) for k, v in expected.items()),
        "WRN300 delivery final weights, recipe, source or zero-development claim differs")
    require(document.get("refit_evidence_sha256") == snapshots(paths), "Complete WRN300 refit evidence changed")
    data_dir = Path(frozen["data_dir"])
    test_paths = test_inputs(data_dir)
    test_manifest = Path(document["test_manifest_path"])
    validate_test_manifest(data_dir, test_manifest)
    require(document.get("test_files_sha256") == snapshots([*test_paths, test_manifest]), "Official WRN300 test evidence changed")
    csv_path = Path(document["csv_path"]).resolve()
    require(csv_path == manifest_path.parent/"submission.csv", "Use the new WRN300 delivery's actual canonical CSV")
    digest = shared.validate_csv(csv_path, data_dir/"sample_submission.csv", shared.SAMPLE_SHA)
    require(digest == document.get("submission_sha256") and document.get("rows") == 2000 and
        document.get("columns") == ["id", "label"] and document.get("ids_match_official_order") is True,
        "Actual WRN300 CSV rows, IDs, labels or hash differ")
    validate_prediction_receipt(manifest_path, document, info, paths, test_paths)
    return {**{key: expected[key] for key in ("run", "competition", "checkpoint_sha256", "parameter_count", "policy", "seed",
        "training_examples", "validation_examples", "development_accuracy", "independent_development_evaluated",
        "parent_delivery_sha256", "epochs_completed", "checkpoint_epoch", "weight_variant", "training_precision")},
        "csv_path": str(csv_path), "submission_sha256": digest, "manifest_sha256": shared.sha256(manifest_path),
        "frozen_manifest_sha256": info["plan_sha256"]}


def predict(plan_path, output_dir, device="cuda", guard_dir=None):
    """Invoke only inside the new controller's authenticated prediction worker."""
    from research.run_wrn300_refit import validate_worker
    output_dir, plan_path = Path(output_dir).resolve(), Path(plan_path).resolve()
    require(guard_dir is not None and Path(guard_dir).resolve() == output_dir, "A supervised prediction OS guard is required")
    inputs_path = output_dir/"refit_inputs.json"
    authorization = validate_worker(output_dir, shared.sha256(inputs_path))
    require(authorization.get("mode") == "predict" and Path(authorization["plan_path"]).resolve() == plan_path,
        "Prediction worker authorization belongs to another plan or mode")
    # No Torch/model/hidden-image imports precede the live resource boundary.
    import runtime
    import torch
    from data import load_kaggle_test
    from model import build_model, count_parameters
    from train import code_provenance
    from research import benchmark as suite
    info, parent, frozen, paths = validate_refit_run(plan_path)
    evidence = snapshots(paths)
    csv_path, manifest_path = output_dir/"submission.csv", output_dir/"delivery.json"
    require(not csv_path.exists() and not manifest_path.exists(), "WRN300 prediction outputs are immutable")
    device = torch.device(device)
    require(device.type == "cuda" and torch.cuda.is_available(), "Actual native CUDA is required for the frozen inference context")
    context = {"device_type": device.type, "amp_dtype": str(suite.benchmark_dtype(device)),
        "torch_version": torch.__version__, "training_core_sha256": code_provenance()["source_sha256"]}
    require(context == frozen["evaluation_context"] and context["amp_dtype"] == "torch.bfloat16",
        "WRN300 inference must use the separately frozen local native-BF16 context")
    data_dir = Path(frozen["data_dir"])
    test_paths = test_inputs(data_dir)
    input_hashes = snapshots(test_paths)
    runtime.seed_all(2026)
    checkpoint = torch.load(info["checkpoint_path"], map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    require(count_parameters(model) == count_parameters(model, trainable_only=True) == 4166298,
        "Actual single residual CNN exceeds or differs from the fixed parameter budget")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    ids, images = load_kaggle_test(data_dir)
    test_manifest_path = data_dir/"kaggle_test_manifest.json"
    validate_test_manifest(data_dir, test_manifest_path)
    require(len(ids) == len(set(ids)) == 2000, "Official WRN300 IDs must contain2000 unique examples")
    require(input_hashes == snapshots(test_paths), "Official test source changed while loading")
    test_hashes = snapshots([*test_paths, test_manifest_path])
    predictions = []
    batch_size = read(frozen["benchmark_path"])["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.as_tensor(images[start:start+batch_size], device=device)
            with torch.autocast(device.type, dtype=suite.benchmark_dtype(device)):
                logits = suite.infer(model, batch, "center80_shift20")
            require(torch.isfinite(logits).all().item(), "Nonfinite WRN300 inference logits")
            predictions.extend(logits.argmax(1).cpu().tolist())
    require(evidence == snapshots(paths) and test_hashes == snapshots([*test_paths, test_manifest_path]),
        "WRN300 training or official test evidence changed during inference")
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, predictions, strict=True))
    digest = shared.validate_csv(csv_path, data_dir/"sample_submission.csv", shared.SAMPLE_SHA)
    document = {"schema": SCHEMA, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata(plan_path, info, parent, frozen), "refit_evidence_sha256": evidence,
        "csv_path": str(csv_path), "submission_sha256": digest, "rows": 2000, "columns": ["id", "label"],
        "ids_match_official_order": True, "test_manifest_path": str(test_manifest_path), "test_files_sha256": test_hashes,
        "prediction_input_manifest_sha256": shared.sha256(inputs_path),
        "interpretation": "Scratch full50k fixed-final300 refit of the exact qualified40k parent; no independent development accuracy belongs to this refit."}
    shared.durable_json(manifest_path, document, exclusive=True)
    return {"manifest_path": str(manifest_path), "manifest_sha256": shared.sha256(manifest_path), "submission_sha256": digest}
