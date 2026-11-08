"""Attest a real Colab ResizeMix repeat followed by a local v3 benchmark.

This is a composition of two preserved receipts, never a local-training receipt.
All private artifacts remain in their original locations; validation is CPU-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import runtime  # Apply the existing Windows cuDNN isolation before importing torch.
import torch
from model import build_model, count_parameters
from train import DEFAULT_CONFIG
from research import delivery_v2 as delivery
from research import run_colab as colab

SCHEMA = "research_cloud_repeat_attestation_v1"
SOURCE_FILES = (*colab.sources_for_trainer("resizemix"), "research/run_colab.py",
    "research/run_diagnostic.py", "research/delivery_v2.py", "research/paired_compare_v2.py",
    "research/paired_compare.py", "research/repeat_attestation.py")
REQUIRED_OUTPUTS = {"baseline/benchmark.json", "remote_receipt.json", "native_amp_smoke.json",
    "method_preflight.json", *("candidate/"+name for name in ("best.pt", "last.pt", "metrics.json",
    "history.json", "config.json", "environment.json", "provenance.json", "benchmark.json", "benchmark.core.json"))}
read, require, stamp = delivery.read, delivery.require, delivery.stamp


def sha256(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def zip_hash(archive, name):
    with archive.open(name) as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def safe_members(archive):
    names = archive.namelist()
    require(len(set(names)) == len(names) and all(not n.endswith("/") and "\\" not in n and
        not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and
        ":" not in n and not (archive.getinfo(n).external_attr >> 16) & 0o170000 == 0o120000
        for n in names), "ZIP contains duplicate, unsafe or non-file members")
    return set(names)


def validate_cleanup(folder, receipt):
    require(receipt.get("runtime_stop_verified") is True and receipt.get("stop_exit_code") == 0 and
        receipt.get("stop_reported_termination") is True and receipt.get("allocation_identity_verified") is True and
        receipt.get("usage_after", {}).get("active_assignments") == 0, "Cloud cleanup was not verified")
    paths = []
    reconciliation = receipt.get("cleanup_reconciliation")
    if reconciliation:
        for key, digest_key in (("server_absence_proof", "server_absence_sha256"),
                ("usage_proof", "usage_sha256"), ("preserved_original_receipt", "original_receipt_sha256")):
            name = reconciliation.get(key, "")
            require(name and Path(name).name == name, "Cleanup proof path must remain inside its original run")
            path = folder/name
            require(sha256(path) == reconciliation.get(digest_key), "Preserved cleanup proof fingerprint differs")
            paths.append(path)
        absence, usage, original = map(read, paths)
        require(original.get("status") == "cleanup_unverified" and original.get("runtime_stop_verified") is False and
            all(receipt.get(k) == original.get(k) for k in original if k not in {"status", "runtime_stop_verified", "usage_after"}),
            "Cleanup reconciliation changed original training evidence")
        require(absence.get("status") == "read_only_checked" and absence.get("server_assignment_count") == 0 and
            absence.get("own_endpoint_identified") is True and absence.get("own_endpoint_present") is False and
            absence.get("server_side_runtime_absent_verified") is True and absence.get("official_termination_history_present") is True and
            absence.get("no_unassign_performed") is True and absence.get("controller_receipt_sha256") == sha256(paths[2]) and
            usage.get("status") == "read_only_checked" and usage.get("usage") == receipt["usage_after"] and
            usage.get("prior_absence_proof") == paths[0].name and usage.get("prior_absence_proof_sha256") == sha256(paths[0]) and
            usage.get("no_stop_or_unassign_performed") is True and reconciliation.get("no_second_stop_or_unassign") is True and
            stamp(receipt["finished_at"]) <= stamp(absence["checked_at"]) <= stamp(usage["checked_at"]) == stamp(receipt["cleanup_verified_at"]) <= delivery.utc_now(),
            "Read-only cloud cleanup reconciliation is incomplete")
    return paths


def validate_input(folder, frozen, config):
    path, bundle = folder/"input_manifest.json", folder/"input.private.zip"
    document = read(path)
    expected = set(colab.sources_for_trainer("resizemix")) | {"data/"+n for n in colab.CACHE} | {
        "baseline/best.pt", "baseline/metrics.json", "baseline/history.json", "research/configs/candidate.json"}
    require(document.get("schema") == "private_colab_input_v1" and document.get("trainer") == "resizemix" and
        document.get("hidden_images_packaged") is False and document.get("external_data_packaged") is False and
        document.get("confirmation_evaluated") is False and set(document.get("files_sha256", {})) == expected and
        document.get("bundle_sha256") == sha256(bundle), "Cloud original input manifest/bundle differs")
    paths = [path, bundle]
    with zipfile.ZipFile(bundle) as archive:
        require(safe_members(archive) == expected, "Cloud input archive allowlist differs")
        for name, digest in document["files_sha256"].items():
            require(zip_hash(archive, name) == digest, "Cloud input ZIP member fingerprint differs")
            if name == "research/configs/candidate.json": continue
            current = (Path(frozen["data_dir"])/name[5:] if name.startswith("data/") else
                ROOT/"results/regularized"/name[9:] if name.startswith("baseline/") else ROOT/name)
            require(sha256(current) == digest, "Cloud packaged source/official cache/trusted baseline differs")
            paths.append(current)
        raw = archive.read("research/configs/candidate.json")
    raw_config = json.loads(raw)
    raw_hash = hashlib.sha256(raw).hexdigest()
    require(document.get("config") == raw_config and {**DEFAULT_CONFIG, **raw_config} == config and
        raw_hash == frozen["primary_decision"].get("planned_repeat_config_sha256"),
        "Cloud raw configuration was not the predeclared exact repeat")
    chunks = document.get("chunks", [])
    require(bool(chunks), "Original cloud input chunks are missing")
    combined = hashlib.sha256()
    for index, part in enumerate(chunks):
        require(part.get("name") == f"input_{index:03d}.private.part", "Cloud input chunk sequence differs")
        chunk = folder/part["name"]
        require(type(part.get("bytes")) is int and 0 < part["bytes"] <= colab.CHUNK_BYTES and
            chunk.stat().st_size == part["bytes"] and sha256(chunk) == part.get("sha256"), "Cloud original chunk changed")
        with chunk.open("rb") as stream:
            while block := stream.read(1024*1024): combined.update(block)
        paths.append(chunk)
    require(combined.hexdigest() == document["bundle_sha256"], "Cloud input chunks do not reconstruct the bundle")
    return document, raw_hash, paths


def validate_last(folder, checkpoint, metrics, history):
    last = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
    require(last.get("epoch") == 160 and last.get("history") == history and
        last.get("config") == checkpoint["config"] and last.get("data_signature") == checkpoint["data_signature"] and
        last.get("provenance") == checkpoint["provenance"] and last.get("updates") == metrics["successful_updates"] and
        last.get("skipped_updates") == metrics["skipped_updates"] and
        last.get("grad_scaler", {}).get("scale") == history[-1]["grad_scaler_scale"], "Final 160-epoch training state differs")
    model = build_model(checkpoint["model_config"])
    require(count_parameters(model) == checkpoint["parameter_count"] <= 5000000, "Constructed repeat exceeds parameter budget")
    for key in ("model", "ema"):
        model.load_state_dict(last[key], strict=True)
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()),
                "Final repeat training state contains nonfinite parameters")


def derive(manifest_path, cloud_run, local_benchmark, local_receipt):
    manifest_path, cloud_run, local_benchmark, local_receipt = [Path(p).resolve() for p in
        (manifest_path, cloud_run, local_benchmark, local_receipt)]
    frozen, bundle = delivery.validate_frozen(manifest_path)
    folder, checkpoint_path = cloud_run/"candidate", cloud_run/"candidate/best.pt"
    cloud_path, local = cloud_run/"supervisor.json", read(local_receipt)
    cloud = read(cloud_path)
    require(cloud.get("status") == "complete" and cloud.get("remote_status") == "complete" and
        cloud.get("trainer") == "resizemix" and cloud.get("precision") == "fp16" and
        cloud.get("assignment_attempts") == 1 and cloud.get("no_new_purchases") is True and
        cloud.get("resource_policy") == "existing_Colab_entitlement_only_no_purchases" and
        cloud.get("allocation_returned_success") is True and cloud.get("remote_guard_verified") is True and
        cloud.get("confirmation_evaluated") is False and cloud.get("submitted") is False, "Complete single-allocation cloud repeat is required")
    cleanup_paths = validate_cleanup(cloud_run, cloud)
    require(stamp(frozen["primary_decision"]["decided_at"]) <= stamp(frozen["frozen_at"]) < stamp(cloud["started_at"]) <
        stamp(cloud["finished_at"]) <= stamp(local["started_at"]) <= stamp(local["finished_at"]) <= delivery.utc_now(),
        "Primary freeze, real cloud training and local benchmark times are out of order")
    if cloud.get("cleanup_verified_at"):
        require(stamp(cloud["cleanup_verified_at"]) <= stamp(local["started_at"]), "Cloud cleanup was unresolved before local benchmark")
    checkpoint, local_document = delivery.audit_experiment(checkpoint_path, local_benchmark, bundle["data"], frozen["evaluation_context"])
    primary, config = bundle["checkpoint"], checkpoint["config"]
    marker = checkpoint.get("provenance", {}).get("research_training", {})
    require(config.get("epochs") == 160 and config.get("seed") == frozen["primary_decision"]["planned_repeat_seed"] != 2026 and
        {k:v for k,v in config.items() if k != "seed"} == {k:v for k,v in primary["config"].items() if k != "seed"} and
        marker.get("schema") == "research_resizemix_training_v1" and marker.get("precision") == "fp16" and
        {k:v for k,v in marker.items() if k != "training_seed"} == {k:v for k,v in primary["provenance"]["research_training"].items() if k != "training_seed"},
        "Repeat source, precision or frozen 160-epoch recipe differs")
    input_document, raw_config_sha, paths = validate_input(cloud_run, frozen, config)
    require(cloud.get("run") == input_document["run"] and cloud.get("input_bundle_sha256") == input_document["bundle_sha256"],
        "Outer cloud receipt does not bind original inputs")
    require(positive(cloud.get("deadline_unix")) and stamp(cloud["finished_at"]).timestamp() <= cloud["deadline_unix"] <=
        stamp(cloud["started_at"]).timestamp()+10800, "Cloud run exceeded its bounded deadline")
    artifacts = cloud_run/"artifacts.private.zip"
    require(sha256(artifacts) == cloud.get("artifact_sha256"), "Cloud artifact archive fingerprint differs")
    with zipfile.ZipFile(artifacts) as archive:
        members = safe_members(archive)
        require(REQUIRED_OUTPUTS <= members <= set(colab.OUTPUTS), "Cloud artifact allowlist/completeness differs")
        for name in members:
            path = cloud_run/name
            require(zip_hash(archive, name) == sha256(path), "Extracted cloud artifact differs from original ZIP")
            paths.append(path)
    remote = read(cloud_run/"remote_receipt.json")
    require(remote.get("status") == "complete" and remote.get("run") == cloud["run"] and
        remote.get("trainer") == "resizemix" and remote.get("confirmation_evaluated") is False and
        remote.get("hidden_images_loaded") is False and remote.get("source_files_sha256") == input_document["files_sha256"],
        "Original remote training receipt differs")
    for filename, generated in (("remote_program.py", colab.remote_program(input_document, cloud["session"], cloud["deadline_unix"])),
                               ("remote_guard.py", colab.remote_guard(cloud["deadline_unix"]))):
        path = cloud_run/filename
        require(path.read_text(encoding="utf-8") == generated, "Cloud generated scratch command/guard source differs")
        paths.append(path)
    environment = read(folder/"environment.json")
    primary_environment = read(Path(frozen["checkpoint_path"]).parent/"environment.json")
    require(environment.get("device") == "cuda" and environment.get("gpu") == primary_environment.get("gpu") == "Tesla T4" and
        environment.get("packages", {}).get("torch") == primary_environment.get("packages", {}).get("torch") and
        environment.get("packages", {}).get("torchvision") == primary_environment.get("packages", {}).get("torchvision") and
        environment.get("packages", {}).get("torch", "").startswith("2.11.") and
        environment.get("cuda_runtime") == primary_environment.get("cuda_runtime"), "Cloud training hardware/software profile differs from primary T4")
    context = {"device_type": "cuda", "amp_dtype": "torch.float16", "torch_version": environment["packages"]["torch"],
               "training_core_sha256": bundle["data"][1]}
    require(cloud.get("runtime_preflight", {}).get("status") == "armed" and
        cloud["runtime_preflight"].get("device") == "Tesla T4" and cloud["runtime_preflight"].get("amp_dtype") == "float16" and
        cloud["runtime_preflight"].get("torch_version") == context["torch_version"], "Cloud native T4 guard profile differs")
    smoke, preflight = read(cloud_run/"native_amp_smoke.json"), read(cloud_run/"method_preflight.json")
    require(remote.get("native_amp_smoke") == smoke and remote.get("method_preflight") == preflight and
        smoke.get("status") == "complete" and smoke.get("finite_update_verified") is True and
        smoke.get("finite_loss_overflow_skipped") is True and smoke.get("overflow_parameter_unchanged") is True and
        smoke.get("torch_version") == context["torch_version"] and
        smoke.get("trainer") == "resizemix" and positive(smoke.get("scale_before_overflow")) and
        positive(smoke.get("scale_after_overflow")) and smoke["scale_after_overflow"] < smoke["scale_before_overflow"] and
        smoke.get("shadow_source_sha256") == marker["source_sha256"]["research/train_shadow.py"] and
        smoke.get("trainer_source_sha256") == marker["source_sha256"]["research/train_resizemix.py"] and
        preflight.get("status") == "complete" and preflight.get("method") == "resizemix" and
        preflight.get("synthetic_only") is True and preflight.get("training_data_loaded") is False and
        preflight.get("hidden_images_loaded") is False and preflight.get("geometry_verified") is True and
        preflight.get("native_fp16_verified") is True and preflight.get("real_parameter_updates") == 13 and
        preflight.get("warmup_steps") == 3 and preflight.get("timed_steps") == 10 and
        preflight.get("batch_size") == config["batch_size"] and preflight.get("seed") == 2026 and
        type(preflight.get("actual_mixed_batches")) is int and preflight["actual_mixed_batches"] > 0 and
        positive(preflight.get("median_step_seconds")) and positive(preflight.get("p90_step_seconds")) and
        positive(preflight.get("peak_gpu_memory_bytes")) and
        preflight.get("parameters") == checkpoint["parameter_count"] and preflight.get("model_config") == checkpoint["model_config"] and
        preflight.get("trainer_source_sha256") == marker["source_sha256"]["research/train_resizemix.py"] and
        preflight.get("helper_source_sha256") == marker["source_sha256"]["research/resizemix.py"] and preflight.get("runtime_budget_fit") is True,
        "Cloud geometry/native-FP16 preflight evidence differs")
    _, cloud_document = delivery.audit_experiment(checkpoint_path, folder/"benchmark.json", bundle["data"], context)
    cloud_baseline = read(cloud_run/"baseline/benchmark.json")
    delivery.suite.validate_baseline(cloud_baseline, cloud_document["policies"], frozen["data_signature"],
        cloud_document["batch_size"], context, sha256(ROOT/"research/benchmark.py"))
    require(cloud_document.get("baseline_sha256") == sha256(cloud_run/"baseline/benchmark.json"), "Original T4 benchmark baseline differs")
    for policy, report in cloud_document["policies"].items():
        require(cloud_document.get("promotion", {}).get(policy) == delivery.suite.promotion_gate(report, cloud_baseline["policies"]["flip"]), "T4 promotion metadata differs")
    metrics, history, provenance = [read(folder/name) for name in ("metrics.json", "history.json", "provenance.json")]
    require(provenance.get("train_examples") == 40000 and provenance.get("validation_examples") == 5000, "Actual training provenance is not 40k/5k")
    validate_last(folder, checkpoint, metrics, history)
    entry = ROOT/"research/benchmark_resizemix.py"
    require(local.get("status") == "complete" and local.get("mode") == "resizemix-benchmark" and
        local.get("role") == "diagnostic" and local.get("backend") == "local" and
        local.get("confirmation_evaluated") is False and local.get("submitted") is False and
        local.get("artifact") == local_benchmark.name and local.get("artifact_sha256") == sha256(local_benchmark) and
        local.get("supervisor_source_sha256") == sha256(ROOT/"research/run_diagnostic.py") and
        local.get("input_sha256") == delivery.snapshots([checkpoint_path, Path(frozen["baseline_path"]), entry]),
        "Local diagnostic receipt source/input/output fingerprints differ")
    require(local_document.get("baseline_sha256") == sha256(frozen["baseline_path"]), "Local repeat baseline differs")
    delivery.passed_policy(local_document, bundle["baseline"], frozen["policy"])
    paths += [manifest_path, cloud_path, artifacts, local_receipt, local_benchmark, *cleanup_paths,
        Path(frozen["checkpoint_path"]).parent/"environment.json", *map(Path, frozen["evidence_sha256"])]
    paths += delivery.artifact_paths(checkpoint_path, local_benchmark, checkpoint)
    paths += [ROOT/name for name in SOURCE_FILES]
    evidence = delivery.snapshots(paths)
    return {"schema": SCHEMA, "status": "complete", "composition": True,
        "backend": "cloud_training_local_benchmark", "trainer": "resizemix", "precision": "fp16",
        "frozen_manifest_path": str(manifest_path), "frozen_manifest_sha256": sha256(manifest_path),
        "cloud_run_path": str(cloud_run), "local_receipt_path": str(local_receipt),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "benchmark_path": str(local_benchmark), "benchmark_sha256": sha256(local_benchmark),
        "config_sha256": raw_config_sha, "training_examples": 40000, "validation_examples": 5000, "epochs": 160,
        "policy": frozen["policy"], "seed": config["seed"], "started_at": cloud["started_at"], "finished_at": local["finished_at"],
        "cloud_run_finished_at": cloud["finished_at"], "local_benchmark_started_at": local["started_at"],
        "confirmation_evaluated": False, "submitted": False, "hidden_images_loaded": False,
        "source_sha256": {name: sha256(ROOT/name) for name in SOURCE_FILES}, "evidence_sha256": evidence,
        "interpretation": "Preserved Colab scratch training plus separate local benchmark; no local training is claimed."}


def validate_attestation(path, manifest_path=None):
    path = Path(path).resolve()
    document = read(path)
    require(document.get("schema") == SCHEMA and document.get("status") == "complete", "Unsupported or incomplete cloud repeat attestation")
    if manifest_path is not None:
        require(Path(document["frozen_manifest_path"]).resolve() == Path(manifest_path).resolve(), "Attestation belongs to another frozen primary")
    delivery.verify_snapshots(document.get("evidence_sha256"))
    expected = derive(document["frozen_manifest_path"], document["cloud_run_path"], document["benchmark_path"], document["local_receipt_path"])
    require({k:v for k,v in document.items() if k != "created_at"} == expected, "Cloud repeat attestation does not match rederived original evidence")
    require(stamp(document["finished_at"]) <= stamp(document["created_at"]) <= delivery.utc_now(), "Attestation creation time is invalid")
    delivery.verify_snapshots(document["evidence_sha256"])
    return {**document, "attestation_path": str(path), "attestation_sha256": sha256(path)}


def build(manifest_path, cloud_run, local_benchmark, local_receipt, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Attestations are immutable; use a new output path")
    document = derive(manifest_path, cloud_run, local_benchmark, local_receipt)
    document["created_at"] = delivery.utc_now().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".attestation-", suffix=".writing", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        validate_attestation(temporary, manifest_path)
        delivery.verify_snapshots(document["evidence_sha256"])
        os.link(temporary, output)  # Atomic, exclusive publication; never overwrites evidence.
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return {**document, "attestation_path": str(output), "attestation_sha256": sha256(output)}


def predict(manifest_path, attestation_path, output_dir, device="cuda"):
    document = validate_attestation(attestation_path, manifest_path)
    result = delivery.predict(manifest_path, document["checkpoint_path"], document["benchmark_path"], attestation_path, output_dir, device)
    validate_attestation(attestation_path, manifest_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("build", "validate", "predict"): action.add_argument("--"+name, action="store_true")
    for name in ("manifest", "cloud-run", "local-benchmark", "local-receipt", "output", "attestation", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.build:
        if not all((args.manifest, args.cloud_run, args.local_benchmark, args.local_receipt, args.output)):
            parser.error("--build requires --manifest --cloud-run --local-benchmark --local-receipt --output")
        result = build(args.manifest, args.cloud_run, args.local_benchmark, args.local_receipt, args.output)
    elif args.predict:
        if not all((args.manifest, args.attestation, args.output_dir)):
            parser.error("--predict requires --manifest --attestation --output-dir")
        result = predict(args.manifest, args.attestation, args.output_dir, args.device)
    else:
        if not args.attestation: parser.error("--validate requires --attestation")
        result = validate_attestation(args.attestation, args.manifest)
    print(json.dumps(result))


if __name__ == "__main__": main()
