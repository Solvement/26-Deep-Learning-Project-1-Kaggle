"""Pure, exact lineage for the one permitted scratch-distillation teacher.

This module reads JSON and file hashes only. It never loads tensors, evaluates
images, imports a numerical runtime, chooses a teacher or grants admission.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "scratch_kd_teacher_reference_v1"
FREEZE_REL = "results/research/wrn34_3_300_delivery/frozen.json"
CHECKPOINT_REL = "results/research/colab_wrn34_3_300_fp16_s2026_v3/candidate/best.pt"
BENCHMARK_REL = "results/research/wrn34_3_300_s2026_local_v3/benchmark.json"
FREEZE_SHA = "b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3"
TEACHER_SHA = "53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc"
BENCHMARK_SHA = "e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b"
CORE = {
    "train.py": "4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
    "model.py": "79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
    "data.py": "4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
    "runtime.py": "75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
}
METHOD = {
    "research/train_shadow.py": "9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
    "research/augmentations.py": "718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
}
SIGNATURE = {
    "archive_sha256": "d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60",
    "split_sha256": "73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e",
}
MODEL = {"depth": 34, "widen_factor": 3, "dropout": 0.0, "num_classes": 10}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(exact(left[k], v) for k, v in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def expected_config(root):
    tree = ast.parse((root/"train.py").read_text(encoding="utf-8"))
    defaults = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in n.targets))
    return {**defaults, **MODEL, "epochs": 300, "seed": 2026}


def build_teacher_reference(data_dir=None, root=ROOT):
    """Return the immutable eight-field reference after a complete byte audit."""
    from research import incumbent_compare as aggregate
    from research.preflight_core_control import official_inputs, DATA_IDENTITY
    root = Path(root).resolve()
    frozen_path, checkpoint_path, benchmark_path = (root/n for n in (FREEZE_REL, CHECKPOINT_REL, BENCHMARK_REL))
    for path, digest in ((frozen_path, FREEZE_SHA), (checkpoint_path, TEACHER_SHA), (benchmark_path, BENCHMARK_SHA)):
        require(path.is_file() and sha256(path) == digest, "Exact original primary26 teacher artifacts required")
    for name, digest in {**CORE, **METHOD}.items():
        require(sha256(root/name) == digest, "Original teacher core/method source bytes changed")
    frozen, benchmark = read(frozen_path), read(benchmark_path)
    config = expected_config(root)
    required = {"schema": "research_candidate_freeze_v2", "status": "frozen",
        "config": config, "model_config": MODEL, "parameter_count": 4166298,
        "selected_epoch": 298, "weight_variant": "model", "policy": "center80_shift20",
        "data_signature": SIGNATURE, "confirmation_evaluated": False,
        "repeat_prediction_authorized": False, "full_refit_supported": False,
        "source_git_commit": None}
    require(all(exact(frozen.get(k), v) for k, v in required.items()), "Teacher freeze identity/config/source purpose differs")
    require(Path(frozen["checkpoint_path"]).resolve() == checkpoint_path and
        Path(frozen["benchmark_path"]).resolve() == benchmark_path, "Only the original primary26 locations are teachers")
    decision = frozen.get("primary_decision", {})
    require(decision.get("seed") == 2026 and type(decision.get("seed")) is int and
        decision.get("checkpoint_sha256") == TEACHER_SHA and decision.get("benchmark_sha256") == BENCHMARK_SHA and
        decision.get("policy") == "center80_shift20", "Actual teacher primary decision differs")
    decided, stamped = (datetime.fromisoformat(x) for x in (decision["decided_at"], frozen["frozen_at"]))
    require(decided.tzinfo is not None and stamped.tzinfo is not None and
        decided <= stamped < datetime.now(timezone.utc), "Teacher decision/freeze chronology differs")
    evidence = frozen.get("evidence_sha256")
    require(isinstance(evidence, dict) and len(evidence) == 29, "Complete original29 teacher evidence files required")
    canonical = {}
    for name, digest in evidence.items():
        require(isinstance(name, str) and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Teacher evidence hash shape differs")
        path = Path(name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha256(path) == digest, "Original teacher evidence bytes changed")
        require(path not in canonical, "Aliased teacher evidence paths are not permitted")
        canonical[path] = digest
    require(canonical.get(checkpoint_path) == TEACHER_SHA and canonical.get(benchmark_path) == BENCHMARK_SHA,
        "Original checkpoint and native benchmark must belong to frozen teacher evidence")
    data_dir = Path(data_dir).resolve() if data_dir is not None else Path(frozen["data_dir"]).resolve()
    require(data_dir == Path(frozen["data_dir"]).resolve(), "Use the same official teacher/student data context")
    official_paths, identity = official_inputs(data_dir)
    require(identity == DATA_IDENTITY and len(official_paths) == 5, "Exact official40k training input identity required")
    aggregate.validate_document(benchmark)
    header = {"schema": "official_development_stress_v3", "status": "complete", "examples": 5000,
        "training_examples": 40000, "checkpoint_sha256": TEACHER_SHA, "checkpoint_epoch": 298,
        "checkpoint_variant": "model", "parameter_count": 4166298, "model_config": MODEL,
        "config": config, "data_signature": SIGNATURE, "bn_recalibrated": False,
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}
    require(all(exact(benchmark.get(k), v) for k, v in header.items()), "Teacher native benchmark header differs")
    context = {"device_type": "cuda", "amp_dtype": "torch.bfloat16", "torch_version": "2.12.0+cu130",
        "training_core_sha256": CORE}
    require(exact(benchmark.get("evaluation_context"), context) and exact(frozen.get("evaluation_context"), context),
        "Fixed nativeBF16 teacher evaluation context required")
    integrity = benchmark.get("research_training_integrity", {})
    training = {"schema": "research_shadow_training_v1", "source_sha256": METHOD, "precision": "fp16",
        "grad_scaler_enabled": True, "training_seed": 2026, "split_seed": 2026,
        "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}
    require(integrity.get("schema") == "research_shadow_integrity_v1" and exact(integrity.get("training"), training) and
        type(integrity.get("successful_updates")) is int and integrity["successful_updates"] == 93864 and
        type(integrity.get("skipped_updates")) is int and integrity["skipped_updates"] == 36,
        "Preserve the actual original FP16 teacher93864 successful/36 skipped updates")
    provenance_path = checkpoint_path.parent/"provenance.json"
    provenance = read(provenance_path)
    require(canonical.get(provenance_path) == sha256(provenance_path) == integrity.get("provenance_document_sha256") and
        provenance.get("code", {}).get("git_commit") is None and provenance.get("train_examples") == 40000 and
        provenance.get("validation_examples") == 5000 and provenance.get("parameter_count") == 4166298 and
        exact(provenance.get("data_signature"), SIGNATURE), "Actual teacher provenance/data/null training commit must be preserved")
    reference = {"schema": SCHEMA, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": TEACHER_SHA,
        "frozen_manifest_path": str(frozen_path), "frozen_manifest_sha256": FREEZE_SHA,
        "benchmark_path": str(benchmark_path), "benchmark_sha256": BENCHMARK_SHA,
        "evidence_sha256": {**evidence, str(frozen_path): FREEZE_SHA}}
    json.dumps(reference, allow_nan=False)
    return reference


def validate_teacher_reference(reference, data_dir=None, root=ROOT):
    require(isinstance(reference, dict), "An explicit original teacher reference is required")
    expected = build_teacher_reference(data_dir, root)
    require(exact(reference, expected), "Teacher reference or complete lineage differs from the sole permitted primary26")
    return expected
