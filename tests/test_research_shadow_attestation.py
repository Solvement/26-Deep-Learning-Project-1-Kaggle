"""CPU-only original-archive and real-method audits for composed repeat receipts."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from model import build_model, count_parameters
from train import DEFAULT_CONFIG, code_provenance
from research import shadow_repeat_attestation as att
from research.benchmark_shadow import validate_shadow
from research.train_shadow import research_provenance


class ShadowAttestationTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cloud = self.root/"cloud"
        self.folder = self.cloud/"candidate"
        self.folder.mkdir(parents=True)
        self.local = self.root/"local"
        self.local.mkdir()
        self.data = self.root/"data"
        self.data.mkdir()
        for name in att.SOURCE_FILES:
            dest = self.root/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(att.ROOT/name, dest)
        for name in att.colab.CACHE: (self.data/name).write_bytes(b"synthetic official cache fixture "+name.encode())
        for name in ("best.pt", "metrics.json", "history.json"):
            path = self.root/"results/regularized"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"temporary trusted fixture "+name.encode())
        public = Path(__file__).resolve().parents[1]/"research/configs/wrn34_3_300_seed2027.json"
        self.raw_config = public.read_bytes()
        self.config = {**DEFAULT_CONFIG, **json.loads(self.raw_config)}
        model_config = {key: self.config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
        model = build_model(model_config)
        self.core = code_provenance()["source_sha256"]
        self.signature = {"archive_sha256": "a"*64, "split_sha256": "b"*64}
        self.cloud_context = {"device_type": "cuda", "amp_dtype": "torch.float16", "torch_version": "2.11.0+cu130", "training_core_sha256": self.core}
        self.context = {**self.cloud_context, "amp_dtype": "torch.bfloat16", "torch_version": torch.__version__}
        marker = research_provenance("fp16", self.config)
        provenance = {"source_sha256": self.core, "research_training": marker}
        self.checkpoint = {"model": model.state_dict(), "model_config": model_config, "config": self.config,
            "epoch": 300, "variant": "model", "validation_accuracy": .9, "validation_loss": .3,
            "parameter_count": count_parameters(model), "data_signature": self.signature, "provenance": provenance}
        self.history = [{"epoch": epoch, "validation_accuracy": .9, "validation_loss": .3,
            "ema_validation_accuracy": .8, "ema_validation_loss": .4, "successful_updates": 313,
            "skipped_updates": 0, "grad_scaler_scale": 256.} for epoch in range(1, 301)]
        self.metrics = {"status": "complete", "epochs_completed": 300, "config": self.config,
            "training_examples": 40000, "validation_examples": 5000, "parameter_count": count_parameters(model),
            "best_validation_accuracy": .9, "best_validation_loss": .3,
            "successful_updates": 93900, "skipped_updates": 0, "research_training": marker}
        self.environment = {"device": "cuda", "gpu": "Tesla T4", "cuda_runtime": "13.0",
            "packages": {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"},
            "shadow_training_precision": "fp16", "grad_scaler_enabled": True}
        self.provenance = {"code": provenance, "data_signature": self.signature, "train_examples": 40000, "validation_examples": 5000}
        for name, value in (("config.json", self.config), ("metrics.json", self.metrics), ("history.json", self.history),
                ("environment.json", self.environment), ("provenance.json", self.provenance)):
            self.write(self.folder/name, value)
        self.checkpoint_path = self.folder/"best.pt"
        torch.save(self.checkpoint, self.checkpoint_path)
        self.last = {"model": model.state_dict(), "ema": copy.deepcopy(model.state_dict()), "epoch": 300,
            "history": self.history, "config": self.config, "data_signature": self.signature, "provenance": provenance,
            "updates": 93900, "skipped_updates": 0, "grad_scaler": {"scale": 256., "growth_factor": 2., "backoff_factor": .5, "growth_interval": 2000, "_growth_tracker": 0},
            "python_rng": __import__("random").Random(2027).getstate(), "numpy_rng": np.random.RandomState(2027).get_state(),
            "torch_rng": torch.Generator().manual_seed(2027).get_state(), "cuda_rng": [torch.ones(16,dtype=torch.uint8)],
            "optimizer": {"param_groups": [{"momentum": .9, "weight_decay": .0005, "nesterov": True, "dampening": 0, "lr": .0001, "params": list(range(len(list(model.parameters()))))}],
                "state": {i: {"momentum_buffer": torch.zeros_like(p)} for i,p in enumerate(model.parameters())}}}
        torch.save(self.last, self.folder/"last.pt")
        self.baseline = self.report_document(.8, .6, self.context)
        self.baseline_path = self.root/"baseline.json"
        self.write(self.baseline_path, self.baseline)
        cloud_baseline = self.report_document(.8, .6, self.cloud_context)
        self.write(self.cloud/"baseline/benchmark.json", cloud_baseline)
        self.seal_benchmark(self.folder/"benchmark.json", self.cloud_context, self.cloud/"baseline/benchmark.json")
        self.benchmark = self.local/"benchmark.json"
        self.seal_benchmark(self.benchmark, self.context, self.baseline_path)
        self.primary = self.root/"primary/best.pt"
        self.primary.parent.mkdir()
        self.primary.write_bytes(b"already qualified primary fixture")
        self.write(self.primary.parent/"environment.json", self.environment)
        raw_config = json.loads(self.raw_config)
        files = {name: att.sha256(self.root/name) for name in att.colab.sources_for_trainer("shadow")}
        files.update({"data/"+name: att.sha256(self.data/name) for name in att.colab.CACHE})
        files.update({"baseline/"+name: att.sha256(self.root/"results/regularized"/name) for name in ("best.pt", "metrics.json", "history.json")})
        files["research/configs/candidate.json"] = hashlib.sha256(self.raw_config).hexdigest()
        input_zip = self.cloud/"input.private.zip"
        with zipfile.ZipFile(input_zip, "w") as archive:
            for name in files:
                if name == "research/configs/candidate.json": archive.writestr(name, self.raw_config)
                else:
                    source = (self.data/name[5:] if name.startswith("data/") else
                        self.root/"results/regularized"/name[9:] if name.startswith("baseline/") else self.root/name)
                    archive.write(source, name)
        chunk = self.cloud/"input_000.private.part"
        chunk.write_bytes(input_zip.read_bytes())
        self.inputs = {"schema": "private_colab_input_v1", "run": "repeat_s2027", "trainer": "shadow",
            "files_sha256": files, "bundle_sha256": att.sha256(input_zip), "config": raw_config,
            "chunks": [{"name": chunk.name, "sha256": att.sha256(chunk), "bytes": chunk.stat().st_size}],
            "hidden_images_packaged": False, "external_data_packaged": False, "confirmation_evaluated": False}
        self.write(self.cloud/"input_manifest.json", self.inputs)
        self.smoke = {"status": "complete", "finite_update_verified": True, "finite_loss_overflow_skipped": True,
            "overflow_parameter_unchanged": True, "torch_version": "2.11.0+cu130",
            "trainer": "shadow", "scale_before_overflow": 65536., "scale_after_overflow": 32768.,
            "shadow_source_sha256": marker["source_sha256"]["research/train_shadow.py"],
            "trainer_source_sha256": marker["source_sha256"]["research/train_shadow.py"]}
        self.preflight = {"status": "complete", "method": "shadow_mixup_cutout", "synthetic_only": True,
            "engineering_only": True, "training_data_loaded": False, "hidden_images_loaded": False,
            "native_fp16_verified": True, "finite_parameters_verified": True, "finite_losses_verified": True,
            "real_parameter_updates": 13, "parameters": count_parameters(model), "warmup_steps": 3, "timed_steps": 10,
            "batch_size": 128, "seed": 2026, "actual_mixed_batches": 3, "median_step_seconds": .04,
            "p90_step_seconds": .045, "peak_gpu_memory_bytes": 500000000, "model_config": model_config,
            "trainer_source_sha256": marker["source_sha256"]["research/train_shadow.py"],
            "augmentation_source_sha256": self.core["train.py"], "runtime_budget_fit": True, "epochs": 300,
            "mixup_alpha": .2, "mixup_probability": .5, "cutout": 16, "runtime_safety_multiplier": 1.5,
            "pipeline_overhead_allowance_seconds": 300, "transfer_cleanup_reserve_seconds": 180,
            "grad_scaler_initial_scale": 256., "grad_scaler_final_scale": 256.,
            "estimated_training_only_seconds": .04*313*300, "conservative_pipeline_seconds": .04*313*300*1.5+300}
        self.remote = {"status": "complete", "run": "repeat_s2027", "trainer": "shadow",
            "confirmation_evaluated": False, "hidden_images_loaded": False, "source_files_sha256": files,
            "native_amp_smoke": self.smoke, "method_preflight": self.preflight}
        self.cloud_receipt = {"status": "complete", "remote_status": "complete", "run": "repeat_s2027", "trainer": "shadow",
            "precision": "fp16", "session": "saber-repeat", "deadline_unix": datetime(2026, 10, 31, 22, tzinfo=timezone.utc).timestamp(),
            "assignment_attempts": 1, "no_new_purchases": True, "resource_policy": "existing_Colab_entitlement_only_no_purchases",
            "allocation_returned_success": True, "remote_guard_verified": True, "confirmation_evaluated": False,
            "submitted": False, "runtime_stop_verified": True, "stop_exit_code": 0, "stop_reported_termination": True,
            "allocation_identity_verified": True, "usage_after": {"active_assignments": 0, "rate_compute_units_per_hour": 0., "balance_compute_units": 0.},
            "input_bundle_sha256": self.inputs["bundle_sha256"], "started_at": "2026-10-31T19:00:00+00:00",
            "finished_at": "2026-10-31T19:40:00+00:00", "runtime_preflight": {"status": "armed", "device": "Tesla T4", "amp_dtype": "float16", "torch_version": "2.11.0+cu130"}}
        self.reseal_cloud()
        self.receipt_path = self.local/"supervisor.json"
        self.local_receipt = {"status": "complete", "mode": "shadow-benchmark", "role": "diagnostic", "backend": "local",
            "confirmation_evaluated": False, "submitted": False, "artifact": self.benchmark.name, "artifact_sha256": att.sha256(self.benchmark),
            "supervisor_source_sha256": att.sha256(self.root/"research/run_diagnostic.py"),
            "input_sha256": att.delivery.snapshots([self.checkpoint_path, self.baseline_path, self.root/"research/benchmark_shadow.py"]),
            "started_at": "2026-10-31T20:00:00+00:00", "finished_at": "2026-10-31T20:01:00+00:00"}
        self.write(self.receipt_path, self.local_receipt)
        self.frozen = {"frozen_at": "2026-10-31T18:35:00+00:00", "primary_decision": {"decided_at": "2026-10-31T18:34:00+00:00",
            "planned_repeat_seed": 2027, "planned_repeat_config_sha256": hashlib.sha256(self.raw_config).hexdigest()},
            "data_dir": str(self.data), "checkpoint_path": str(self.primary), "baseline_path": str(self.baseline_path),
            "data_signature": self.signature, "evaluation_context": self.context, "policy": "center80_shift20",
            "evidence_sha256": att.delivery.snapshots([self.primary, self.primary.parent/"environment.json"])}
        self.manifest = self.root/"frozen.json"
        self.write(self.manifest, self.frozen)
        primary_pin = patch.object(att, "PRIMARY_MANIFEST_SHA", att.sha256(self.manifest))
        primary_pin.start(); self.addCleanup(primary_pin.stop)
        primary_marker = {**marker, "training_seed": 2026}
        self.bundle = {"checkpoint": {"config": {**self.config, "seed": 2026}, "parameter_count": 4166298,
                "provenance": {"research_training": primary_marker}},
            "baseline": self.baseline, "data": (self.signature, self.core, np.arange(40000), np.arange(40000, 45000))}
        for patcher in (patch.object(att, "ROOT", self.root), patch.object(att.delivery, "ROOT", self.root),
                patch.object(att.delivery, "validate_frozen", return_value=(self.frozen, self.bundle)),
                patch.object(att.delivery, "utc_now", return_value=datetime(2026, 10, 31, 21, tzinfo=timezone.utc))):
            patcher.start(); self.addCleanup(patcher.stop)

    def report_document(self, clean, stress, context):
        cases = {name: {"examples": 5000, "input_pixel_sha256": "c"*64,
            "accuracy": clean if name == "clean" else stress} for name in att.delivery.suite.CASES}
        report = {"cases": cases, "clean_accuracy": clean, "stress_accuracy": stress}
        return {"schema": att.delivery.suite.SCHEMA, "status": "complete", "examples": 5000,
            "training_examples": 40000, "data_signature": self.signature, "evaluation_context": context,
            "stress_cases": list(att.delivery.suite.CASES), "bn_recalibrated": False,
            "checkpoint_sha256": att.delivery.suite.BASELINE_CHECKPOINT_SHA, "batch_size": 128,
            "benchmark_source_sha256": att.sha256(self.root/"research/benchmark.py"), "policies": {"flip": report}}

    def seal_benchmark(self, path, context, baseline):
        document = self.report_document(.9, .7, context)
        document.update(checkpoint_sha256=att.sha256(self.checkpoint_path), training_metrics_sha256=att.sha256(self.folder/"metrics.json"),
            training_history_sha256=att.sha256(self.folder/"history.json"), config=self.config,
            model_config=self.checkpoint["model_config"], checkpoint_epoch=300, checkpoint_variant="model",
            parameter_count=self.checkpoint["parameter_count"], source={"source_sha256": self.core},
            confirmation_evaluated=False, hidden_images_loaded=False, external_data_loaded=False, baseline_sha256=att.sha256(baseline))
        report = document["policies"]["flip"]
        document["policies"] = {p: copy.deepcopy(report) for p in att.delivery.suite.POLICIES}
        document["promotion"] = {p: att.delivery.suite.promotion_gate(r, att.read(baseline)["policies"]["flip"]) for p,r in document["policies"].items()}
        core = path.with_name(path.stem+".core.json")
        self.write(core, document)
        integrity = validate_shadow(self.checkpoint, self.metrics, self.history, self.provenance, self.environment)
        document["research_training_integrity"] = {**integrity,
            "validator_source_sha256": att.sha256(self.root/"research/benchmark_shadow.py"), "core_benchmark_sha256": att.sha256(core),
            "provenance_document_sha256": att.sha256(self.folder/"provenance.json"), "environment_document_sha256": att.sha256(self.folder/"environment.json")}
        self.write(path, document)

    def reseal_cloud(self):
        for name, doc in (("remote_receipt.json", self.remote), ("native_amp_smoke.json", self.smoke), ("method_preflight.json", self.preflight)):
            self.write(self.cloud/name, doc)
        self.cloud.joinpath("remote_program.py").write_text(att.colab.remote_program(self.inputs, self.cloud_receipt["session"], self.cloud_receipt["deadline_unix"]), encoding="utf-8")
        self.cloud.joinpath("remote_guard.py").write_text(att.colab.remote_guard(self.cloud_receipt["deadline_unix"]), encoding="utf-8")
        with zipfile.ZipFile(self.cloud/"artifacts.private.zip", "w") as archive:
            for name in att.REQUIRED_OUTPUTS: archive.write(self.cloud/name, name)
        self.cloud_receipt["artifact_sha256"] = att.sha256(self.cloud/"artifacts.private.zip")
        self.write(self.cloud/"supervisor.json", self.cloud_receipt)

    def build(self):
        return att.build(self.manifest, self.cloud, self.benchmark, self.receipt_path, self.root/"attestation.json")

    def test_real_cpu_artifact_audit_builds_truthful_composition_and_revalidates_without_cuda(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("No GPU access")):
            result = self.build()
            validated = att.validate_attestation(self.root/"attestation.json", self.manifest)
        self.assertEqual(result["started_at"], self.cloud_receipt["started_at"])
        self.assertEqual(result["finished_at"], self.local_receipt["finished_at"])
        self.assertEqual(validated["backend"], "cloud_training_local_benchmark")
        self.assertEqual(validated["config_sha256"], hashlib.sha256(self.raw_config).hexdigest())
        self.assertIn(str(self.cloud/"input.private.zip"), result["evidence_sha256"])
        self.assertIn(str(self.cloud/"artifacts.private.zip"), result["evidence_sha256"])
        with self.assertRaises(FileExistsError): self.build()

    def test_only_exact_shadow_wrn34_300_mixup_cutout_and_predeclared_seed_are_supported(self):
        for change in ({"epochs": 160}, {"seed": 2026}, {"seed": 2028}, {"depth": 22}, {"train_all": True},
                       {"cutmix_alpha": 0., "cutmix_probability": 0.}, {"mixup_alpha": 0.}, {"cutout": 0},
                       {"resizemix_probability": .5}, {"sam_rho": .05}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError): att.validate_recipe({**self.config, **change}, 2027)
        for trainer in ("resizemix", "sam", "frozen"):
            self.write(self.cloud/"supervisor.json", {**self.cloud_receipt, "trainer": trainer})
            with self.assertRaisesRegex(ValueError, "single-allocation cloud repeat"):
                self.build()

    def test_original_rz_schema_other_primary_and_modified_generic_validator_are_rejected(self):
        path = self.root/"old_attestation.json"
        self.write(path, {"schema": "research_cloud_repeat_attestation_v1", "status": "complete"})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            att.validate_attestation(path)
        original = self.manifest.read_bytes()
        self.manifest.write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError, "explicitly frozen"):
            self.build()
        self.manifest.write_bytes(original)
        helper = self.root/"research/repeat_attestation.py"
        helper.write_bytes(helper.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "Generic ZIP/cleanup validator source"):
            self.build()

    def test_last_state_requires_real_rng_complete_scaler_and_finite_sgd_momentum(self):
        original = self.last.copy()
        for key, value in (("numpy_rng", None), ("cuda_rng", []), ("grad_scaler", {"scale": 256.})):
            with self.subTest(key=key):
                self.last[key] = value
                torch.save(self.last, self.folder/"last.pt")
                with self.assertRaises(ValueError):
                    att.validate_last(self.folder, self.checkpoint, self.metrics, self.history)
                self.last = original.copy()
        optimizer = self.last["optimizer"]
        first = next(iter(optimizer["state"]))
        momentum = optimizer["state"][first]["momentum_buffer"]
        self.last["optimizer"] = {**optimizer, "state": {**optimizer["state"],
            first: {"momentum_buffer": torch.full_like(momentum, float("inf"))}}}
        torch.save(self.last, self.folder/"last.pt")
        with self.assertRaisesRegex(ValueError, "momentum is invalid"):
            att.validate_last(self.folder, self.checkpoint, self.metrics, self.history)

    def test_partial_history_or_excessive_fp16_overflow_fails_real_shadow_validator(self):
        with self.assertRaisesRegex(ValueError, "every completed"):
            validate_shadow(self.checkpoint, self.metrics, self.history[:-1], self.provenance, self.environment)
        history = [{**row, "successful_updates": 309, "skipped_updates": 4} for row in self.history]
        metrics = {**self.metrics, "successful_updates": 309*300, "skipped_updates": 4*300}
        with self.assertRaisesRegex(ValueError, "Too many overflow"):
            validate_shadow(self.checkpoint, metrics, history, self.provenance, self.environment)

    def test_packaged_current_source_and_complete_policy_coverage_are_required(self):
        document = att.read(self.benchmark)
        document["policies"].pop("plain")
        document["promotion"].pop("plain")
        integrity = document.pop("research_training_integrity")
        core = self.benchmark.with_name("benchmark.core.json")
        self.write(core, document)
        document["research_training_integrity"] = {**integrity, "core_benchmark_sha256": att.sha256(core)}
        self.write(self.benchmark, document)
        self.local_receipt["artifact_sha256"] = att.sha256(self.benchmark)
        self.write(self.receipt_path, self.local_receipt)
        with self.assertRaisesRegex(ValueError, "every fixed policy"):
            self.build()
        source = self.root/"research/train_shadow.py"
        source.write_bytes(source.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "packaged source"):
            self.build()

    def test_cloud_failure_or_unknown_cleanup_never_creates_complete_attestation(self):
        for key, value in (("status", "failed"), ("runtime_stop_verified", False), ("assignment_attempts", 2), ("no_new_purchases", False)):
            with self.subTest(key=key):
                self.write(self.cloud/"supervisor.json", {**self.cloud_receipt, key: value})
                with self.assertRaises(ValueError): self.build()
                self.assertFalse((self.root/"attestation.json").exists())

    def test_reconciled_cleanup_preserves_original_receipt_and_both_read_only_proofs(self):
        original_path = self.cloud/"supervisor_before_reconciliation.json"
        self.write(original_path, {**self.cloud_receipt, "status": "cleanup_unverified",
            "runtime_stop_verified": False, "usage_after": {"active_assignments": 1}})
        absence_path, usage_path = self.cloud/"server_absence.json", self.cloud/"usage_absence.json"
        self.write(absence_path, {"status": "read_only_checked", "server_assignment_count": 0,
            "own_endpoint_identified": True, "own_endpoint_present": False, "server_side_runtime_absent_verified": True,
            "official_termination_history_present": True, "no_unassign_performed": True,
            "controller_receipt_sha256": att.sha256(original_path), "checked_at": "2026-10-31T19:42:00+00:00"})
        self.write(usage_path, {"status": "read_only_checked", "usage": self.cloud_receipt["usage_after"],
            "prior_absence_proof": absence_path.name, "prior_absence_proof_sha256": att.sha256(absence_path),
            "no_stop_or_unassign_performed": True, "checked_at": "2026-10-31T19:43:00+00:00"})
        self.cloud_receipt.update(cleanup_verified_at="2026-10-31T19:43:00+00:00", cleanup_reconciliation={
            "server_absence_proof": absence_path.name, "server_absence_sha256": att.sha256(absence_path),
            "usage_proof": usage_path.name, "usage_sha256": att.sha256(usage_path),
            "preserved_original_receipt": original_path.name, "original_receipt_sha256": att.sha256(original_path),
            "no_second_stop_or_unassign": True})
        self.write(self.cloud/"supervisor.json", self.cloud_receipt)
        result = self.build()
        for path in (original_path, absence_path, usage_path): self.assertIn(str(path), result["evidence_sha256"])
        original_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            att.validate_attestation(self.root/"attestation.json")

    def test_primary_freeze_must_precede_real_cloud_start_and_cloud_must_finish_before_local_start(self):
        for frozen_time, cloud_finish, local_start in ((self.cloud_receipt["started_at"], self.cloud_receipt["finished_at"], self.local_receipt["started_at"]),
                (self.frozen["frozen_at"], "2026-10-31T20:00:01+00:00", self.local_receipt["started_at"])):
            self.frozen["frozen_at"] = frozen_time
            self.write(self.cloud/"supervisor.json", {**self.cloud_receipt, "finished_at": cloud_finish})
            self.write(self.receipt_path, {**self.local_receipt, "started_at": local_start})
            with self.assertRaisesRegex(ValueError, "times are out of order"): self.build()

    def test_original_raw_config_sha_is_required_even_when_merged_configuration_matches(self):
        self.frozen["primary_decision"]["planned_repeat_config_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "predeclared exact repeat"): self.build()

    def test_input_chunk_or_archive_member_mutation_is_rejected(self):
        chunk = self.cloud/"input_000.private.part"
        chunk.write_bytes(b"changed original chunk")
        with self.assertRaisesRegex(ValueError, "chunk changed"): self.build()

    def test_extracted_artifact_cannot_differ_from_preserved_zip(self):
        (self.cloud/"remote_receipt.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "differs from original ZIP"): self.build()

    def test_true_training_count_history_scaler_and_last_epoch_are_checked(self):
        self.last["epoch"] = 9
        torch.save(self.last, self.folder/"last.pt")
        self.reseal_cloud()
        with self.assertRaisesRegex(ValueError, "Final complete 300-epoch"): self.build()

    def test_false_full_data_count_and_zero_scaler_fail_actual_training_validator(self):
        self.write(self.folder/"metrics.json", {**self.metrics, "training_examples": 50000})
        with self.assertRaises(ValueError): self.build()
        self.write(self.folder/"metrics.json", self.metrics)
        changed = copy.deepcopy(self.history)
        changed[0]["grad_scaler_scale"] = 0.
        self.write(self.folder/"history.json", changed)
        with self.assertRaises(ValueError): self.build()

    def test_fixed_policy_gate_cannot_be_replaced_with_another_good_policy(self):
        document = att.read(self.benchmark)
        bad = {"cases": {name: {"examples": 5000, "input_pixel_sha256": "c"*64,
            "accuracy": .8 if name == "clean" else .6} for name in att.delivery.suite.CASES},
            "clean_accuracy": .8, "stress_accuracy": .6}
        document["policies"]["flip"] = copy.deepcopy(document["policies"]["center80_shift20"])
        document["policies"]["center80_shift20"] = bad
        document["promotion"] = {key: att.delivery.suite.promotion_gate(value, self.baseline["policies"]["flip"])
            for key, value in document["policies"].items()}
        document.pop("research_training_integrity")
        core = self.benchmark.with_name("benchmark.core.json")
        self.write(core, document)
        integrity = att.read(self.folder/"benchmark.json")["research_training_integrity"]
        document["research_training_integrity"] = {**integrity, "core_benchmark_sha256": att.sha256(core)}
        self.write(self.benchmark, document)
        self.local_receipt["artifact_sha256"] = att.sha256(self.benchmark)
        self.write(self.receipt_path, self.local_receipt)
        with self.assertRaisesRegex(ValueError, "did not actually pass"): self.build()

    def test_method_or_fp16_source_cannot_change(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["provenance"]["research_training"]["precision"] = "bf16"
        torch.save(checkpoint, self.checkpoint_path)
        with self.assertRaises(ValueError): self.build()

    def test_local_receipt_requires_actual_artifact_and_all_three_inputs(self):
        for change in ({"artifact_sha256": "e"*64}, {"input_sha256": {}}, {"mode": "resizemix-benchmark"}):
            with self.subTest(change=change):
                self.write(self.receipt_path, {**self.local_receipt, **change})
                with self.assertRaisesRegex(ValueError, "Local diagnostic receipt"): self.build()

    def test_composition_fields_and_inputs_are_rederived_before_prediction(self):
        self.build()
        path = self.root/"attestation.json"
        document = att.read(path)
        self.write(path, {**document, "training_examples": 50000})
        with patch.object(att.delivery, "predict") as predict:
            with self.assertRaisesRegex(ValueError, "rederived original evidence"):
                att.predict(self.manifest, path, self.root/"predictions", "cpu")
            predict.assert_not_called()
        self.write(path, document)
        (self.cloud/"input_000.private.part").write_bytes(b"changed later")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"): att.validate_attestation(path)

    def test_prediction_adapter_uses_attested_paths_and_never_renames_cloud_training_local(self):
        result = self.build()
        with patch.object(att.delivery, "predict", return_value={"validated": True}) as predict:
            att.predict(self.manifest, self.root/"attestation.json", self.root/"predictions", "cpu")
        predict.assert_called_once_with(self.manifest, result["checkpoint_path"], result["benchmark_path"],
            self.root/"attestation.json", self.root/"predictions", "cpu")
        self.assertEqual(att.read(self.cloud/"supervisor.json")["started_at"], self.cloud_receipt["started_at"])


if __name__ == "__main__": unittest.main()
