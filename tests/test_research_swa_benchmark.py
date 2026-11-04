"""CPU audits for fixed SWA selection, real tensors and unchanged v3 scoring."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import runtime
import numpy as np
import torch
from model import build_model, count_parameters
from train import DEFAULT_CONFIG, code_provenance
from research import benchmark_swa as audit
from research import train_swa


class SWABenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {**DEFAULT_CONFIG, "depth": 34, "widen_factor": 3, "epochs": 200,
            "swa_prefix_epochs": 160, "swa_tail_epochs": 40, "swa_lr": .01, "swa_bn_seed": 2026}
        cls.model_config = {key: cls.config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
        cls.model = build_model(cls.model_config)
        cls.count = count_parameters(cls.model)
        cls.state = cls.model.state_dict()
        cls.parameter_names = list(dict(cls.model.named_parameters()))
        cls.momenta = {index: {"momentum_buffer": torch.zeros_like(value)}
            for index, value in enumerate(cls.model.parameters())}

    def setUp(self):
        marker = train_swa.research_provenance("bf16", self.config)
        self.provenance_code = {**code_provenance(), "research_training": marker}
        self.parent = {"checkpoint_path": "/own/prefix/last.pt", "checkpoint_sha256": "a"*64, "epoch": 160}
        self.signature = {"archive_sha256": "b"*64, "split_sha256": "c"*64}
        self.history = {"prefix": [{"epoch": e} for e in range(1, 161)], "tail": [
            {"epoch": 160+e, "lr": .01, "successful_updates": 313, "skipped_updates": 0, "swa_count": e,
             "train_loss": .1, "train_accuracy": .95, "validation_accuracy": None, "validation_loss": None,
             "ema_validation_accuracy": None, "ema_validation_loss": None} for e in range(1, 41)]}
        self.bn = {"schema": "research_swa_bn_recompute_v1", "training_examples": 40000, "batch_size": 128,
            "batches": 313, "last_batch_examples": 64, "seed": 2026, "precision": "bf16", "device_type": "cuda",
            "shuffle": False, "augmentation": "frozen_crop_flip_cutout0", "mixup": False, "cutout": 0,
            "momentum": "sample_weighted_batch_size_over_seen", "grad_enabled": False, "parameter_count": self.count,
            "parameters_unchanged": True, "training_modes_restored": True, "momenta_restored": True, "rng_restored": True,
            "augmented_input_sha256": "d"*64, "helper_source_sha256": marker["source_sha256"]["research/swa.py"]}
        self.final = {"loss": .2, "accuracy": .96, "examples": 5000}
        self.counts = {"swa_count": 40, "prefix_updates": 50080, "tail_updates": 12520,
            "successful_updates": 62600, "skipped_updates": 0}
        self.checkpoint = {"schema": "research_swa_checkpoint_v1", "model": self.state, "model_config": self.model_config,
            "config": self.config, "epoch": 200, "variant": "swa", "parameter_count": self.count,
            "data_signature": self.signature, "provenance": self.provenance_code, "parent_training": self.parent,
            **self.counts, "bn_recomputation": self.bn, "final_validation": self.final,
            "selection": "fixed_swa_final", "no_best_selection": True}
        self.metrics = {"status": "complete", "config": self.config, "selection": "fixed_swa_final", "no_best_selection": True,
            "epochs_completed": 200, "prefix_epochs": 160, "tail_epochs": 40, "training_examples": 40000,
            "validation_examples": 5000, "parameter_count": self.count, "research_training": marker,
            "parent_training": self.parent, **self.counts, "final_validation": {v: self.final for v in ("swa", "raw_tail_control")},
            "bn_recomputation": {v: self.bn for v in ("swa", "raw_tail_control")}, "ema_tracked": True, "ema_selected": False,
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False}
        self.provenance = {"code": self.provenance_code, "data_signature": self.signature, "train_examples": 40000,
            "validation_examples": 5000, "parameter_count": self.count, "parent_training": self.parent}
        self.environment = {"device": "cuda", "swa_training_precision": "bf16", "grad_scaler_enabled": False,
            "research_training": marker}
        self.last = {"schema": "research_swa_last_v1", "epoch": 200, "model": self.state, "swa_model": self.state,
            "ema": self.state,
            "config": self.config, "data_signature": self.signature, "provenance": self.provenance_code,
            "parent_training": self.parent, "history": self.history, "updates": 62600, "tail_updates": 12520, "swa_count": 40,
            "python_rng": random.Random(2026).getstate(), "numpy_rng": np.random.RandomState(2026).get_state(),
            "torch_rng": torch.Generator().manual_seed(2026).get_state(), "cuda_rng": [torch.ones(16, dtype=torch.uint8)],
            "optimizer": {"param_groups": [{"lr": .01, "momentum": .9, "weight_decay": .0005,
                "nesterov": True, "dampening": 0,
                "params": list(self.momenta)}], "state": self.momenta}}

    def validate(self):
        return audit.validate_swa_training(self.checkpoint, self.metrics, self.history, self.provenance,
            self.environment, self.last, self.parent, self.history["prefix"])

    def test_fixed_epoch_swa_validates_real_wrn_tensors_without_cuda_or_legacy_best(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            result = self.validate()
        self.assertEqual(self.count, 4166298)
        self.assertEqual((result["epoch"], result["swa_count"], result["successful_updates"]), (200, 40, 62600))
        self.assertEqual(result["selection"], "fixed_swa_final")

    def test_raw_tail_is_explicit_control_and_cannot_change_fixed_primary_selection(self):
        self.checkpoint.update(variant="raw_tail_control", selection="fixed_raw_tail_control")
        result = self.validate()
        self.assertFalse(result["raw_control_prediction_authorized"])
        self.metrics["selection"] = "fixed_raw_tail_control"
        with self.assertRaisesRegex(ValueError, "fixed epoch-200"):
            self.validate()

    def test_eligibility_requires_both_predeclared_weighted_gates_and_never_control(self):
        passing = {policy: {"passed": True} for policy in audit.suite.POLICIES}
        inherited = {**passing, audit.FIXED_POLICY: {"passed": False}}
        self.assertTrue(audit.candidate_eligible("swa", passing, passing))
        self.assertFalse(audit.candidate_eligible("swa", passing, inherited))
        self.assertFalse(audit.candidate_eligible("swa", inherited, passing))
        self.assertFalse(audit.candidate_eligible("raw_tail_control", passing, passing))

    def test_partial_wrong_average_or_legacy_best_is_rejected(self):
        for change in ({"epoch": 199}, {"swa_count": 39}, {"variant": "ema"}, {"validation_accuracy": .99}):
            with self.subTest(change=change):
                previous = self.checkpoint.copy()
                self.checkpoint.update(change)
                with self.assertRaises(ValueError): self.validate()
                self.checkpoint = previous

    def test_tail_never_uses_development_selection_or_changes_lr(self):
        for key, value in (("validation_accuracy", .99), ("lr", .02), ("successful_updates", 312), ("swa_count", 2)):
            with self.subTest(key=key):
                original = self.history["tail"][0].copy()
                self.history["tail"][0][key] = value
                with self.assertRaisesRegex(ValueError, "tail update"):
                    self.validate()
                self.history["tail"][0] = original

    def test_bn_recomputation_cannot_include_dev_or_change_modes_rng_or_pixels(self):
        for key, value in (("training_examples", 45000), ("rng_restored", False), ("mixup", True)):
            original = self.bn.copy()
            self.bn[key] = value
            with self.assertRaisesRegex(ValueError, "BN proof"):
                self.validate()
            self.bn.clear(); self.bn.update(original)
        self.metrics["bn_recomputation"] = {"swa": self.bn, "raw_tail_control": {**self.bn, "augmented_input_sha256": "e"*64}}
        with self.assertRaisesRegex(ValueError, "shared BN"):
            self.validate()

    def test_parent_source_precision_and_full_training_counts_are_bound(self):
        self.provenance["parent_training"] = {**self.parent, "epoch": 159}
        with self.assertRaisesRegex(ValueError, "parent prefix"):
            self.validate()
        self.provenance["parent_training"] = self.parent
        self.metrics["training_examples"] = 50000
        with self.assertRaisesRegex(ValueError, "counts"):
            self.validate()

    def test_final_state_requires_rng_momentum_and_exact_averaged_parameters(self):
        del self.last["numpy_rng"]
        with self.assertRaisesRegex(ValueError, "RNG"):
            self.validate()
        self.last["numpy_rng"] = np.random.RandomState(2026).get_state()
        self.last["optimizer"] = {**self.last["optimizer"], "state": {}}
        with self.assertRaisesRegex(ValueError, "momentum"):
            self.validate()
        self.last["optimizer"]["state"] = self.momenta
        key = self.parameter_names[0]
        self.checkpoint["model"] = {**self.state, key: self.state[key]+1}
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.validate()

    def test_invalid_artifact_is_rejected_before_official_array_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"benchmark.json"
            args = ["benchmark_swa.py", "--checkpoint", "invalid.pt", "--baseline", "baseline.json", "--output", str(output), "--device", "cpu"]
            with patch.object(audit.sys, "argv", args), patch.object(audit, "validate_run", side_effect=ValueError("invalid source")), \
                    patch.object(audit, "load_training_arrays") as loader:
                with self.assertRaisesRegex(ValueError, "invalid source"):
                    audit.main()
            loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_driver_calls_unchanged_v3_for_all_policies_and_records_both_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint_path, output = folder/"swa.pt", folder/"benchmark.json"
            checkpoint_path.write_bytes(b"audited checkpoint fixture")
            for name in ("metrics.json", "history.json"):
                (folder/name).write_text("{}")
            (folder/"training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "c"*64}}))
            def report(clean, stress):
                cases = {case: {"accuracy": clean if case == "clean" else stress, "examples": 5000,
                    "loss": .2, "input_pixel_sha256": "f"*64} for case in audit.suite.CASES}
                return {"cases": cases, "clean_accuracy": clean, "clean_loss": .2, "stress_accuracy": stress}
            context = {"device_type": "cpu", "amp_dtype": "torch.float32", "torch_version": torch.__version__,
                "training_core_sha256": code_provenance()["source_sha256"]}
            baseline = {"schema": audit.suite.SCHEMA, "status": "complete", "data_signature": self.signature,
                "examples": 5000, "stress_cases": list(audit.suite.CASES), "bn_recalibrated": False,
                "checkpoint_sha256": audit.suite.BASELINE_CHECKPOINT_SHA, "batch_size": 128, "evaluation_context": context,
                "benchmark_source_sha256": audit.suite.sha256(Path(audit.suite.__file__)), "policies": {"flip": report(.9, .8)}}
            baseline_path = folder/"baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            parent = {"policies": {policy: report(.95, .84) for policy in audit.suite.POLICIES}}
            args = ["benchmark_swa.py", "--checkpoint", str(checkpoint_path), "--baseline", str(baseline_path),
                "--output", str(output), "--data-dir", str(folder), "--device", "cpu"]
            arrays = (np.zeros((50000, 1), dtype=np.uint8), np.zeros(50000, dtype=np.int64),
                np.arange(40000), np.arange(40000, 45000))
            with patch.object(audit.sys, "argv", args), patch.object(audit, "validate_run", return_value=(self.checkpoint, self.validate(), {})), \
                patch.object(audit, "load_training_arrays", return_value=arrays), patch.object(audit, "seed_all"), \
                patch.object(audit, "training_source", return_value={"archive_sha256": "b"*64}), \
                patch.object(audit, "validate_parent_benchmark", return_value=(parent, {})), \
                patch.object(audit, "build_model", return_value=self.model), \
                patch.object(audit.suite, "score", side_effect=lambda *args: report(.96, .85)) as score:
                audit.main()
            result = json.loads(output.read_text())
            self.assertEqual([call.args[4] for call in score.call_args_list], list(audit.suite.POLICIES))
            self.assertEqual(result["benchmark_source_sha256"], audit.suite.sha256(Path(audit.suite.__file__)))
            self.assertEqual(result["swa_benchmark_source_sha256"], audit.suite.sha256(Path(audit.__file__)))
            self.assertTrue(result["candidate_eligible"])
            self.assertTrue(result["promotion"][audit.FIXED_POLICY]["passed"])
            self.assertTrue(result["promotion_vs_parent"][audit.FIXED_POLICY]["passed"])
            self.assertTrue(result["bn_recalibrated"])


if __name__ == "__main__": unittest.main()
