"""Research eligibility and primary-only prediction boundaries; synthetic CPU data."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from research import delivery
from train import DEFAULT_CONFIG


class ResearchDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/"data"
        self.data.mkdir()
        self.primary = self.root/"primary/best.pt"
        self.primary.parent.mkdir()
        self.primary.write_bytes(b"primary checkpoint")
        self.benchmark_path = self.primary.parent/"benchmark.json"
        self.baseline_path = self.root/"baseline.json"
        self.paired_path = self.root/"paired.json"
        self.source = self.root/"source.py"
        self.source.write_text("fixed own source")
        self.context = {"device_type": "cpu", "amp_dtype": "torch.float32", "torch_version": torch.__version__,
                        "training_core_sha256": {"train.py": "frozen"}}
        self.signature = {"archive_sha256": "official", "split_sha256": "fixed"}
        self.config = {**DEFAULT_CONFIG, "seed": 2026}
        self.checkpoint = {"config": self.config, "provenance": {}, "model_config": {"tiny": True}}
        def report(clean, stress):
            cases = {name: {"accuracy": clean if name == "clean" else stress,
                            "examples": 5000, "input_pixel_sha256": "a"*64} for name in delivery.suite.CASES}
            return {"clean_accuracy": clean, "stress_accuracy": stress, "cases": cases}
        self.baseline = {"schema": delivery.suite.SCHEMA, "status": "complete", "examples": 5000,
            "data_signature": self.signature, "stress_cases": list(delivery.suite.CASES), "bn_recalibrated": False,
            "checkpoint_sha256": delivery.suite.BASELINE_CHECKPOINT_SHA, "batch_size": 128,
            "evaluation_context": self.context, "benchmark_source_sha256": delivery.suite.sha256(Path(delivery.suite.__file__)),
            "policies": {"flip": report(.9628, .77628)}}
        self.benchmark = {"schema": delivery.suite.SCHEMA, "status": "complete", "examples": 5000,
            "data_signature": self.signature, "evaluation_context": self.context, "batch_size": 128,
            "parameter_count": 4166298, "policies": {"center80_shift20": report(.9632, .78592)}}
        self.benchmark["promotion"] = {"center80_shift20": delivery.suite.promotion_gate(
            self.benchmark["policies"]["center80_shift20"], self.baseline["policies"]["flip"])}
        self.baseline_path.write_text(json.dumps(self.baseline))
        self.benchmark_path.write_text(json.dumps(self.benchmark))
        self.benchmark["baseline_sha256"] = delivery.suite.sha256(self.baseline_path)
        self.pair = {"schema": "exploratory_paired_image_bootstrap_v1", "status": "complete",
            "candidate_policy": "center80_shift20", "baseline_policy": "flip", "data_signature": self.signature,
            "evaluation_context": self.context, "candidate_checkpoint_sha256": delivery.suite.sha256(self.primary),
            "baseline_checkpoint_sha256": delivery.suite.BASELINE_CHECKPOINT_SHA,
            "candidate_benchmark_sha256": delivery.suite.sha256(self.benchmark_path),
            "baseline_benchmark_sha256": delivery.suite.sha256(self.baseline_path),
            "source_sha256": delivery.suite.sha256(Path(delivery.paired.__file__)),
            "benchmark_source_sha256": delivery.suite.sha256(Path(delivery.suite.__file__)),
            "confirmation_evaluated": False, "hidden_images_loaded": False, "external_data_loaded": False,
            "automatic_promotion": False, "bootstrap": {"original_images": 5000,
            "replicates": delivery.paired.REPLICATES, "seed": delivery.paired.SEED, "confidence_level": .95,
            "sampling_unit": "original_image_with_all_five_stress_views",
            "clean": {"delta_accuracy": .0004, "percentile_ci95": [-.004, .0048],
                      "candidate_only_correct": 65, "baseline_only_correct": 63, "both_correct": 4751, "both_wrong": 121},
            "stress": {"delta_accuracy": .00964, "percentile_ci95": [.00568, .01348],
                       "candidate_only_correct": 1189, "baseline_only_correct": 948, "both_correct": 18459, "both_wrong": 4404}}}
        self.paired_path.write_text(json.dumps(self.pair))
        self.bundle = {"checkpoint": self.checkpoint, "benchmark": self.benchmark, "baseline": self.baseline,
                       "data": (self.signature, {"train.py": "frozen"}, None, None),
                       "paths": [self.primary, self.benchmark_path, self.baseline_path, self.paired_path, self.source]}

    def freeze_fixture(self):
        manifest = self.root/"frozen.json"
        with patch.object(delivery, "primary_bundle", return_value=self.bundle), \
             patch.object(delivery, "utc_now", return_value=datetime(2026, 10, 31, 16, 44, tzinfo=timezone.utc)):
            value = delivery.freeze(self.primary, self.benchmark_path, self.paired_path, "center80_shift20", manifest,
                                    self.data, self.baseline_path)
        return manifest, value

    def test_robust_promotion_accepts_clean_ci_crossing_zero_but_positive_stress_ci(self):
        delivery.validate_pair(self.pair, self.benchmark, self.baseline, self.primary,
                               self.benchmark_path, self.baseline_path, "center80_shift20")
        for lower in (0.0, -.001):
            altered = copy.deepcopy(self.pair)
            altered["bootstrap"]["stress"]["percentile_ci95"][0] = lower
            with self.assertRaisesRegex(ValueError, "positive paired stress"):
                delivery.validate_pair(altered, self.benchmark, self.baseline, self.primary,
                                       self.benchmark_path, self.baseline_path, "center80_shift20")

    def test_paired_source_policy_point_delta_and_counts_are_bound(self):
        mutations = [("source_sha256", "changed"), ("candidate_policy", "flip"),
                     ("candidate_checkpoint_sha256", "changed")]
        for key, value in mutations:
            altered = copy.deepcopy(self.pair)
            altered[key] = value
            with self.assertRaises(ValueError):
                delivery.validate_pair(altered, self.benchmark, self.baseline, self.primary,
                                       self.benchmark_path, self.baseline_path, "center80_shift20")
        altered = copy.deepcopy(self.pair)
        altered["bootstrap"]["stress"]["delta_accuracy"] = .02
        with self.assertRaisesRegex(ValueError, "point deltas"):
            delivery.validate_pair(altered, self.benchmark, self.baseline, self.primary,
                                   self.benchmark_path, self.baseline_path, "center80_shift20")

    def test_selected_policy_cannot_use_another_policy_or_partial_case_gate(self):
        with self.assertRaisesRegex(ValueError, "policy is missing"):
            delivery.passed_policy(self.benchmark, self.baseline, "flip")
        partial = copy.deepcopy(self.benchmark)
        partial["policies"]["center80_shift20"]["cases"]["clean"]["examples"] = 100
        with self.assertRaises(ValueError):
            delivery.passed_policy(partial, self.baseline, "center80_shift20")
        failed = copy.deepcopy(self.benchmark)
        failed["promotion"]["center80_shift20"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "did not actually pass"):
            delivery.passed_policy(failed, self.baseline, "center80_shift20")

    def test_freeze_without_repeat_is_immutable_and_cannot_authorize_prediction(self):
        manifest, value = self.freeze_fixture()
        self.assertFalse(value["repeat_prediction_authorized"])
        self.assertFalse(value["full_refit_supported"])
        with self.assertRaisesRegex(ValueError, "repeat and completed prediction"):
            delivery.validate_delivery(manifest)
        with patch.object(delivery, "primary_bundle", return_value=self.bundle):
            with self.assertRaisesRegex(ValueError, "already exists"):
                delivery.freeze(self.primary, self.benchmark_path, self.paired_path, "center80_shift20", manifest)

    def test_changed_frozen_source_is_rejected_without_loading_hidden_images(self):
        manifest, _ = self.freeze_fixture()
        self.source.write_text("changed own source")
        with patch.object(delivery, "load_kaggle_test") as hidden:
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                delivery.predict(manifest, "repeat.pt", "repeat.json", "supervisor.json", self.root/"prediction", "cpu")
        hidden.assert_not_called()

    def test_partial_or_ineligible_primary_never_reaches_hidden_loader(self):
        with patch.object(delivery, "validate_frozen", side_effect=ValueError("partial benchmark")), \
             patch.object(delivery, "load_kaggle_test") as hidden:
            with self.assertRaisesRegex(ValueError, "partial"):
                delivery.predict(self.root/"frozen.json", "repeat.pt", "repeat.json", "supervisor.json", self.root/"prediction")
        hidden.assert_not_called()

    def test_same_seed_or_different_recipe_repeat_is_rejected_before_hidden_loader(self):
        _, frozen = self.freeze_fixture()
        repeat = self.root/"repeat/best.pt"
        repeat.parent.mkdir()
        repeat.write_bytes(b"different weights")
        for config in (self.config, {**self.config, "seed": 2027, "weight_decay": .1}):
            checkpoint = {**self.checkpoint, "config": config}
            with patch.object(delivery, "validate_frozen", return_value=(frozen, self.bundle)), \
                 patch.object(delivery, "audit_experiment", return_value=(checkpoint, self.benchmark)), \
                 patch.object(delivery, "load_kaggle_test") as hidden:
                with self.assertRaisesRegex(ValueError, "except for its predeclared seed"):
                    delivery.predict(self.root/"frozen.json", repeat, self.root/"repeat.json", self.root/"receipt.json", self.root/"prediction")
            hidden.assert_not_called()

    def test_original_stability_checkpoint_is_excluded_even_if_renamed(self):
        with patch.object(delivery.suite, "sha256", return_value=delivery.paired.STABILITY_SHA), \
             patch.object(delivery.torch, "load") as checkpoint_load:
            with self.assertRaisesRegex(ValueError, "stability checkpoint"):
                delivery.audit_experiment(self.root/"renamed.pt", self.benchmark_path, self.bundle["data"], self.context)
        checkpoint_load.assert_not_called()

    def test_completed_independent_repeat_must_follow_precommitted_primary_selection(self):
        _, frozen = self.freeze_fixture()
        repeat = self.root/"repeat/best.pt"
        repeat.parent.mkdir()
        repeat.write_bytes(b"independent repeat")
        checkpoint = {**self.checkpoint, "config": {**self.config, "seed": 2027}}
        benchmark_path, receipt_path = repeat.parent/"benchmark.json", repeat.parent/"supervisor.json"
        benchmark_path.write_text(json.dumps(self.benchmark))
        receipt = {"status": "complete", "confirmation_evaluated": False, "submitted": False,
                   "benchmark_sha256": delivery.suite.sha256(benchmark_path),
                   "started_at": "2026-10-31T16:45:00+00:00", "finished_at": "2026-10-31T16:46:00+00:00"}
        receipt_path.write_text(json.dumps(receipt))
        with patch.object(delivery, "audit_experiment", return_value=(checkpoint, self.benchmark)), \
             patch.object(delivery, "utc_now", return_value=datetime(2026, 10, 31, 17, tzinfo=timezone.utc)):
            paths = delivery.validate_repeat(frozen, self.bundle, repeat, benchmark_path, receipt_path)
            self.assertIn(receipt_path, paths)
            receipt["started_at"] = "2026-10-31T16:40:00+00:00"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "after repeat outcomes"):
                delivery.validate_repeat(frozen, self.bundle, repeat, benchmark_path, receipt_path)

    def test_prediction_backend_must_match_benchmark_before_hidden_loader(self):
        manifest, frozen = self.freeze_fixture()
        frozen["evaluation_context"] = {**self.context, "amp_dtype": "torch.float16"}
        repeat = self.root/"repeat.pt"
        repeat.write_bytes(b"repeat")
        with patch.object(delivery, "validate_frozen", return_value=(frozen, self.bundle)), \
             patch.object(delivery, "validate_repeat", return_value=[repeat]), \
             patch.object(delivery, "load_kaggle_test") as hidden:
            with self.assertRaisesRegex(ValueError, "backend/precision"):
                delivery.predict(manifest, repeat, "repeat.json", "receipt.json", self.root/"prediction", "cpu")
        hidden.assert_not_called()

    def test_predict_uses_frozen_primary_weights_and_policy_not_repeat_weights(self):
        manifest, frozen = self.freeze_fixture()
        model = torch.nn.Linear(1, 10, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        self.bundle["checkpoint"]["model"] = copy.deepcopy(model.state_dict())
        sample = self.data/"sample_submission.csv"
        sample.write_text("id,label\n"+"".join(f"{i},0\n" for i in range(2000)))
        test_manifest = self.data/"kaggle_test_manifest.json"
        test_manifest.write_text(json.dumps({"competition": delivery.COMPETITION, "count": 2000, "labels_loaded": False}))
        repeat_file = self.root/"repeat.pt"
        repeat_file.write_bytes(b"repeat validation-only")
        seen = []
        def infer(actual_model, images, policy):
            seen.append((policy, bool(torch.equal(actual_model.weight, torch.zeros_like(actual_model.weight)))))
            result = torch.zeros(len(images), 10)
            result[:, 4] = 1
            return result
        def check_csv(path, sample_path):
            import csv
            with Path(path).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2000)
            self.assertEqual([r["id"] for r in rows], [str(i) for i in range(2000)])
            self.assertTrue(all(r["label"] == "4" for r in rows))
            return delivery.suite.sha256(path)
        with patch.object(delivery, "validate_frozen", return_value=(frozen, self.bundle)), \
             patch.object(delivery, "validate_repeat", return_value=[repeat_file]), \
             patch.object(delivery, "validate_test_files", return_value=[sample]), \
             patch.object(delivery, "code_provenance", return_value={"source_sha256": self.context["training_core_sha256"]}), \
             patch.object(delivery, "build_model", return_value=model), \
             patch.object(delivery.suite, "infer", side_effect=infer), \
             patch.object(delivery, "load_kaggle_test", return_value=([str(i) for i in range(2000)], np.zeros((2000, 32, 32, 3), dtype=np.uint8))), \
             patch.object(delivery, "validate_csv", side_effect=check_csv), \
             patch.object(delivery, "validate_delivery", return_value={"validated": True}):
            delivery.predict(manifest, repeat_file, "repeat.json", "receipt.json", self.root/"prediction", "cpu")
        self.assertTrue(seen and all(policy == "center80_shift20" and primary for policy, primary in seen))
        metadata = json.loads((self.root/"prediction/delivery.json").read_text())
        self.assertEqual(metadata["seed"], 2026)
        self.assertEqual(metadata["checkpoint_sha256"], delivery.suite.sha256(self.primary))


if __name__ == "__main__":
    unittest.main()
