"""Full-refit delivery boundaries with synthetic CPU files and no hidden inputs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import runtime
import torch
from train import DEFAULT_CONFIG
from research import refit_delivery as recipe


class RefitDeliveryTests(unittest.TestCase):
    def fixture(self, root):
        job = root/"job"
        job.mkdir()
        (root/"research").mkdir()
        (root/"research/run_refit.py").write_text("# audited supervisor fixture")
        (root/"train.py").write_text("# frozen core fixture")
        parent_checkpoint, parent_path, frozen_path = root/"parent.pt", root/"parent_delivery.json", root/"frozen.json"
        torch.save({"provenance": {}}, parent_checkpoint)
        config = {**DEFAULT_CONFIG, "depth": 34, "widen_factor": 3}
        frozen = {"config": config, "weight_variant": "model", "policy": "center80_shift20", "parameter_count": 4166298,
                  "data_signature": {"archive": "official"}, "checkpoint_path": str(parent_checkpoint), "data_dir": str(root),
                  "model_config": {key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")},
                  "evaluation_context": {"device_type": "cuda"}, "benchmark_path": str(root/"benchmark.json")}
        frozen_path.write_text(json.dumps(frozen))
        parent_path.write_text(json.dumps({"schema": recipe.primary.PREDICTION_SCHEMA, "frozen_manifest_path": str(frozen_path)}))
        parent = {"manifest_sha256": recipe.suite.sha256(parent_path), "frozen_manifest_sha256": recipe.suite.sha256(frozen_path),
                  "policy": frozen["policy"], "training_examples": 40000, "parameter_count": 4166298}
        for name in ("best.pt", "last.pt", "input_config.json", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json", "supervisor.json"):
            (job/name).write_text("{}")
        plan = job/"refit_plan.json"
        plan.write_text("{}")
        info = {"run": "wrn34_refit_s2026", "config": {**config, "train_all": True, "final_variant": "model"},
                "policy": frozen["policy"], "parent_manifest_path": str(parent_path), "parent_manifest_sha256": parent["manifest_sha256"],
                "checkpoint_path": str(job/"best.pt"), "checkpoint_sha256": recipe.suite.sha256(job/"best.pt"),
                "parameter_count": 4166298, "data_signature": frozen["data_signature"], "source_sha256": {"train.py": "frozen"},
                "plan_path": str(plan), "plan_sha256": recipe.suite.sha256(plan), "training_examples": 50000, "validation_examples": 0}
        runner = ModuleType("research.run_refit")
        runner.validate_refit = Mock(side_effect=lambda path: copy.deepcopy(info))
        runner.derive_refit_config = lambda value: {**value["config"], "train_all": True, "final_variant": value["weight_variant"]}
        return plan, info, parent, frozen, runner

    def guards(self, root, parent, runner):
        return (patch.dict(sys.modules, {"research.run_refit": runner}), patch.object(recipe, "ROOT", root),
                patch.object(recipe.primary, "validate_delivery", return_value=parent),
                patch.object(recipe, "code_provenance", return_value={"source_sha256": {"train.py": "frozen"}}))

    def test_complete_refit_requires_real_parent_chain_and_fixed_derived_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, info, parent, _, runner = self.fixture(root)
            a, b, c, d = self.guards(root, parent, runner)
            with a, b, c as qualified, d:
                checked, _, _, paths = recipe.validate_refit_run(plan)
            runner.validate_refit.assert_called_once_with(plan)
            qualified.assert_called_once_with(Path(info["parent_manifest_path"]), require_prediction=True)
            self.assertEqual((checked["training_examples"], checked["validation_examples"]), (50000, 0))
            self.assertIn(plan.parent/"history.json", paths)
            self.assertIn(plan.parent/"last.pt", paths)

    def test_wrong_counts_overlap_recipe_epochs_finalvariant_or_parent_stop_before_hidden_loader(self):
        changes = ({"training_examples": 40000}, {"validation_examples": 5000}, {"parent_manifest_sha256": "changed"},
                   {"config": {"train_all": True}}, {"policy": "flip"}, {"parameter_count": 5000001},
                   {"source_sha256": {"train.py": "changed"}})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, info, parent, _, runner = self.fixture(root)
                runner.validate_refit.side_effect = None
                runner.validate_refit.return_value = {**info, **change}
                a, b, c, d = self.guards(root, parent, runner)
                with a, b, c, d, patch.object(recipe, "load_kaggle_test", side_effect=AssertionError("no hidden inputs")) as hidden:
                    with self.assertRaises(ValueError): recipe.predict(plan, root/"predictions", "cpu")
                    hidden.assert_not_called()
        for change in ({"epochs": 159}, {"final_variant": "ema"}, {"train_all": False}, {"seed": 2027}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, info, parent, _, runner = self.fixture(root)
                runner.validate_refit.side_effect = None
                runner.validate_refit.return_value = {**info, "config": {**info["config"], **change}}
                a, b, c, d = self.guards(root, parent, runner)
                with a, b, c, d, patch.object(recipe, "load_kaggle_test", side_effect=AssertionError("no hidden inputs")) as hidden:
                    with self.assertRaises(ValueError): recipe.predict(plan, root/"predictions", "cpu")
                    hidden.assert_not_called()

    def test_missing_independent_repeat_or_unsupported_parent_never_loads_hidden_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _, parent, frozen, runner = self.fixture(root)
            a, b, c, d = self.guards(root, parent, runner)
            with a, b, c as qualified, d, patch.object(recipe, "load_kaggle_test") as hidden:
                qualified.side_effect = ValueError("independent repeat incomplete")
                with self.assertRaisesRegex(ValueError, "repeat"): recipe.predict(plan, root/"predictions", "cpu")
                hidden.assert_not_called()
            for schema in ("research_shadow_training_v1", "research_sam_training_v1", "research_resizemix_training_v1"):
                torch.save({"provenance": {"research_training": {"schema": schema}}}, frozen["checkpoint_path"])
                a, b, c, d = self.guards(root, parent, runner)
                with a, b, c, d, patch.object(recipe, "load_kaggle_test") as hidden:
                    with self.assertRaisesRegex(ValueError, "unsupported"): recipe.predict(plan, root/"predictions", "cpu")
                    hidden.assert_not_called()

    def test_prediction_backend_is_checked_before_hidden_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _, parent, _, runner = self.fixture(root)
            a, b, c, d = self.guards(root, parent, runner)
            with a, b, c, d, patch.object(recipe, "load_kaggle_test") as hidden:
                with self.assertRaisesRegex(ValueError, "backend/precision"):
                    recipe.predict(plan, root/"predictions", "cpu")
                hidden.assert_not_called()

    def test_refit_manifest_cannot_claim_independent_development_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, info, parent, frozen, _ = self.fixture(root)
            manifest = root/"refit_delivery.json"
            document = {"schema": recipe.SCHEMA, "status": "complete", "refit_plan_path": str(plan),
                "run": info["run"], "competition": recipe.COMPETITION, "policy": parent["policy"], "seed": 2026,
                "training_examples": 50000, "validation_examples": 0, "development_accuracy": None,
                "independent_development_evaluated": False, "checkpoint_sha256": info["checkpoint_sha256"],
                "parameter_count": info["parameter_count"], "refit_plan_sha256": info["plan_sha256"],
                "config": info["config"], "model_config": frozen["model_config"], "data_signature": info["data_signature"],
                "training_source_sha256": info["source_sha256"], "weight_variant": "model", "epochs_completed": 160,
                "checkpoint_epoch": 160, "initialization": "scratch_seeded", "parent_weights_loaded": False,
                "parent_delivery_path": str(Path(info["parent_manifest_path"]).resolve()),
                "parent_delivery_sha256": parent["manifest_sha256"], "parent_frozen_manifest_sha256": parent["frozen_manifest_sha256"],
                "evaluation_context": frozen["evaluation_context"], "generator_source_sha256": recipe.suite.sha256(Path(recipe.__file__)),
                "confirmation_evaluated": False, "hidden_test_labels_loaded": False, "external_training_data_loaded": False,
                "refit_evidence_sha256": {"sealed": "snapshot"}, "test_files_sha256": {"sealed": "snapshot"},
                "test_manifest_path": str(root/"test_manifest.json"), "csv_path": str(root/"submission.csv"),
                "submission_sha256": "f"*64, "rows": 2000, "columns": ["id", "label"], "ids_match_official_order": True}
            manifest.write_text(json.dumps(document))
            with patch.object(recipe, "validate_refit_run", return_value=(info, parent, frozen, [])), \
                    patch.object(recipe.primary, "validate_test_files", return_value=[]) as test_files, \
                    patch.object(recipe.primary, "verify_snapshots"), \
                    patch.object(recipe.primary, "snapshots", return_value={"sealed": "snapshot"}), \
                    patch.object(recipe, "validate_test_manifest"), patch.object(recipe.primary, "validate_csv", return_value="f"*64):
                result = recipe.validate_refit_delivery(manifest)
                self.assertEqual((result["training_examples"], result["validation_examples"]), (50000, 0))
                self.assertIsNone(result["development_accuracy"])
                for change in ({"development_accuracy": .99}, {"independent_development_evaluated": True},
                               {"weight_variant": "ema"}, {"checkpoint_epoch": 159}, {"training_examples": 40000}):
                    manifest.write_text(json.dumps({**document, **change}))
                    test_files.reset_mock()
                    with self.assertRaisesRegex(ValueError, "no-development"):
                        recipe.validate_refit_delivery(manifest)
                    test_files.assert_not_called()


if __name__ == "__main__": unittest.main()
