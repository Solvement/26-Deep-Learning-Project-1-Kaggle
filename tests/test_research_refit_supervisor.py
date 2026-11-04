"""Scratch-only full refit derivation, no-dev evidence and shared budget controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil
import torch
from research import run_refit as refit
from train import DEFAULT_CONFIG


class RefitSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in refit.SOURCE_FILES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen test source " + name)
        (self.root/"research/campaign.json").write_text(json.dumps({
            "stops_at": "2099-01-01T00:00:00+00:00", "local_gpu_seconds_per_utc_day": 21600,
            "local_job_timeout_seconds": 10800}))
        self.parent_path = self.root/"parent_delivery.json"
        self.parent_path.write_text("{}")
        (self.root/"data").mkdir()
        self.core = {name: refit.sha256(self.root/name) for name in ("train.py", "model.py", "data.py", "runtime.py")}
        self.context = {"device_type": "cuda", "amp_dtype": "torch.bfloat16", "torch_version": torch.__version__,
                        "training_core_sha256": self.core}
        self.parent_config = {**DEFAULT_CONFIG, "depth": 34, "widen_factor": 3}
        self.frozen = {"config": self.parent_config, "weight_variant": "model", "data_dir": str(self.root/"data"),
                       "data_signature": {"archive_sha256": "official", "split_sha256": "fixed"},
                       "evaluation_context": self.context}
        self.summary = {"manifest_sha256": refit.sha256(self.parent_path), "frozen_manifest_sha256": "frozen-manifest",
                        "checkpoint_sha256": "primary-40k", "policy": "center80_shift20", "parameter_count": 10}
        self.parent = {"frozen_manifest_path": str(self.root/"frozen.json")}
        self.full_config = refit.derive_refit_config(self.frozen)

    def prepared(self):
        with patch.object(refit, "ROOT", self.root), \
             patch.object(refit, "load_parent", return_value=(self.summary, self.parent, self.frozen, self.full_config)), \
             patch.object(refit, "code_provenance", return_value={"source_sha256": self.core}):
            path = refit.prepare("full_refit", self.parent_path)
            plan = refit.validate_plan(path)
        return path, plan

    def records(self):
        _, plan = self.prepared()
        model = torch.nn.Linear(1, 10, bias=False)
        model_config = {key: self.full_config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "model_config": model_config,
            "config": self.full_config, "epoch": 160, "variant": "model", "validation_accuracy": None,
            "validation_loss": None, "parameter_count": 10, "data_signature": plan["data_signature"],
            "provenance": {"source_sha256": self.core}}
        history = [{"epoch": epoch, "selected_variant": "model", "train_loss": .5, "train_accuracy": .8,
                    "lr": .01, "elapsed_seconds": float(epoch), **{key: None for key in refit.VALIDATION_KEYS}}
                   for epoch in range(1, 161)]
        last = {"model": copy.deepcopy(model.state_dict()), "ema": copy.deepcopy(model.state_dict()),
                "epoch": 160, "updates": 391*160, "history": history, "config": self.full_config,
                "provenance": {"source_sha256": self.core}, "data_signature": plan["data_signature"],
                "best_accuracy": None, "best_loss": None}
        metrics = {"status": "complete", "epochs_completed": 160, "config": self.full_config,
                   "training_examples": 50000, "validation_examples": 0, "best_validation_accuracy": None,
                   "best_validation_loss": None, "parameter_count": 10}
        provenance = {"code": {"source_sha256": self.core}, "data_signature": plan["data_signature"],
                      "train_examples": 50000, "validation_examples": 0, "parameter_count": 10}
        environment = {"device": "cuda", "amp_dtype": "torch.bfloat16", "packages": {"torch": torch.__version__}}
        return plan, checkpoint, last, metrics, history, provenance, environment

    def test_derivation_changes_only_train_all_and_preselected_variant(self):
        before = copy.deepcopy(self.frozen)
        config = refit.derive_refit_config(self.frozen)
        self.assertEqual(self.frozen, before)
        self.assertEqual({key for key in config if config[key] != self.parent_config[key]}, {"train_all", "final_variant"})
        self.assertEqual(config["seed"], 2026)
        self.assertEqual(config["epochs"], 160)
        self.assertEqual(config["fixed_flip_tta"], self.parent_config["fixed_flip_tta"])
        self.assertTrue(config["train_all"])
        for addition in ({"sam_rho": .05}, {"cutmix_alpha": 1., "cutmix_probability": .5}, {"resizemix_tau_min": .1}):
            with self.assertRaisesRegex(ValueError, "core-compatible"):
                refit.derive_refit_config({**self.frozen, "config": {**self.parent_config, **addition}})
        with self.assertRaisesRegex(ValueError, "seed-2026"):
            refit.derive_refit_config({**self.frozen, "config": {**self.parent_config, "seed": 2027}})

    def test_refit_preserves_a_different_qualified_training_duration(self):
        frozen = {**self.frozen, "config": {**self.parent_config, "epochs": 300}}
        config = refit.derive_refit_config(frozen)
        self.assertEqual(config["epochs"], 300)
        self.assertEqual({key for key in config if config[key] != frozen["config"][key]}, {"train_all", "final_variant"})
        for invalid in (0, -1, True, 160.0):
            with self.assertRaises(ValueError):
                refit.derive_refit_config({**frozen, "config": {**frozen["config"], "epochs": invalid}})

    def test_prepare_creates_only_immutable_plan_and_separate_input_config(self):
        path, plan = self.prepared()
        self.assertEqual(path.name, "refit_plan.json")
        self.assertTrue((path.parent/"input_config.json").is_file())
        self.assertFalse((path.parent/"config.json").exists())
        self.assertFalse((path.parent/"supervisor.json").exists())
        self.assertFalse((path.parent.parent/"local_lease.json").exists())
        self.assertFalse(plan["parent_weights_loaded"])
        self.assertEqual(plan["policy"], "center80_shift20")

    def test_changed_source_or_parent_chain_invalidates_prepared_plan(self):
        path, _ = self.prepared()
        with patch.object(refit, "ROOT", self.root), \
             patch.object(refit, "load_parent", return_value=({**self.summary, "manifest_sha256": "changed"}, self.parent, self.frozen, self.full_config)):
            with self.assertRaisesRegex(ValueError, "parent chain"):
                refit.validate_plan(path)
        (self.root/"research/watchdog.py").write_text("changed source")
        with patch.object(refit, "ROOT", self.root), \
             patch.object(refit, "load_parent", return_value=(self.summary, self.parent, self.frozen, self.full_config)), \
             patch.object(refit, "code_provenance", return_value={"source_sha256": self.core}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                refit.validate_plan(path)

    def test_active_gpu_lease_blocks_launch_without_creating_a_supervisor(self):
        path, plan = self.prepared()
        process = psutil.Process()
        lease = path.parent.parent/"local_lease.json"
        lease.write_text(json.dumps({"pid": process.pid, "process_created_at": process.create_time()}))
        with patch.object(refit, "ROOT", self.root), patch.object(refit, "validate_plan", return_value=plan), \
             patch.object(refit, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                refit.launch(path)
        child.assert_not_called()
        self.assertFalse((path.parent/"supervisor.json").exists())
        self.assertTrue(lease.is_file())

    def test_insufficient_budget_fails_without_child_and_releases_owned_lease(self):
        path, plan = self.prepared()
        policy = json.loads((self.root/"research/campaign.json").read_text())
        policy["local_gpu_seconds_per_utc_day"] = 100
        (self.root/"research/campaign.json").write_text(json.dumps(policy))
        with patch.object(refit, "ROOT", self.root), patch.object(refit, "validate_plan", return_value=plan), \
             patch.object(refit, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "Insufficient local budget"):
                refit.launch(path, minimum_seconds=2400)
        child.assert_not_called()
        self.assertFalse((path.parent.parent/"local_lease.json").exists())
        self.assertEqual(json.loads((path.parent/"supervisor.json").read_text())["status"], "failed")

    def test_launch_is_scratch_core_training_without_resume_or_parent_weight_argument(self):
        path, plan = self.prepared()
        result = {"checkpoint_sha256": "new-full-refit"}
        def child(command, log, timeout, receipt_path, receipt):
            self.assertGreater(timeout, 0)
            for name in ("best.pt", "last.pt", "config.json", "metrics.json", "history.json", "provenance.json", "environment.json"):
                (path.parent/name).write_bytes(b"mock completed child artifact")
        with patch.object(refit, "ROOT", self.root), patch.object(refit, "validate_plan", return_value=plan), \
             patch.object(refit.torch.cuda, "is_available", return_value=True), \
             patch.object(refit.torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(refit, "supervised_command", side_effect=child) as launch, \
             patch.object(refit, "audit_artifacts", return_value=result):
            receipt = refit.launch(path)
        command = launch.call_args.args[0]
        self.assertEqual(command[1], "train.py")
        self.assertNotIn("--resume", command)
        self.assertNotIn("--checkpoint", command)
        self.assertEqual(command[command.index("--config")+1], str(path.parent/"input_config.json"))
        self.assertEqual(receipt["status"], "complete")
        self.assertFalse(receipt["parent_weights_loaded"])
        self.assertEqual((receipt["training_examples"], receipt["validation_examples"]), (50000, 0))

    def test_valid_full_records_have_no_development_or_confirmation_evaluation(self):
        records = self.records()
        with patch.object(refit, "build_model", side_effect=lambda config: torch.nn.Linear(1, 10, bias=False)):
            refit.validate_records(*records)

    def test_development_values_partial_epochs_and_stale_final_weights_reject(self):
        original = self.records()
        mutations = [lambda d: d[3].update(validation_examples=5000),
                     lambda d: d[4][0].update(validation_accuracy=.99),
                     lambda d: d[1].update(epoch=152),
                     lambda d: d[1].update(variant="ema"),
                     lambda d: d[2].update(updates=1),
                     lambda d: d[2]["model"]["weight"].add_(1),
                     lambda d: d[1]["provenance"]["source_sha256"].update({"train.py": "changed"}),
                     lambda d: d[6].update(amp_dtype="torch.float16")]
        with patch.object(refit, "build_model", side_effect=lambda config: torch.nn.Linear(1, 10, bias=False)):
            for mutate in mutations:
                records = list(copy.deepcopy(original))
                mutate(records)
                with self.assertRaises(ValueError):
                    refit.validate_records(*records)

    def test_overlapping_development_benchmark_is_rejected_before_checkpoint_load(self):
        path, plan = self.prepared()
        (path.parent/"benchmark.json").write_text("{}")
        with patch.object(refit.torch, "load") as checkpoint:
            with self.assertRaisesRegex(ValueError, "overlapping development"):
                refit.audit_artifacts(path, plan)
        checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
