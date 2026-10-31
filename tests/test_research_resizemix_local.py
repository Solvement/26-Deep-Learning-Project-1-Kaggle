"""Independent repeat supervision: mocked CPU children, no real GPU allocation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil
from research import run_resizemix_local as runner
from train import DEFAULT_CONFIG


class ResizeMixLocalSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in runner.SOURCE_FILES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixed own source " + name)
        (self.root/"data").mkdir()
        (self.root/"data/training_manifest.json").write_text("{}")
        (self.root/"baseline.json").write_text(json.dumps({"schema": runner.suite.SCHEMA, "status": "complete",
            "examples": 5000, "checkpoint_sha256": runner.suite.BASELINE_CHECKPOINT_SHA}))
        self.policy = {"stops_at": "2099-01-01T00:00:00+00:00", "local_job_timeout_seconds": 10800,
                       "local_gpu_seconds_per_utc_day": 21600, "baseline_benchmark": "baseline.json",
                       "benchmark_schema": "official_development_stress_v3"}
        self.policy_path = self.root/"research/campaign.json"
        self.policy_path.write_text(json.dumps(self.policy))
        self.config = {**DEFAULT_CONFIG, "seed": 2027, "cutout": 0, "mixup_alpha": 0., "mixup_probability": 0.,
                       "resizemix_probability": .5, "resizemix_tau_min": .1, "resizemix_tau_max": .8}
        self.config_path = self.root/"repeat_config.json"
        self.config_path.write_text(json.dumps(self.config, separators=(",", ":")))
        self.folder = self.root/"results/research/resize_repeat"

    def completed_child(self, command, log, seconds, receipt_path, receipt):
        self.assertGreater(seconds, 0)
        if command[1] == "research/benchmark_resizemix.py":
            (receipt_path.parent/"benchmark.json").write_text(json.dumps({
                "schema": "official_development_stress_v3", "status": "complete", "examples": 5000,
                "training_examples": 40000, "config": self.config, "confirmation_evaluated": False,
                "promotion": {"center80_shift20": {"passed": True}}}))

    def test_commands_require_explicit_precision_and_never_request_resume_or_profile(self):
        for precision in ("fp16", "bf16", "fp32"):
            with patch.object(runner, "ROOT", self.root):
                training, benchmark = runner.build_commands(self.config_path, self.folder, self.root/"baseline.json", precision)
            self.assertEqual(training[1], "research/train_resizemix.py")
            self.assertEqual(training[training.index("--precision")+1], precision)
            self.assertEqual(benchmark[1], "research/benchmark_resizemix.py")
            self.assertNotIn("--resume", training)
            self.assertNotIn("--benchmark-steps", training)
        with self.assertRaisesRegex(ValueError, "explicitly"):
            runner.build_commands(self.config_path, self.folder, self.root/"baseline.json", "auto")

    def test_invalid_config_primary_seed_or_missing_source_rejects_before_a_lease(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "acquire_lease") as lease:
            self.config_path.write_text(json.dumps({**self.config, "cutout": 16}))
            with self.assertRaisesRegex(ValueError, "replace both"):
                runner.run("resize_repeat", self.config_path, "fp16")
            self.config_path.write_text(json.dumps({**self.config, "seed": 2026}))
            with self.assertRaisesRegex(ValueError, "different seed"):
                runner.run("resize_repeat", self.config_path, "fp16")
            self.config_path.write_text(json.dumps(self.config))
            (self.root/"research/resizemix.py").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.run("resize_repeat", self.config_path, "fp16")
        lease.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_existing_directory_is_immutable_and_does_not_acquire_gpu(self):
        self.folder.mkdir(parents=True)
        proof = self.folder/"keep.json"
        proof.write_text("untouched")
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "acquire_lease") as lease:
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                runner.run("resize_repeat", self.config_path, "fp16")
        lease.assert_not_called()
        self.assertEqual(proof.read_text(), "untouched")

    def test_partial_baseline_rejects_before_creating_repeat_or_lease(self):
        (self.root/"baseline.json").write_text(json.dumps({"schema": runner.suite.SCHEMA, "status": "smoke", "examples": 100}))
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "acquire_lease") as lease:
            with self.assertRaisesRegex(ValueError, "trusted-original baseline"):
                runner.run("resize_repeat", self.config_path, "fp16")
        lease.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_active_owner_blocks_without_creating_repeat_artifacts(self):
        base = self.folder.parent
        base.mkdir(parents=True)
        process = psutil.Process()
        lease = base/"local_lease.json"
        original = json.dumps({"pid": process.pid, "process_created_at": process.create_time()})
        lease.write_text(original)
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                runner.run("resize_repeat", self.config_path, "fp16")
        child.assert_not_called()
        self.assertFalse(self.folder.exists())
        self.assertEqual(lease.read_text(), original)

    def test_success_copies_exact_config_and_runs_complete_benchmark_with_cloud_excluded(self):
        cloud = self.folder.parent/"cloud"
        cloud.mkdir(parents=True)
        (cloud/"supervisor.json").write_text(json.dumps({"status": "training_and_benchmarking", "session": "other-cloud",
                                                       "assignment_attempts": 1}))
        original = self.config_path.read_bytes()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner.torch.cuda, "is_available", return_value=True), \
             patch.object(runner, "supervised_command", side_effect=self.completed_child) as child:
            result = runner.run("resize_repeat", self.config_path, "fp16")
        self.assertEqual([call.args[0][1] for call in child.call_args_list],
                         ["research/train_resizemix.py", "research/benchmark_resizemix.py"])
        self.assertEqual((self.folder/"input_config.json").read_bytes(), original)
        self.assertEqual(result["config_sha256"], runner.sha256(self.config_path))
        self.assertEqual((result["trainer"], result["precision"], result["role"]), ("resizemix", "fp16", "development_repeat"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["benchmark_passed"])
        self.assertIn("benchmark_sha256", result)
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_minimum_budget_refuses_child_and_releases_owned_lease(self):
        self.policy["local_gpu_seconds_per_utc_day"] = 2399
        self.policy_path.write_text(json.dumps(self.policy))
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "Insufficient local budget"):
                runner.run("resize_repeat", self.config_path, "fp16")
        child.assert_not_called()
        self.assertFalse((self.folder.parent/"local_lease.json").exists())
        self.assertEqual(json.loads((self.folder/"supervisor.json").read_text())["status"], "failed")

    def test_parent_interruption_preserves_failure_and_releases_lease(self):
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner.torch.cuda, "is_available", return_value=True), \
             patch.object(runner, "supervised_command", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run("resize_repeat", self.config_path, "fp16")
        result = json.loads((self.folder/"supervisor.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "KeyboardInterrupt")
        self.assertIn("finished_at", result)
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_incomplete_development_benchmark_cannot_complete_repeat(self):
        def partial(command, log, seconds, receipt_path, receipt):
            self.completed_child(command, log, seconds, receipt_path, receipt)
            path = receipt_path.parent/"benchmark.json"
            if path.exists():
                doc = json.loads(path.read_text())
                doc["examples"] = 100
                path.write_text(json.dumps(doc))
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner.torch.cuda, "is_available", return_value=True), \
             patch.object(runner, "supervised_command", side_effect=partial):
            with self.assertRaisesRegex(ValueError, "complete fixed official"):
                runner.run("resize_repeat", self.config_path, "fp16")
        self.assertEqual(json.loads((self.folder/"supervisor.json").read_text())["status"], "failed")

    def test_changed_source_after_child_cannot_complete_repeat(self):
        def changed(command, log, seconds, receipt_path, receipt):
            self.completed_child(command, log, seconds, receipt_path, receipt)
            if command[1] == "research/benchmark_resizemix.py":
                (self.root/"research/resizemix.py").write_text("changed after training")
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner.torch.cuda, "is_available", return_value=True), \
             patch.object(runner, "supervised_command", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "Immutable repeat inputs/source changed"):
                runner.run("resize_repeat", self.config_path, "fp16")
        self.assertEqual(json.loads((self.folder/"supervisor.json").read_text())["status"], "failed")
        self.assertFalse((self.folder.parent/"local_lease.json").exists())


if __name__ == "__main__":
    unittest.main()
