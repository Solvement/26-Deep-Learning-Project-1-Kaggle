"""SWA routing and bounded cleanup with CPU-only mocked children."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil
from train import DEFAULT_CONFIG
from research import run_swa as runner
from research import train_swa


class SWASupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in runner.SOURCE_FILES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen owned source "+name)
        (self.root/"data").mkdir()
        (self.root/"data/training_manifest.json").write_text("{}")
        self.policy = {"stops_at": "2099-01-01T00:00:00+00:00", "local_job_timeout_seconds": 10800,
            "local_gpu_seconds_per_utc_day": 21600, "baseline_benchmark": "baseline.json"}
        (self.root/"research/campaign.json").write_text(json.dumps(self.policy))
        (self.root/"baseline.json").write_text("{}")
        self.config = {**DEFAULT_CONFIG, "depth": 34, "widen_factor": 3, "epochs": 200,
            "swa_prefix_epochs": 160, "swa_tail_epochs": 40, "swa_lr": .01, "swa_bn_seed": 2026}
        self.path = self.root/"config.json"
        self.path.write_text(json.dumps(self.config))
        self.folder = self.root/"results/research/swa_run"
        self.parent = self.root/"own/last.pt"
        self.parent.parent.mkdir()
        self.parent.write_bytes(b"own prefix checkpoint fixture")
        self.parent_info = {"metadata": {"checkpoint_path": str(self.parent), "checkpoint_sha256": runner.sha256(self.parent)},
            "paths": [self.parent]}
        self.preflight = self.root/"results/research/preflight/engineering_benchmark.json"
        self.preflight.parent.mkdir(parents=True)
        self.proof = {"median_step_seconds": .02, "p90_step_seconds": .03}

    def engineering(self):
        marker = train_swa.research_provenance("bf16", self.config)
        return {"schema": "research_swa_engineering_v1", "status": "complete", "synthetic_only": True,
            "precision": "bf16", "config_sha256": train_swa.config_hash(self.config), "research_training": marker,
            "source_sha256": marker["source_sha256"], "model_config": {k: self.config[k] for k in
                ("depth", "widen_factor", "dropout", "num_classes")}, "parameter_count": 4166298,
            "steps": 20, "timed_steps": 20, "warmup_steps": 3, "successful_updates": 20,
            "total_successful_updates": 23, "parameter_changed_steps": 23, "averaging_calls": 23,
            "native_bf16_verified": True, "finite_losses_verified": True, "finite_parameters_verified": True,
            "ema_tracked": True, "ema_selected": False,
            "training_data_loaded": False, "hidden_images_loaded": False, "confirmation_evaluated": False,
            **self.proof}

    def completed_child(self, command, log, seconds, receipt_path, receipt):
        self.assertGreater(seconds, 0)
        receipt.update(child_pid=123, child_created_at=123., watchdog_pid=124)
        folder = receipt_path.parent
        if "--benchmark-steps" in command:
            (folder/"engineering_benchmark.json").write_text(json.dumps(self.engineering()))
        elif command[1] == "train.py":
            prefix = folder/"prefix"
            (prefix/"config.json").write_text(json.dumps(train_swa.prefix_config(self.config)))
            (prefix/"last.pt").write_bytes(b"new scratch prefix checkpoint fixture")
        elif command[1] == "research/benchmark_swa.py":
            output = Path(command[command.index("--output")+1])
            variant = "swa" if output.name == "benchmark.json" else "raw_tail_control"
            output.write_text(json.dumps({"schema": runner.suite.SCHEMA, "status": "complete", "examples": 5000,
                "training_examples": 40000, "checkpoint_variant": variant, "checkpoint_epoch": 200,
                "policies": {policy: {"cases": {case: {"examples": 5000} for case in runner.suite.CASES}}
                    for policy in runner.suite.POLICIES}, "config": self.config, "research_swa_integrity": {"swa_count": 40},
                "confirmation_evaluated": False, "fixed_primary_policy": "center80_shift20",
                "candidate_eligible": variant == "swa",
                "promotion": {policy: {"passed": True} for policy in runner.suite.POLICIES},
                "promotion_vs_parent": {policy: {"passed": True} for policy in runner.suite.POLICIES}}))

    def formal_patches(self):
        return (patch.object(runner, "ROOT", self.root),
            patch.object(runner, "training_source", return_value={"archive_sha256": "a"*64}),
            patch.object(runner.suite, "validate_baseline"),
            patch.object(runner, "validate_preflight", return_value=(self.proof, [])),
            patch.object(train_swa, "validate_parent", return_value=self.parent_info),
            patch.object(runner.torch.cuda, "is_available", return_value=True),
            patch.object(runner.torch.cuda, "is_bf16_supported", return_value=True))

    def test_routes_primary_tail_fresh_prefix_and_preflight_without_resume(self):
        with patch.object(runner, "ROOT", self.root):
            primary = runner.build_commands(self.path, self.folder, self.root/"baseline.json", self.parent)
            fresh = runner.build_commands(self.path, self.folder, self.root/"baseline.json", fresh_prefix=True)
            smoke = runner.build_commands(self.path, self.folder, self.root/"baseline.json", preflight=True)
        self.assertEqual([stage for stage, _ in primary], ["tail", "swa", "control"])
        self.assertEqual([stage for stage, _ in fresh], ["prefix", "tail", "swa", "control"])
        self.assertEqual(fresh[0][1][1], "train.py")
        self.assertEqual(smoke[0][1][-2:], ["--benchmark-steps", "20"])
        self.assertFalse(any("--resume" in command for _, command in fresh))

    def test_bad_mode_config_or_missing_preflight_fails_before_lease_or_artifacts(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "acquire_lease") as lease:
            with self.assertRaisesRegex(ValueError, "requires preflight"):
                runner.run("swa_run", self.path, "formal", self.parent)
            self.path.write_text(json.dumps({**self.config, "swa_tail_epochs": 80}))
            with self.assertRaises(ValueError): runner.run("swa_run", self.path, "preflight")
        lease.assert_not_called()
        self.assertFalse(self.folder.exists())

    def test_existing_or_active_owner_does_not_create_swa_artifacts(self):
        base = self.folder.parent
        base.mkdir(parents=True, exist_ok=True)
        owner = psutil.Process()
        lease = base/"local_lease.json"
        original = json.dumps({"pid": owner.pid, "process_created_at": owner.create_time()})
        lease.write_text(original)
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                runner.run("swa_run", self.path, "preflight")
        child.assert_not_called()
        self.assertFalse(self.folder.exists())
        self.assertEqual(lease.read_text(), original)

    def test_preflight_uses_shared_lease_and_rejects_bad_update_proof(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner.torch.cuda, "is_available", return_value=True), \
            patch.object(runner.torch.cuda, "is_bf16_supported", return_value=True), \
            patch.object(runner, "supervised_command", side_effect=self.completed_child) as child:
            result = runner.run("swa_run", self.path, "preflight")
            proof, _ = runner.validate_preflight(self.folder/"engineering_benchmark.json", self.config, result["source_sha256"])
            self.assertEqual(proof["parameter_changed_steps"], 23)
            doc = json.loads((self.folder/"engineering_benchmark.json").read_text())
            doc["parameter_changed_steps"] = 22
            (self.folder/"engineering_benchmark.json").write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                runner.validate_preflight(self.folder/"engineering_benchmark.json", self.config, result["source_sha256"])
        self.assertEqual(child.call_count, 1)
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_primary_runs_tail_and_both_full_benchmarks_in_one_receipt(self):
        (self.root/"data/training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "b"*64}}))
        patches = self.formal_patches()
        for item in patches: item.start(); self.addCleanup(item.stop)
        with patch.object(runner, "supervised_command", side_effect=self.completed_child) as child:
            result = runner.run("swa_run", self.path, "formal", self.parent, preflight_path=self.preflight)
        self.assertEqual([call.args[0][1] for call in child.call_args_list],
            ["research/train_swa.py", "research/benchmark_swa.py", "research/benchmark_swa.py"])
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["raw_control_prediction_authorized"])
        self.assertEqual((self.folder/"input_config.json").read_bytes(), self.path.read_bytes())
        self.assertLessEqual(result["reserved_seconds"], 10800)
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_fresh_repeat_times_and_origin_include_entire_scratch_prefix(self):
        self.config["seed"] = 2027
        self.path.write_text(json.dumps(self.config))
        (self.root/"data/training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "b"*64}}))
        for item in self.formal_patches(): item.start(); self.addCleanup(item.stop)
        with patch.object(runner, "supervised_command", side_effect=self.completed_child):
            result = runner.run("swa_run", self.path, "formal", fresh_prefix=True, preflight_path=self.preflight)
        origin = json.loads((self.folder/"prefix/prefix_origin.json").read_text())
        self.assertEqual(origin["init"], "scratch_seeded")
        self.assertFalse(origin["parent_weights_loaded"])
        self.assertLessEqual(result["started_at"], origin["started_at"])
        self.assertLessEqual(origin["finished_at"], result["stages"][1]["started_at"])
        self.assertEqual(origin["rawinputconfig_sha256"], runner.sha256(self.folder/"prefix/input_config.json"))
        self.assertEqual(json.loads((self.folder/"prefix/input_config.json").read_text())["epochs"], 160)

    def test_raw_control_or_other_policy_win_cannot_replace_failed_weighted_parent_gate(self):
        (self.root/"data/training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "b"*64}}))
        for item in self.formal_patches(): item.start(); self.addCleanup(item.stop)
        def not_incremental(command, log, seconds, receipt_path, receipt):
            self.completed_child(command, log, seconds, receipt_path, receipt)
            if command[1] == "research/benchmark_swa.py":
                path = Path(command[command.index("--output")+1])
                if path.name == "benchmark.json":
                    value = json.loads(path.read_text())
                    value["promotion_vs_parent"]["center80_shift20"]["passed"] = False
                    value["candidate_eligible"] = False
                    path.write_text(json.dumps(value))
        with patch.object(runner, "supervised_command", side_effect=not_incremental):
            result = runner.run("swa_run", self.path, "formal", self.parent, preflight_path=self.preflight)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["benchmark_passed"])
        self.assertFalse(result["raw_control_prediction_authorized"])

    def test_partial_case_or_wrong_variant_cannot_complete_formal_job(self):
        (self.root/"data/training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "b"*64}}))
        for item in self.formal_patches(): item.start(); self.addCleanup(item.stop)
        def partial(command, log, seconds, receipt_path, receipt):
            self.completed_child(command, log, seconds, receipt_path, receipt)
            if command[1] == "research/benchmark_swa.py":
                path = Path(command[command.index("--output")+1])
                value = json.loads(path.read_text())
                value["policies"]["plain"]["cases"]["clean"]["examples"] = 100
                path.write_text(json.dumps(value))
        with patch.object(runner, "supervised_command", side_effect=partial):
            with self.assertRaisesRegex(ValueError, "full development"):
                runner.run("swa_run", self.path, "formal", self.parent, preflight_path=self.preflight)
        self.assertEqual(json.loads((self.folder/"supervisor.json").read_text())["status"], "failed")
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_daily_budget_and_measured_overrun_fail_before_child_and_cleanup(self):
        self.policy["local_gpu_seconds_per_utc_day"] = 59
        (self.root/"research/campaign.json").write_text(json.dumps(self.policy))
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "supervised_command") as child, \
                patch.object(runner.torch.cuda, "is_available", side_effect=AssertionError("No GPU query")):
            with self.assertRaisesRegex(RuntimeError, "Insufficient local budget"):
                runner.run("swa_run", self.path, "preflight")
        child.assert_not_called()
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_fixed_measured_runtime_margin_rejects_a_formal_job_that_cannot_finish(self):
        (self.root/"data/training_manifest.json").write_text(json.dumps({"split": {"indices_sha256": "b"*64}}))
        self.proof["median_step_seconds"] = 1.
        for item in self.formal_patches(): item.start(); self.addCleanup(item.stop)
        with patch.object(runner, "supervised_command") as child:
            with self.assertRaisesRegex(RuntimeError, "Insufficient local budget"):
                runner.run("swa_run", self.path, "formal", self.parent, preflight_path=self.preflight)
        child.assert_not_called()
        self.assertFalse((self.folder.parent/"local_lease.json").exists())

    def test_parent_failure_is_preserved_and_shared_lease_released(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner.torch.cuda, "is_available", return_value=True), \
            patch.object(runner.torch.cuda, "is_bf16_supported", return_value=True), \
            patch.object(runner, "supervised_command", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): runner.run("swa_run", self.path, "preflight")
        record = json.loads((self.folder/"supervisor.json").read_text())
        self.assertEqual((record["status"], record["error_type"]), ("failed", "KeyboardInterrupt"))
        self.assertIn("finished_at", record)
        self.assertFalse((self.folder.parent/"local_lease.json").exists())


if __name__ == "__main__": unittest.main()
