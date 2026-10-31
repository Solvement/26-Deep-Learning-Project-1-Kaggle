"""Colab preparation and cleanup boundaries; no network or real allocation."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from research import run_colab as runner


class ColabRunnerTests(unittest.TestCase):
    def test_trainer_routes_are_allowlisted_and_legacy_default_remains_shadow(self):
        for trainer, entries in {"shadow": ("research/train_shadow.py", "research/benchmark_shadow.py"),
                                 "resizemix": ("research/train_resizemix.py", "research/benchmark_resizemix.py")}.items():
            self.assertEqual(runner.trainer_routes(trainer), entries)
            manifest = {"run": "abc", "trainer": trainer, "chunks": [], "files_sha256": {}, "bundle_sha256": "x"}
            program = runner.remote_program(manifest, "owned", 1900000000)
            compile(program, "remote.py", "exec")
            self.assertIn("TRAIN = " + repr(entries[0]), program)
            self.assertIn("VALIDATOR = " + repr(entries[1]), program)
            self.assertIn("'trainer_source_sha256':M['files_sha256'][TRAIN]", program)
            self.assertIn("from research.train_shadow import optimizer_step", program)
        legacy = runner.remote_program({"run": "old"}, "owned", 1900000000)
        self.assertIn("TRAIN = 'research/train_shadow.py'", legacy)
        self.assertNotIn("research/resizemix.py", runner.sources_for_trainer("shadow"))
        self.assertTrue(set(runner.RESIZEMIX_SOURCES) <= set(runner.sources_for_trainer("resizemix")))

    def test_unknown_trainer_rejected_before_cloud_lease_or_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"input_manifest.json").write_text(json.dumps({"trainer": "arbitrary_command"}))
            with patch.object(runner, "acquire_lease") as lease, patch.object(runner, "ColabCLI") as client:
                with self.assertRaisesRegex(ValueError, "Unsupported Colab training method"):
                    runner.launch(folder, "Ubuntu")
            lease.assert_not_called()
            client.assert_not_called()
        with self.assertRaises(ValueError):
            runner.remote_program({"trainer": "../../custom.py"}, "owned", 1900000000)

    def test_full_training_and_unsafe_ids_rejected(self):
        for value in ("../new", "bad-name", "stability_seed2027", ""):
            with self.assertRaises(ValueError):
                runner.validate_run(value)
        for value in ({"train_all": True, "split_seed": 2026, "epochs": 160},
                      {"train_all": False, "split_seed": 2027, "epochs": 160}):
            with self.assertRaises(ValueError):
                runner.validate_config(value)

    def test_usage_requires_all_verified_fields(self):
        output = "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
        self.assertEqual(runner.parse_usage(output)["active_assignments"], 0)
        with self.assertRaisesRegex(ValueError, "could not be verified"):
            runner.parse_usage("balance 0 active 0")

    def test_existing_t4_accepts_positive_cu_rate_at_zero_balance(self):
        usage = runner.parse_usage("Current balance: 0.00 compute units\nUsage rate: 1.15/hr\nActive assignments: 1")
        runner.validate_resource_usage(usage, expected_assignments=1)
        self.assertEqual(usage["balance_compute_units"], 0)
        self.assertEqual(usage["rate_compute_units_per_hour"], 1.15)

    def test_resource_usage_rejects_nonfinite_negative_and_multiple_assignments(self):
        valid = {"balance_compute_units": 0.0, "rate_compute_units_per_hour": 1.15,
                 "active_assignments": 1}
        for key in ("balance_compute_units", "rate_compute_units_per_hour"):
            for value in (float("nan"), float("inf"), -1.0, True):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    runner.validate_resource_usage({**valid, key: value}, expected_assignments=1)
        for count in (0, 2, True, 1.0):
            with self.assertRaisesRegex(RuntimeError, "active Colab assignments"):
                runner.validate_resource_usage({**valid, "active_assignments": count}, expected_assignments=1)

    def test_cloud_lease_never_silently_reclaims_dead_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease.json"
            runner.acquire_lease(path, {"pid": 99999999, "session": "old"})
            with self.assertRaises(FileExistsError):
                runner.acquire_lease(path, {"session": "new"})
            self.assertEqual(json.loads(path.read_text())["session"], "old")

    def test_prepare_allowlists_official_cache_not_external_or_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "research").mkdir()
            (root / "results/regularized").mkdir(parents=True)
            for name in ("core.py", "research/train_shadow.py", *runner.RESIZEMIX_SOURCES):
                (root / name).write_text("own code")
            hashes = {}
            for name in runner.CACHE[:-1]:
                (root / "data" / name).write_bytes(b"audited allowed input")
                hashes[name] = runner.sha256(root / "data" / name)
            manifest = {"source_type": "official_kaggle_competition_training",
                        "external_training_data_loaded": False, "official_test_labels_loaded": False,
                        "split": {"seed": 2026}, "files_sha256": hashes}
            (root / "data/training_manifest.json").write_text(json.dumps(manifest))
            for name in ("best.pt", "metrics.json", "history.json"):
                (root / "results/regularized" / name).write_bytes(b"baseline")
            (root / "data/official_test.npz").write_bytes(b"never package")
            (root / "data/cifar-10-python.tar.gz").write_bytes(b"external never package")
            config = root / "config.json"
            config.write_text(json.dumps({"train_all": False, "split_seed": 2026, "epochs": 160}))
            for trainer in ("shadow", "resizemix"):
                output = root / ("prepared_" + trainer)
                output.mkdir()
                with patch.object(runner, "CORE", {"core.py": runner.sha256(root / "core.py")}), \
                     patch.object(runner, "SOURCES", ("core.py", "research/train_shadow.py")), \
                     patch.object(runner, "BASELINE_SHA", runner.sha256(root / "results/regularized/best.pt")):
                    result = runner.prepare(root, config, output, "candidate1", trainer)
                with zipfile.ZipFile(output / "input.private.zip") as archive:
                    names = set(archive.namelist())
                self.assertNotIn("data/official_test.npz", names)
                self.assertNotIn("data/cifar-10-python.tar.gz", names)
                self.assertIn("data/official_split_indices.npz", names)
                self.assertEqual(result["trainer"], trainer)
                if trainer == "resizemix":
                    self.assertTrue(set(runner.RESIZEMIX_SOURCES) <= names)
                else:
                    self.assertTrue(set(runner.RESIZEMIX_SOURCES).isdisjoint(names))
                self.assertFalse(result["confirmation_evaluated"])
                self.assertEqual(result["bundle_sha256"], runner.sha256(output / "input.private.zip"))
                restored = b"".join((output / p["name"]).read_bytes() for p in result["chunks"])
                self.assertEqual(hashlib.sha256(restored).hexdigest(), result["bundle_sha256"])

    def test_unapproved_output_and_corrupt_download_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "result.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("../../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                runner.extract_artifacts(archive, Path(tmp) / "out", runner.sha256(archive))
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.extract_artifacts(archive, Path(tmp) / "out", "wrong")

    def test_stop_verification_checks_actual_usage(self):
        cli = MagicMock()
        cli.run.side_effect = [(0, "[colab] Session terminated."), (0, "Current balance: 0 compute units\nUsage rate: 0/hr\nActive assignments: 1")]
        self.assertFalse(runner.stop_and_verify(cli, "owned", allocation_known=True)["runtime_stop_verified"])
        self.assertEqual(cli.run.call_args_list[0].args[0], ["stop", "--session", "owned"])

    def test_unknown_allocation_cannot_be_reconciled_by_noop_stop_and_zero_usage(self):
        usage = "Current balance: 0 compute units\nUsage rate: 0/hr\nActive assignments: 0"
        cli = MagicMock()
        cli.run.side_effect = [(0, "[colab] Session 'owned' not found."), (0, usage)]
        self.assertFalse(runner.stop_and_verify(cli, "owned")["runtime_stop_verified"])
        cli.run.side_effect = [(0, "[colab] Session terminated."), (0, usage)]
        self.assertFalse(runner.stop_and_verify(cli, "owned", allocation_known=False)["runtime_stop_verified"])
        cli.run.side_effect = [(0, "[colab] Session terminated."), (0, usage)]
        self.assertTrue(runner.stop_and_verify(cli, "owned", allocation_known=True)["runtime_stop_verified"])

    def test_failed_allocation_is_not_retried_and_uncertain_cleanup_keeps_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "research").mkdir()
            (root / "results/research/job").mkdir(parents=True)
            (root / "research/campaign.json").write_text(json.dumps({
                "colab_additional_purchases_authorized": False,
                "stops_at": "2999-01-01T00:00:00+00:00"}))
            folder = root / "results/research/job"
            (folder / "input.private.zip").write_bytes(b"sealed")
            (folder / "input_manifest.json").write_text(json.dumps({
                "run": "research1", "chunks": [], "config": {
                    "train_all": False, "split_seed": 2026, "epochs": 160},
                "bundle_sha256": runner.sha256(folder / "input.private.zip")}))
            cli = MagicMock()
            cli.require.side_effect = ["Current balance: 0 compute units\nUsage rate: 0/hr\nActive assignments: 0",
                                       RuntimeError("Unknown assignment response")]
            cli.run.side_effect = [(0, "[colab] Session 'owned' not found."),
                                   (0, "Current balance: 0 compute units\nUsage rate: 0/hr\nActive assignments: 0")]
            process = MagicMock()
            process.poll.return_value = None
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "ColabCLI", return_value=cli), \
                 patch.object(runner, "private_directory", side_effect=lambda p: p.mkdir(exist_ok=True)), \
                 patch.object(runner.subprocess, "Popen", return_value=process), \
                 patch.object(runner, "stop_and_verify", wraps=runner.stop_and_verify) as cleanup:
                with self.assertRaisesRegex(RuntimeError, "Unknown assignment"):
                    runner.launch(folder, "Ubuntu")
            allocation_calls = [call for call in cli.require.call_args_list if call.args[0][0] == "new"]
            self.assertEqual(len(allocation_calls), 1)
            self.assertEqual(cleanup.call_count, 1)
            self.assertTrue((root / "results/research/cloud_lease.json").exists())
            receipt = json.loads((folder / "supervisor.json").read_text())
            self.assertFalse(receipt["runtime_stop_verified"])
            self.assertEqual(receipt["assignment_attempts"], 1)
            self.assertFalse(receipt["allocation_returned_success"])
            self.assertFalse(receipt["remote_guard_verified"])
            self.assertEqual(receipt["stop_exit_code"], 0)
            self.assertEqual(receipt["usage_after"]["active_assignments"], 0)
            self.assertEqual(receipt["status"], "failed")

    def test_remote_program_has_self_release_and_no_hidden_or_submission_calls(self):
        guard = runner.remote_guard(1900000000)
        program = runner.remote_program({"run": "abc", "chunks": [], "files_sha256": {}, "bundle_sha256": "x"}, "owned", 1900000000)
        compile(guard, "guard.py", "exec")
        compile(program, "program.py", "exec")
        self.assertIn("colab_runtime.unassign()", guard)
        self.assertIn("'--precision','fp16'", program)
        self.assertIn("research/benchmark_shadow.py", program)
        for forbidden in ("load_confirmation_indices", "load_kaggle_test", "kaggle competitions submit"):
            self.assertNotIn(forbidden, program)

    def test_resizemix_native_method_preflight_is_before_formal_baseline_and_training(self):
        manifest = {"run": "resize", "trainer": "resizemix", "chunks": [], "files_sha256": {}, "bundle_sha256": "x"}
        program = runner.remote_program(manifest, "owned", 1900000000)
        compile(program, "resizemix_remote.py", "exec")
        self.assertIn("if M.get('trainer','shadow')=='resizemix':", program)
        self.assertIn("from research.train_resizemix import prepare_batch, optimizer_step", program)
        self.assertIn("{**cfg,'resizemix_probability':1.}", program)
        self.assertIn("torch.amp.GradScaler('cuda',init_scale=256.)", program)
        self.assertIn("torch.autocast('cuda',dtype=torch.float16)", program)
        self.assertIn("for index in range(13)", program)
        self.assertIn("assert mixed_calls>0 and changed_steps==13", program)
        self.assertIn("'warmup_steps':3,'timed_steps':10", program)
        self.assertIn("'runtime_budget_fit':safe_estimate<D-time.time()-180", program)
        self.assertLess(program.index("(R/'method_preflight.json')"), program.index("run(['research/benchmark.py'"))
        self.assertIn("method_preflight.json", runner.OUTPUTS)

    def test_cli_command_has_no_broad_auth_or_shell(self):
        cli = runner.ColabCLI(Path("unused"), python="/home/example/.local/share/uv/tools/google-colab-cli/bin/python")
        command = cli.command(["usage"])
        self.assertNotIn("sh", command)
        self.assertIn("COLAB_CLI_ASSIGN_RETRIES=0", command)
        self.assertTrue(any(arg.endswith("research/colab_cli.py") for arg in command))
        self.assertEqual(command[-1], "usage")


if __name__ == "__main__":
    unittest.main()
