"""Bounded diagnostic routing, shared GPU exclusion and accounting; mocked children."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
from research import run_diagnostic as diagnostic
from research.run_local import used_seconds


class DiagnosticSupervisorTests(unittest.TestCase):
    def setup_root(self, root, daily=21600):
        (root/"research").mkdir()
        (root/"results/regularized").mkdir(parents=True)
        (root/"results/regularized/best.pt").write_bytes(b"trusted-presence-only")
        (root/"research/campaign.json").write_text(json.dumps({
            "stops_at": "2099-01-01T00:00:00+00:00", "local_gpu_seconds_per_utc_day": daily,
            "local_job_timeout_seconds": 10800, "baseline_benchmark": "baseline.json"}))
        for name in ("train_sam.py", "benchmark_shadow.py", "benchmark_sam.py", "paired_compare.py"):
            (root/"research"/name).write_text("mock child entrypoint")
        for name in ("config.json", "baseline.json", "candidate.json"):
            (root/name).write_text("{}")
        (root/"best.pt").write_bytes(b"checkpoint-presence-only")
        return ["run_diagnostic.py", "--run", "check", "--mode", "sam-speed", "--config", str(root/"config.json")]

    def test_all_modes_route_only_to_allowlisted_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_root(root)
            args = SimpleNamespace(mode="sam-speed", config=root/"config.json", checkpoint=root/"best.pt",
                                   candidate_benchmark=root/"candidate.json", policy="flip")
            entries = {"sam-speed": "research/train_sam.py", "shadow-benchmark": "research/benchmark_shadow.py",
                       "sam-benchmark": "research/benchmark_sam.py", "paired": "research/paired_compare.py"}
            with patch.object(diagnostic, "ROOT", root):
                for mode, entry in entries.items():
                    args.mode = mode
                    command, output, inputs = diagnostic.build_command(args, root/"new", root/"baseline.json")
                    self.assertEqual(command[1], entry)
                    self.assertTrue(all(path.is_file() for path in inputs))
                    self.assertEqual(command[command.index("--device")+1], "cuda")
                    if mode == "sam-speed":
                        self.assertEqual(command[command.index("--benchmark-steps")+1], "20")
                        self.assertEqual(command[command.index("--precision")+1], "bf16")
                        self.assertEqual(output.name, "engineering_benchmark.json")
                    elif mode == "paired":
                        self.assertEqual(command[command.index("--baseline-checkpoint")+1], str(root/"results/regularized/best.pt"))
                        self.assertEqual(output.name, "paired.json")

    def test_missing_required_mode_inputs_rejected_before_a_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_root(root)
            with patch.object(diagnostic, "ROOT", root), patch.object(diagnostic, "acquire_lease") as lease:
                for mode in diagnostic.MODES:
                    with patch("sys.argv", ["run_diagnostic.py", "--run", "missing", "--mode", mode]):
                        with self.assertRaises(ValueError):
                            diagnostic.main()
            lease.assert_not_called()

    def test_active_local_owner_refuses_without_creating_diagnostic_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root)
            base = root/"results/research"
            base.mkdir()
            lease = base/"local_lease.json"
            process = psutil.Process()
            original = json.dumps({"pid": process.pid, "process_created_at": process.create_time()})
            lease.write_text(original)
            with patch.object(diagnostic, "ROOT", root), patch("sys.argv", argv), \
                 patch.object(diagnostic, "supervised_command") as child:
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    diagnostic.main()
            child.assert_not_called()
            self.assertFalse((base/"check").exists())
            self.assertEqual(lease.read_text(), original)

    def test_complete_diagnostic_has_default_bound_and_counts_once_ignoring_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root)
            cloud = root/"results/research/cloud"
            cloud.mkdir(parents=True)
            (cloud/"supervisor.json").write_text(json.dumps({"session": "owned-cloud", "assignment_attempts": 1,
                                                           "status": "training_and_benchmarking"}))
            def complete(command, log, timeout, receipt_path, receipt):
                (receipt_path.parent/"engineering_benchmark.json").write_text(json.dumps({"status": "complete"}))
            with patch.object(diagnostic, "ROOT", root), patch("sys.argv", argv), \
                 patch.object(diagnostic, "supervised_command", side_effect=complete) as child:
                diagnostic.main()
            self.assertGreater(child.call_args.args[2], 0)
            self.assertLessEqual(child.call_args.args[2], 180)
            base = root/"results/research"
            record = json.loads((base/"check/supervisor.json").read_text())
            self.assertEqual((record["status"], record["role"], record["backend"]), ("complete", "diagnostic", "local"))
            self.assertEqual(record["reserved_seconds"], 180)
            day = diagnostic.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.assertAlmostEqual(used_seconds(base, day, day+timedelta(days=1)),
                                   (diagnostic.datetime.fromisoformat(record["finished_at"])-
                                    diagnostic.datetime.fromisoformat(record["started_at"])).total_seconds())
            self.assertFalse((base/"local_lease.json").exists())

    def test_exhausted_budget_refuses_child_and_releases_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root, daily=0)
            with patch.object(diagnostic, "ROOT", root), patch("sys.argv", argv), \
                 patch.object(diagnostic, "supervised_command") as child:
                with self.assertRaisesRegex(RuntimeError, "runtime remains"):
                    diagnostic.main()
            child.assert_not_called()
            self.assertFalse((root/"results/research/local_lease.json").exists())

    def test_campaign_hardstop_caps_the_child_below_requested_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root)
            current = diagnostic.now()
            policy_path = root/"research/campaign.json"
            policy = json.loads(policy_path.read_text())
            policy["stops_at"] = (current+timedelta(seconds=5)).isoformat()
            policy_path.write_text(json.dumps(policy))
            def complete(command, log, timeout, receipt_path, receipt):
                (receipt_path.parent/"engineering_benchmark.json").write_text(json.dumps({"status": "complete"}))
            with patch.object(diagnostic, "ROOT", root), patch.object(diagnostic, "now", return_value=current), \
                 patch("sys.argv", argv), patch.object(diagnostic, "supervised_command", side_effect=complete) as child:
                diagnostic.main()
            self.assertGreater(child.call_args.args[2], 0)
            self.assertLessEqual(child.call_args.args[2], 5)

    def test_existing_completed_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root)
            folder = root/"results/research/check"
            folder.mkdir(parents=True)
            evidence = folder/"engineering_benchmark.json"
            evidence.write_text("original immutable evidence")
            with patch.object(diagnostic, "ROOT", root), patch("sys.argv", argv), \
                 patch.object(diagnostic, "acquire_lease") as lease:
                with self.assertRaisesRegex(FileExistsError, "immutable"):
                    diagnostic.main()
            lease.assert_not_called()
            self.assertEqual(evidence.read_text(), "original immutable evidence")

    def test_parent_exception_preserves_failure_receipt_and_releases_owned_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.setup_root(root)
            with patch.object(diagnostic, "ROOT", root), patch("sys.argv", argv), \
                 patch.object(diagnostic, "supervised_command", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    diagnostic.main()
            record = json.loads((root/"results/research/check/supervisor.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "KeyboardInterrupt")
            self.assertIn("finished_at", record)
            self.assertFalse((root/"results/research/local_lease.json").exists())


if __name__ == "__main__":
    unittest.main()
