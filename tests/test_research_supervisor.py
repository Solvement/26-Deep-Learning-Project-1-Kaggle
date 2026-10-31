"""Interrupted jobs must resume without overlapping a live owner or erasing budget."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from research.run_local import (assert_stopped, close_interrupted_attempt, used_seconds, main, sha256,
                                build_commands, trainer_identity, validate_resume_identity)


class ResearchSupervisorTests(unittest.TestCase):
    def test_trainer_commands_route_to_the_matching_training_and_integrity_validator(self):
        expected = {"frozen": ("train.py", "research/benchmark.py"),
                    "shadow": ("research/train_shadow.py", "research/benchmark_shadow.py"),
                    "sam": ("research/train_sam.py", "research/benchmark_sam.py")}
        for trainer, entries in expected.items():
            with self.subTest(trainer=trainer):
                command, benchmark = build_commands(trainer, "bf16", Path("config.json"),
                                                      Path("run"), Path("baseline.json"), resume=True)
                self.assertEqual(command[1], entries[0])
                self.assertEqual(benchmark[1], entries[1])
                self.assertIn("--resume", command)
                if trainer == "frozen":
                    self.assertNotIn("--precision", command)
                else:
                    self.assertEqual(command[command.index("--precision")+1], "bf16")
                self.assertEqual(benchmark[benchmark.index("--baseline")+1], "baseline.json")

    def test_frozen_precision_and_legacy_resume_are_bound(self):
        self.assertEqual(trainer_identity("frozen", "auto"), ("frozen", "bf16"))
        validate_resume_identity({}, "frozen", "auto")
        validate_resume_identity({}, "frozen", "bf16")
        validate_resume_identity({"trainer": "frozen", "precision": "bfloat16"}, "frozen", "auto")
        for precision in ("fp16", "fp32"):
            with self.assertRaisesRegex(ValueError, "hardcodes BF16"):
                trainer_identity("frozen", precision)
        for trainer in ("shadow", "sam"):
            with self.assertRaisesRegex(ValueError, "trainer or precision changed"):
                validate_resume_identity({}, trainer, "bf16")
        with self.assertRaisesRegex(ValueError, "trainer or precision changed"):
            validate_resume_identity({"trainer": "sam", "precision": "bf16"}, "sam", "fp16")

    def test_live_process_refuses_resume_but_reused_pid_does_not(self):
        process = psutil.Process()
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            assert_stopped({"pid": process.pid, "process_created_at": process.create_time()})
        assert_stopped({"pid": process.pid, "process_created_at": process.create_time()-100})

    def test_interrupted_reservation_is_charged_and_raw_receipt_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            job = base/"run"
            job.mkdir()
            began = datetime(2026, 10, 31, 6, tzinfo=timezone.utc)
            record = {"status": "training", "started_at": began.isoformat(), "reserved_seconds": 10800}
            path = job/"supervisor.json"
            original = json.dumps(record).encode()
            path.write_bytes(original)
            with patch("research.run_local.now", return_value=began+timedelta(hours=9)):
                archived = close_interrupted_attempt(path)
            self.assertEqual((job/"supervisor_raw_1.json").read_bytes(), original)
            self.assertEqual(path.read_bytes(), original)
            closed = json.loads(archived.read_text())
            self.assertEqual(closed["status"], "interrupted")
            self.assertEqual(closed["elapsed_seconds"], 10800)
            self.assertEqual(used_seconds(base, began.replace(hour=0), began.replace(hour=0)+timedelta(days=1), exclude=path), 10800)

    def test_ledger_counts_prior_attempt_and_cross_midnight_overlap_once(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            job = base/"run"
            job.mkdir()
            day = datetime(2026, 11, 1, tzinfo=timezone.utc)
            records = [("supervisor_attempt_1.json", day-timedelta(hours=1), day+timedelta(minutes=30)),
                       ("supervisor.json", day+timedelta(hours=1), day+timedelta(hours=2))]
            for name, start, end in records:
                (job/name).write_text(json.dumps({"status": "complete", "started_at": start.isoformat(), "finished_at": end.isoformat()}))
            self.assertEqual(used_seconds(base, day, day+timedelta(days=1)), 5400)

    def test_unfinished_other_job_blocks_new_work(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base/"other").mkdir()
            (base/"other/supervisor.json").write_text(json.dumps({"status": "training"}))
            day = datetime(2026, 11, 1, tzinfo=timezone.utc)
            with self.assertRaisesRegex(RuntimeError, "reconciliation"):
                used_seconds(base, day, day+timedelta(days=1))

    def test_active_colab_does_not_block_or_charge_the_independent_local_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base/"cloud").mkdir()
            (base/"cloud/supervisor.json").write_text(json.dumps({
                "status": "training_and_benchmarking", "session": "colab-owned",
                "assignment_attempts": 1}))
            day = datetime(2026, 11, 1, tzinfo=timezone.utc)
            self.assertEqual(used_seconds(base, day, day+timedelta(days=1)), 0)

    def test_cross_trainer_resume_rejected_before_archiving_lease_or_child_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"research").mkdir()
            (root/"research/campaign.json").write_text(json.dumps({"stops_at": "2099-01-01T00:00:00+00:00"}))
            config = root/"config.json"
            config.write_text(json.dumps({"train_all": False, "split_seed": 2026}))
            job = root/"results/research/resume_case"
            job.mkdir(parents=True)
            (job/"last.pt").write_bytes(b"SAM-checkpoint-presence-only")
            receipt = job/"supervisor.json"
            original = json.dumps({"status": "failed", "trainer": "sam", "precision": "bf16",
                                   "config_sha256": sha256(config)}).encode()
            receipt.write_bytes(original)
            with patch("research.run_local.ROOT", root), \
                 patch("sys.argv", ["run_local.py", "--run", "resume_case", "--config", str(config), "--resume"]), \
                 patch("research.run_local.acquire_lease") as lease, \
                 patch("research.run_local.close_interrupted_attempt") as archive, \
                 patch("research.run_local.supervised_command") as child:
                with self.assertRaisesRegex(ValueError, "trainer or precision changed"):
                    main()
            lease.assert_not_called()
            archive.assert_not_called()
            child.assert_not_called()
            self.assertEqual(receipt.read_bytes(), original)

    def test_new_sam_job_records_trainer_and_precision_and_routes_both_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"research").mkdir()
            (root/"research/campaign.json").write_text(json.dumps({
                "stops_at": "2099-01-01T00:00:00+00:00", "local_job_timeout_seconds": 10800,
                "local_gpu_seconds_per_utc_day": 21600, "baseline_benchmark": "baseline.json"}))
            config = root/"config.json"
            config.write_text(json.dumps({"train_all": False, "split_seed": 2026}))
            def mock_child(command, log, timeout, receipt_path, receipt):
                if command[1] == "research/benchmark_sam.py":
                    (receipt_path.parent/"benchmark.json").write_text(json.dumps({"promotion": {"flip": {"passed": False}}}))
            with patch("research.run_local.ROOT", root), \
                 patch("sys.argv", ["run_local.py", "--run", "sam_case", "--config", str(config),
                                    "--trainer", "sam", "--precision", "bf16"]), \
                 patch("research.run_local.supervised_command", side_effect=mock_child) as child:
                main()
            self.assertEqual([call.args[0][1] for call in child.call_args_list],
                             ["research/train_sam.py", "research/benchmark_sam.py"])
            record = json.loads((root/"results/research/sam_case/supervisor.json").read_text())
            self.assertEqual((record["trainer"], record["precision"], record["backend"]), ("sam", "bf16", "local"))
            self.assertEqual(record["status"], "complete")
            self.assertFalse((root/"results/research/local_lease.json").exists())

    def test_resume_archive_failure_leaves_original_receipt_and_releases_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"research").mkdir()
            (root/"research/campaign.json").write_text(json.dumps({"stops_at": "2099-01-01T00:00:00+00:00"}))
            config = root/"config.json"
            config.write_text(json.dumps({"train_all": False, "split_seed": 2026}))
            job = root/"results/research/resume_case"
            job.mkdir(parents=True)
            (job/"last.pt").write_bytes(b"checkpoint-presence-only")
            path = job/"supervisor.json"
            original = json.dumps({"status": "training", "config_sha256": sha256(config)}).encode()
            path.write_bytes(original)
            with patch("research.run_local.ROOT", root), patch("sys.argv", ["run_local.py", "--run", "resume_case", "--config", str(config), "--resume"]), patch("research.run_local.close_interrupted_attempt", side_effect=OSError("archive failed")):
                with self.assertRaisesRegex(OSError, "archive failed"):
                    main()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((root/"results/research/local_lease.json").exists())


if __name__ == "__main__":
    unittest.main()
