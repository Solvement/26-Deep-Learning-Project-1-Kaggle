"""Interrupted jobs must resume without overlapping a live owner or erasing budget."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from research.run_local import assert_stopped, close_interrupted_attempt, used_seconds, main, sha256


class ResearchSupervisorTests(unittest.TestCase):
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
