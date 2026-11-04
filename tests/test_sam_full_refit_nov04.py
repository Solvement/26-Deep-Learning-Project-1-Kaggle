from datetime import timedelta
import json
import unittest

from research import run_sam_full_refit as previous
from research import run_sam_full_refit_nov04 as current


class SamNov04AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(current.POLICY_PATH.read_text(encoding="utf-8"))

    def test_admission_requires_current_utc_window(self):
        self.assertNotEqual(current.SCHEMA, previous.SCHEMA)
        self.assertNotEqual(current.INPUT_SCHEMA, previous.INPUT_SCHEMA)
        self.assertEqual(current.budget_seconds(self.policy, current.START, 0), 180)
        for outside in (current.START - timedelta(seconds=1), current.STOP):
            with self.subTest(outside=outside), self.assertRaises(ValueError):
                current.budget_seconds(self.policy, outside, 0)

    def test_training_applies_same_guard_to_all_three_limits(self):
        report = {"median_step_seconds": .1085018}
        self.assertEqual(current.guard_seconds(report), previous.guard_seconds(report))
        self.assertEqual(current.budget_seconds(self.policy, current.START, 0, report), 10800)
        for when, used, measurement in (
            (current.STOP - timedelta(seconds=9000), 0, report),
            (current.START, 12000, report),
            (current.START, 0, {"median_step_seconds": .12}),
        ):
            with self.subTest(when=when, used=used, measurement=measurement):
                with self.assertRaises(ValueError):
                    current.budget_seconds(self.policy, when, used, measurement)


if __name__ == "__main__":
    unittest.main()
