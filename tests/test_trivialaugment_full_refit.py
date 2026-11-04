from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from research import run_trivialaugment_full_refit as controller
from research import submit_trivialaugment_full_refit as uploader
from research import train_trivialaugment_full_refit as engine


class TrivialAugmentFullRefitTests(unittest.TestCase):
    def test_exact_full_data_recipe_keeps_terminal_ema(self):
        config = engine.fixed_config()
        engine.validate_config(config)
        for overrides in ({"train_all": False}, {"seed": 2027}, {"final_variant": "model"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                engine.validate_config({**config, **overrides})

    def test_guard_preserves_job_window_and_daily_limits(self):
        policy = json.loads(controller.POLICY_PATH.read_text(encoding="utf-8"))
        begin, stop = controller.START, controller.STOP
        measured = {"median_step_seconds": .1}
        self.assertEqual(controller.guard_seconds(measured), 9684)
        self.assertEqual(controller.budget_seconds(policy, begin, 0, measured), 10800)
        for when, used, proof in (
            (begin - timedelta(seconds=1), 0, measured),
            (stop - timedelta(seconds=9000), 0, measured),
            (begin, 12000, measured),
            (begin, 0, {"median_step_seconds": .12}),
        ):
            with self.subTest(when=when, used=used, proof=proof), self.assertRaises(ValueError):
                controller.budget_seconds(policy, when, used, proof)

    def test_uploader_rejects_legacy_ta_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "delivery.json"
            old.write_text('{"schema":"research_trivialaugment_prediction_delivery_v1"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                uploader.validate_manifest(old)


if __name__ == "__main__":
    unittest.main()
