"""Independent trainer source and overflow history must be valid before promotion."""
import copy
import math
import unittest

from research.benchmark_shadow import validate_shadow
from research.train_shadow import research_provenance
from train import DEFAULT_CONFIG


class ShadowBenchmarkTests(unittest.TestCase):
    def evidence(self, precision="fp16"):
        config = {**DEFAULT_CONFIG, "epochs": 6, "warmup_epochs": 1, "train_all": False}
        marker = research_provenance(precision, config)
        steps = math.ceil(40000/config["batch_size"])
        history = [{"epoch": e, "successful_updates": steps, "skipped_updates": 0,
                    "grad_scaler_scale": 65536.0} for e in range(1, 7)]
        checkpoint = {"config": config, "provenance": {"research_training": marker}}
        metrics = {"research_training": copy.deepcopy(marker), "successful_updates": 6*steps, "skipped_updates": 0}
        provenance = {"code": {"research_training": copy.deepcopy(marker)}}
        environment = {"shadow_training_precision": precision, "grad_scaler_enabled": precision == "fp16"}
        return checkpoint, metrics, history, provenance, environment

    def test_full_matched_source_and_steps_are_accepted(self):
        result = validate_shadow(*self.evidence())
        self.assertEqual(result["skipped_fraction"], 0)

    def test_changed_shadow_source_is_rejected(self):
        documents = self.evidence()
        documents[0]["provenance"]["research_training"]["source_sha256"]["research/train_shadow.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "source"):
            validate_shadow(*documents)

    def test_missing_or_fabricated_update_history_is_rejected(self):
        documents = self.evidence()
        documents[2][0]["successful_updates"] -= 1
        with self.assertRaisesRegex(ValueError, "history"):
            validate_shadow(*documents)

    def test_excessive_real_overflow_skips_are_rejected(self):
        documents = self.evidence()
        for row in documents[2]:
            row["successful_updates"] -= 10
            row["skipped_updates"] = 10
        documents[1]["successful_updates"] -= 60
        documents[1]["skipped_updates"] = 60
        with self.assertRaisesRegex(ValueError, "overflow"):
            validate_shadow(*documents)

    def test_precision_and_scaler_must_agree(self):
        documents = self.evidence()
        documents[4]["grad_scaler_enabled"] = False
        with self.assertRaisesRegex(ValueError, "GradScaler"):
            validate_shadow(*documents)


if __name__ == "__main__":
    unittest.main()
