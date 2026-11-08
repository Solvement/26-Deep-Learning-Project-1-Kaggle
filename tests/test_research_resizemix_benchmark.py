"""New augmentation cannot inherit a trusted benchmark from different training."""
import copy
import math
import unittest

from research.benchmark_resizemix import validate_resizemix_training
from research.train_resizemix import research_provenance
from train import DEFAULT_CONFIG


class ResizeMixBenchmarkTests(unittest.TestCase):
    def evidence(self):
        config = {**DEFAULT_CONFIG, "epochs": 6, "warmup_epochs": 1, "train_all": False,
                  "cutout": 0, "mixup_alpha": 0., "mixup_probability": 0.,
                  "resizemix_probability": .5, "resizemix_tau_min": .1, "resizemix_tau_max": .8}
        marker = research_provenance("fp16", config)
        steps = math.ceil(40000/config["batch_size"])
        rows = [{"epoch": i, "successful_updates": steps, "skipped_updates": 0,
                 "resizemix_batches": 100, "grad_scaler_scale": 256.} for i in range(1, 7)]
        return ({"config": config, "provenance": {"research_training": marker}},
                {"research_training": copy.deepcopy(marker), "successful_updates": 6*steps,
                 "skipped_updates": 0, "resizemix_batches": 600}, rows,
                {"code": {"research_training": copy.deepcopy(marker)}},
                {"resizemix_training_precision": "fp16", "grad_scaler_enabled": True})

    def test_complete_source_bound_augmentation_accepts(self):
        self.assertEqual(validate_resizemix_training(*self.evidence())["resizemix_batches"], 600)

    def test_changed_source_or_step_counts_reject(self):
        data = self.evidence()
        data[0]["provenance"]["research_training"]["source_sha256"]["research/resizemix.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "source"):
            validate_resizemix_training(*data)
        data = self.evidence()
        data[2][0]["successful_updates"] -= 1
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_resizemix_training(*data)

    def test_enabled_but_unused_augmentation_rejects(self):
        data = self.evidence()
        for row in data[2]: row["resizemix_batches"] = 0
        data[1]["resizemix_batches"] = 0
        with self.assertRaisesRegex(ValueError, "never ran"):
            validate_resizemix_training(*data)

    def test_scaler_or_augmented_batch_overflow_rejects(self):
        data = self.evidence()
        data[4]["grad_scaler_enabled"] = False
        with self.assertRaisesRegex(ValueError, "scaler"):
            validate_resizemix_training(*data)
        data = self.evidence()
        data[2][0]["resizemix_batches"] = 10000
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_resizemix_training(*data)


if __name__ == "__main__":
    unittest.main()
