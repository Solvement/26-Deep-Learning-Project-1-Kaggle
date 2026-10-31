"""Safety boundaries for exploratory development benchmarks (synthetic inputs)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from research.benchmark import promotion_gate, stress, validate_checkpoint, recalibrate_training_bn
from train import normalize
from scripts.submit_verified import ALLOWED_RUNS as UPLOAD_RUNS, validate_run
from scripts.make_benchmark_submissions import ALLOWED_RUNS as PREDICTION_RUNS


class ResearchBenchmarkTests(unittest.TestCase):
    def test_full_data_models_cannot_claim_held_out_development(self):
        with self.assertRaisesRegex(ValueError, "full-data"):
            validate_checkpoint({"config": {"train_all": True}}, [], [], {})

    def test_overlap_and_fingerprint_mismatch_are_rejected(self):
        train = np.arange(40000)
        dev = np.arange(40000, 45000)
        checkpoint = {"config": {"train_all": False}, "data_signature": {"split": "a"}}
        validate_checkpoint(checkpoint, train, dev, {"split": "a"})
        with self.assertRaisesRegex(ValueError, "source/split"):
            validate_checkpoint(checkpoint, train, dev, {"split": "b"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_checkpoint(checkpoint, train, np.arange(5000), {"split": "a"})

    def test_noise_is_paired_and_clean_is_exact(self):
        images = torch.arange(2*3*32*32).remainder(256).to(torch.uint8).reshape(2, 32, 32, 3)
        self.assertTrue(torch.equal(stress(images, "clean"), images))
        self.assertTrue(torch.equal(stress(images, "noise_004", 128), stress(images, "noise_004", 128)))
        self.assertFalse(torch.equal(stress(images, "noise_004", 0), stress(images, "noise_004", 128)))

    def test_gate_rejects_clean_collapse_and_unchanged_candidate(self):
        baseline = {"clean_accuracy": .96, "stress_accuracy": .8}
        self.assertFalse(promotion_gate(baseline, baseline)["passed"])
        self.assertFalse(promotion_gate({"clean_accuracy": .95, "stress_accuracy": .9}, baseline)["passed"])
        self.assertTrue(promotion_gate({"clean_accuracy": .961, "stress_accuracy": .8}, baseline)["passed"])
        self.assertTrue(promotion_gate({"clean_accuracy": .959, "stress_accuracy": .806}, baseline)["passed"])

    def test_bn_recalibration_weights_tail_by_sample_count(self):
        images = torch.zeros((3, 32, 32, 3), dtype=torch.uint8)
        images[-1] = 255
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3)).eval()
        calibrated = recalibrate_training_bn(model, images, batch_size=2)
        expected = normalize(images).mean(dim=(0, 2, 3))
        self.assertTrue(torch.allclose(calibrated[0].running_mean, expected, atol=1e-5))
        self.assertTrue(torch.equal(model[0].running_mean, torch.zeros(3)))

    def test_stability_repeat_is_excluded_before_any_file_access(self):
        self.assertNotIn("stability_seed2027", UPLOAD_RUNS)
        self.assertNotIn("stability_seed2027", PREDICTION_RUNS)
        with self.assertRaisesRegex(ValueError, "not authorized"):
            validate_run("stability_seed2027", Path("never-created.csv"))


if __name__ == "__main__":
    unittest.main()
