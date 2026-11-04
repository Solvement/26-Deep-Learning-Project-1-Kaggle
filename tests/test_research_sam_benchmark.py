"""SAM cannot reach a benchmark with one-pass, mismatched or unsafe training evidence."""
import copy
import math
import unittest

from research.benchmark_sam import validate_sam
from research.train_sam import research_provenance
from train import DEFAULT_CONFIG


class SAMBenchmarkTests(unittest.TestCase):
    def evidence(self):
        config = {**DEFAULT_CONFIG, "epochs": 6, "warmup_epochs": 1, "train_all": False, "sam_rho": .05}
        marker = research_provenance("bf16", config)
        steps = math.ceil(40000/config["batch_size"])
        history = [{"epoch": e, "successful_updates": steps, "skipped_updates": 0,
                    "sam_forward_passes": 2*steps, "sam_loss": .5, "mean_perturbation_norm": .05} for e in range(1, 7)]
        return ({"config": config, "provenance": {"research_training": marker}},
                {"research_training": copy.deepcopy(marker), "successful_updates": 6*steps,
                 "skipped_updates": 0, "sam_forward_passes": 12*steps}, history,
                {"code": {"research_training": copy.deepcopy(marker)}},
                {"sam_training_precision": "bf16", "grad_scaler_enabled": False})

    def test_complete_two_pass_evidence_accepts(self):
        evidence = validate_sam(*self.evidence())
        self.assertEqual(evidence["sam_forward_passes"], 2*evidence["successful_updates"])

    def test_one_pass_or_skipped_updates_reject(self):
        for key, value in (("sam_forward_passes", 1), ("skipped_updates", 1)):
            data = self.evidence()
            data[2][0][key] = value
            with self.assertRaisesRegex(ValueError, "two passes"):
                validate_sam(*data)

    def test_changed_method_source_rejects(self):
        data = self.evidence()
        data[0]["provenance"]["research_training"]["source_sha256"]["research/sam.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "source"):
            validate_sam(*data)

    def test_invalid_or_excessive_perturbations_reject(self):
        for value in (float("nan"), .1, -.01):
            data = self.evidence()
            data[2][0]["mean_perturbation_norm"] = value
            with self.assertRaisesRegex(ValueError, "perturbation"):
                validate_sam(*data)

    def test_scaler_or_fp16_path_rejects(self):
        data = self.evidence()
        data[4]["grad_scaler_enabled"] = True
        with self.assertRaisesRegex(ValueError, "scaler"):
            validate_sam(*data)
        data = self.evidence()
        data[0]["provenance"]["research_training"]["precision"] = "fp16"
        with self.assertRaisesRegex(ValueError, "BF16"):
            validate_sam(*data)


if __name__ == "__main__":
    unittest.main()
