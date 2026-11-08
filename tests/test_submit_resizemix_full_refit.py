import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from research import submit_resizemix_full_refit as uploader


class ResizeMixRefitUploadTests(unittest.TestCase):
    def test_rejects_existing_delivery_schema_before_typed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delivery.json"
            path.write_text(json.dumps({"schema": "research_prediction_delivery_v2"}))
            with self.assertRaisesRegex(ValueError, "explicit ResizeMix"):
                uploader.validate_manifest(path)

    def test_rejects_nonterminal_or_nonema_result(self):
        valid = {
            "competition": uploader.shared.COMPETITION,
            "training_examples": 50000,
            "validation_examples": 0,
            "development_accuracy": None,
            "independent_development_evaluated": False,
            "seed": 2026,
            "parameter_count": 4298970,
            "policy": "center80_shift20",
            "epochs_completed": 160,
            "checkpoint_epoch": 160,
            "weight_variant": "ema",
            "training_precision": "fp16",
        }
        module_name = "research.resizemix_full_refit_delivery"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delivery.json"
            path.write_text(json.dumps({"schema": uploader.SCHEMA}))
            for change in ({"checkpoint_epoch": 150}, {"weight_variant": "model"}, {"validation_examples": 5000}):
                fake = types.SimpleNamespace(validate_refit_delivery=lambda path, root: {**valid, **change})
                with self.subTest(change=change), patch.dict("sys.modules", {module_name: fake}):
                    with self.assertRaisesRegex(ValueError, "fixed epoch-160 EMA"):
                        uploader.validate_manifest(path)


if __name__ == "__main__":
    unittest.main()
