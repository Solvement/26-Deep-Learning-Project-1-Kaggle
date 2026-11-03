import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from research import submit_sam_full_refit as uploader


class SAMRefitUploadTests(unittest.TestCase):
    def test_rejects_existing_delivery_schema_before_typed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delivery.json"
            path.write_text(json.dumps({"schema": "research_prediction_delivery_v2"}))
            with self.assertRaisesRegex(ValueError, "explicit SAM"):
                uploader.validate_manifest(path)

    def test_rejects_nonterminal_or_nononline_result(self):
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
            "weight_variant": "model",
            "training_precision": "bf16",
        }
        module_name = "research.sam_full_refit_delivery"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delivery.json"
            path.write_text(json.dumps({"schema": uploader.SCHEMA}))
            for change in ({"checkpoint_epoch": 150}, {"weight_variant": "ema"}, {"validation_examples": 5000}):
                fake = types.SimpleNamespace(validate_refit_delivery=lambda path, root: {**valid, **change})
                with self.subTest(change=change), patch.dict("sys.modules", {module_name: fake}):
                    with self.assertRaisesRegex(ValueError, "fixed epoch-160 online model"):
                        uploader.validate_manifest(path)


    def test_status_bearing_delivery_yields_ledger_safe_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delivery.json"
            path.write_text(json.dumps({"schema": uploader.SCHEMA}))
            digest = "a" * 64
            document = {
                "status": "complete", "submitted": False, "submission_ref": None,
                "competition": uploader.shared.COMPETITION, "training_examples": 50000,
                "validation_examples": 0, "development_accuracy": None,
                "independent_development_evaluated": False, "seed": 2026,
                "parameter_count": 4298970, "policy": "center80_shift20",
                "epochs_completed": 160, "checkpoint_epoch": 160, "weight_variant": "model",
                "training_precision": "bf16", "run": "sam_refit_test",
                "csv_path": str(path.parent / "submission.csv"),
                **dict.fromkeys(("submission_sha256", "checkpoint_sha256", "manifest_sha256",
                                 "frozen_manifest_sha256", "parent_delivery_sha256"), digest),
            }
            fake = types.SimpleNamespace(validate_refit_delivery=lambda path, root: document)
            with (patch.dict("sys.modules", {"research.sam_full_refit_delivery": fake}),
                  patch.object(uploader.shared, "validate_csv", return_value=digest),
                  patch.object(uploader.shared, "sha256", return_value=digest)):
                validated = uploader.validate_manifest(path)
            intent = {**validated, "status": "upload_attempt_started", "submission_ref": None}
            self.assertNotIn("status", validated)
            self.assertNotIn("submission_ref", validated)
            self.assertTrue(all(intent[key] == value for key, value in validated.items()))


if __name__ == "__main__":
    unittest.main()
