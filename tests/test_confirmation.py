"""CPU guards for one-time confirmation receipts; no real competition inputs."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime  # Configure Windows DLL resolution before torch.
import numpy as np
import torch

import confirm


class ConstantClassifier(torch.nn.Module):
    """Exercise the real CPU evaluator with predictable aggregate outcomes."""

    def forward(self, images):
        return images.new_zeros((len(images), 10))


class ConfirmationReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="confirmation-regression-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.checkpoint_path = self.base / "best.pt"
        self.checkpoint_path.write_bytes(b"synthetic checkpoint bytes, never model weights")
        self.selection_path = self.base / "selection.json"
        self.output_path = self.base / "confirmation.json"
        self.data_dir = self.base / "data"
        self.data_dir.mkdir()
        self.split_hash = "a" * 64
        self._write_manifest(self.split_hash)
        self.selection = {
            "run": "regularized",
            "flip_tta": True,
            "checkpoint_sha256": hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
        }
        self._write(self.selection_path, self.selection)
        self.checkpoint = {
            "config": {"train_all": False},
            "data_signature": {"split_sha256": self.split_hash},
            "model_config": {}, "model": {},
        }
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.int64)
        empty = np.array([], dtype=np.int64)
        self.seed = self._patch(confirm, "seed_all")
        self.load_arrays = self._patch(confirm, "load_training_arrays", return_value=(images, labels, empty, empty))
        self.load_confirmation = self._patch(confirm, "load_confirmation_indices", return_value=np.array([0, 1]))
        self.load_checkpoint = self._patch(confirm.torch, "load", return_value=self.checkpoint)
        self.build_model = self._patch(confirm, "build_model", side_effect=lambda _: ConstantClassifier())
        self.evaluate = self._patch(confirm, "evaluate", wraps=confirm.evaluate)

    def _patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _write_manifest(self, split_hash):
        self._write(self.data_dir / "training_manifest.json", {"split": {"indices_sha256": split_hash}})

    def _main(self):
        arguments = [
            "confirm.py", "--checkpoint", str(self.checkpoint_path),
            "--selection", str(self.selection_path), "--output", str(self.output_path),
            "--data-dir", str(self.data_dir), "--device", "cpu",
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            confirm.main()

    def _fresh_receipt(self):
        self._main()
        receipt = json.loads(self.output_path.read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["examples"], 2)
        self.assertEqual(receipt["accuracy"], 0.5)
        self.assertEqual(sum(map(sum, receipt["confusion_matrix"])), 2)
        self.assertEqual(receipt["selection_file_sha256"], hashlib.sha256(self.selection_path.read_bytes()).hexdigest())
        self.assertEqual(receipt["confirmation_source_sha256"], hashlib.sha256(Path(confirm.__file__).read_bytes()).hexdigest())
        self.assertEqual(receipt["split_sha256"], self.split_hash)
        self.evaluate.assert_called_once()
        for mocked in (self.load_arrays, self.load_confirmation, self.load_checkpoint, self.build_model, self.evaluate):
            mocked.reset_mock()
        return receipt

    def _assert_cache_not_reevaluated(self):
        for mocked in (self.load_arrays, self.load_confirmation, self.load_checkpoint, self.build_model, self.evaluate):
            mocked.assert_not_called()

    def test_identical_completed_receipt_is_reused_without_another_evaluation(self):
        receipt = self._fresh_receipt()
        original_bytes = self.output_path.read_bytes()
        self._main()
        self.assertEqual(self.output_path.read_bytes(), original_bytes)
        self.assertEqual(json.loads(self.output_path.read_text()), receipt)
        self._assert_cache_not_reevaluated()

    def test_cached_confirmation_rejects_changed_selection_file_hash(self):
        self._fresh_receipt()
        self._write(self.selection_path, {**self.selection, "audit_note": "selection changed"})
        with self.assertRaisesRegex(RuntimeError, "different frozen selection"):
            self._main()
        self._assert_cache_not_reevaluated()

    def test_cached_confirmation_rejects_changed_flip_even_if_hash_was_updated(self):
        receipt = self._fresh_receipt()
        self._write(self.selection_path, {**self.selection, "flip_tta": False})
        # Isolate the explicit policy guard from the selection-file hash guard.
        receipt["selection_file_sha256"] = hashlib.sha256(self.selection_path.read_bytes()).hexdigest()
        self._write(self.output_path, receipt)
        with self.assertRaisesRegex(RuntimeError, "different frozen selection"):
            self._main()
        self._assert_cache_not_reevaluated()

    def test_cached_confirmation_rejects_changed_run_even_if_hash_was_updated(self):
        receipt = self._fresh_receipt()
        self._write(self.selection_path, {**self.selection, "run": "architecture"})
        receipt["selection_file_sha256"] = hashlib.sha256(self.selection_path.read_bytes()).hexdigest()
        self._write(self.output_path, receipt)
        with self.assertRaisesRegex(RuntimeError, "different frozen selection"):
            self._main()
        self._assert_cache_not_reevaluated()

    def test_cached_confirmation_rejects_a_changed_split_before_reuse(self):
        self._fresh_receipt()
        self._write_manifest("b" * 64)
        with self.assertRaisesRegex(RuntimeError, "different data split"):
            self._main()
        self._assert_cache_not_reevaluated()

    def test_full_data_checkpoint_is_rejected_without_evaluation_or_receipt(self):
        self.checkpoint["config"]["train_all"] = True
        with self.assertRaisesRegex(ValueError, "full-data model cannot be evaluated"):
            self._main()
        self.evaluate.assert_not_called()
        self.build_model.assert_not_called()
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
