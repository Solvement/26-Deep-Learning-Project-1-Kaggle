"""CPU regressions for interrupted training, full-data refits and cache reuse.

All training examples are tiny synthetic fixtures in temporary directories.
These tests perform no downloads, GPU tensor operations or Kaggle submissions.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load process-local DLL configuration before torch on Windows.
import runtime
import numpy as np
import torch

import run_campaign
import train


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _training_manifest() -> dict:
    """A provenance fixture, never an accepted official competition dataset."""

    return {
        "competition": "synthetic-unit-test",
        "source_type": "synthetic-unit-test",
        "source_provenance": {
            "official_train_npz_sha256": "synthetic-archive-sha256",
            "ordered_training_png_sha256": "synthetic-png-sha256",
            "train_labels_csv_sha256": "synthetic-csv-sha256",
        },
        "split": {"seed": 2026, "indices_sha256": "synthetic-split-sha256"},
    }


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms, self.previous_deterministic)
        temporary = tempfile.TemporaryDirectory(prefix="cifar-training-regression-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.data_dir = self.base / "data"
        _write_json(self.data_dir / "training_manifest.json", _training_manifest())
        generator = np.random.default_rng(11)
        self.images = generator.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8)
        self.labels = np.arange(12, dtype=np.int64) % 10
        self.train_ids = np.arange(8, dtype=np.int64)
        self.validation_ids = np.arange(8, 12, dtype=np.int64)
        self.config = dict(
            train.DEFAULT_CONFIG,
            depth=10, widen_factor=1, epochs=2, warmup_epochs=0,
            batch_size=4, seed=37, cutout=8,
            mixup_alpha=0.2, mixup_probability=1.0,
        )
        arrays = (self.images, self.labels, self.train_ids, self.validation_ids)
        self.addCleanup(patch.stopall)
        patch.object(train, "load_training_arrays", return_value=arrays).start()
        # Avoid any environment query touching a real GPU on the host.
        patch.object(train, "environment", return_value={"device": "cpu", "synthetic_test": True}).start()

    def _run(self, folder: Path, *, config=None, stop_after_epoch=None, resume=False):
        folder.mkdir(exist_ok=True)
        random.seed(37)
        np.random.seed(37)
        # Seed only the CPU generator; do not initialize CUDA in a unit test.
        torch.random.default_generator.manual_seed(37)
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(
                config or self.config, folder, self.data_dir, torch.device("cpu"),
                stop_after_epoch=stop_after_epoch, resume=resume,
            )

    def _assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self._assert_same(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_model_ema_optimizer_and_all_cpu_rng(self):
        continuous = self.base / "continuous"
        resumed = self.base / "resumed"
        self._run(continuous)
        self._run(resumed, stop_after_epoch=1)
        paused = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(paused["epoch"], 1)
        self.assertEqual(paused["updates"], 2)
        self._run(resumed, resume=True)
        actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        expected = torch.load(continuous / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(actual["epoch"], 2)
        self.assertEqual(actual["updates"], 4)
        self.assertTrue(actual["optimizer"]["state"])
        for key in (
            "model", "ema", "optimizer", "python_rng", "numpy_rng", "torch_rng",
            "cuda_rng", "best_accuracy", "best_loss", "data_signature",
        ):
            with self.subTest(checkpoint_component=key):
                self._assert_same(expected[key], actual[key])
        self.assertEqual(actual["cuda_rng"], [])
        # Preserve actual source fingerprints instead of mocking provenance away.
        self.assertEqual(
            actual["provenance"]["source_sha256"],
            train.code_provenance()["source_sha256"],
        )

    def test_full_data_refit_uses_all_examples_without_holdout_evaluation(self):
        folder = self.base / "full_data"
        config = dict(self.config, train_all=True, final_variant="ema")
        with patch.object(train, "evaluate", side_effect=AssertionError("Holdout evaluated during full-data refit")) as evaluate:
            self._run(folder, config=config)
        evaluate.assert_not_called()
        metrics = json.loads((folder / "metrics.json").read_text())
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(metrics["training_examples"], 12)
        self.assertEqual(metrics["validation_examples"], 0)
        self.assertEqual(last["updates"], 6)  # 3 batches/epoch, rather than the 8-example split's 2.
        self.assertIsNone(metrics["best_validation_accuracy"])
        self.assertIsNone(metrics["best_validation_loss"])
        self.assertIsNone(checkpoint["validation_accuracy"])
        self.assertEqual(checkpoint["epoch"], config["epochs"])
        self.assertEqual(checkpoint["variant"], "ema")
        self.assertTrue(checkpoint["config"]["train_all"])
        for record in last["history"]:
            self.assertIsNone(record["validation_accuracy"])
            self.assertIsNone(record["ema_validation_accuracy"])


class CampaignCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cifar-campaign-regression-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.folder = self.base / "results" / "candidate"
        self.folder.mkdir(parents=True)
        _write_json(self.base / "data" / "training_manifest.json", _training_manifest())
        self.config = dict(train.DEFAULT_CONFIG, epochs=2, warmup_epochs=0)
        self.provenance = {"code": train.code_provenance()}
        self.checkpoint = {
            "config": self.config,
            "data_signature": {
                "split_sha256": "synthetic-split-sha256",
                "archive_sha256": "synthetic-archive-sha256",
            },
        }
        _write_json(self.folder / "config.json", self.config)
        _write_json(self.folder / "metrics.json", {
            "status": "complete", "config": self.config, "epochs_completed": 2,
        })
        _write_json(self.folder / "provenance.json", self.provenance)
        torch.save(self.checkpoint, self.folder / "best.pt")
        root_patch = patch.object(run_campaign, "ROOT", self.base)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_completed_cache_accepts_matching_artifacts_and_rejects_changed_recipe(self):
        self.assertTrue(run_campaign.verify_completed(self.folder, self.config))
        changed = dict(self.config, batch_size=self.config["batch_size"] * 2)
        with self.assertRaisesRegex(RuntimeError, "Stale completed experiment configuration"):
            run_campaign.verify_completed(self.folder, changed)

    def test_completed_cache_rejects_changed_training_source(self):
        self.provenance["code"]["source_sha256"]["train.py"] = "0" * 64
        _write_json(self.folder / "provenance.json", self.provenance)
        with self.assertRaisesRegex(RuntimeError, "Stale completed experiment source"):
            run_campaign.verify_completed(self.folder, self.config)

    def test_completed_cache_rejects_checkpoint_recipe_or_data_mismatch(self):
        variants = [
            {**self.checkpoint, "config": dict(self.config, train_all=True)},
            {**self.checkpoint, "data_signature": {
                **self.checkpoint["data_signature"], "split_sha256": "different-split",
            }},
        ]
        for checkpoint in variants:
            with self.subTest(checkpoint=checkpoint):
                torch.save(checkpoint, self.folder / "best.pt")
                with self.assertRaisesRegex(RuntimeError, "Stale completed checkpoint inputs"):
                    run_campaign.verify_completed(self.folder, self.config)

    def test_evaluation_cannot_be_reused_after_checkpoint_changes(self):
        checkpoint_path = self.folder / "best.pt"
        _write_json(self.folder / "evaluation.json", {
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        })
        self.assertTrue(run_campaign.verify_evaluation(self.folder))
        torch.save({**self.checkpoint, "epoch": 1}, checkpoint_path)
        with self.assertRaisesRegex(RuntimeError, "Stale evaluation checkpoint"):
            run_campaign.verify_evaluation(self.folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
