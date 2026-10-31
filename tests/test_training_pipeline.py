"""CPU-only behavior checks for architecture, provenance and training helpers.

Fixtures are synthetic unit-test inputs, never competition training examples.
No network requests or GPU tensor operations are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import runtime before torch to isolate Windows DLL resolution.
import runtime
import numpy as np
import torch
from torch import nn

import data
from model import PARAMETER_LIMIT, PreActWideBlock, build_model, count_parameters
from train import DEFAULT_CONFIG, MEAN, STD, augment, normalize, save_checkpoint, schedule


class ModelBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2026)

    def test_default_budget_and_oversized_model_rejection(self):
        default = build_model()
        self.assertLessEqual(count_parameters(default), PARAMETER_LIMIT)
        self.assertEqual(count_parameters(default), count_parameters(default, trainable_only=True))
        with self.assertRaisesRegex(ValueError, "project limit"):
            build_model(depth=28, widen_factor=4)

    def test_identity_residual_preserves_values_and_input_gradient(self):
        block = PreActWideBlock(8, 8, 1, 0.0).eval()
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
        inputs = torch.randn(2, 8, 16, 16, requires_grad=True)
        outputs = block(inputs)
        torch.testing.assert_close(outputs, inputs, rtol=0, atol=0)
        outputs.sum().backward()
        torch.testing.assert_close(inputs.grad, torch.ones_like(inputs), rtol=0, atol=0)

    def test_projection_residual_and_features_receive_gradients(self):
        block = PreActWideBlock(8, 16, 2, 0.0)
        inputs = torch.randn(2, 8, 16, 16, requires_grad=True)
        outputs = block(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 16, 8, 8))
        outputs.square().mean().backward()
        for weight in (block.projection.weight, block.conv1.weight, block.conv2.weight):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all().item())
            self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_cpu_training_step_and_checkpoint_preserve_inference(self):
        model = build_model(depth=10, widen_factor=1).cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        inputs = torch.rand(4, 3, 32, 32)
        targets = torch.tensor([0, 2, 5, 9])
        original_stem = model.stem.weight.detach().clone()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        optimizer.step()
        self.assertFalse(torch.equal(original_stem, model.stem.weight))
        model.eval()
        with torch.no_grad():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "best.pt"
            save_checkpoint(checkpoint_path, {
                "model": model.state_dict(), "model_config": model.export_config(),
                "optimizer": optimizer.state_dict(),
            })
            self.assertTrue(checkpoint_path.is_file())
            self.assertFalse(checkpoint_path.with_suffix(".tmp").exists())
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            restored = build_model(saved["model_config"]).cpu().eval()
            restored.load_state_dict(saved["model"])
            self.assertTrue(saved["optimizer"]["state"])
            with torch.no_grad():
                actual = restored(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TrainingTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_normalization_matches_known_channel_values(self):
        pixels = torch.tensor([0, 128, 255], dtype=torch.uint8).view(1, 1, 1, 3).expand(2, 32, 32, 3)
        normalized = normalize(pixels)
        expected = (torch.tensor([0.0, 128 / 255, 1.0]) - torch.tensor(MEAN)) / torch.tensor(STD)
        self.assertEqual(tuple(normalized.shape), (2, 3, 32, 32))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertTrue(normalized.is_contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(normalized.mean(dim=(0, 2, 3)), expected)
        for channel in range(3):
            torch.testing.assert_close(normalized[:, channel], torch.full((2, 32, 32), expected[channel]))

    def test_crop_flip_cutout_are_seeded_and_keep_expected_shape(self):
        pixels = (torch.arange(4 * 32 * 32 * 3) % 256).to(torch.uint8).view(4, 32, 32, 3)
        torch.manual_seed(37)
        first = augment(pixels, cutout=16)
        torch.manual_seed(37)
        repeated = augment(pixels, cutout=16)
        torch.manual_seed(38)
        different = augment(pixels, cutout=16)
        self.assertEqual(tuple(first.shape), (4, 3, 32, 32))
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertTrue(first.is_contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, different))
        self.assertTrue((first == 0).all(dim=1).flatten(1).any(dim=1).all().item())

    def test_constant_images_without_cutout_preserve_color(self):
        pixels = torch.full((3, 32, 32, 3), 128, dtype=torch.uint8)
        torch.testing.assert_close(augment(pixels, cutout=0), normalize(pixels), rtol=0, atol=0)

    def test_warmup_and_cosine_have_correct_endpoints_and_direction(self):
        config = {**DEFAULT_CONFIG, "lr": 0.1, "min_lr": 0.001, "warmup_epochs": 2}
        rates = [schedule(step, 30, 5, config) for step in range(30)]
        self.assertAlmostEqual(rates[0], 0.01)
        self.assertAlmostEqual(rates[9], 0.1)
        self.assertAlmostEqual(rates[10], 0.1)
        self.assertAlmostEqual(rates[-1], 0.001)
        self.assertTrue(all(a <= b for a, b in zip(rates[:9], rates[1:10])))
        self.assertTrue(all(a >= b for a, b in zip(rates[10:-1], rates[11:])))
        config["warmup_epochs"] = 0
        self.assertAlmostEqual(schedule(0, 30, 5, config), config["lr"])
        self.assertAlmostEqual(schedule(29, 30, 5, config), config["min_lr"])


class OfficialDataProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def _export_fixture(self, **manifest_overrides):
        archive = self.folder / "official_train.npz"
        # Deliberately too small for training: unit tests must not manufacture
        # an accepted competition dataset.
        np.savez_compressed(archive,
            images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
            labels=np.array([0, 1], dtype=np.int64), ids=np.array(["0", "1"]))
        manifest = {
            "source": "official_kaggle_competition", "competition": data.COMPETITION,
            "train_csv_sha256": data.CSV_SHA256["train_labels.csv"],
            "archive_sha256": data.file_hash(archive),
            "no_external_data": True, "hidden_labels_loaded": False,
        }
        manifest.update(manifest_overrides)
        (self.folder / "official_source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_external_training_cache_is_rejected_before_reading_arrays(self):
        (self.folder / "training_manifest.json").write_text(
            json.dumps({"source": "external CIFAR archive"}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not verified official Kaggle"):
            data.load_training_arrays(self.folder)

    def test_old_two_way_official_split_requires_rebuild(self):
        (self.folder / "training_manifest.json").write_text(json.dumps({
            "source_type": data.SOURCE_TYPE, "competition": data.COMPETITION,
            "split": {"train_count": 45000, "validation_count": 5000},
        }), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "40k/5k/5k"):
            data.load_training_arrays(self.folder)

    def test_three_way_split_is_seeded_stratified_and_disjoint(self):
        labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
        labels = np.random.default_rng(71).permutation(labels)
        partitions = data._stratified_split(labels, 2026)
        repeated = data._stratified_split(labels, 2026)
        self.assertEqual(tuple(len(part) for part in partitions), (40000, 5000, 5000))
        np.testing.assert_array_equal(np.sort(np.concatenate(partitions)), np.arange(50000))
        for part, same, per_class in zip(partitions, repeated, (4000, 500, 500)):
            np.testing.assert_array_equal(part, same)
            np.testing.assert_array_equal(np.bincount(labels[part], minlength=10), np.full(10, per_class))
        self.assertEqual(len(np.intersect1d(partitions[1], partitions[2])), 0)

    def test_csv_content_must_match_official_audit(self):
        (self.folder / "train_labels.csv").write_text("id,filename,label\n0,train_0000.png,0\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "CSV SHA-256 mismatch"):
            data._official_csv(self.folder, "train_labels.csv", ["id", "filename", "label"])

    def test_export_rejects_external_source_wrong_competition_or_hidden_labels(self):
        scenarios = [
            ({"source": "external_dataset"}, "not from this official competition"),
            ({"competition": "different_competition"}, "not from this official competition"),
            ({"hidden_labels_loaded": True}, "no hidden labels"),
            ({"no_external_data": False}, "official-only inputs"),
        ]
        for overrides, message in scenarios:
            with self.subTest(overrides=overrides):
                self._export_fixture(**overrides)
                with self.assertRaisesRegex(RuntimeError, message):
                    data._load_official_training(self.folder)

    def test_export_rejects_wrong_label_csv_or_npz_fingerprint(self):
        for field, message in [
            ("train_csv_sha256", "audited official train_labels.csv"),
            ("archive_sha256", "NPZ SHA-256 mismatch"),
        ]:
            with self.subTest(field=field):
                self._export_fixture(**{field: "0" * 64})
                with self.assertRaisesRegex(RuntimeError, message):
                    data._load_official_training(self.folder)

    def test_export_rejects_validly_hashed_but_incomplete_training_arrays(self):
        self._export_fixture()
        with self.assertRaisesRegex(RuntimeError, "exactly 50,000"):
            data._load_official_training(self.folder)

    def test_export_rejects_missing_ids_even_with_matching_archive_hash(self):
        self._export_fixture()
        archive = self.folder / "official_train.npz"
        np.savez_compressed(archive, images=np.zeros((2, 32, 32, 3), dtype=np.uint8), labels=np.array([0, 1]))
        source_path = self.folder / "official_source_manifest.json"
        manifest = json.loads(source_path.read_text())
        manifest["archive_sha256"] = data.file_hash(archive)
        source_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "images, labels and ids"):
            data._load_official_training(self.folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
