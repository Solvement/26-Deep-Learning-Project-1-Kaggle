"""CPU contract tests. Fixture images are synthetic and are not competition results."""
import contextlib
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from cifar_project.data import CsvImages, research_split, stratified_split
from cifar_project.confirm import main as confirm
from cifar_project.model import CifarResNet
from cifar_project.predict import main as predict, validate_submission
from cifar_project.train import Config, main as train
from cifar_project.utils import choose_device


def write_csv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "train_images").mkdir()
        (self.root / "test_images").mkdir()
        rng = np.random.default_rng(1)
        rows = []
        for label in range(10):
            for index in range(4):
                name = f"{label}-{index}.png"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(self.root / "train_images" / name)
                rows.append([name, label])
        write_csv(self.root / "train_labels.csv", ["filename", "label"], rows)
        for name in ["b.png", "a.png"]:
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(self.root / "test_images" / name)
        write_csv(self.root / "test.csv", ["id", "filename"], [[9, "b.png"], [2, "a.png"]])
        write_csv(self.root / "sample_submission.csv", ["id", "label"], [[2, 0], [9, 0]])

    def test_residual_model_parameter_limit_and_gradients(self):
        torch.set_num_threads(2)
        model = CifarResNet()
        self.assertLessEqual(model.parameter_count, 5_000_000)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "limit"):
            CifarResNet(base_width=128)
        with self.assertRaisesRegex(ValueError, "32"):
            model(torch.randn(2, 3, 64, 64))

    def test_stratified_split_is_disjoint_complete_and_reproducible(self):
        labels = [label for label in range(10) for _ in range(20)]
        training, validation = stratified_split(labels, 0.1, 42)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(200)))
        self.assertEqual([labels[i] for i in validation].count(0), 2)
        self.assertEqual((training, validation), stratified_split(labels, 0.1, 42))
        self.assertNotEqual(validation, stratified_split(labels, 0.1, 43)[1])
        with self.assertRaises(ValueError):
            stratified_split([0, 0, 1, 1], 0.1, 42)

    def test_data_fingerprint_tracks_image_changes(self):
        dataset = CsvImages(self.root / "train_labels.csv", self.root / "train_images")
        before = dataset.fingerprint()
        Image.new("RGB", (32, 32), color="red").save(dataset.paths[0])
        self.assertNotEqual(before, dataset.fingerprint())
        self.assertEqual(tuple(dataset[0][0].shape), (3, 32, 32))

    def test_confirmation_is_not_queried_during_training(self):
        config = Config(base_width=4, blocks_per_stage=1, epochs=1, warmup_epochs=0,
                        batch_size=10, workers=0, validation_fraction=0.25, amp=False)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(asdict(config)))
        out = self.root / "isolated"
        seen = set()
        original = CsvImages.__getitem__
        def observed(dataset, index):
            seen.add(index)
            return original(dataset, index)
        with patch.object(CsvImages, "__getitem__", observed), contextlib.redirect_stdout(io.StringIO()):
            train(["--config", str(config_path), "--data-root", str(self.root), "--train-csv", "train_labels.csv",
                   "--output-dir", str(out), "--device", "cpu", "--smoke-test"])
        split = json.loads((out / "split.json").read_text())
        self.assertFalse(seen.intersection(split["confirmation"]))
        self.assertEqual(seen, set(split["training"]) | set(split["validation"]))
        command = ["--checkpoint", str(out / "best.pt"), "--split", str(out / "split.json"),
                   "--data-root", str(self.root), "--campaign-dir", str(self.root / "campaign"),
                   "--device", "cpu", "--allow-smoke-checkpoint"]
        with contextlib.redirect_stdout(io.StringIO()):
            result = confirm(command)
        self.assertEqual(result["metrics"]["examples"], len(split["confirmation"]))
        with self.assertRaises(FileExistsError):
            confirm(command)

    def test_research_split_keeps_all_three_partitions_disjoint(self):
        labels = [label for label in range(10) for _ in range(100)]
        split = research_split(labels, 0.1, 0.1, 42)
        self.assertEqual([len(split[k]) for k in ["training", "validation", "confirmation"]], [800, 100, 100])
        self.assertEqual(set.union(*(set(v) for v in split.values())), set(range(1000)))
        self.assertFalse(set(split["training"]) & set(split["confirmation"]))
        self.assertFalse(set(split["validation"]) & set(split["confirmation"]))

    def test_invalid_paths_labels_duplicates_and_dimensions_fail(self):
        for rows in [[["../escape.png", 0]], [["0-0.png", 10]], [["0-0.png", 0], ["0-0.png", 1]]]:
            write_csv(self.root / "bad.csv", ["filename", "label"], rows)
            with self.assertRaises(ValueError):
                CsvImages(self.root / "bad.csv", self.root / "train_images")
        dataset = CsvImages(self.root / "train_labels.csv", self.root / "train_images")
        Image.new("RGB", (16, 16)).save(dataset.paths[0])
        with self.assertRaisesRegex(ValueError, "32"):
            dataset[0]

    def test_submission_rejects_wrong_ids_order_labels_and_columns(self):
        sample = self.root / "sample_submission.csv"
        output = self.root / "candidate.csv"
        write_csv(output, ["id", "label"], [[2, 3], [9, 4]])
        self.assertEqual(validate_submission(output, sample), 2)
        for rows in [[[9, 3], [2, 4]], [[2, 0], [2, 0]], [[2, 10], [9, 0]], [[2, "1.5"], [9, 0]], [[2, 0]]]:
            write_csv(output, ["id", "label"], rows)
            with self.assertRaises(ValueError):
                validate_submission(output, sample)
        write_csv(output, ["id", "label", "extra"], [[2, 3, 1], [9, 4, 1]])
        with self.assertRaises(ValueError):
            validate_submission(output, sample)

    def test_training_resume_matches_uninterrupted_and_inference_contract(self):
        config = Config(base_width=4, blocks_per_stage=1, epochs=2, warmup_epochs=0,
                        batch_size=10, workers=0, validation_fraction=0.25, amp=False)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(asdict(config)))
        base = ["--config", str(config_path), "--data-root", str(self.root),
                "--train-csv", "train_labels.csv", "--device", "cpu", "--smoke-test"]
        full, resumed = self.root / "full", self.root / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            train(base + ["--output-dir", str(full)])
            train(base + ["--output-dir", str(resumed), "--stop-after-epoch", "1"])
            train(base + ["--output-dir", str(resumed), "--resume", str(resumed / "last.pt")])
        first = torch.load(full / "last.pt", weights_only=True)
        second = torch.load(resumed / "last.pt", weights_only=True)
        self.assertEqual(first["split_sha256"], second["split_sha256"])
        for key, tensor in first["model"].items():
            torch.testing.assert_close(tensor, second["model"][key], rtol=0, atol=0)
        command = ["--checkpoint", str(resumed / "best.pt"), "--data-root", str(self.root),
                   "--output", str(self.root / "submission.csv"), "--device", "cpu"]
        with self.assertRaisesRegex(ValueError, "Smoke"):
            predict(command)
        with contextlib.redirect_stdout(io.StringIO()):
            result = predict(command + ["--allow-smoke-checkpoint"])
        self.assertEqual(result["rows"], 2)
        self.assertTrue(result["smoke_test"])
        self.assertFalse(result["submitted_to_kaggle"])
        self.assertIsNone(result["official_test_accuracy"])
        self.assertEqual(validate_submission(self.root / "submission.csv", self.root / "sample_submission.csv"), 2)
        with self.assertRaises(FileExistsError):
            predict(command + ["--allow-smoke-checkpoint"])
        Image.new("RGB", (32, 32)).save(self.root / "train_images/0-0.png")
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            train(base + ["--output-dir", str(resumed), "--resume", str(resumed / "last.pt")])

    def test_invalid_config_and_unavailable_cuda_fail_early(self):
        with self.assertRaises(ValueError):
            Config(epochs=0)
        with self.assertRaises(ValueError):
            Config(learning_rate=float("nan"))
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                choose_device("cuda")


if __name__ == "__main__":
    unittest.main()
