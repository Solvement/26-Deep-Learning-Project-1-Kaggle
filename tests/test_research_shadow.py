"""CPU checks for shadow precision, updates, resume state and partition bounds."""
import copy
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from research.train_shadow import (capture_rng, optimizer_step, research_provenance, resolve_precision,
                                   restore_rng, train, validate_config, validate_partition, validate_resume)
from train import DEFAULT_CONFIG


class ResearchShadowTests(unittest.TestCase):
    def test_cpu_fp32_updates_parameters_and_disables_scaler(self):
        precision, dtype = resolve_precision("auto", torch.device("cpu"))
        self.assertEqual((precision, dtype), ("fp32", torch.float32))
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.assertFalse(scaler.is_enabled())
        self.assertTrue(optimizer_step(parameter.square().sum(), optimizer, scaler))
        self.assertTrue(torch.allclose(parameter, torch.tensor([.8])))
        with self.assertRaisesRegex(ValueError, "CPU shadow"):
            resolve_precision("fp16", torch.device("cpu"))

    def test_t4_bf16_rejected_before_real_data_loading(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(resolve_precision("auto", torch.device("cuda")), ("fp16", torch.float16))
            with self.assertRaisesRegex(RuntimeError, "Native BF16"):
                resolve_precision("bf16", torch.device("cuda"))

    def test_nonfinite_loss_and_unscaled_gradient_never_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            optimizer_step(parameter.sum()*float("nan"), optimizer, scaler)
        # Finite loss with an infinite derivative must fail before optimizer.step.
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            optimizer_step((parameter-1).sqrt().sum(), optimizer, scaler)
        self.assertEqual(parameter.item(), 1.0)

    def test_overflow_skip_is_not_counted_as_successful_update(self):
        class OverflowScaler:
            scale_value = 1024.0
            def get_scale(self): return self.scale_value
            def is_enabled(self): return True
            def scale(self, value): return value
            def step(self, optimizer): pass  # mimic GradScaler suppressing step
            def update(self): self.scale_value /= 2
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        self.assertFalse(optimizer_step(parameter.square().sum(), optimizer, OverflowScaler()))
        self.assertEqual(parameter.item(), 1.0)

    def test_forbidden_full_data_and_split_changes_rejected(self):
        validate_config(DEFAULT_CONFIG.copy())
        for changed in ({"train_all": True}, {"split_seed": 2027}, {"seed": True}, {"typo": .1}):
            with self.assertRaises(ValueError):
                validate_config({**DEFAULT_CONFIG, **changed})
        train, dev = np.arange(40000), np.arange(40000, 45000)
        validate_partition(train, dev, {"split": {"seed": 2026}}, DEFAULT_CONFIG)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_partition(train, np.arange(5000), {"split": {"seed": 2026}}, DEFAULT_CONFIG)

    def test_cutmix_configuration_replaces_mixup_and_cutout_without_changing_legacy_keys(self):
        legacy = DEFAULT_CONFIG.copy()
        validate_config(legacy)
        self.assertEqual(set(legacy), set(DEFAULT_CONFIG))
        config = {**legacy, "cutmix_alpha": 1.0, "cutmix_probability": .5,
                  "mixup_alpha": 0.0, "mixup_probability": 0.0, "cutout": 0}
        validate_config(config)
        for changes in ({"cutout": 16}, {"mixup_alpha": .2}, {"cutmix_alpha": 0},
                        {"cutmix_probability": float("inf")}, {"cutmix_probability": True}):
            with self.assertRaises(ValueError):
                validate_config({**config, **changes})
        with self.assertRaises(ValueError):
            validate_config({**legacy, "cutmix_alpha": 1.0})
        marker = research_provenance("fp16", config)
        self.assertEqual(set(marker["source_sha256"]), {"research/train_shadow.py", "research/augmentations.py"})

    def test_resume_restores_all_cpu_random_streams(self):
        random.seed(2718); np.random.seed(2718); torch.manual_seed(2718)
        state = capture_rng(torch.device("cpu"))
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        restore_rng(state, torch.device("cpu"))
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_resume_binds_shadow_source_precision_and_training_seed(self):
        config = DEFAULT_CONFIG.copy()
        research = research_provenance("fp16", config)
        self.assertEqual(research["training_seed"], 2026)
        self.assertFalse(research["confirmation_evaluated"])
        provenance = {"source_sha256": {"train.py": "frozen"}, "research_training": research}
        last = {"config": config, "data_signature": {"split": "a"}, "provenance": provenance, "grad_scaler": {}}
        validate_resume(last, config, {"split": "a"}, provenance)
        for key, value in (("precision", "bf16"), ("training_seed", 2027), ("source_sha256", {"shadow": "changed"})):
            changed = copy.deepcopy(provenance)
            changed["research_training"][key] = value
            with self.assertRaisesRegex(ValueError, "source, precision and seeds"):
                validate_resume(last, config, {"split": "a"}, changed)

    def test_actual_cpu_training_resume_matches_uninterrupted_parameters(self):
        # Only this unit fixture replaces the strict official loader/partition.
        # The public CLI always uses the unpatched 40k/5k validation above.
        config = {**DEFAULT_CONFIG, "depth": 10, "widen_factor": 1, "dropout": .2,
                  "epochs": 2, "warmup_epochs": 0, "batch_size": 4}
        rng = np.random.default_rng(3)
        arrays = (rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8),
                  np.arange(12, dtype=np.int64) % 10, np.arange(8), np.arange(8, 12))
        variants = {"mixup": config, "cutmix": {**config, "cutout": 0, "mixup_alpha": 0.0,
                    "mixup_probability": 0.0, "cutmix_alpha": 1.0, "cutmix_probability": 1.0}}
        for mode, fixture_config in variants.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory/"training_manifest.json").write_text(json.dumps({"split": {"seed": 2026, "indices_sha256": "synthetic"}}))
                with patch("research.train_shadow.load_training_arrays", return_value=arrays), \
                        patch("research.train_shadow.training_source", return_value={"archive_sha256": "synthetic"}), \
                        patch("research.train_shadow.validate_partition"), contextlib.redirect_stdout(io.StringIO()):
                    train(fixture_config, directory/"whole", directory, torch.device("cpu"), "fp32")
                    train(fixture_config, directory/"resumed", directory, torch.device("cpu"), "fp32", stop_after_epoch=1)
                    train(fixture_config, directory/"resumed", directory, torch.device("cpu"), "fp32", resume=True)
                whole = torch.load(directory/"whole"/"last.pt", weights_only=False)
                resumed = torch.load(directory/"resumed"/"last.pt", weights_only=False)
                self.assertEqual(whole["updates"], resumed["updates"])
                self.assertEqual(whole["grad_scaler"], resumed["grad_scaler"])
                self.assertEqual(resumed["epoch"], 2)
                for variant in ("model", "ema"):
                    for key, value in whole[variant].items():
                        self.assertTrue(torch.equal(value, resumed[variant][key]), f"{variant}.{key}")
                if mode == "cutmix":
                    self.assertEqual([record["cutmix_batches"] for record in resumed["history"]], [2, 2])


if __name__ == "__main__":
    unittest.main()
