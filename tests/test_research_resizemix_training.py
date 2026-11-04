"""CPU-only ResizeMix trainer source/precision/recovery checks on synthetic data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from research import train_resizemix as recipe
from train import DEFAULT_CONFIG


class ResizeMixTrainingTests(unittest.TestCase):
    def config(self, **changes):
        return {**DEFAULT_CONFIG, "cutout": 0, "mixup_alpha": 0., "mixup_probability": 0.,
                "resizemix_probability": .5, "resizemix_tau_min": .1, "resizemix_tau_max": .8, **changes}

    def arrays(self):
        rng = np.random.default_rng(43)
        return (rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8),
                np.arange(12, dtype=np.int64) % 10, np.arange(8), np.arange(8, 12))

    def manifest(self, directory):
        (directory/"training_manifest.json").write_text(json.dumps({"split": {"seed": 2026, "indices_sha256": "synthetic"}}))

    def test_controlled_configuration_and_forbidden_inputs_reject_before_loader(self):
        saved = {**DEFAULT_CONFIG, **json.loads((ROOT/"research/configs/wrn22_resizemix.json").read_text())}
        self.assertEqual(saved, self.config())
        recipe.validate_config(saved)
        for changes in ({"train_all": True}, {"split_seed": 2027}, {"cutout": 16}, {"mixup_alpha": .2},
                        {"mixup_probability": .5}, {"resizemix_tau_min": float("nan")}, {"cutmix_alpha": 1.},
                        {"resizemix_tau_min": .9, "resizemix_tau_max": .8}):
            with self.subTest(changes=changes), patch.object(recipe, "load_training_arrays", side_effect=AssertionError("no official loader")):
                with self.assertRaises(ValueError):
                    recipe.train(self.config(**changes), Path("unused"), Path("unused"), torch.device("cpu"), "fp32")

    def test_native_t4_precision_and_source_contract(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=False) as supported:
            self.assertEqual(recipe.resolve_precision("auto", torch.device("cuda")), ("fp16", torch.float16))
            supported.assert_called_once_with(including_emulation=False)
            with self.assertRaisesRegex(RuntimeError, "Native BF16"):
                recipe.resolve_precision("bf16", torch.device("cuda"))
        marker = recipe.research_provenance("fp16", self.config())
        self.assertTrue(marker["grad_scaler_enabled"])
        self.assertEqual(set(marker["source_sha256"]), set(recipe.SOURCE_KEYS))
        self.assertEqual(marker["method"], "resizemix")
        self.assertEqual(marker["tau_sampling"], "uniform_side_ratio")
        self.assertFalse(marker["confirmation_evaluated"])
        self.assertFalse(marker["hidden_images_loaded"])

    def test_cpu_interruption_restores_model_ema_optimizer_scaler_and_all_random_streams(self):
        config = self.config(depth=10, widen_factor=1, dropout=.2, epochs=2, warmup_epochs=0, batch_size=4,
                             resizemix_probability=1.)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.manifest(directory)
            with patch.object(recipe, "load_training_arrays", return_value=self.arrays()), \
                    patch.object(recipe, "training_source", return_value={"archive_sha256": "synthetic"}), \
                    patch.object(recipe, "validate_partition"), contextlib.redirect_stdout(io.StringIO()):
                recipe.train(config, directory/"whole", directory, torch.device("cpu"), "fp32")
                recipe.train(config, directory/"resumed", directory, torch.device("cpu"), "fp32", stop_after_epoch=1)
                recipe.train(config, directory/"resumed", directory, torch.device("cpu"), "fp32", resume=True)
            whole = torch.load(directory/"whole/last.pt", weights_only=False)
            resumed = torch.load(directory/"resumed/last.pt", weights_only=False)
            self.assertEqual((whole["updates"], resumed["updates"]), (4, 4))
            self.assertEqual(whole["grad_scaler"], resumed["grad_scaler"])
            for variant in ("model", "ema"):
                for name, value in whole[variant].items():
                    self.assertTrue(torch.equal(value, resumed[variant][name]), f"{variant}.{name}")
            self.assertEqual(whole["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
            for key, value in whole["optimizer"]["state"].items():
                self.assertTrue(torch.equal(value["momentum_buffer"], resumed["optimizer"]["state"][key]["momentum_buffer"]))
            self.assertEqual(whole["python_rng"], resumed["python_rng"])
            self.assertTrue(np.array_equal(whole["numpy_rng"][1], resumed["numpy_rng"][1]))
            self.assertTrue(torch.equal(whole["torch_rng"], resumed["torch_rng"]))
            metrics = json.loads((directory/"resumed/metrics.json").read_text())
            self.assertEqual(metrics["status"], "complete")
            self.assertEqual(metrics["resizemix_batches"], 4)
            for row in resumed["history"]:
                self.assertEqual((row["successful_updates"], row["skipped_updates"], row["resizemix_batches"]), (2, 0, 2))
                self.assertGreater(row["grad_scaler_scale"], 0)
            env = json.loads((directory/"resumed/environment.json").read_text())
            self.assertEqual(env["resizemix_training_precision"], "fp32")
            self.assertEqual(env["research_training"], metrics["research_training"])
            self.assertFalse(env["grad_scaler_enabled"])

    def test_skipped_update_does_not_advance_ema_decay_or_success_counter(self):
        config = self.config(depth=10, widen_factor=1, epochs=1, warmup_epochs=0, batch_size=4)
        recipe.seed_all(config["seed"])
        initial = recipe.build_model({key: config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")})
        initial_probe = next(initial.parameters()).detach().clone()
        real_step = recipe.optimizer_step
        count = 0
        def first_skip(loss, optimizer, scaler):
            nonlocal count
            count += 1
            return False if count == 1 else real_step(loss, optimizer, scaler)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.manifest(directory)
            with patch.object(recipe, "load_training_arrays", return_value=self.arrays()), \
                    patch.object(recipe, "training_source", return_value={"archive_sha256": "synthetic"}), \
                    patch.object(recipe, "validate_partition"), patch.object(recipe, "optimizer_step", side_effect=first_skip), \
                    contextlib.redirect_stdout(io.StringIO()):
                recipe.train(config, directory/"run", directory, torch.device("cpu"), "fp32")
            last = torch.load(directory/"run/last.pt", weights_only=False)
            self.assertEqual((last["updates"], last["skipped_updates"]), (1, 1))
            probe_key = next(iter(initial.state_dict()))
            expected = torch.lerp(initial_probe, last["model"][probe_key], 1-2/11)
            self.assertTrue(torch.equal(expected, last["ema"][probe_key]))

    def test_resume_rejects_changed_helper_precision_bounds_or_update_totals(self):
        config = self.config(epochs=3, warmup_epochs=0)
        provenance = {"source_sha256": {"train.py": "frozen"}, "research_training": recipe.research_provenance("fp16", config)}
        last = {"config": config, "data_signature": {"split": "official"}, "provenance": provenance, "epoch": 1,
                "history": [{"epoch": 1, "successful_updates": 2, "skipped_updates": 0,
                             "resizemix_batches": 1, "grad_scaler_scale": 1024.}],
                "grad_scaler": {}, "updates": 2, "skipped_updates": 0}
        recipe.validate_resume(last, config, last["data_signature"], provenance, 2)
        for change in ({"precision": "bf16"}, {"tau_min": .2}, {"source_sha256": {"helper": "changed"}}):
            changed = copy.deepcopy(provenance)
            changed["research_training"].update(change)
            with self.assertRaisesRegex(ValueError, "sources"):
                recipe.validate_resume(last, config, last["data_signature"], changed, 2)
        with self.assertRaisesRegex(ValueError, "totals"):
            recipe.validate_resume({**last, "updates": 3}, config, last["data_signature"], provenance, 2)


if __name__ == "__main__": unittest.main()
