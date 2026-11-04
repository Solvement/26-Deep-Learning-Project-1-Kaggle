"""CPU synthetic SAM training, exact interruption recovery and engineering mode."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from research import train_sam as sam
from train import DEFAULT_CONFIG


class SAMTrainingTests(unittest.TestCase):
    def config(self, **changes):
        return {**DEFAULT_CONFIG, "sam_rho": .05, **changes}

    def test_configuration_keeps_original_recipe_and_forbids_full_data(self):
        candidate = {**DEFAULT_CONFIG, **json.loads((ROOT/"research/configs/wrn22_sam.json").read_text())}
        sam.validate_config(candidate)
        self.assertEqual({k: v for k, v in candidate.items() if k != "sam_rho"}, DEFAULT_CONFIG)
        for changes in ({"train_all": True}, {"split_seed": 2027}, {"sam_rho": 0}, {"sam_rho": float("nan")},
                        {"sam_rho": True}, {"cutmix_alpha": 1}, {"num_classes": 100}):
            with self.assertRaises(ValueError): sam.validate_config(self.config(**changes))

    def test_fp16_and_unsupported_gpu_fail_before_loading_training_data(self):
        with patch.object(sam, "load_training_arrays", side_effect=AssertionError("no official loader")):
            with self.assertRaisesRegex(ValueError, "FP16"):
                sam.train(self.config(), Path("unused"), Path("unused"), torch.device("cuda"), "fp16")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "native CUDA BF16"):
                sam.resolve_sam_precision("auto", torch.device("cuda"))
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True) as supported:
            self.assertEqual(sam.resolve_sam_precision("bf16", torch.device("cuda")), ("bf16", torch.bfloat16))
            supported.assert_called_once_with(including_emulation=False)
        self.assertEqual(sam.resolve_sam_precision("auto", torch.device("cpu")), ("fp32", torch.float32))

    def test_resume_binds_every_research_source_precision_and_step_count(self):
        config = self.config(epochs=3, warmup_epochs=0)
        marker = sam.research_provenance("fp32", config)
        self.assertEqual(set(marker["source_sha256"]), set(sam.SOURCE_KEYS))
        self.assertFalse(marker["grad_scaler_enabled"])
        provenance = {"source_sha256": {"train.py": "frozen"}, "research_training": marker}
        last = {"config": config, "data_signature": {"split": "official"}, "provenance": provenance,
                "epoch": 1, "history": [{"epoch": 1}], "updates": 2, "skipped_updates": 0}
        sam.validate_resume(last, config, last["data_signature"], provenance, 2)
        for change in ({"precision": "bf16"}, {"rho": .1}, {"source_sha256": {"helper": "changed"}}):
            altered = copy.deepcopy(provenance)
            altered["research_training"].update(change)
            with self.assertRaisesRegex(ValueError, "sources"):
                sam.validate_resume(last, config, last["data_signature"], altered, 2)
        for change in ({"updates": 3}, {"skipped_updates": 1}, {"epoch": 3}):
            with self.assertRaisesRegex(ValueError, "history"):
                sam.validate_resume({**last, **change}, config, last["data_signature"], provenance, 2)

    def test_two_sam_passes_reuse_one_prepared_batch_without_reaugmenting(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*32*32, 10))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        config = self.config()
        images, labels = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8), torch.tensor([0, 1, 2, 3])
        original = sam.augment
        seen = []
        handle = model.register_forward_pre_hook(lambda _model, inputs: seen.append(inputs[0].clone()))
        try:
            with patch.object(sam, "augment", wraps=original) as augmentation:
                batch = sam.prepare_batch(images, labels, config)
                report = sam.sam_step(model, optimizer, sam.loss_closure(model, batch, config, torch.device("cpu"), torch.float32, {}), rho=.05)
                augmentation.assert_called_once()
        finally: handle.remove()
        self.assertEqual(report["forward_backward_passes"], 2)
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], seen[1]))

    def test_cpu_training_resume_matches_uninterrupted_model_ema_optimizer_rng_and_bn(self):
        config = self.config(depth=10, widen_factor=1, dropout=.2, epochs=2, warmup_epochs=0, batch_size=4)
        rng = np.random.default_rng(13)
        arrays = (rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8),
                  np.arange(12, dtype=np.int64) % 10, np.arange(8), np.arange(8, 12))
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"training_manifest.json").write_text(json.dumps({"split": {"seed": 2026, "indices_sha256": "synthetic"}}))
            with patch.object(sam, "load_training_arrays", return_value=arrays), \
                    patch.object(sam, "training_source", return_value={"archive_sha256": "synthetic"}), \
                    patch.object(sam, "validate_partition"), contextlib.redirect_stdout(io.StringIO()):
                sam.train(config, directory/"whole", directory, torch.device("cpu"), "fp32")
                sam.train(config, directory/"resumed", directory, torch.device("cpu"), "fp32", stop_after_epoch=1)
                sam.train(config, directory/"resumed", directory, torch.device("cpu"), "fp32", resume=True)
            whole = torch.load(directory/"whole/last.pt", weights_only=False)
            resumed = torch.load(directory/"resumed/last.pt", weights_only=False)
            self.assertEqual((whole["updates"], resumed["updates"]), (4, 4))
            for variant in ("model", "ema"):
                for name, value in whole[variant].items():
                    self.assertTrue(torch.equal(value, resumed[variant][name]), f"{variant}.{name}")
                    if name.endswith("num_batches_tracked"): self.assertEqual(value.item(), 4)
            self.assertEqual(whole["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
            for key, value in whole["optimizer"]["state"].items():
                self.assertTrue(torch.equal(value["momentum_buffer"], resumed["optimizer"]["state"][key]["momentum_buffer"]))
            self.assertEqual(whole["python_rng"], resumed["python_rng"])
            self.assertTrue(np.array_equal(whole["numpy_rng"][1], resumed["numpy_rng"][1]))
            self.assertTrue(torch.equal(whole["torch_rng"], resumed["torch_rng"]))
            metrics = json.loads((directory/"resumed/metrics.json").read_text())
            self.assertEqual(metrics["status"], "complete")
            self.assertEqual(metrics["sam_forward_passes"], 8)
            for row in resumed["history"]:
                self.assertEqual((row["successful_updates"], row["skipped_updates"], row["sam_forward_passes"]), (2, 0, 4))
                self.assertTrue(np.isfinite(row["sam_loss"]))
                self.assertLessEqual(row["mean_perturbation_norm"], .05+1e-6)
            env = json.loads((directory/"resumed/environment.json").read_text())
            self.assertEqual(env["sam_training_precision"], "fp32")
            self.assertEqual(env["research_training"], metrics["research_training"])

    def test_synthetic_engineering_mode_never_loads_real_data_and_cannot_overwrite(self):
        config = self.config(depth=10, widen_factor=1, batch_size=4, epochs=2, warmup_epochs=0)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sam, "load_training_arrays", side_effect=AssertionError("no real data")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = sam.benchmark(config, Path(folder), torch.device("cpu"), "fp32", 2)
            self.assertTrue(result["synthetic_only"])
            self.assertFalse(result["training_data_loaded"])
            self.assertEqual(result["parameter_changed_steps"], 5)
            self.assertEqual(result["sam_forward_passes"], 10)
            self.assertGreater(result["median_step_seconds"], 0)
            self.assertIsNone(result["peak_gpu_memory_bytes"])
            with self.assertRaises(FileExistsError): sam.benchmark(config, Path(folder), torch.device("cpu"), "fp32", 2)


if __name__ == "__main__":
    unittest.main()
