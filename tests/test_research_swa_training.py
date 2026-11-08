"""CPU-only typed SWA lineage, raw SGD continuity and interruption recovery."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import runtime
import numpy as np
import torch
import train as frozen
from model import build_model, count_parameters
from research import train_swa as swa


class SWATrainingTests(unittest.TestCase):
    def config(self, seed=2026):
        return {**swa.BASE_CONFIG, "seed": seed, "epochs": swa.PREFIX_EPOCHS+swa.TAIL_EPOCHS,
            "swa_prefix_epochs": swa.PREFIX_EPOCHS, "swa_tail_epochs": swa.TAIL_EPOCHS,
            "swa_lr": .01, "swa_bn_seed": 2026}

    def test_public_configs_are_exactly_fixed_recipe_and_only_seed_differs(self):
        primary = {**frozen.DEFAULT_CONFIG, **swa.read(swa.ROOT/"research/configs/wrn34_3_swa40.json")}
        repeat = {**frozen.DEFAULT_CONFIG, **swa.read(swa.ROOT/"research/configs/wrn34_3_swa40_seed2027.json")}
        swa.validate_config(primary); swa.validate_config(repeat)
        self.assertEqual({k:v for k,v in primary.items() if k != "seed"}, {k:v for k,v in repeat.items() if k != "seed"})
        original = {**frozen.DEFAULT_CONFIG, **swa.read(swa.ROOT/"research/configs/wrn34_3.json")}
        self.assertEqual(swa.prefix_config(primary), original)
        for change in ({"train_all": True}, {"swa_tail_epochs": 80}, {"swa_lr": .05}, {"swa_bn_seed": 2027},
                {"epochs": 160}, {"weight_decay": .0001}, {"seed": True}, {"extra": 1}):
            with self.assertRaises(ValueError): swa.validate_config({**primary, **change})

    @contextlib.contextmanager
    def tiny(self):
        base = {**frozen.DEFAULT_CONFIG, "depth": 10, "widen_factor": 1, "dropout": .2,
            "epochs": 2, "warmup_epochs": 0, "batch_size": 4}
        parameters = count_parameters(build_model({key: base[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}))
        with patch.multiple(swa, BASE_CONFIG=base, PREFIX_EPOCHS=2, TAIL_EPOCHS=2, TRAIN_EXAMPLES=8,
                VALIDATION_EXAMPLES=4, PARAMETERS=parameters):
            yield

    def parent(self, directory):
        data = directory/"data"
        data.mkdir()
        (data/"training_manifest.json").write_text(json.dumps({"split": {"seed": 2026, "indices_sha256": "synthetic"}}))
        rng = np.random.default_rng(17)
        arrays = (rng.integers(0,256,(12,32,32,3),dtype=np.uint8), np.arange(12,dtype=np.int64)%10, np.arange(8), np.arange(8,12))
        source = {"archive_sha256": "synthetic", "source": "official_kaggle_competition_training"}
        folder = directory/"prefix"
        folder.mkdir()
        config = swa.prefix_config(self.config())
        raw = folder/"input_config.json"
        raw.write_text(json.dumps(config))
        with patch.object(frozen, "load_training_arrays", return_value=arrays), patch.object(frozen, "training_source", return_value=source), \
                patch.object(frozen, "environment", return_value={"device": "cpu", "amp_dtype": "torch.float32"}), contextlib.redirect_stdout(io.StringIO()):
            frozen.train(config, folder, data, torch.device("cpu"))
        now = datetime.now(timezone.utc)
        origin = {"schema": "research_swa_prefix_origin_v1", "status": "complete", "init": "scratch_seeded",
            "parent_weights_loaded": False, "resumed": False, "seed": 2026,
            "config_sha256": swa.sha256(folder/"config.json"), "checkpoint_sha256": swa.sha256(folder/"last.pt"),
            "source_sha256": frozen.code_provenance()["source_sha256"],
            "rawinputconfig_path": str(raw.resolve()), "rawinputconfig_sha256": swa.sha256(raw),
            "command": [sys.executable,"train.py","--config",str(raw.resolve()),"--output-dir",str(folder.resolve()),
                "--data-dir",str((swa.ROOT/"data").resolve()),"--device","cpu"],
            "started_at": (now-timedelta(minutes=2)).isoformat(), "finished_at": (now-timedelta(minutes=1)).isoformat()}
        (folder/"prefix_origin.json").write_text(json.dumps(origin))
        return folder/"last.pt", data, arrays, source

    @contextlib.contextmanager
    def fake_data(self, arrays, source):
        with patch.object(swa, "load_training_arrays", return_value=arrays), patch.object(swa, "training_source", return_value=source), \
                patch.object(swa, "validate_partition"), patch.object(swa, "environment", return_value={"device": "cpu", "amp_dtype": "torch.float32"}), \
                patch.object(swa, "recompute_training_bn_", side_effect=self.small_bn), contextlib.redirect_stdout(io.StringIO()):
            yield

    def small_bn(self, model, images, **options):
        # Strict 40k helper mathematics are tested separately; this trainer test
        # shrinks only the synthetic dataset and verifies identical call inputs.
        self.assertEqual((len(images), options["seed"], options["batch_size"], options["precision"]), (8,2026,4,"fp32"))
        return {"schema": "research_swa_bn_recompute_v1", "training_examples": len(images), "seed": options["seed"],
            "augmented_input_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(), "parameters_unchanged": True}

    def test_actual_frozen_cpu_prefix_is_audited_and_rejects_fake_momentum_or_origin(self):
        with self.tiny(), tempfile.TemporaryDirectory() as folder:
            parent, _, _, _ = self.parent(Path(folder))
            valid = swa.validate_parent(parent, self.config())
            self.assertEqual(valid["metadata"]["successful_updates"], 4)
            self.assertEqual(valid["metadata"]["precision"], "fp32")
            original = torch.load(parent, weights_only=False)
            for change in ({"epoch": 1}, {"updates": 3}, {"cuda_rng": [torch.tensor([1], dtype=torch.uint8)]}):
                torch.save({**original, **change}, parent)
                with self.assertRaises(ValueError): swa.validate_parent(parent, self.config())
            altered = copy.deepcopy(original)
            altered["optimizer"]["state"][0]["momentum_buffer"].fill_(float("nan"))
            torch.save(altered, parent)
            with self.assertRaisesRegex(ValueError, "momentum tensors"): swa.validate_parent(parent, self.config())
            torch.save(original, parent)
            origin_path = parent.parent/"prefix_origin.json"
            origin = swa.read(origin_path)
            origin["command"].append("--resume")
            origin_path.write_text(json.dumps(origin))
            with self.assertRaises(ValueError): swa.validate_parent(parent, self.config())

    def test_fp16_and_invalid_prefix_fail_before_official_data_loader(self):
        with patch.object(swa, "load_training_arrays", side_effect=AssertionError("no official data")):
            with self.assertRaisesRegex(ValueError, "FP16"):
                swa.train(self.config(), "missing", Path("unused"), Path("unused"), torch.device("cuda"), "fp16")
            with patch.object(swa, "validate_parent", side_effect=ValueError("untrusted prefix")):
                with self.assertRaisesRegex(ValueError, "untrusted prefix"):
                    swa.train(self.config(), "missing", Path("unused"), Path("unused"), torch.device("cpu"), "fp32")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Native CUDA BF16"): swa.resolve_precision("bf16", torch.device("cuda"))

    def test_stop_resume_preserves_raw_ema_swa_optimizer_rng_and_never_selects_tail_dev(self):
        with self.tiny(), tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            parent, data, arrays, source = self.parent(directory)
            config = self.config()
            with self.fake_data(arrays, source), patch.object(swa, "evaluate", wraps=swa.evaluate) as evaluation:
                swa.train(config, parent, directory/"whole", data, torch.device("cpu"), "fp32")
                self.assertEqual(evaluation.call_count, 2)
                swa.train(config, parent, directory/"resume", data, torch.device("cpu"), "fp32", stop_after_epoch=3)
                self.assertEqual(evaluation.call_count, 2)
                swa.train(config, parent, directory/"resume", data, torch.device("cpu"), "fp32", resume=True)
                self.assertEqual(evaluation.call_count, 4)
            whole = torch.load(directory/"whole/last.pt", weights_only=False)
            resumed = torch.load(directory/"resume/last.pt", weights_only=False)
            for variant in ("model", "ema", "swa_model"):
                for name,value in whole[variant].items(): self.assertTrue(torch.equal(value, resumed[variant][name]), (variant,name))
            self.assertEqual((resumed["swa_count"],resumed["updates"],resumed["tail_updates"]), (2,8,4))
            self.assertEqual(whole["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
            for key,value in whole["optimizer"]["state"].items():
                self.assertTrue(torch.equal(value["momentum_buffer"], resumed["optimizer"]["state"][key]["momentum_buffer"]))
            self.assertEqual(whole["python_rng"], resumed["python_rng"])
            self.assertTrue(np.array_equal(whole["numpy_rng"][1], resumed["numpy_rng"][1]))
            self.assertTrue(torch.equal(whole["torch_rng"], resumed["torch_rng"]))
            for row in resumed["history"]["tail"]:
                self.assertEqual(row["lr"], .01)
                self.assertIsNone(row["validation_accuracy"])
                self.assertIsNone(row["ema_validation_accuracy"])
            metrics = swa.read(directory/"resume/metrics.json")
            self.assertTrue(metrics["no_best_selection"] and metrics["ema_tracked"])
            self.assertFalse(metrics["ema_selected"])
            self.assertEqual(metrics["status"], "complete")
            self.assertFalse((directory/"resume/best.pt").exists())
            for filename, variant in (("swa.pt","swa"),("raw_tail.pt","raw_tail_control")):
                checkpoint = torch.load(directory/"resume"/filename, weights_only=False)
                self.assertEqual(checkpoint["variant"],variant)
                self.assertEqual(checkpoint["epoch"],4)

    def test_first_tail_epoch_matches_independent_frozen_sgd_rng_and_equal_average(self):
        with self.tiny(), tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            parent_path,data,arrays,source = self.parent(directory)
            config = self.config()
            with self.fake_data(arrays, source):
                swa.train(config,parent_path,directory/"actual",data,torch.device("cpu"),"fp32",stop_after_epoch=3)
            actual = torch.load(directory/"actual/last.pt",weights_only=False)
            parent = torch.load(parent_path,weights_only=False)
            model = build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")}).to(memory_format=torch.channels_last).train()
            model.load_state_dict(parent["model"])
            optimizer = torch.optim.SGD(model.parameters(), lr=.01,momentum=.9,weight_decay=.0005,nesterov=True)
            optimizer.load_state_dict(parent["optimizer"])
            for group in optimizer.param_groups: group["lr"] = .01
            swa.restore_rng(parent,torch.device("cpu"))
            images = torch.as_tensor(arrays[0][:8]); labels = torch.as_tensor(arrays[1][:8])
            order = torch.randperm(8)
            for start in range(0,8,4):
                ids = order[start:start+4]; y = labels[ids]; x = frozen.augment(images[ids],16)
                mix = random.random()<.5
                lam = float(np.random.beta(.2,.2)) if mix else 1.
                permutation = torch.randperm(len(y)) if mix else None
                if mix: x = x*lam+x[permutation]*(1-lam)
                optimizer.zero_grad(set_to_none=True); logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits,y)*lam
                if mix: loss += torch.nn.functional.cross_entropy(logits,y[permutation])*(1-lam)
                loss.backward(); optimizer.step()
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,actual["model"][name]),name)
            for name,value in model.named_parameters(): self.assertTrue(torch.equal(value,actual["swa_model"][name]),name)

    def test_synthetic_preflight_never_loads_parent_or_real_data_and_is_immutable(self):
        with self.tiny(), tempfile.TemporaryDirectory() as folder, \
                patch.object(swa,"validate_parent",side_effect=AssertionError("no parent")), \
                patch.object(swa,"load_training_arrays",side_effect=AssertionError("no official labels")):
            result = swa.benchmark(self.config(),Path(folder),torch.device("cpu"),"fp32",2)
            self.assertTrue(result["synthetic_only"] and result["finite_losses_verified"] and result["finite_parameters_verified"])
            self.assertFalse(result["training_data_loaded"])
            self.assertEqual((result["successful_updates"],result["total_successful_updates"],result["averaging_calls"]),(2,5,5))
            self.assertEqual(result["source_sha256"],swa.research_provenance("fp32",self.config())["source_sha256"])
            with self.assertRaises(FileExistsError): swa.benchmark(self.config(),Path(folder),torch.device("cpu"),"fp32",2)


if __name__ == "__main__": unittest.main()
