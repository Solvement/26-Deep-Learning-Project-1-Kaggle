"""CPU synthetic checks for independent three-view AugMix training and audit.

Toy models, images, controlled scaler branches and metadata fixtures below are
not official-data/native-GPU experiments or competition-score evidence.
"""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
import psutil
from torch.nn import functional as F

from research import augmix
from research import train_augmix as trainer
from research import benchmark_augmix as validator
from research import run_augmix_local as supervisor
from research import run_local
from model import build_model
from train import DEFAULT_CONFIG


def assert_state_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_state_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a,b in zip(left, right):
            assert_state_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class CountingToy(torch.nn.Module):
    """One tiny 10-class pass; BN and dropout make replay checks meaningful."""
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.dropout = torch.nn.Dropout(.2)
        self.classifier = torch.nn.Linear(3, 10)
        self.forward_batches = []
        self.last_logits = None

    def forward(self, images):
        self.forward_batches.append(len(images))
        features = self.dropout(self.bn(images)).mean(dim=(2, 3))
        logits = self.classifier(features)
        self.last_logits = logits.detach().clone()
        return logits


class ForcedSkipScaler:
    """CPU test double, never usable as a native overflow/preflight receipt."""
    def __init__(self):
        self.value = 65536.
    def is_enabled(self):
        return True
    def get_scale(self):
        return self.value
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        return None
    def update(self):
        self.value /= 2


class AugMixCPUCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
    def setUp(self):
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
    def model_optimizer(self):
        model = CountingToy().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9,
                                    weight_decay=.0005, nesterov=True)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        return model, optimizer, ema


class AugMixBatchTests(AugMixCPUCase):
    def images(self, batch=2):
        return np.random.default_rng(33).integers(0, 256, (batch, 32, 32, 3), dtype=np.uint8)

    def test_common_crop_flip_matches_reflect_pad_and_single_explicit_rng_draws(self):
        image = self.images(1)[0]
        original = image.copy()
        for seed in range(12):
            with self.subTest(seed=seed):
                rng, oracle = np.random.default_rng(seed), np.random.default_rng(seed)
                top, left = int(oracle.integers(0, 9)), int(oracle.integers(0, 9))
                padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="reflect")
                expected = padded[top:top+32, left:left+32]
                if oracle.random() < .5:
                    expected = expected[:, ::-1]
                actual = trainer.common_crop_flip(image, rng)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertTrue(np.array_equal(actual, expected))
                self.assertFalse(np.shares_memory(actual, image))
        self.assertTrue(np.array_equal(image, original))

    def test_preparation_crops_once_per_original_and_concatenates_view_major(self):
        images = self.images()
        labels = np.array([2, 7], dtype=np.int64)
        rng, oracle = np.random.default_rng(15), np.random.default_rng(15)
        expected_views = []
        seeds = oracle.integers(0, 2**64, size=len(images), dtype=np.uint64)
        for image,seed in zip(images, seeds):
            per_source = np.random.default_rng(int(seed))
            base = trainer.common_crop_flip(image, per_source)
            expected_views.append(augmix.augmix_views(base, per_source))
        expected = torch.stack([expected_views[index][view] for view in range(3) for index in range(2)])
        original_crop, original_views = trainer.common_crop_flip, trainer.augmix_views
        with patch.object(trainer, "common_crop_flip", wraps=original_crop) as crop, \
                patch.object(trainer, "augmix_views", wraps=original_views) as views:
            prepared, targets = trainer.prepare_batch(images, labels, rng, torch.device("cpu"))
        self.assertEqual(crop.call_count, 2)
        self.assertEqual(views.call_count, 2)
        self.assertEqual(prepared.shape, (6, 3, 32, 32))
        self.assertEqual(prepared.dtype, torch.float32)
        self.assertTrue(torch.equal(prepared, expected))
        self.assertTrue(torch.equal(targets, torch.tensor([2, 7], dtype=torch.int64)))

    def test_last_source_batch64_uses_one_forward_on192_views_and_one_bn_update(self):
        images = self.images(64)
        labels = np.arange(64, dtype=np.int64) % 10
        prepared = trainer.prepare_batch(images, labels, np.random.default_rng(7), torch.device("cpu"))
        model, optimizer, ema = self.model_optimizer()
        report = trainer.train_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
            prepared, torch.device("cpu"), torch.float32, ema=ema, successful_updates=0)
        self.assertEqual(model.forward_batches, [192])
        self.assertEqual(model.bn.num_batches_tracked.item(), 1)
        self.assertEqual((report["source_examples"], report["model_views"]), (64, 192))
        self.assertTrue(report["updated"])
        self.assertEqual(report["successful_updates_after"], 1)
        self.assertAlmostEqual(report["cross_entropy"], F.cross_entropy(model.last_logits[:64], prepared[1]).item(), places=6)
        self.assertAlmostEqual(report["loss"], report["cross_entropy"]+12*report["jsd"], places=5)

    def test_controlled_overflow_skip_preserves_parameters_optimizer_and_ema(self):
        prepared = trainer.prepare_batch(self.images(), np.array([0, 1], dtype=np.int64),
                                         np.random.default_rng(9), torch.device("cpu"))
        model, optimizer, ema = self.model_optimizer()
        parameters = [value.detach().clone() for value in model.parameters()]
        before_optimizer, before_ema = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(ema.state_dict())
        report = trainer.train_step(model, optimizer, ForcedSkipScaler(), prepared,
            torch.device("cpu"), torch.float32, ema=ema, successful_updates=11)
        self.assertFalse(report["updated"])
        self.assertEqual(report["successful_updates_after"], 11)
        self.assertLess(report["scale_after"], report["scale_before"])
        for value,before in zip(model.parameters(), parameters):
            self.assertTrue(torch.equal(value.detach(), before))
        assert_state_equal(self, optimizer.state_dict(), before_optimizer)
        assert_state_equal(self, ema.state_dict(), before_ema)
        self.assertEqual(model.forward_batches, [6])

    def test_successful_update_advances_ema_using_success_count(self):
        prepared = trainer.prepare_batch(self.images(), np.array([0, 1], dtype=np.int64),
                                         np.random.default_rng(9), torch.device("cpu"))
        model, optimizer, ema = self.model_optimizer()
        before = [value.detach().clone() for value in ema.parameters()]
        report = trainer.train_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
            prepared, torch.device("cpu"), torch.float32, ema=ema, successful_updates=11, ema_decay=.995)
        self.assertTrue(report["updated"])
        self.assertEqual(report["successful_updates_after"], 12)
        decay = min(.995, 13/22)
        for average,online,previous in zip(ema.parameters(), model.parameters(), before):
            self.assertTrue(torch.allclose(average, previous.lerp(online.detach(), 1-decay), rtol=0, atol=1e-7))

    def test_rng_checkpoint_replays_pixels_dropout_bn_sgd_ema_and_scaler_exactly(self):
        images, labels = self.images(), np.array([0, 9], dtype=np.int64)
        rng = np.random.default_rng(2026)
        model, optimizer, ema = self.model_optimizer()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        first = trainer.prepare_batch(images, labels, rng, torch.device("cpu"))
        trainer.train_step(model, optimizer, scaler, first, torch.device("cpu"), torch.float32,
                           ema=ema, successful_updates=0)
        state = {"model": copy.deepcopy(model.state_dict()), "ema": copy.deepcopy(ema.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "grad_scaler": scaler.state_dict(),
            **trainer.capture_state_rng(rng, torch.device("cpu"))}
        expected_batch = trainer.prepare_batch(images, labels, rng, torch.device("cpu"))
        expected_report = trainer.train_step(model, optimizer, scaler, expected_batch,
            torch.device("cpu"), torch.float32, ema=ema, successful_updates=1)
        expected_states = (copy.deepcopy(model.state_dict()), copy.deepcopy(ema.state_dict()),
                           copy.deepcopy(optimizer.state_dict()), trainer.capture_state_rng(rng, torch.device("cpu")))
        resumed, resumed_optimizer, resumed_ema = self.model_optimizer()
        resumed.load_state_dict(state["model"])
        resumed_ema.load_state_dict(state["ema"])
        resumed_optimizer.load_state_dict(state["optimizer"])
        resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
        resumed_scaler.load_state_dict(state["grad_scaler"])
        resumed_rng = np.random.default_rng(123)
        trainer.restore_state_rng(state, resumed_rng, torch.device("cpu"))
        actual_batch = trainer.prepare_batch(images, labels, resumed_rng, torch.device("cpu"))
        self.assertTrue(torch.equal(actual_batch[0], expected_batch[0]))
        actual_report = trainer.train_step(resumed, resumed_optimizer, resumed_scaler, actual_batch,
            torch.device("cpu"), torch.float32, ema=resumed_ema, successful_updates=1)
        self.assertEqual(actual_report, expected_report)
        for actual,expected in zip((resumed.state_dict(), resumed_ema.state_dict(), resumed_optimizer.state_dict(),
                trainer.capture_state_rng(resumed_rng, torch.device("cpu"))), expected_states):
            assert_state_equal(self, actual, expected)

    def test_invalid_source_or_three_view_packets_fail_before_model_forward(self):
        for images,labels in ((self.images(1).astype(np.float32), np.array([0], dtype=np.int64)),
                (self.images(1), np.array([0.], dtype=np.float32)),
                (self.images(1), np.array([10], dtype=np.int64)),
                (self.images(2), np.array([0], dtype=np.int64)),
                (self.images(129), np.arange(129, dtype=np.int64)%10)):
            with self.assertRaises(ValueError):
                trainer.prepare_batch(images, labels, np.random.default_rng(0), torch.device("cpu"))
        model, optimizer, ema = self.model_optimizer()
        with self.assertRaises(ValueError):
            trainer.train_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
                (torch.zeros(5, 3, 32, 32), torch.tensor([0, 1])), torch.device("cpu"), torch.float32, ema=ema)
        self.assertEqual(model.forward_batches, [])


class AugMixSourceAuditTests(AugMixCPUCase):
    def history(self, epochs):
        return [{"epoch": epoch, "successful_updates": 313, "skipped_updates": 0,
            "attempted_updates": 313, "source_examples": 40000, "model_views": 120000,
            "forward_calls": 313, "last_source_batch_size": 64,
            "train_loss": 2.42, "cross_entropy": 2.3, "jsd": .01,
            "validation_loss": .3, "ema_validation_loss": .31,
            "epoch_seconds": 1., "elapsed_seconds": float(epoch), "grad_scaler_scale": 1., "lr": .1,
            "train_accuracy": .8, "validation_accuracy": .9, "ema_validation_accuracy": .89,
            "selected_variant": "model"} for epoch in range(1, epochs+1)]

    def resume_fixture(self, seed=2026):
        config = trainer.fixed_config(seed)
        signature = copy.deepcopy(trainer.OFFICIAL_SIGNATURE)
        marker = trainer.research_provenance("fp32", config)
        provenance = {"source_sha256": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("train.py", "model.py", "data.py", "runtime.py")}, "research_training": marker}
        model_config = {key:config[key] for key in ("depth", "widen_factor", "dropout", "num_classes")}
        model = build_model(model_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005, nesterov=True)
        saved_optimizer = optimizer.state_dict()
        saved_optimizer["state"] = {index:{"momentum_buffer": torch.zeros_like(parameter)}
                                    for index,parameter in enumerate(model.parameters())}
        last = {"schema": trainer.SCHEMA, "config": copy.deepcopy(config), "data_signature": signature,
            "provenance": copy.deepcopy(provenance), "model_config": model_config,
            "parameter_count": 4298970, "epoch": 1, "history": self.history(1),
            "updates": 313, "skipped_updates": 0, "attempted_updates": 313,
            "source_examples_seen": 40000, "model_views_seen": 120000,
            "model": model.state_dict(), "ema": copy.deepcopy(model.state_dict()),
            "optimizer": saved_optimizer, "grad_scaler": {},
            **trainer.capture_state_rng(np.random.default_rng(seed), torch.device("cpu"), seed=seed)}
        return config, signature, provenance, last

    def test_config_replaces_mixup_cutout_and_locks_primary_three_view_recipe(self):
        config = trainer.fixed_config()
        trainer.validate_config(config)
        expected = {**DEFAULT_CONFIG, "cutout": 0, "mixup_alpha": 0.0, "mixup_probability": 0.0}
        self.assertEqual({key:config[key] for key in DEFAULT_CONFIG}, expected)
        self.assertNotIn("sam_rho", config)
        self.assertNotIn("architecture", config)
        for changes in ({"cutout": 16}, {"mixup_alpha": .2}, {"mixup_probability": .5},
                {"batch_size": 64}, {"epochs": 80}, {"seed": 2028}, {"split_seed": 2027},
                {"train_all": True}, {"augmix_width": 2}, {"augmix_depth": 1}, {"augmix_severity": 5},
                {"augmix_alpha": .5}, {"augmix_all_ops": True}, {"augmix_jsd_weight": 0.}, {"sam_rho": .05}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    trainer.validate_config({**config, **changes})

    def test_precision_and_raw_source_identity_are_explicit_before_any_data_access(self):
        config = trainer.fixed_config()
        cpu = trainer.research_provenance("fp32", config, "1"*64)
        native = trainer.research_provenance("fp16", config, "1"*64)
        self.assertFalse(cpu["grad_scaler_enabled"])
        self.assertTrue(native["grad_scaler_enabled"])
        self.assertNotEqual(cpu, native)
        self.assertNotEqual(cpu, trainer.research_provenance("fp32", config, "2"*64))
        self.assertEqual(cpu["model_batch_size"], 384)
        self.assertEqual(cpu["forward"], "one_view_major_3B_forward")
        with self.assertRaises(ValueError):
            trainer.research_provenance("bf16", config)
        with patch.object(trainer, "load_training_arrays", side_effect=AssertionError("no data read")) as loader:
            with self.assertRaises(ValueError):
                trainer.train(config, Path("unused"), Path("unused"), torch.device("cpu"), "fp16")
            loader.assert_not_called()

    def test_history_requires_full_source_view_forward_and_attempt_ledgers(self):
        original = self.history(160)
        self.assertEqual(trainer.validate_history(original, 160), (50080, 0))
        for key,value in (("source_examples", 50000), ("model_views", 80000), ("forward_calls", 626),
                ("attempted_updates", 312), ("successful_updates", 312), ("last_source_batch_size", 128),
                ("jsd", float("nan")), ("grad_scaler_scale", 0.), ("validation_accuracy", 1.1)):
            with self.subTest(field=key):
                altered = copy.deepcopy(original)
                altered[17][key] = value
                with self.assertRaises(ValueError):
                    trainer.validate_history(altered, 160)
        with self.assertRaises(ValueError):
            trainer.validate_history(original[:-1], 160)
        altered = copy.deepcopy(original)
        altered[3].update(successful_updates=312, skipped_updates=1)
        self.assertEqual(trainer.validate_history(altered, 160), (50079, 1))

    def test_resume_binds_pcg64_sources_config_real_weights_sgd_scaler_and_counts(self):
        config, signature, provenance, last = self.resume_fixture()
        trainer.validate_resume(last, config, signature, provenance)
        mutations = [lambda x:x.update(schema="research_shadow_training_v1"),
            lambda x:x["provenance"]["research_training"].update(input_config_sha256="f"*64),
            lambda x:x["provenance"]["research_training"]["source_sha256"].update({"research/augmix.py": "f"*64}),
            lambda x:x.update(model_views_seen=80000), lambda x:x.update(attempted_updates=312),
            lambda x:x["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1)),
            lambda x:x["model"].pop(next(iter(x["model"]))),
            lambda x:x.update(grad_scaler={"scale": 65536.}),
            lambda x:x["augmix_rng"].update(bit_generator="MT19937"),
            lambda x:x.update(torch_rng=torch.zeros(1, dtype=torch.uint8)),
            lambda x:x.pop("augmix_rng"), lambda x:x.pop("sampler_rng"), lambda x:x.pop("loader_rng")]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                altered = copy.deepcopy(last)
                mutation(altered)
                with self.assertRaises(ValueError):
                    trainer.validate_resume(altered, config, signature, provenance)
        with self.assertRaises(ValueError):
            trainer.validate_resume(last, config, {**signature, "split_sha256": "c"*64}, provenance)

    def test_unrelated_commit_does_not_replace_original_source_provenance_on_resume(self):
        config, signature, provenance, last = self.resume_fixture()
        last["provenance"]["git_commit"] = "old-reporting-commit"
        current = {**provenance, "git_commit": "new-reporting-only-commit"}
        original = copy.deepcopy(last["provenance"])
        trainer.validate_resume(last, config, signature, current)
        self.assertEqual(last["provenance"], original)

    def test_fp16_scaler_state_cannot_be_substituted_by_disabled_cpu_or_changed_growth(self):
        trainer.validate_scaler({}, "fp32")
        native = {"scale": 32768., "growth_factor": 2., "backoff_factor": .5,
                  "growth_interval": 2000, "_growth_tracker": 12}
        trainer.validate_scaler(native, "fp16")
        with self.assertRaises(ValueError):
            trainer.validate_scaler({}, "fp16")
        with self.assertRaises(ValueError):
            trainer.validate_scaler(native, "fp32")
        for changes in ({"scale": float("nan")}, {"scale": 0.}, {"growth_factor": 1.},
                        {"backoff_factor": 1.}, {"growth_interval": 1000}, {"_growth_tracker": 2000}):
            with self.subTest(change=changes):
                with self.assertRaises(ValueError):
                    trainer.validate_scaler({**native, **changes}, "fp16")

    def complete_fixture(self, precision="fp32", skipped=0, seed=2026):
        config, signature, provenance, last = self.resume_fixture(seed)
        marker = trainer.research_provenance(precision, config)
        provenance["research_training"] = marker
        history = self.history(160)
        remaining = skipped
        for row in history:
            count = min(remaining, 312)
            row.update(successful_updates=313-count, skipped_updates=count)
            remaining -= count
        self.assertEqual(remaining, 0)
        successful = 50080-skipped
        last.update(epoch=160, history=history, provenance=copy.deepcopy(provenance), updates=successful,
            skipped_updates=skipped, attempted_updates=50080, source_examples_seen=6400000, model_views_seen=19200000)
        if precision == "fp16":
            last["grad_scaler"] = {"scale": 32768., "growth_factor": 2., "backoff_factor": .5,
                "growth_interval": 2000, "_growth_tracker": 0}
            # Synthetic bytes only; no CUDA generator is allocated by tests.
            last["cuda_rng"] = [last["torch_rng"].clone()]
        checkpoint = {"schema": trainer.SCHEMA, "config": copy.deepcopy(config),
            "provenance": copy.deepcopy(provenance), "data_signature": signature,
            "model_config": copy.deepcopy(last["model_config"]), "parameter_count": 4298970,
            "epoch": 160, "variant": "model", "model": last["model"],
            "validation_accuracy": .9, "validation_loss": .3}
        metrics = {"schema": trainer.SCHEMA, "config": copy.deepcopy(config), "research_training": copy.deepcopy(marker),
            "parameter_count": 4298970, "status": "complete", "epochs_completed": 160,
            "training_examples": 40000, "validation_examples": 5000,
            "successful_updates": successful, "skipped_updates": skipped, "attempted_updates": 50080,
            "forward_calls": 50080, "source_examples_seen": 6400000, "model_views_seen": 19200000,
            "best_validation_accuracy": .9, "best_validation_loss": .3}
        artifact = {"schema": trainer.SCHEMA, "code": copy.deepcopy(provenance),
            "parameter_count": 4298970, "train_examples": 40000, "validation_examples": 5000,
            "data_signature": signature}
        environment = {"schema": trainer.SCHEMA, "research_training": copy.deepcopy(marker),
            "augmix_training_precision": precision, "grad_scaler_enabled": precision == "fp16",
            "amp_dtype": "torch.float16" if precision == "fp16" else "torch.float32"}
        return checkpoint, metrics, history, artifact, environment, trainer.raw_config(config), last

    def test_complete_augmix_audit_rejects_relabelled_methods_partial_or_wrong_view_ledgers(self):
        original = self.complete_fixture()
        proof = validator.validate_augmix(*original)
        self.assertEqual(proof["schema"], "research_augmix_integrity_v1")
        self.assertEqual((proof["attempted_updates"], proof["source_examples_seen"], proof["model_views_seen"]),
                         (50080, 6400000, 19200000))
        mutations = [lambda a:a[0].update(schema="research_shadow_training_v1"),
            lambda a:a[0]["provenance"]["research_training"].update(method="sam"),
            lambda a:a[1].update(epochs_completed=159),
            lambda a:a[1].update(training_examples=50000, validation_examples=0),
            lambda a:a[1].update(source_examples_seen=6400001),
            lambda a:a[1].update(model_views_seen=12800000),
            lambda a:a[1].update(forward_calls=100160),
            lambda a:a[1].update(attempted_updates=50079),
            lambda a:a[2][20].update(model_views=80000),
            lambda a:a[2][20].update(attempted_updates=312),
            lambda a:a[4].update(grad_scaler_enabled=True),
            lambda a:a[0]["provenance"]["source_sha256"].update({"train.py": "f"*64}),
            lambda a:a[6]["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1))]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                documents = copy.deepcopy(original)
                mutation(documents)
                with self.assertRaises(ValueError):
                    validator.validate_augmix(*documents)
        documents = list(copy.deepcopy(original))
        documents[5] += b" "
        with self.assertRaises(ValueError):
            validator.validate_augmix(*documents)

    def test_native_skip_counts_are_bounded_and_cpu_disabled_scaler_never_claims_skips(self):
        legitimate = self.complete_fixture("fp16", skipped=1)
        proof = validator.validate_augmix(*legitimate)
        self.assertEqual((proof["successful_updates"], proof["skipped_updates"]), (50079, 1))
        with self.assertRaises(ValueError):
            validator.validate_augmix(*self.complete_fixture("fp16", skipped=502))
        with self.assertRaises(ValueError):
            validator.validate_augmix(*self.complete_fixture("fp32", skipped=1))

    def test_same_unchanged_validator_accepts_future_seed2027_typed_engine_evidence(self):
        # This is a synthetic complete-history fixture, not a trained repeat.
        documents = self.complete_fixture(seed=2027)
        proof = validator.validate_augmix(*documents)
        self.assertEqual(proof["training"]["training_seed"], 2027)
        self.assertEqual(proof["training"]["split_seed"], 2026)
        self.assertEqual(documents[0]["config"]["seed"], 2027)


class AugMixEngineSeedTests(AugMixCPUCase):
    def test_only_exact_integer26_or27_engine_seeds_and_same_recipe_are_allowed(self):
        primary, repeat = trainer.fixed_config(), trainer.fixed_config(2027)
        trainer.validate_config(repeat)
        self.assertEqual({k:v for k,v in primary.items() if k != "seed"}, {k:v for k,v in repeat.items() if k != "seed"})
        for seed in (True, False, 2028, 2026., np.int64(2027), None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError): trainer.fixed_config(seed)
                with self.assertRaises(ValueError): trainer.pipeline_generators(seed)
                with self.assertRaises(ValueError): trainer.validate_config({**primary,"seed":seed})
        for key,value in (("epochs",80),("amp_init_scale",65536.),("augmix_num_workers",0),("mixup_alpha",.2)):
            with self.subTest(field=key):
                with self.assertRaises(ValueError): trainer.validate_config({**repeat,key:value})

    def test_primary_raw_bytes_and_default_sampler_loader_sequence_are_unchanged(self):
        raw = (ROOT/"research/configs/wrn22_augmix_jsd.json").read_bytes()
        self.assertIn(hashlib.sha256(raw).hexdigest(), {
            "3c38e5e638894d605101d9ae7aeb34edb26fb1e4a10ba96ee95d246f916924ff",  # registered Windows bytes
            "c7ba91cb25ff128dfe816b303cde43cd4678673e0847a9f5b5bca7c4c4541df8",  # Git's LF checkout of the same input
        })
        self.assertEqual(trainer.raw_config(trainer.fixed_config(), raw), raw)
        # The registered Windows input uses CRLF; the unchanged canonical
        # in-memory default has always encoded LF. Neither representation is
        # rewritten by the engine when explicit input bytes are supplied.
        self.assertEqual(trainer.raw_config(trainer.fixed_config()), raw.replace(b"\r\n", b"\n"))
        sampler, loader = trainer.pipeline_generators()
        self.assertTrue(torch.equal(sampler.get_state(), torch.Generator().manual_seed(3027).get_state()))
        self.assertTrue(torch.equal(loader.get_state(), torch.Generator().manual_seed(4027).get_state()))

    def test_model_augmentation_sampler_loader_differ_between_seeds_and_each_replays(self):
        images = np.random.default_rng(33).integers(0,256,(2,32,32,3),dtype=np.uint8)
        labels = np.array([2,8],dtype=np.int64)
        def fingerprint(seed):
            torch.manual_seed(seed)
            config = trainer.fixed_config(seed)
            model = build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
            probe = next(model.parameters()).detach().clone()
            del model
            parent = np.random.Generator(np.random.PCG64(seed))
            sampler, loader = trainer.pipeline_generators(seed)
            views, targets = trainer.prepare_batch(images,labels,parent,torch.device("cpu"))
            order = torch.randperm(32,generator=sampler)
            worker_seed = torch.empty((),dtype=torch.int64).random_(generator=loader).item()
            return probe, views, order, worker_seed, trainer.capture_state_rng(parent,torch.device("cpu"),sampler,loader,seed=seed)
        first, second = fingerprint(2026), fingerprint(2027)
        for seed,expected in ((2026,first),(2027,second)):
            assert_state_equal(self,fingerprint(seed),expected)
        for index in (0,1,2): self.assertFalse(torch.equal(first[index],second[index]))
        self.assertNotEqual(first[3],second[3])
        self.assertNotEqual(first[4]["augmix_rng"],second[4]["augmix_rng"])

    def test_train_and_preflight_take_config_seed_before_any_official_data_or_gpu_work(self):
        config = trainer.fixed_config(2027)
        for entry,device,precision in ((trainer.train,torch.device("cpu"),"fp32"),
                                       (trainer.preflight,torch.device("cuda"),"fp16")):
            with self.subTest(entry=entry.__name__), tempfile.TemporaryDirectory() as directory, \
                    patch.object(trainer,"seed_all") as seed, \
                    patch.object(trainer,"pipeline_generators",wraps=trainer.pipeline_generators) as generators, \
                    patch.object(trainer,"resolve_augmix_precision",return_value=(precision,torch.float32 if precision == "fp32" else torch.float16)), \
                    patch.object(trainer,"current_sources",return_value={}), \
                    patch.object(trainer,"load_training_arrays",side_effect=RuntimeError("synthetic stop before data")) as data:
                with self.assertRaisesRegex(RuntimeError,"stop before data"):
                    entry(config,Path(directory)/"new",Path("unused"),device,precision)
                seed.assert_called_once_with(2027)
                generators.assert_called_once_with(2027)
                data.assert_called_once()

    def test_primary_supervisors_still_refuse27_before_lease_allocation_or_upload(self):
        from research import run_colab_augmix as cloud_supervisor
        repeat = trainer.fixed_config(2027)
        with self.assertRaisesRegex(ValueError,"seed2026"):
            cloud_supervisor.validate_config(repeat)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/"repeat.json"
            config.write_text(json.dumps(repeat))
            arguments = ["run_augmix_local.py","--run","future_repeat","--mode","train","--config",str(config),
                         "--preflight",str(Path(directory)/"absent.json")]
            with patch.object(sys,"argv",arguments), \
                    patch.object(supervisor,"inputs",side_effect=AssertionError("no data access")), \
                    patch.object(supervisor,"acquire_lease",side_effect=AssertionError("no resource allocation")):
                with self.assertRaisesRegex(ValueError,"primary2026"):
                    supervisor.main()


class AugMixPreflightTests(unittest.TestCase):
    def fixture(self):
        config = trainer.fixed_config()
        raw_sha = "1"*64
        smoke = {"status": "passed", "updated": False, "finite_loss": True,
            "infinite_gradient_injected": True, "all_parameters_unchanged": True,
            "optimizer_state_unchanged": True, "ema_parameters_unchanged": True,
            "source_examples": 128, "model_views": 384, "forward_calls": 1,
            "scale_before": 256., "scale_after": 128.}
        sources = trainer.current_sources()
        report = {"schema": trainer.PREFLIGHT_SCHEMA, "status": "complete", "synthetic_only": False,
            "official_training_only": True, "training_data_loaded": True, "precision": "fp16",
            "grad_scaler_enabled": True, "initial_grad_scaler_scale":256.0, "parameters": 4298970,
            "model_config": {k:config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")},
            "config": config, "input_config_sha256": raw_sha,
            "research_training": trainer.research_provenance("fp16", config, raw_sha),
            "source_sha256": sources, "batch_size": 128, "model_batch_size": 384,
            "warmup_steps": 3, "timed_steps": 20, "attempted_updates": 23,
            "successful_updates": 23, "skipped_updates": 0, "parameter_changed_steps": 23,
            "forward_calls": 23, "source_examples_seen": 2944, "model_views_seen": 8832,
            "training_examples": 40000, "validation_examples": 5000, "source_ids_unique_count": 2944,
            "single_forward_per_step": True, "source_ids_are_training_only": True,
            "validation_evaluated": False, "confirmation_evaluated": False,
            "hidden_images_loaded": False, "external_data_loaded": False,
            "preflight_source_ids_sha256": "a"*64,
            "data_signature": copy.deepcopy(trainer.OFFICIAL_SIGNATURE),
            "overflow_smoke": smoke, "median_step_seconds": .0625, "p90_step_seconds": .07,
            "required_training_and_benchmark_seconds": 1.5*.0625*313*160+300,
            "mean_loss": 2.42, "mean_cross_entropy": 2.3, "mean_jsd": .01,
            "num_workers": 2, "prefetch_factor": 2, "multiprocessing_context": "spawn",
            "persistent_workers": False, "worker_cuda_initialized": False, "all_workers_stopped": True,
            "environment": {"device": "cuda", "amp_dtype": "torch.float16", "gpu": "Tesla T4",
                "packages": {"torch": "synthetic metadata fixture, no actual GPU"}}}
        return report, config, raw_sha, sources

    def test_native_preflight_cannot_relabel_synthetic_or_wrong_source_view_hardware(self):
        original = self.fixture()
        self.assertEqual(trainer.validate_preflight(*original, require_t4=True), original[0])
        mutations = [lambda a:a[0].update(synthetic_only=True),
            lambda a:a[0].update(training_data_loaded=False),
            lambda a:a[0].update(precision="fp32"), lambda a:a[0].update(grad_scaler_enabled=False),
            lambda a:a[0].update(successful_updates=22), lambda a:a[0].update(skipped_updates=1),
            lambda a:a[0].update(model_batch_size=256), lambda a:a[0].update(forward_calls=46),
            lambda a:a[0].update(source_ids_unique_count=128),
            lambda a:a[0].update(source_ids_are_training_only=False),
            lambda a:a[0].update(confirmation_evaluated=True),
            lambda a:a[0]["environment"].update(device="cpu"),
            lambda a:a[0]["environment"].update(gpu="unregistered GPU"),
            lambda a:a[0].update(input_config_sha256="2"*64),
            lambda a:a[0].update(source_sha256={**a[0]["source_sha256"], "research/augmix.py": "f"*64}),
            lambda a:a[0]["data_signature"].update(split_sha256="invalid"),
            lambda a:a[0].update(preflight_source_ids_sha256="invalid"),
            lambda a:a[0].update(mean_jsd=float("nan")),
            lambda a:a[0].update(initial_grad_scaler_scale=65536.),
            lambda a:a[0].update(required_training_and_benchmark_seconds=100.)]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                documents = copy.deepcopy(original)
                mutation(documents)
                with self.assertRaises(ValueError):
                    trainer.validate_preflight(*documents, require_t4=True)

    def test_separate_overflow_diagnostic_cannot_be_counted_as_normal_update_or_change_ema(self):
        original = self.fixture()
        for key,value in (("updated", True), ("scale_after", 65536.),
                ("finite_loss", False), ("infinite_gradient_injected", False),
                ("all_parameters_unchanged", False), ("optimizer_state_unchanged", False),
                ("ema_parameters_unchanged", False), ("source_examples", 64),
                ("model_views", 128), ("forward_calls", 2)):
            with self.subTest(field=key):
                documents = copy.deepcopy(original)
                documents[0]["overflow_smoke"][key] = value
                with self.assertRaises(ValueError):
                    trainer.validate_preflight(*documents)


class AugMixWorkerPipelineTests(AugMixCPUCase):
    def data_and_generators(self):
        images = np.random.default_rng(10).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(8, dtype=np.int64) % 10
        rng = np.random.default_rng(2026)
        sampler = torch.Generator(device="cpu").manual_seed(2026)
        loader = torch.Generator(device="cpu").manual_seed(991)
        return images, labels, np.arange(8), rng, sampler, loader

    def assert_owned_workers_stopped(self, records):
        for item in records:
            try:
                process = psutil.Process(item["pid"])
            except psutil.NoSuchProcess:
                continue
            self.assertGreaterEqual(abs(process.create_time()-item["created_at"]), .01,
                                    "An exact test-owned worker remained alive")

    def test_serial_and_two_spawn_workers_produce_identical_order_seeds_pixels_and_rng(self):
        torch_state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            serial_data = self.data_and_generators()
            with trainer.build_epoch_loader(*serial_data, workers=0, output=directory, tag="serial") as serial:
                serial_views = [(images.clone(), targets.clone()) for images,targets in serial]
            source_ids, sample_seeds = serial.source_ids.copy(), serial.sample_seeds.copy()
            parallel_data = self.data_and_generators()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "parent-mask-must-survive"}):
                with trainer.build_epoch_loader(*parallel_data, workers=2, output=directory, tag="parallel") as parallel:
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "parent-mask-must-survive")
                    parallel_views = [(images.clone(), targets.clone()) for images,targets in parallel]
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "parent-mask-must-survive")
            self.assertTrue(serial.closed)
            self.assertTrue(parallel.closed)
            self.assertEqual(len(parallel.worker_records), 2)
            self.assert_owned_workers_stopped(parallel.worker_records)
            self.assertTrue(np.array_equal(source_ids, parallel.source_ids))
            self.assertTrue(np.array_equal(sample_seeds, parallel.sample_seeds))
            self.assertEqual(len(serial_views), len(parallel_views))
            for first,second in zip(serial_views, parallel_views):
                self.assertTrue(torch.equal(first[0], second[0]))
                self.assertTrue(torch.equal(first[1], second[1]))
            self.assertEqual(serial_data[3].bit_generator.state, parallel_data[3].bit_generator.state)
            self.assertTrue(torch.equal(serial_data[4].get_state(), parallel_data[4].get_state()))
            self.assertTrue(torch.equal(serial_data[5].get_state(), parallel_data[5].get_state()))
            receipt = json.loads((Path(directory)/"worker_pool_parallel.json").read_text())
            self.assertEqual(receipt["context"], "spawn")
            self.assertEqual(receipt["prefetch_factor"], 2)
            self.assertTrue(receipt["all_workers_stopped"])
            self.assertFalse(receipt["worker_cuda_initialized"])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_spawn_pipeline_exception_closes_only_its_workers_and_restores_parent_mask(self):
        present, previous = "CUDA_VISIBLE_DEVICES" in os.environ, os.environ.get("CUDA_VISIBLE_DEVICES")
        with tempfile.TemporaryDirectory() as directory:
            pipeline = trainer.build_epoch_loader(*self.data_and_generators(), workers=2,
                output=directory, tag="exception")
            with self.assertRaisesRegex(RuntimeError, "synthetic consumer failure"):
                with pipeline:
                    next(pipeline)
                    raise RuntimeError("synthetic consumer failure")
            self.assertTrue(pipeline.closed)
            self.assert_owned_workers_stopped(pipeline.worker_records)
            receipt = json.loads((Path(directory)/"worker_pool_exception.json").read_text())
            self.assertTrue(receipt["all_workers_stopped"])
            self.assertEqual(receipt["status"], "closed")
        self.assertEqual("CUDA_VISIBLE_DEVICES" in os.environ, present)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), previous)

    def test_worker_startup_failure_cleans_only_new_spawn_children_and_restores_mask(self):
        class BrokenLoader:
            def __iter__(self): raise RuntimeError("synthetic partial spawn failure")
        previous, present = os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES" in os.environ
        with tempfile.TemporaryDirectory() as directory:
            pipeline = trainer.build_epoch_loader(*self.data_and_generators(), workers=2, output=directory, tag="startup")
            pipeline.loader = BrokenLoader()
            existing = Mock(pid=1001)
            existing.create_time.return_value = 10.
            spawned = Mock(pid=1002)
            spawned.create_time.return_value = 20.
            spawned.cmdline.return_value = ["python", "from multiprocessing.spawn import spawn_main"]
            tracker = Mock(pid=1003)
            tracker.create_time.return_value = 20.
            tracker.cmdline.return_value = ["python", "multiprocessing.resource_tracker"]
            owner = Mock()
            owner.children.side_effect = [[existing], [existing, spawned, tracker]]
            with patch.object(trainer.psutil, "Process", return_value=owner), \
                    patch.object(trainer, "cleanup_workers", return_value=True) as cleanup:
                with self.assertRaisesRegex(RuntimeError, "partial spawn"):
                    pipeline.__enter__()
            cleanup.assert_called_once_with([{"pid":1002,"created_at":20.}])
            record = json.loads((Path(directory)/"worker_pool_startup.json").read_text())
            self.assertEqual(record["status"], "startup_failed")
            self.assertTrue(record["all_workers_stopped"])
        self.assertEqual("CUDA_VISIBLE_DEVICES" in os.environ, present)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), previous)

    def test_early_worker_exit_during_identity_capture_still_cleans_surviving_worker(self):
        class ReturnedIterator:
            def __init__(self): self._workers = [Mock(pid=1002), Mock(pid=1003)]
            def __iter__(self): return self
            def __next__(self): raise StopIteration
        class ReturnedLoader:
            def __iter__(self): return ReturnedIterator()
        previous, present = os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES" in os.environ
        with tempfile.TemporaryDirectory() as directory:
            pipeline = trainer.build_epoch_loader(*self.data_and_generators(), workers=2, output=directory, tag="earlyexit")
            pipeline.loader = ReturnedLoader()
            survivor = Mock(pid=1003)
            survivor.create_time.return_value = 20.
            survivor.cmdline.return_value = ["python", "from multiprocessing.spawn import spawn_main"]
            owner = Mock()
            owner.children.side_effect = [[], [survivor]]
            def process(pid=None):
                if pid is None: return owner
                if pid == 1002: raise psutil.NoSuchProcess(pid)
                if pid == 1003: return survivor
                raise AssertionError("Only exact test-owned processes may be inspected")
            with patch.object(trainer.psutil, "Process", side_effect=process), \
                    patch.object(trainer, "cleanup_workers", return_value=True) as cleanup:
                with self.assertRaises(psutil.NoSuchProcess):
                    pipeline.__enter__()
            cleanup.assert_called_once_with([{"pid":1003,"created_at":20.}])
            record = json.loads((Path(directory)/"worker_pool_earlyexit.json").read_text())
            self.assertEqual(record["status"], "startup_failed")
            self.assertEqual(record["workers"], [{"pid":1003,"created_at":20.}])
            self.assertTrue(record["all_workers_stopped"])
            self.assertTrue(pipeline.closed)
        self.assertEqual("CUDA_VISIBLE_DEVICES" in os.environ, present)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), previous)

    def test_checkpoint_replays_sampler_loader_views_and_toy_update_across_worker_counts(self):
        images, labels, ids, rng, sampler, loader = self.data_and_generators()
        model, optimizer, ema = self.model_optimizer()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with trainer.build_epoch_loader(images, labels, ids, rng, sampler, loader, workers=0) as first:
            batch = next(first)
        trainer.train_step(model, optimizer, scaler, batch, torch.device("cpu"), torch.float32,
                           ema=ema, successful_updates=0)
        saved = {"model": copy.deepcopy(model.state_dict()), "ema": copy.deepcopy(ema.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "grad_scaler": scaler.state_dict(),
            **trainer.capture_state_rng(rng, torch.device("cpu"), sampler, loader)}
        with trainer.build_epoch_loader(images, labels, ids, rng, sampler, loader, workers=0) as second:
            expected_batch = next(second)
        expected_order, expected_seeds = second.source_ids.copy(), second.sample_seeds.copy()
        expected_report = trainer.train_step(model, optimizer, scaler, expected_batch,
            torch.device("cpu"), torch.float32, ema=ema, successful_updates=1)
        expected_states = (copy.deepcopy(model.state_dict()), copy.deepcopy(ema.state_dict()),
            copy.deepcopy(optimizer.state_dict()), trainer.capture_state_rng(rng, torch.device("cpu"), sampler, loader))
        resumed, resumed_optimizer, resumed_ema = self.model_optimizer()
        resumed.load_state_dict(saved["model"])
        resumed_ema.load_state_dict(saved["ema"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
        resumed_scaler.load_state_dict(saved["grad_scaler"])
        resumed_rng = np.random.default_rng(10)
        resumed_sampler, resumed_loader = trainer.pipeline_generators()
        trainer.restore_state_rng(saved, resumed_rng, torch.device("cpu"), resumed_sampler, resumed_loader)
        with trainer.build_epoch_loader(images, labels, ids, resumed_rng, resumed_sampler, resumed_loader,
                                        workers=2) as restarted:
            actual_batch = next(restarted)
        self.assertTrue(np.array_equal(restarted.source_ids, expected_order))
        self.assertTrue(np.array_equal(restarted.sample_seeds, expected_seeds))
        self.assertTrue(torch.equal(actual_batch[0], expected_batch[0]))
        actual_report = trainer.train_step(resumed, resumed_optimizer, resumed_scaler, actual_batch,
            torch.device("cpu"), torch.float32, ema=resumed_ema, successful_updates=1)
        self.assertEqual(actual_report, expected_report)
        actual_states = (resumed.state_dict(), resumed_ema.state_dict(), resumed_optimizer.state_dict(),
            trainer.capture_state_rng(resumed_rng, torch.device("cpu"), resumed_sampler, resumed_loader))
        for actual,expected in zip(actual_states, expected_states):
            assert_state_equal(self, actual, expected)
        self.assert_owned_workers_stopped(restarted.worker_records)


class AugMixSupervisorTests(unittest.TestCase):
    def policy(self):
        return {"stops_at": "2026-11-05T05:00:00+00:00", "concurrent_local_gpu_jobs": 1,
                "local_job_timeout_seconds": 10800, "local_gpu_seconds_per_utc_day": 21600,
                "baseline_benchmark": "baseline.json"}

    def test_native_pipeline_admission_keeps_daily_job_hardstop_and_fixed_margin(self):
        policy = self.policy()
        current = datetime(2026, 11, 1, tzinfo=timezone.utc)
        self.assertEqual(supervisor.admission_budget(policy, current, 0), 10800)
        self.assertEqual(supervisor.admission_budget(policy, current, 21000), 600)
        deadline = datetime.fromisoformat(policy["stops_at"])
        self.assertEqual(supervisor.admission_budget(policy, deadline-timedelta(seconds=15), 0), 15)
        preflight = {"median_step_seconds": .0625}
        required = 1.5*.0625*313*160+300
        self.assertEqual(supervisor.admission_budget(policy, current, 21600-required, preflight), required)
        with self.assertRaises(ValueError):
            supervisor.admission_budget(policy, current, 21601-required, preflight)
        for used in (True, -1, float("nan"), float("inf"), 21600):
            with self.subTest(used=used):
                with self.assertRaises(ValueError):
                    supervisor.admission_budget(policy, current, used)
        with self.assertRaises(ValueError):
            supervisor.admission_budget(policy, deadline, 0)
        for changes in ({"concurrent_local_gpu_jobs": 2}, {"local_job_timeout_seconds": 21600},
                        {"local_gpu_seconds_per_utc_day": 86400}):
            with self.assertRaises(ValueError):
                supervisor.admission_budget({**policy, **changes}, current, 0)

    def test_existing_gpu_lease_accounting_and_timeout_cleanup_remain_shared(self):
        self.assertIs(supervisor.acquire_lease, run_local.acquire_lease)
        self.assertIs(supervisor.used_seconds, run_local.used_seconds)
        from research import process_tree_guard
        self.assertIs(supervisor.run_guarded, process_tree_guard.run_guarded)
        owned = [{"pid": 999991, "created_at": 123.}]
        receipt = {"child_pid": 12345, "child_created_at": 789., "process_tree_cleanup_verified": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,pid,created in (("own", 12345, 789.), ("foreign", 12346, 789.), ("reused", 12345, 790.)):
                (root/f"worker_pool_{name}.json").write_text(json.dumps({
                    "owner_pid": pid, "owner_created_at": created, "workers": owned}))
            def verified_tree_timeout(*args, **kwargs):
                receipt["process_tree_cleanup_verified"] = True
                raise TimeoutError("mock child timeout")
            with patch.object(supervisor, "cleanup_workers", return_value=True) as cleanup, \
                    patch.object(supervisor, "run_guarded", side_effect=verified_tree_timeout):
                with self.assertRaisesRegex(TimeoutError, "mock child"):
                    supervisor.execute(["mock-only"], root/"log", 1, root, receipt)
                cleanup.assert_called_once_with(owned)
                self.assertTrue(receipt["process_tree_cleanup_verified"])

    def resume_budget_failure(self, used_sequence):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"research").mkdir()
            (root/"research/campaign.json").write_text(json.dumps(self.policy()))
            config = root/"config.json"
            config.write_text(json.dumps(trainer.fixed_config()))
            sources = {"method": "synthetic-source-binding"}
            preflight_dir = root/"results/research/preflight"
            preflight_dir.mkdir(parents=True)
            preflight = preflight_dir/"engineering_benchmark.json"
            preflight.write_text(json.dumps({"status": "complete", "median_step_seconds": .0625}))
            preflight_receipt = preflight_dir/"supervisor.json"
            preflight_receipt.write_text(json.dumps({"schema": supervisor.SCHEMA, "mode": "preflight",
                "status": "complete", "source_sha256": sources,
                "artifact_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest()}))
            binding = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (preflight, preflight_receipt)}
            run = root/"results/research/augmix_resume"
            run.mkdir()
            (run/"last.pt").write_bytes(b"synthetic placeholder, never loaded or trained")
            previous = {"schema": supervisor.SCHEMA, "mode": "train", "run": "augmix_resume", "status": "failed",
                "source_sha256": sources, "input_sha256": binding,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "pid": 2147483647,
                "process_created_at": 0., "started_at": "2026-11-01T00:00:00+00:00",
                "finished_at": "2026-11-01T01:00:00+00:00", "reserved_seconds": 10800,
                "elapsed_seconds": 3600.}
            receipt = run/"supervisor.json"
            receipt.write_text(json.dumps(previous))
            original = receipt.read_bytes()
            args = ["run_augmix_local.py", "--run", "augmix_resume", "--mode", "train", "--config", str(config),
                    "--preflight", str(preflight), "--resume"]
            with patch.object(supervisor, "ROOT", root), patch.object(sys, "argv", args), \
                    patch.object(supervisor, "now", return_value=datetime(2026, 11, 1, 2, tzinfo=timezone.utc)), \
                    patch.object(supervisor, "inputs", return_value={}), \
                    patch.object(supervisor, "bound_sources", return_value=sources), \
                    patch.object(supervisor, "validate_preflight", side_effect=lambda report,*args:report), \
                    patch.object(supervisor, "assert_stopped"), \
                    patch.object(supervisor, "used_seconds", side_effect=used_sequence), \
                    patch.object(supervisor, "environment", side_effect=AssertionError("no GPU query")), \
                    patch.object(supervisor, "execute", side_effect=AssertionError("no child may start")) as child:
                with self.assertRaisesRegex(ValueError, "budget"):
                    supervisor.main()
                child.assert_not_called()
            self.assertFalse((root/"results/research/local_lease.json").exists())
            return original, receipt.read_bytes(), {path.name:path.read_bytes() for path in run.glob("supervisor_*_*.json")}

    def test_resume_failure_before_reconciliation_never_overwrites_old_receipt(self):
        original, canonical, archives = self.resume_budget_failure([21600])
        self.assertEqual(original, canonical)
        self.assertEqual(archives, {})

    def test_resume_failure_after_reconciliation_preserves_raw_bytes_reservation_and_link(self):
        original, canonical, archives = self.resume_budget_failure([0, 21600])
        self.assertEqual(archives["supervisor_raw_1.json"], original)
        self.assertEqual(json.loads(archives["supervisor_attempt_1.json"])["reserved_seconds"], 10800)
        failed = json.loads(canonical)
        self.assertEqual(failed["previous_attempt"], "supervisor_attempt_1.json")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["reserved_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
