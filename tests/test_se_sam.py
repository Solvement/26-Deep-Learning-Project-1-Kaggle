"""CPU-only SE/SAM controls; all batches and checkpoint fixtures are synthetic.

These tests never load official examples, confirmation or hidden images. A
synthetic complete-history fixture exercises rejection rules, not experiment
evidence. Native GPU engineering and official benchmark results remain pending.
"""
import copy
from datetime import datetime, timedelta, timezone
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
from torch.nn import functional as F

from model import count_parameters
from research import benchmark as core_benchmark
from research import benchmark_se_sam as se_benchmark
from research import run_se_sam as se_run
from research import run_local
from research import train_se_sam as se_train
from research.se_model import ARCHITECTURE, PARAMETER_COUNT, build_model
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
        for a, b in zip(left, right):
            assert_state_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def synthetic_saved_state(config):
    """Real SE tensor topology, synthetic zero momentum; no training claim."""
    model = build_model(se_train.model_configuration(config)).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9,
                                weight_decay=.0005, nesterov=True)
    state = optimizer.state_dict()
    state["state"] = {index:{"momentum_buffer": torch.zeros_like(parameter)}
                      for index,parameter in enumerate(model.parameters())}
    return {"model": model.state_dict(), "ema": copy.deepcopy(model.state_dict()), "optimizer": state}


class SECPUCase(unittest.TestCase):
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
        model = build_model(se_train.model_configuration(se_train.fixed_config())).cpu().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9,
                                    weight_decay=.0005, nesterov=True)
        return model, optimizer

    def synthetic_step(self, model, optimizer, images, labels):
        config = se_train.fixed_config()
        batch = se_train.prepare_batch(images, labels, config)
        closure = se_train.loss_closure(model, batch, config, torch.device("cpu"), torch.float32, {})
        return se_train.sam_step(model, optimizer, closure, rho=.05, precision="fp32")


class SEArchitectureSAMTests(SECPUCase):
    def test_real_se_gate_gradients_updates_and_one_bn_update_per_sam_step(self):
        model, optimizer = self.model_optimizer()
        self.assertEqual(count_parameters(model), PARAMETER_COUNT)
        self.assertEqual(count_parameters(model, True), 4_332_654)
        gates = se_train.gate_parameters(model)
        self.assertEqual(len(gates), 36)
        self.assertEqual(len({name.split(".se.")[0] for name in gates}), 9)
        before = {name: parameter.detach().clone() for name, parameter in gates.items()}
        gradients = {}
        handles = [parameter.register_hook(lambda gradient, name=name:
                    gradients.__setitem__(name, gradient.detach().clone()))
                   for name, parameter in gates.items()]
        images = torch.randn(4, 3, 8, 8)
        labels = torch.tensor([0, 3, 6, 9])
        bn = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        counters = [module.num_batches_tracked.detach().clone() for module in bn]
        try:
            evidence = se_train.sam_step(model, optimizer,
                lambda: F.cross_entropy(model(images), labels), rho=.05, precision="fp32")
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(evidence["updated"])
        self.assertEqual(evidence["forward_backward_passes"], 2)
        self.assertLessEqual(evidence["perturbation_norm"], .050001)
        self.assertEqual(set(gradients), set(gates))
        for gradient in gradients.values():
            self.assertTrue(torch.isfinite(gradient).all())
        for module, previous in zip(bn, counters):
            self.assertTrue(torch.equal(module.num_batches_tracked, previous+1))
        se_train.gate_update_evidence(model, before, gradients)
        # A real SGD change on the base stem alone must not qualify SE gates.
        unchanged = {name: parameter.detach().clone() for name, parameter in gates.items()}
        with self.assertRaises((ValueError, FloatingPointError, RuntimeError)):
            se_train.gate_update_evidence(model, unchanged, gradients)
        broken = dict(gradients)
        first = next(iter(broken))
        broken[first] = torch.full_like(broken[first], float("nan"))
        with self.assertRaises((ValueError, FloatingPointError, RuntimeError)):
            se_train.gate_update_evidence(model, before, broken)

    def test_se_second_pass_failure_restores_gate_base_bn_and_sgd_state(self):
        model, optimizer = self.model_optimizer()
        images, labels = torch.randn(3, 3, 8, 8), torch.tensor([1, 4, 8])
        se_train.sam_step(model, optimizer, lambda: F.cross_entropy(model(images), labels))
        before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            loss = F.cross_entropy(model(images), labels)
            if calls == 2:
                raise RuntimeError("synthetic second-pass failure")
            return loss
        with self.assertRaisesRegex(RuntimeError, "synthetic second-pass"):
            se_train.sam_step(model, optimizer, closure)
        self.assertEqual(calls, 2)
        assert_state_equal(self, model.state_dict(), before_model)
        assert_state_equal(self, optimizer.state_dict(), before_optimizer)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_cpu_resume_replays_augmentation_mixup_se_weights_bn_and_momentum_exactly(self):
        model, optimizer = self.model_optimizer()
        images = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
        labels = torch.tensor([0, 4, 9])
        self.synthetic_step(model, optimizer, images, labels)
        saved = {"model": copy.deepcopy(model.state_dict()),
                 "optimizer": copy.deepcopy(optimizer.state_dict()),
                 **se_train.capture_rng(torch.device("cpu"))}
        expected_report = self.synthetic_step(model, optimizer, images, labels)
        expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        expected_rng = se_train.capture_rng(torch.device("cpu"))
        resumed, resumed_optimizer = self.model_optimizer()
        resumed.load_state_dict(saved["model"], strict=True)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        se_train.restore_rng(saved, torch.device("cpu"))
        actual_report = self.synthetic_step(resumed, resumed_optimizer, images, labels)
        self.assertEqual(actual_report, expected_report)
        assert_state_equal(self, resumed.state_dict(), expected_model)
        assert_state_equal(self, resumed_optimizer.state_dict(), expected_optimizer)
        assert_state_equal(self, se_train.capture_rng(torch.device("cpu")), expected_rng)


class SERecipeControlTests(SECPUCase):
    def test_only_architecture_metadata_differs_from_fixed_sam_recipe(self):
        reference = {**DEFAULT_CONFIG, **json.loads((ROOT/"research/configs/wrn22_sam.json").read_text())}
        config = se_train.fixed_config()
        self.assertEqual({key:value for key,value in config.items()
                          if key not in {"architecture", "se_reduction"}}, reference)
        self.assertEqual(config["architecture"], ARCHITECTURE)
        self.assertEqual(config["se_reduction"], 16)
        se_train.validate_config(config)
        repeat = se_train.fixed_config(2027)
        self.assertEqual({key:value for key,value in repeat.items() if key != "seed"},
                         {key:value for key,value in config.items() if key != "seed"})
        se_train.validate_config(repeat)

    def test_nonarchitecture_recipe_changes_cannot_hide_in_se_comparison(self):
        changes = {"epochs": 80, "batch_size": 64, "lr": .2, "min_lr": .002,
            "warmup_epochs": 3, "momentum": .8, "weight_decay": .0001,
            "label_smoothing": .1, "cutout": 0, "mixup_alpha": 1., "mixup_probability": .2,
            "ema_decay": .9, "dropout": .1, "depth": 34, "widen_factor": 3, "num_classes": 100,
            "seed": 2028, "split_seed": 2027, "train_all": True, "final_variant": "model",
            "fixed_flip_tta": True, "sam_rho": .1, "architecture": "other", "se_reduction": 8,
            "unknown_method": 1}
        for key,value in changes.items():
            with self.subTest(field=key):
                config = se_train.fixed_config()
                config[key] = value
                with self.assertRaises(ValueError):
                    se_train.validate_config(config)

    def test_effective_config_and_raw_input_have_distinct_bound_fingerprints(self):
        config = se_train.fixed_config()
        first = se_train.research_provenance("fp32", config, "1"*64)
        second = se_train.research_provenance("fp32", config, "2"*64)
        self.assertEqual(first["schema"], "research_se_sam_training_v1")
        self.assertNotEqual(first, second)
        self.assertFalse(first["grad_scaler_enabled"])
        self.assertFalse(first["confirmation_evaluated"])
        self.assertFalse(first["hidden_images_loaded"])
        self.assertFalse(first["external_data_loaded"])
        with self.assertRaises(ValueError):
            se_train.research_provenance("fp16", config)

    def resume_fixture(self):
        config = se_train.fixed_config()
        provenance = {"source_sha256": {"train.py": "synthetic-frozen-core"},
                      "research_training": se_train.research_provenance("fp32", config)}
        signature = {"archive_sha256": "a"*64, "split_sha256": "b"*64}
        last = {"schema": se_train.SCHEMA, "config": config, "data_signature": signature,
            "model_config": se_train.model_configuration(config), "parameter_count": PARAMETER_COUNT,
            "provenance": copy.deepcopy(provenance), "epoch": 1, "updates": 313,
            "skipped_updates": 0, "sam_forward_passes": 626,
            "history": [{"epoch": 1, "successful_updates": 313, "skipped_updates": 0,
                         "sam_forward_passes": 626}],
            **synthetic_saved_state(config),
            **se_train.capture_rng(torch.device("cpu"))}
        return config, provenance, signature, last

    def test_resume_rejects_source_method_raw_config_and_optimizer_drift(self):
        config, provenance, signature, last = self.resume_fixture()
        se_train.validate_resume(last, config, signature, provenance, 313)
        changes = [lambda x:x.update(schema="research_sam_training_v1"),
            lambda x:x["provenance"]["source_sha256"].update({"train.py": "other-core"}),
            lambda x:x["provenance"]["research_training"].update(precision="bf16"),
            lambda x:x["provenance"]["research_training"].update(input_config_sha256="f"*64),
            lambda x:x["model_config"].update(reduction=8),
            lambda x:x.update(parameter_count=4_298_970),
            lambda x:x["optimizer"]["param_groups"][0].update(weight_decay=.0001),
            lambda x:x["optimizer"]["param_groups"][0].update(nesterov=False),
            lambda x:x["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1)),
            lambda x:x["optimizer"]["state"].pop(0),
            lambda x:x.update(grad_scaler={"scale": 65536.}),
            lambda x:x.pop("numpy_rng"), lambda x:x.update(torch_rng=torch.ones(3))]
        for mutation in changes:
            with self.subTest(change=changes.index(mutation)):
                altered = copy.deepcopy(last)
                mutation(altered)
                with self.assertRaises(ValueError):
                    se_train.validate_resume(altered, config, signature, provenance, 313)
        with self.assertRaises(ValueError):
            se_train.validate_resume(last, config, {**signature, "split_sha256": "c"*64}, provenance, 313)

    def test_resume_rejects_missing_twopass_updates_complete_or_disordered_history(self):
        config, provenance, signature, last = self.resume_fixture()
        changes = [lambda x:x.update(updates=312), lambda x:x.update(skipped_updates=1),
            lambda x:x.update(sam_forward_passes=625), lambda x:x.update(epoch=160),
            lambda x:x["history"][0].update(successful_updates=312),
            lambda x:x["history"][0].update(sam_forward_passes=313),
            lambda x:x["history"][0].update(epoch=2)]
        for index,mutation in enumerate(changes):
            with self.subTest(change=index):
                altered = copy.deepcopy(last)
                mutation(altered)
                with self.assertRaises(ValueError):
                    se_train.validate_resume(altered, config, signature, provenance, 313)
        with self.assertRaises(ValueError):
            se_train.validate_resume(last, config, signature, provenance, 312)

    def test_bf16_resume_cannot_use_an_empty_cuda_rng_as_native_proof(self):
        config, provenance, signature, last = self.resume_fixture()
        marker = se_train.research_provenance("bf16", config)
        last["provenance"]["research_training"] = copy.deepcopy(marker)
        provenance["research_training"] = marker
        with self.assertRaisesRegex(ValueError, "RNG"):
            se_train.validate_resume(last, config, signature, provenance, 313)


class SEBenchmarkAuditTests(SECPUCase):
    def complete_training_fixture(self):
        config = se_train.fixed_config()
        raw = se_train.input_config(config)
        marker = se_train.research_provenance("fp32", config)
        sources = {name:core_benchmark.sha256(ROOT/name)
                   for name in ("train.py", "model.py", "data.py", "runtime.py")}
        code = {"source_sha256": sources, "research_training": marker}
        signature = {"archive_sha256": "a"*64, "split_sha256": "b"*64}
        checkpoint = {"schema": se_train.SCHEMA, "config": config,
            "model_config": se_train.model_configuration(config), "parameter_count": PARAMETER_COUNT,
            "provenance": code, "data_signature": signature, "epoch": 160,
            "variant": "model", "validation_accuracy": .9, "validation_loss": .2}
        metrics = {"schema": se_train.SCHEMA, "research_training": copy.deepcopy(marker),
            "config": copy.deepcopy(config), "parameter_count": PARAMETER_COUNT,
            "status": "complete", "epochs_completed": 160, "training_examples": 40000,
            "validation_examples": 5000, "successful_updates": 50080, "skipped_updates": 0,
            "sam_forward_passes": 100160, "best_validation_accuracy": .9, "best_validation_loss": .2}
        history = [{"epoch": epoch, "successful_updates": 313, "sam_forward_passes": 626,
            "skipped_updates": 0, "train_loss": .3, "sam_loss": .4,
            "mean_first_grad_norm": 1., "mean_perturbation_norm": .05,
            "validation_loss": .2, "ema_validation_loss": .21,
            "epoch_seconds": 1., "elapsed_seconds": float(epoch), "lr": .0001,
            "train_accuracy": .85, "validation_accuracy": .9, "ema_validation_accuracy": .89,
            "selected_variant": "model"} for epoch in range(1, 161)]
        provenance = {"schema": se_train.SCHEMA, "code": copy.deepcopy(code),
            "train_examples": 40000, "validation_examples": 5000,
            "parameter_count": PARAMETER_COUNT, "data_signature": signature}
        environment = {"schema": se_train.SCHEMA, "research_training": copy.deepcopy(marker),
            "se_sam_training_precision": "fp32", "grad_scaler_enabled": False}
        return checkpoint, metrics, history, provenance, environment, raw

    def test_complete_training_audit_rejects_relabelled_sam_source_and_partial_history(self):
        original = self.complete_training_fixture()
        proof = se_benchmark.validate_training(*original)
        self.assertEqual(proof["schema"], "research_se_sam_integrity_v1")
        self.assertEqual((proof["successful_updates"], proof["sam_forward_passes"]), (50080, 100160))
        mutations = [lambda a:a[0].update(schema="research_sam_training_v1"),
            lambda a:a[0]["provenance"]["research_training"].update(source_sha256={"method": "old-SAM"}),
            lambda a:a[3]["code"]["source_sha256"].update({"train.py": "changed"}),
            lambda a:a[0]["model_config"].update(architecture="ordinary_wrn"),
            lambda a:a[1].update(parameter_count=4_298_970),
            lambda a:a[1].update(status="running", epochs_completed=159),
            lambda a:a[2].pop(), lambda a:a[2][80].update(successful_updates=312),
            lambda a:a[2][80].update(sam_forward_passes=313),
            lambda a:a[2][80].update(skipped_updates=1),
            lambda a:a[1].update(successful_updates=50079),
            lambda a:a[1].update(sam_forward_passes=100159),
            lambda a:a[1].update(training_examples=50000, validation_examples=0),
            lambda a:a[4].update(grad_scaler_enabled=True)]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                documents = copy.deepcopy(original)
                mutation(documents)
                with self.assertRaises(ValueError):
                    se_benchmark.validate_training(*documents)

    def test_complete_training_requires_finite_loss_norm_and_bound_raw_config(self):
        original = self.complete_training_fixture()
        for key,value in (("sam_loss", float("nan")), ("mean_first_grad_norm", float("inf")),
                ("mean_perturbation_norm", .06), ("validation_accuracy", 1.1)):
            with self.subTest(field=key):
                documents = copy.deepcopy(original)
                documents[2][55][key] = value
                with self.assertRaises(ValueError):
                    se_benchmark.validate_training(*documents)
        changed = list(copy.deepcopy(original))
        changed[-1] = changed[-1]+b" "  # Same effective JSON, different raw bytes.
        with self.assertRaises(ValueError):
            se_benchmark.validate_training(*changed)

    def test_final_state_cannot_relabel_an_ordinary_wrn_as_the_se_architecture(self):
        documents = self.complete_training_fixture()
        checkpoint, _, history, _, _, _ = documents
        config = checkpoint["config"]
        saved = synthetic_saved_state(config)
        last = {"schema": se_train.SCHEMA, "config": copy.deepcopy(config), "epoch": 160,
            "model_config": se_train.model_configuration(config), "parameter_count": PARAMETER_COUNT,
            "provenance": copy.deepcopy(checkpoint["provenance"]), "history": copy.deepcopy(history),
            "data_signature": copy.deepcopy(checkpoint["data_signature"]),
            "updates": 50080, "sam_forward_passes": 100160, "skipped_updates": 0,
            **saved,
            **se_train.capture_rng(torch.device("cpu"))}
        se_benchmark.validate_training(*documents, last=last)
        ordinary_state = {name:value for name,value in saved["model"].items() if ".se." not in name}
        with self.assertRaises((ValueError, RuntimeError)):
            se_benchmark.validate_training(*documents, last={**last, "model": ordinary_state})
        with self.assertRaises((ValueError, RuntimeError)):
            se_benchmark.validate_training(*documents, last={**last, "ema": ordinary_state})
        with self.assertRaises(ValueError):
            se_benchmark.validate_training(*documents, last={**last, "updates": 50079})

    def benchmark_fixture(self, clean=.9686, stress=.794):
        signature = {"archive_sha256": "a"*64, "split_sha256": "b"*64}
        context = {"device_type": "cuda", "amp_dtype": "torch.bfloat16",
            "torch_version": "synthetic-test-context", "training_core_sha256": {"core": "same"}}
        def report(accuracy, stress_accuracy):
            cases = {case: {"accuracy": accuracy if case == "clean" else stress_accuracy,
                "loss": .2, "examples": 5000, "input_pixel_sha256": f"{index+1:064x}"}
                for index,case in enumerate(core_benchmark.CASES)}
            return {"cases": cases, "clean_accuracy": accuracy, "clean_loss": .2,
                    "stress_accuracy": stress_accuracy}
        def document(checkpoint, accuracy, stress_accuracy):
            return {"schema": core_benchmark.SCHEMA, "status": "complete", "examples": 5000,
                "training_examples": 40000, "parameter_count": 4298970,
                "data_signature": copy.deepcopy(signature), "evaluation_context": copy.deepcopy(context),
                "batch_size": 128, "bn_recalibrated": False, "checkpoint_sha256": checkpoint,
                "stress_cases": list(core_benchmark.CASES),
                "benchmark_source_sha256": core_benchmark.sha256(ROOT/"research/benchmark.py"),
                "policies": {policy:report(accuracy, stress_accuracy) for policy in core_benchmark.POLICIES}}
        reports = {policy:report(clean, stress) for policy in core_benchmark.POLICIES}
        original = document(core_benchmark.BASELINE_CHECKPOINT_SHA, .9628, .77628)
        sam = document(se_benchmark.SAM_CHECKPOINT_SHA256, .9674, .79368)
        return reports, original, sam, signature, 128, context

    def test_se_must_improve_fixed_sam_as_well_as_original_baseline(self):
        good = se_benchmark.dual_gate(*self.benchmark_fixture())
        self.assertTrue(good["passed"])
        self.assertEqual(good["policy"], "center80_shift20")
        self.assertFalse(good["alternative_policies_promotable"])
        # Inherits SAM's old improvement but adds no SE gain versus SAM.
        failed = se_benchmark.dual_gate(*self.benchmark_fixture(.9674, .79368))
        self.assertTrue(failed["against_original"]["center80_shift20"]["passed"])
        self.assertFalse(failed["against_fixed_sam"]["center80_shift20"]["passed"])
        self.assertFalse(failed["passed"])
        # Passing alternative policies cannot rescue the predeclared policy.
        documents = self.benchmark_fixture()
        documents[0]["center80_shift20"].update(clean_accuracy=.9674, stress_accuracy=.79368)
        self.assertTrue(se_benchmark.dual_gate(*documents)["against_fixed_sam"]["flip"]["passed"])
        self.assertFalse(se_benchmark.dual_gate(*documents)["passed"])

    def test_se_dual_gate_rejects_cross_context_pixels_partial_cases_or_moving_sam_reference(self):
        original = self.benchmark_fixture()
        mutations = [lambda a:a[2]["evaluation_context"].update(amp_dtype="torch.float16"),
            lambda a:a[2].update(checkpoint_sha256="f"*64),
            lambda a:a[2].update(training_examples=50000),
            lambda a:a[2].update(bn_recalibrated=True),
            lambda a:a[2]["policies"]["center80_shift20"]["cases"]["noise_004"].update(examples=4999),
            lambda a:a[2]["policies"]["center80_shift20"]["cases"]["noise_004"].update(input_pixel_sha256="f"*64),
            lambda a:a[0]["flip"]["cases"]["resize24"].update(input_pixel_sha256="f"*64),
            lambda a:a[1].update(benchmark_source_sha256="f"*64),
            lambda a:a[0].pop("flip")]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                documents = copy.deepcopy(original)
                mutation(documents)
                with self.assertRaises(ValueError):
                    se_benchmark.dual_gate(*documents)


class SEBudgetAndPreflightTests(unittest.TestCase):
    def test_formal_seed2027_is_refused_before_lease_without_a_primary_freeze_route(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/"repeat.json"
            config.write_text(json.dumps(se_train.fixed_config(2027)))
            arguments = ["run_se_sam.py", "--run", "se_repeat", "--mode", "train", "--config", str(config),
                         "--preflight", str(Path(directory)/"not-yet-existing-preflight.json")]
            with patch.object(sys, "argv", arguments), \
                    patch.object(se_run, "inputs", side_effect=AssertionError("no official data access")), \
                    patch.object(se_run, "acquire_lease", side_effect=AssertionError("no GPU lease")):
                with self.assertRaisesRegex(ValueError, "primary pairing/freeze"):
                    se_run.main()

    def policy(self):
        return {"stops_at": "2026-11-05T05:00:00+00:00", "concurrent_local_gpu_jobs": 1,
                "local_job_timeout_seconds": 10800, "local_gpu_seconds_per_utc_day": 21600}

    def preflight_fixture(self):
        config = se_train.fixed_config()
        raw_sha = "1"*64
        gates = {f"stage{stage}.{block}": {"gradient_l2": .01, "update_l2": .001}
                 for stage in (1, 2, 3) for block in range(3)}
        report = {"schema": se_train.ENGINEERING_SCHEMA, "status": "complete", "synthetic_only": True,
            "precision": "bf16", "parameters": PARAMETER_COUNT,
            "model_config": se_train.model_configuration(config), "config": config,
            "input_config_sha256": raw_sha, "research_training": se_train.research_provenance("bf16", config, raw_sha),
            "batch_size": 128, "warmup_steps": 3, "timed_steps": 20, "successful_updates": 23,
            "sam_forward_passes": 46, "skipped_updates": 0, "gate_count": 9,
            "gate_changed_steps": 23, "gate_gradient_verified_steps": 23,
            "second_pass_bn_verified": True, "gate_evidence": [copy.deepcopy(gates) for _ in range(23)],
            "median_step_seconds": .04, "p90_step_seconds": .045,
            "required_training_and_benchmark_seconds": 1.5*.04*313*160+300,
            "mean_first_loss": .3, "mean_sam_loss": .4, "mean_perturbation_norm": .05,
            "environment": {"device": "cuda", "amp_dtype": "torch.bfloat16",
                "gpu": "synthetic CPU test fixture, not measured GPU", "packages": {"torch": "fixture"}},
            "training_data_loaded": False, "confirmation_evaluated": False,
            "hidden_images_loaded": False, "external_data_loaded": False}
        return report, config, raw_sha, se_run.bound_sources()

    def test_budget_caps_daily_job_hardstop_and_declared_pipeline_margin(self):
        policy = self.policy()
        current = datetime(2026, 11, 1, tzinfo=timezone.utc)
        self.assertEqual(se_run.admission_budget(policy, current, 0), 10800)
        self.assertEqual(se_run.admission_budget(policy, current, 21000), 600)
        deadline = datetime.fromisoformat(policy["stops_at"])
        self.assertEqual(se_run.admission_budget(policy, deadline-timedelta(seconds=30), 0), 30)
        # A binary-exact duration makes the equality boundary exact; real
        # admission remains conservative and never rounds a budget upward.
        preflight = {"median_step_seconds": .0625}
        required = 1.5*.0625*313*160+300
        self.assertEqual(se_run.admission_budget(policy, current, 21600-required, preflight), required)
        with self.assertRaises(ValueError):
            se_run.admission_budget(policy, current, 21600-required+1, preflight)
        for invalid in (True, -1, float("nan"), float("inf"), 21600):
            with self.subTest(used=invalid):
                with self.assertRaises(ValueError):
                    se_run.admission_budget(policy, current, invalid)
        for changes in ({"concurrent_local_gpu_jobs": 2}, {"local_job_timeout_seconds": 21600},
                        {"local_gpu_seconds_per_utc_day": 86400}):
            with self.assertRaises(ValueError):
                se_run.admission_budget({**policy, **changes}, current, 0)
        with self.assertRaises(ValueError):
            se_run.admission_budget(policy, deadline, 0)
        with self.assertRaises(ValueError):
            se_run.admission_budget(policy, current.replace(tzinfo=None), 0)

    def test_preflight_rejects_cpu_emulation_incomplete_gates_bn_or_changed_source(self):
        original = self.preflight_fixture()
        self.assertEqual(se_run.validate_preflight(*original), original[0])
        mutations = [lambda a:a[0].update(precision="fp32"),
            lambda a:a[0].update(parameters=4_298_970), lambda a:a[0].update(timed_steps=1),
            lambda a:a[0].update(successful_updates=22), lambda a:a[0].update(sam_forward_passes=45),
            lambda a:a[0].update(gate_changed_steps=22), lambda a:a[0].update(second_pass_bn_verified=False),
            lambda a:a[0]["gate_evidence"][12].pop("stage1.1"),
            lambda a:a[0]["gate_evidence"][12]["stage1.1"].update(gradient_l2=float("nan")),
            lambda a:a[0]["gate_evidence"][12]["stage1.1"].update(update_l2=0.),
            lambda a:a[0]["environment"].update(device="cpu"),
            lambda a:a[0]["environment"].update(amp_dtype="torch.float16"),
            lambda a:a[0].update(input_config_sha256="2"*64),
            lambda a:a[0].update(median_step_seconds=float("nan")),
            lambda a:a[0].update(required_training_and_benchmark_seconds=100.),
            lambda a:a[0].update(training_data_loaded=True),
            lambda a:a[0].update(confirmation_evaluated=True),
            lambda a:a[3].update({"research/se_model.py": "f"*64})]
        for index,mutation in enumerate(mutations):
            with self.subTest(change=index):
                documents = copy.deepcopy(original)
                mutation(documents)
                with self.assertRaises(ValueError):
                    se_run.validate_preflight(*documents)

    def test_se_supervisor_shares_existing_live_lease_and_accounting(self):
        self.assertIs(se_run.acquire_lease, run_local.acquire_lease)
        self.assertIs(se_run.used_seconds, run_local.used_seconds)
        self.assertIs(se_run.supervised_command, run_local.supervised_command)
        with tempfile.TemporaryDirectory() as folder:
            lease = Path(folder)/"local_lease.json"
            run_local.acquire_lease(lease)
            with self.assertRaisesRegex(RuntimeError, "lease"):
                se_run.acquire_lease(lease)

    def test_native_preflight_cannot_hide_nonfinite_sam_loss_or_excessive_rho(self):
        original = self.preflight_fixture()
        for key,value in (("mean_first_loss", float("nan")), ("mean_sam_loss", float("inf")),
                          ("mean_first_loss", -.1), ("mean_perturbation_norm", .06)):
            with self.subTest(field=key, value=value):
                documents = copy.deepcopy(original)
                documents[0][key] = value
                with self.assertRaises(ValueError):
                    se_run.validate_preflight(*documents)

    def interrupted_supervisor_fixture(self, root):
        """Only temporary CPU records; all child execution is mocked out."""
        (root/"research").mkdir()
        policy = {**self.policy(), "baseline_benchmark": "baseline.json"}
        (root/"research/campaign.json").write_text(json.dumps(policy))
        config_path = root/"config.json"
        config_path.write_text(json.dumps(se_train.fixed_config()))
        sources = {"method": "synthetic-source-binding"}
        preflight = root/"results/research/preflight"
        preflight.mkdir(parents=True)
        preflight_path = preflight/"engineering_benchmark.json"
        preflight_path.write_text(json.dumps({"status": "complete", "median_step_seconds": .04}))
        preflight_receipt = preflight/"supervisor.json"
        preflight_receipt.write_text(json.dumps({"schema": se_run.SCHEMA, "mode": "preflight",
            "status": "complete", "source_sha256": sources,
            "artifact_sha256": core_benchmark.sha256(preflight_path)}))
        binding = {str(preflight_path):core_benchmark.sha256(preflight_path),
                   str(preflight_receipt):core_benchmark.sha256(preflight_receipt)}
        run = root/"results/research/se_resume"
        run.mkdir()
        (run/"last.pt").write_bytes(b"synthetic checkpoint placeholder; never loaded or executed")
        previous = {"schema": se_run.SCHEMA, "mode": "train", "run": "se_resume", "status": "failed",
            "source_sha256": sources, "input_sha256": binding,
            "config_sha256": core_benchmark.sha256(config_path), "pid": 2147483647,
            "process_created_at": 0., "started_at": "2026-11-01T00:00:00+00:00",
            "finished_at": "2026-11-01T01:00:00+00:00", "reserved_seconds": 10800,
            "elapsed_seconds": 3600.}
        (run/"supervisor.json").write_text(json.dumps(previous))
        return config_path, preflight_path, run, sources

    def simulate_resume_budget_failure(self, used_sequence):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, preflight, run, sources = self.interrupted_supervisor_fixture(root)
            original = (run/"supervisor.json").read_bytes()
            arguments = ["run_se_sam.py", "--run", "se_resume", "--mode", "train", "--config", str(config),
                         "--preflight", str(preflight), "--resume"]
            with patch.object(se_run, "ROOT", root), patch.object(sys, "argv", arguments), \
                    patch.object(se_run, "now", return_value=datetime(2026, 11, 1, 2, tzinfo=timezone.utc)), \
                    patch.object(se_run, "inputs", return_value={}), \
                    patch.object(se_run, "bound_sources", return_value=sources), \
                    patch.object(se_run, "validate_preflight", side_effect=lambda report,*args:report), \
                    patch.object(se_run, "assert_stopped"), \
                    patch.object(se_run, "environment", side_effect=AssertionError("no GPU environment query")), \
                    patch.object(se_run, "used_seconds", side_effect=used_sequence), \
                    patch.object(se_run, "supervised_command", side_effect=AssertionError("no child may start")) as child:
                with self.assertRaisesRegex(ValueError, "budget"):
                    se_run.main()
                child.assert_not_called()
            self.assertFalse((root/"results/research/local_lease.json").exists())
            return original, (run/"supervisor.json").read_bytes(), {
                p.name:p.read_bytes() for p in run.glob("supervisor_*_*.json")}

    def test_resume_budget_failure_before_archive_preserves_original_receipt_bytes(self):
        original, canonical, archives = self.simulate_resume_budget_failure([21600])
        self.assertEqual(canonical, original)
        self.assertEqual(archives, {})

    def test_resume_budget_failure_after_archive_keeps_original_bytes_and_attempt_link(self):
        original, canonical, archives = self.simulate_resume_budget_failure([0, 21600])
        self.assertEqual(archives["supervisor_raw_1.json"], original)
        attempt = json.loads(archives["supervisor_attempt_1.json"])
        self.assertEqual(attempt["reserved_seconds"], 10800)
        failed = json.loads(canonical)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["previous_attempt"], "supervisor_attempt_1.json")
        self.assertTrue(failed["resumed"])
        previous = json.loads(original)
        for key in ("source_sha256", "input_sha256", "config_sha256"):
            self.assertEqual(failed[key], previous[key])
        self.assertEqual(failed["reserved_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
