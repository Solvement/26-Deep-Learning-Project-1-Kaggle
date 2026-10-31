"""CPU SWA parameter arithmetic and isolated sample-weighted training BN stats."""
import copy
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from model import build_model, count_parameters
from train import MEAN, STD, normalize
from research import swa


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(test, left, right):
    test.assertEqual(left[0], right[0])
    test.assertTrue(all(np.array_equal(a, b) for a, b in zip(left[1], right[1])))
    test.assertTrue(torch.equal(left[2], right[2]))


def images_with_tail():
    # Expanded synthetic images keep this arithmetic test small in RAM.
    pixels = torch.zeros((40000, 1, 1, 3), dtype=torch.uint8)
    pixels[-64:] = 255
    return pixels.expand(-1, 32, 32, -1)


def tiny_augmentation(batch, cutout=0):
    if cutout != 0:
        raise AssertionError("SWA BN must not apply Cutout")
    return normalize(batch[:, :2, :2, :])


class SWAMathTests(unittest.TestCase):
    def test_equal_parameter_mean_includes_bn_affine_but_never_buffers_or_gradients(self):
        source = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
        averaged = copy.deepcopy(source)
        with torch.no_grad():
            for parameter in source.parameters(): parameter.fill_(2.)
            for parameter in averaged.parameters(): parameter.fill_(99.)
            source[0].running_mean.fill_(17.); source[0].num_batches_tracked.fill_(8)
            averaged[0].running_mean.fill_(3.); averaged[0].num_batches_tracked.fill_(4)
        for parameter in source.parameters(): parameter.grad = torch.full_like(parameter, 7.)
        for parameter in averaged.parameters(): parameter.grad = torch.full_like(parameter, 11.)
        old_buffers = {name: value.clone() for name, value in averaged.named_buffers()}
        self.assertEqual(swa.average_parameters_(averaged, source, 0), 1)
        self.assertTrue(all(torch.equal(parameter, torch.full_like(parameter, 2.)) for parameter in averaged.parameters()))
        with torch.no_grad():
            for parameter in source.parameters(): parameter.fill_(4.)
        self.assertEqual(swa.average_parameters_(averaged, source, 1), 2)
        for parameter in averaged.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 3.)))
            self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 11.)))
            self.assertTrue(parameter.is_leaf)
        for parameter in source.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 4.)))
            self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 7.)))
        for name, value in averaged.named_buffers(): self.assertTrue(torch.equal(value, old_buffers[name]))

    def test_invalid_names_shape_dtype_nonfinite_count_or_alias_fail_before_mutation(self):
        target = torch.nn.ParameterDict({"a": torch.nn.Parameter(torch.ones(2)), "b": torch.nn.Parameter(torch.ones(1))})
        cases = [torch.nn.ParameterDict({"other": torch.nn.Parameter(torch.ones(2))}),
                 torch.nn.ParameterDict({"a": torch.nn.Parameter(torch.ones(3)), "b": torch.nn.Parameter(torch.ones(1))}),
                 copy.deepcopy(target).double()]
        nonfinite = copy.deepcopy(target)
        with torch.no_grad(): nonfinite["b"].fill_(float("nan"))
        cases.append(nonfinite)
        original = copy.deepcopy(target.state_dict())
        for source in cases:
            with self.assertRaises(ValueError): swa.average_parameters_(target, source, 1)
        for count in (-1, True, 1.):
            with self.assertRaises(ValueError): swa.average_parameters_(target, copy.deepcopy(target), count)
        with self.assertRaisesRegex(ValueError, "alias"):
            swa.average_parameters_(target, target, 0)
        for name, value in target.state_dict().items(): self.assertTrue(torch.equal(value, original[name]))

    def test_convex_fp32_mean_avoids_overflow_from_subtracting_large_finite_values(self):
        target = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([3e38]))])
        source = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([-3e38]))])
        swa.average_parameters_(target, source, 1)
        self.assertEqual(target[0].item(), 0.)

    def test_one_wrn34_3_inference_model_retains_exact_parameter_budget(self):
        source = build_model({"depth": 34, "widen_factor": 3, "dropout": 0., "num_classes": 10})
        averaged = copy.deepcopy(source)
        identities = [id(value) for value in averaged.parameters()]
        swa.average_parameters_(averaged, source, 0)
        self.assertEqual(count_parameters(averaged), 4166298)
        self.assertEqual(count_parameters(averaged, True), 4166298)
        self.assertEqual(identities, [id(value) for value in averaged.parameters()])

    def test_bn_tail_is_sample_weighted_and_affine_optimizer_modes_are_preserved(self):
        model = torch.nn.Sequential(torch.nn.Dropout2d(.7), torch.nn.BatchNorm2d(3, momentum=.37))
        model.train(); model[1].eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        before_parameters = {name: value.clone() for name, value in model.named_parameters()}
        before_gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        flags = []
        handle = model[1].register_forward_pre_hook(lambda module, inputs: flags.append((module.training, model[0].training, torch.is_grad_enabled())))
        try:
            with patch.object(swa, "augment", side_effect=tiny_augmentation) as augmentation:
                proof = swa.recompute_training_bn_(model, images_with_tail(), batch_size=128, seed=2026, precision="fp32", device="cpu")
        finally:
            handle.remove()
        expected = (torch.tensor([64/40000]*3)-torch.tensor(MEAN))/torch.tensor(STD)
        self.assertTrue(torch.allclose(model[1].running_mean, expected, atol=5e-5))
        self.assertEqual(model[1].num_batches_tracked.item(), 313)
        self.assertEqual((proof["training_examples"], proof["batches"], proof["last_batch_examples"]), (40000, 313, 64))
        self.assertEqual(augmentation.call_count, 313)
        self.assertTrue(all(bn and not dropout and not grad for bn, dropout, grad in flags))
        self.assertEqual((model.training, model[0].training, model[1].training, model[1].momentum), (True, True, False, .37))
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before_parameters[name]))
            self.assertTrue(torch.equal(value.grad, before_gradients[name]))
        self.assertEqual(optimizer.state_dict()["param_groups"], before_optimizer["param_groups"])
        for key, value in optimizer.state_dict()["state"].items():
            self.assertTrue(torch.equal(value["momentum_buffer"], before_optimizer["state"][key]["momentum_buffer"]))

    def test_bn_same_seed_order_and_augmentation_match_while_all_rngs_are_restored(self):
        first = torch.nn.BatchNorm2d(3)
        second = copy.deepcopy(first)
        images = images_with_tail()
        def stochastic(batch, cutout):
            value = tiny_augmentation(batch, cutout)
            return value+torch.rand(len(batch), 1, 1, 1)*.01+random.random()*.001+np.random.random()*.001
        before = rng_state()
        with patch.object(swa, "augment", side_effect=stochastic):
            a = swa.recompute_training_bn_(first, images, batch_size=128, seed=2026)
            assert_rng_equal(self, before, rng_state())
            b = swa.recompute_training_bn_(second, images, batch_size=128, seed=2026)
        assert_rng_equal(self, before, rng_state())
        self.assertEqual(a["augmented_input_sha256"], b["augmented_input_sha256"])
        self.assertTrue(torch.equal(first.running_mean, second.running_mean))
        self.assertTrue(all(a[key] is True for key in ("rng_restored", "parameters_unchanged", "training_modes_restored", "momenta_restored")))

    def test_bn_tail_variance_uses_sample_weighted_unbiased_batch_statistics(self):
        model = torch.nn.BatchNorm2d(3)
        def distinct_variances(batch, cutout):
            value = tiny_augmentation(batch, cutout)
            scale = 3. if batch[0, 0, 0, 0].item() == 255 else 1.
            return value+torch.tensor([0., 1., 2., 3.]).reshape(1, 1, 2, 2)*scale
        with patch.object(swa, "augment", side_effect=distinct_variances):
            swa.recompute_training_bn_(model, images_with_tail(), batch_size=128)
        # PyTorch's running_var uses each batch's unbiased variance. The 64
        # example tail must have weight 64/40000, not one full batch's weight.
        full_variance = 1.25*(128*4)/(128*4-1)
        tail_variance = 1.25*9*(64*4)/(64*4-1)
        expected = (39936*full_variance+64*tail_variance)/40000
        self.assertTrue(torch.allclose(model.running_var, torch.full((3,), expected), atol=2e-6))

    def test_bn_parameter_mutation_is_detected_and_original_values_restored(self):
        class Mutating(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.bn = torch.nn.BatchNorm2d(3)
            def forward(self, batch):
                self.bn.weight.add_(.01)
                return self.bn(batch)
        model = Mutating().eval()
        original = copy.deepcopy(model.state_dict())
        with patch.object(swa, "augment", side_effect=tiny_augmentation):
            with self.assertRaisesRegex(RuntimeError, "changed model parameters"):
                swa.recompute_training_bn_(model, images_with_tail())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))
        self.assertFalse(model.training); self.assertFalse(model.bn.training)

    def test_bn_failure_restores_original_statistics_parameters_modes_and_rng(self):
        class Failing(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.bn = torch.nn.BatchNorm2d(3, momentum=None)
            def forward(self, batch):
                self.bn(batch)
                raise RuntimeError("injected failure")
        model = Failing().eval()
        original = copy.deepcopy(model.state_dict()); before = rng_state()
        with patch.object(swa, "augment", side_effect=tiny_augmentation):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                swa.recompute_training_bn_(model, images_with_tail(), batch_size=128)
        assert_rng_equal(self, before, rng_state())
        for name, value in model.state_dict().items(): self.assertTrue(torch.equal(value, original[name]))
        self.assertFalse(model.training); self.assertFalse(model.bn.training); self.assertIsNone(model.bn.momentum)

    def test_bn_rejects_nontraining_counts_fp16_and_untracked_bn_before_forward(self):
        images = images_with_tail()
        model = torch.nn.BatchNorm2d(3)
        with patch.object(swa, "augment") as augmentation:
            for count in (5000, 50000):
                invalid = torch.zeros((count, 1, 1, 3), dtype=torch.uint8).expand(-1, 32, 32, -1)
                with self.assertRaisesRegex(ValueError, "40,000"):
                    swa.recompute_training_bn_(model, invalid)
            with self.assertRaisesRegex(ValueError, "CPU FP32"):
                swa.recompute_training_bn_(model, images, precision="fp16")
            with self.assertRaisesRegex(ValueError, "tracked"):
                swa.recompute_training_bn_(torch.nn.BatchNorm2d(3, track_running_stats=False), images)
            augmentation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
