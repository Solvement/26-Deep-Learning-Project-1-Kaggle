"""Independent CPU math, pixels and RNG checks for the fixed AugMix package.

All images/logits are synthetic. These checks are not a native GPU preflight,
official training, development accuracy or competition-score evidence.
"""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from research import augmix


class AugMixLossTests(unittest.TestCase):
    def distributions(self):
        return [torch.tensor([[.8, .15, .05], [.2, .3, .5]], dtype=torch.float64),
                torch.tensor([[.1, .7, .2], [.6, .1, .3]], dtype=torch.float64),
                torch.tensor([[.2, .2, .6], [.15, .7, .15]], dtype=torch.float64)]

    def test_jsd_matches_analytic_kl_to_mixture_with_correct_direction_and_reduction(self):
        distributions = self.distributions()
        mixture = sum(distributions)/3
        expected = sum((p*(p.log()-mixture.log())).sum(1).mean() for p in distributions)/3
        reverse = sum((mixture*(mixture.log()-p.log())).sum(1).mean() for p in distributions)/3
        actual = augmix.jsd_three_view(*(p.log() for p in distributions))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        self.assertGreater(abs(actual.item()-reverse.item()), .01)
        self.assertGreater(actual.item(), 0)
        self.assertLessEqual(actual.item(), math.log(3)+1e-6)

    def test_identical_distributions_have_near_zero_jsd_with_roundoff_tolerance(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[2., .1, -1.], [-4., 3., .3]], dtype=dtype)
                value = augmix.jsd_three_view(logits, logits.clone(), logits.clone())
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value))
                self.assertLess(abs(value.item()), 1e-6)
                self.assertGreaterEqual(value.item(), -1e-6)

    def test_three_view_loss_is_clean_cross_entropy_plus_twelve_jsd(self):
        logits = [p.log().float().requires_grad_() for p in self.distributions()]
        targets = torch.tensor([0, 1], dtype=torch.int64)
        expected_ce = F.cross_entropy(logits[0], targets)
        expected_jsd = augmix.jsd_three_view(*logits)
        actual = augmix.augmix_loss(*logits, targets)
        self.assertTrue(torch.allclose(actual, expected_ce+12*expected_jsd, rtol=0, atol=1e-6))
        wrong_ce = sum(F.cross_entropy(value, targets) for value in logits)/3
        self.assertGreater(abs(expected_ce.item()-wrong_ce.item()), .01)
        actual.backward()
        for index,value in enumerate(logits):
            with self.subTest(view=index):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.norm().item(), 0)

    def test_no_jsd_ablation_uses_only_single_view_cross_entropy(self):
        logits = self.distributions()[1].log().requires_grad_()
        targets = torch.tensor([1, 0], dtype=torch.int64)
        actual = augmix.no_jsd_loss(logits, targets)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, F.cross_entropy(logits.float(), targets), atol=1e-7))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_extreme_half_and_bfloat_cpu_logits_use_stable_fp32_math(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                clean = torch.tensor([[60000., -60000., 0.], [-60000., 60000., 0.]], dtype=dtype, requires_grad=True)
                first = torch.tensor([[0., 60000., -60000.], [60000., 0., -60000.]], dtype=dtype, requires_grad=True)
                second = torch.tensor([[-60000., 0., 60000.], [0., -60000., 60000.]], dtype=dtype, requires_grad=True)
                divergence = augmix.jsd_three_view(clean, first, second)
                loss = augmix.augmix_loss(clean, first, second, torch.tensor([1, 0]))
                self.assertEqual(divergence.dtype, torch.float32)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(divergence))
                self.assertGreaterEqual(divergence.item(), -1e-6)
                self.assertLessEqual(divergence.item(), math.log(3)+1e-6)
                loss.backward()
                for value in (clean, first, second):
                    self.assertTrue(torch.isfinite(value.grad).all())

    def test_bad_logits_are_rejected_before_loss_arithmetic(self):
        good = torch.zeros(2, 3)
        bad_values = [torch.zeros(2), torch.empty(0, 3), torch.empty(2, 0), torch.zeros(2, 1),
                      torch.zeros(3, 3), torch.zeros(2, 4), torch.ones(2, 3, dtype=torch.int64),
                      torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf"))]
        for index,bad in enumerate(bad_values):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    augmix.jsd_three_view(good, bad, good)
        with self.assertRaises(ValueError):
            augmix.jsd_three_view(good, torch.empty((2, 3), device="meta"), good)

    def test_invalid_targets_fail_closed_in_three_view_and_single_view_losses(self):
        logits = torch.zeros(2, 3)
        bad_targets = [torch.tensor([0., 1.]), torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([-1, 1]), torch.tensor([0, 3]), torch.tensor([0]),
            torch.tensor([[0], [1]]), torch.empty((2,), dtype=torch.int64, device="meta")]
        for index,target in enumerate(bad_targets):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    augmix.augmix_loss(logits, logits, logits, target)
                with self.assertRaises(ValueError):
                    augmix.no_jsd_loss(logits, target)

    def test_moderate_logits_match_author_probability_target_kl_form(self):
        logits = [p.log().float() for p in self.distributions()]
        probabilities = [F.softmax(value, dim=1) for value in logits]
        mixture = (sum(probabilities)/3).clamp(1e-7, 1)
        author_form = sum(F.kl_div(mixture.log(), p, reduction="batchmean") for p in probabilities)/3
        self.assertTrue(torch.allclose(augmix.jsd_three_view(*logits), author_form, rtol=0, atol=1e-7))

    def test_mixture_probability_floor_is_not_silently_renormalized(self):
        logits = torch.tensor([[0., -20., -21.]])
        probabilities = np.exp(np.array([0., -20., -21.], dtype=np.float64))
        probabilities /= probabilities.sum()
        clamped = np.clip(probabilities, 1e-7, 1.)
        expected = float(np.sum(probabilities*np.log(probabilities/clamped)))
        actual = augmix.jsd_three_view(logits, logits, logits).item()
        self.assertAlmostEqual(actual, expected, delta=5e-9)
        # The author's unnormalized floor permits this tiny negative value;
        # renormalizing the mixture would add a different positive term.
        self.assertGreaterEqual(actual, -1e-6)


class AugMixPixelTests(unittest.TestCase):
    def test_numpy_hot_path_is_bitwise_equal_to_the_torch_reference_for_sampled_traces(self):
        images = np.random.default_rng(2026).integers(0, 256, (32, 32, 32, 3), dtype=np.uint8)

        def reference_normalization(image):
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous().float()/255.
            return (pixels-pixels.new_tensor(augmix.MEAN)[:, None, None])/pixels.new_tensor(augmix.STD)[:, None, None]

        for index, pixels in enumerate(images):
            with self.subTest(seed=1000+index):
                actual, trace = augmix.augmix_view(pixels, np.random.default_rng(1000+index), return_trace=True)
                base = Image.fromarray(pixels)
                chains = []
                for operations in trace.chains:
                    branch = base.copy()
                    for operation in operations:
                        branch = augmix.apply_operation(branch, operation)
                    chains.append(reference_normalization(branch))
                clean = reference_normalization(base)
                expected = augmix.mix_normalized(clean, chains, trace.weights, trace.mixing)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(augmix.normalize_image(base), clean))

    def image(self):
        return (np.arange(32*32*3, dtype=np.uint16).reshape(32, 32, 3) % 256).astype(np.uint8)

    def test_fixed_configuration_and_default_nine_operations_are_explicit(self):
        config = augmix.AugMixConfig()
        config.validate()
        self.assertEqual((config.width, config.depth, config.severity, config.alpha, config.all_ops),
                         (3, -1, 3, 1., False))
        self.assertEqual(set(augmix.OPERATIONS), {"autocontrast", "equalize", "posterize", "rotate", "solarize",
            "shear_x", "shear_y", "translate_x", "translate_y"})
        for values in ({"width": 2}, {"depth": 2}, {"severity": 5}, {"alpha": .5}, {"all_ops": True},
                       {"width": True}, {"alpha": True}):
            with self.subTest(config=values):
                with self.assertRaises(ValueError):
                    augmix.AugMixConfig(**values).validate()

    def test_operation_sampling_matches_author_severity_and_truncation(self):
        # Twin explicit generators provide an independent scalar oracle for
        # magnitude, truncation and sign without mocking numpy internals.
        for seed in range(20):
            for name in augmix.OPERATIONS:
                with self.subTest(seed=seed, operation=name):
                    rng, oracle = np.random.default_rng(seed), np.random.default_rng(seed)
                    sampled = augmix.sample_operation(name, rng)
                    self.assertEqual(sampled.name, name)
                    if name in ("autocontrast", "equalize"):
                        expected = None
                    else:
                        level = oracle.uniform(.1, 3)
                        if name == "posterize":
                            expected = 4-int(level*4/10)
                        elif name == "solarize":
                            expected = 256-int(level*256/10)
                        else:
                            expected = int(level*30/10) if name == "rotate" else (
                                level*.3/10 if name.startswith("shear_") else int(level*(32/3)/10))
                            if oracle.random() > .5:
                                expected = -expected
                    self.assertEqual(sampled.value, expected)

    def test_posterize_and_solarize_boundary_pixels_match_exact_thresholds(self):
        image = self.image()
        for bits in (3, 4):
            actual = np.array(augmix.apply_operation(image, augmix.Operation("posterize", bits)))
            expected = image & np.uint8(256-2**(8-bits))
            self.assertTrue(np.array_equal(actual, expected))
        for threshold in (180, 254):
            actual = np.array(augmix.apply_operation(image, augmix.Operation("solarize", threshold)))
            expected = np.where(image < threshold, image, 255-image).astype(np.uint8)
            self.assertTrue(np.array_equal(actual, expected))

    def test_geometric_pixels_use_bilinear_inverse_affine_and_black_fill(self):
        image = Image.fromarray(self.image())
        bilinear, affine = Image.Resampling.BILINEAR, Image.Transform.AFFINE
        for name,value,coefficients in (("shear_x", .075, (1, .075, 0, 0, 1, 0)),
                ("shear_y", -.075, (1, 0, 0, -.075, 1, 0)),
                ("translate_x", 3, (1, 0, 3, 0, 1, 0)),
                ("translate_y", -3, (1, 0, 0, 0, 1, -3))):
            with self.subTest(operation=name):
                expected = image.transform((32, 32), affine, coefficients, resample=bilinear, fillcolor=(0, 0, 0))
                actual = augmix.apply_operation(image, augmix.Operation(name, value))
                self.assertTrue(np.array_equal(np.array(actual), np.array(expected)))
        expected = image.rotate(8, resample=bilinear, expand=False, fillcolor=(0, 0, 0))
        actual = augmix.apply_operation(image, augmix.Operation("rotate", 8))
        self.assertTrue(np.array_equal(np.array(actual), np.array(expected)))
        shifted = np.array(augmix.apply_operation(image, augmix.Operation("translate_x", 3)))
        self.assertTrue(np.array_equal(shifted[:, :-3], np.array(image)[:, 3:]))
        self.assertTrue((shifted[:, -3:] == 0).all())

    def test_transforms_copy_inputs_and_reject_non_rgb_or_invalid_magnitudes(self):
        source = self.image()
        before = source.copy()
        pil = Image.fromarray(source)
        pil_bytes = pil.tobytes()
        for name in augmix.OPERATIONS:
            operation = augmix.sample_operation(name, np.random.default_rng(2026))
            for image in (source, pil):
                output = augmix.apply_operation(image, operation)
                self.assertIsInstance(output, Image.Image)
                self.assertEqual((output.mode, output.size), ("RGB", (32, 32)))
                self.assertIsNot(output, image)
        self.assertTrue(np.array_equal(source, before))
        self.assertEqual(pil.tobytes(), pil_bytes)
        invalid = [np.zeros((31, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros((32, 32, 4), dtype=np.uint8), Image.new("L", (32, 32)), Image.new("RGBA", (32, 32)),
            torch.zeros(3, 32, 32)]
        for image in invalid:
            with self.assertRaises(ValueError):
                augmix.normalize_image(image)
        for operation in (augmix.Operation("unknown"), augmix.Operation("rotate", 9),
                augmix.Operation("rotate", 3.5), augmix.Operation("shear_x", .1),
                augmix.Operation("translate_x", 4), augmix.Operation("posterize", 2),
                augmix.Operation("solarize", 179), augmix.Operation("equalize", 1),
                augmix.Operation("rotate", float("nan"))):
            with self.assertRaises(ValueError):
                augmix.apply_operation(source, operation)

    def test_normalization_uses_frozen_mean_std_and_mixing_preserves_float_precision(self):
        image = self.image()
        expected = torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255
        expected = (expected-torch.tensor([.4914, .4822, .4465])[:, None, None])/torch.tensor([.2470, .2435, .2616])[:, None, None]
        actual = augmix.normalize_image(image)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(torch.equal(actual, expected))
        clean = torch.zeros(3, 32, 32)
        chains = [torch.full_like(clean, value) for value in (.1, .2, .8)]
        weights = np.array([.2, .3, .5], dtype=np.float32)
        mixed = augmix.mix_normalized(clean, chains, weights, .25)
        self.assertEqual(mixed.dtype, torch.float32)
        self.assertTrue(torch.allclose(mixed, torch.full_like(clean, .12), atol=1e-7, rtol=0))
        self.assertTrue(torch.equal(clean, torch.zeros_like(clean)))
        self.assertTrue(torch.allclose(augmix.mix_normalized(clean, chains, weights, 1), 4*mixed))
        self.assertTrue(torch.equal(augmix.mix_normalized(clean, chains, weights, 0), clean))

    def test_nonconvex_weights_nonfinite_or_invalid_view_ranges_fail_closed(self):
        clean = torch.zeros(3, 32, 32)
        chains = [clean.clone() for _ in range(3)]
        for weights,mixing in (([.2, .3, .4], .5), ([-.1, .6, .5], .5), ([1, 1, -1], .5),
                ([1., 0.], .5), ([float("nan"), .5, .5], .5), ([.2, .3, .5], -.1),
                ([.2, .3, .5], 1.1), ([.2, .3, .5], float("inf")), ([.2, .3, .5], True)):
            with self.assertRaises(ValueError):
                augmix.mix_normalized(clean, chains, weights, mixing)
        for invalid in (torch.full_like(clean, 20.), torch.full_like(clean, float("nan")),
                        clean.double(), torch.zeros(32, 32, 3)):
            with self.assertRaises(ValueError):
                augmix.mix_normalized(invalid, chains, [.2, .3, .5], .5)


class AugMixReplayTests(unittest.TestCase):
    def image(self):
        return (np.arange(32*32*3, dtype=np.uint16).reshape(32, 32, 3) % 256).astype(np.uint8)

    def test_trace_reconstructs_exact_three_chain_convex_view(self):
        image = self.image()
        actual,trace = augmix.augmix_view(image, np.random.default_rng(2026), return_trace=True)
        self.assertEqual(len(trace.chains), 3)
        self.assertTrue(all(1 <= len(chain) <= 3 for chain in trace.chains))
        self.assertTrue(all(weight >= 0 for weight in trace.weights))
        self.assertAlmostEqual(sum(trace.weights), 1., places=6)
        self.assertTrue(0 <= trace.mixing <= 1)
        transformed = []
        for chain in trace.chains:
            current = Image.fromarray(image)
            for operation in chain:
                self.assertIn(operation.name, augmix.OPERATIONS)
                current = augmix.apply_operation(current, operation)
            transformed.append(augmix.normalize_image(current))
        reconstructed = augmix.mix_normalized(augmix.normalize_image(image), transformed, trace.weights, trace.mixing)
        self.assertTrue(torch.equal(actual, reconstructed))

    def test_parent_rng_checkpoint_replays_pair_and_each_child_seed_recreates_its_view(self):
        image = self.image()
        rng = np.random.default_rng(19)
        state = copy.deepcopy(rng.bit_generator.state)
        views,trace = augmix.augmix_views(image, rng, return_trace=True)
        rng.bit_generator.state = state
        repeated,repeated_trace = augmix.augmix_views(image, rng, return_trace=True)
        self.assertEqual(trace, repeated_trace)
        self.assertTrue(all(torch.equal(left, right) for left,right in zip(views, repeated)))
        self.assertTrue(torch.equal(views[0], augmix.normalize_image(image)))
        self.assertNotEqual(trace["child_seeds"][0], trace["child_seeds"][1])
        for index,seed in enumerate(trace["child_seeds"], 1):
            child = augmix.augmix_view(image, np.random.default_rng(seed))
            self.assertTrue(torch.equal(views[index], child))

    def test_first_view_rng_consumption_does_not_change_second_view(self):
        image = self.image()
        expected,expected_trace = augmix.augmix_views(image, np.random.default_rng(37), return_trace=True)
        original = augmix.augmix_view
        calls = 0
        def consume_first(image, rng, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                rng.random(1000)
            return original(image, rng, *args, **kwargs)
        with patch.object(augmix, "augmix_view", side_effect=consume_first):
            actual,trace = augmix.augmix_views(image, np.random.default_rng(37), return_trace=True)
        self.assertEqual(calls, 2)
        self.assertEqual(trace["child_seeds"], expected_trace["child_seeds"])
        self.assertFalse(torch.equal(actual[1], expected[1]))
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertEqual(trace["view_traces"][1], expected_trace["view_traces"][1])

    def test_explicit_rng_and_separate_no_jsd_single_view_do_not_use_global_random_state(self):
        import random
        image = self.image()
        before_image = image.copy()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        single = augmix.augmix_no_jsd(image, np.random.default_rng(45))
        expected = augmix.augmix_view(image, np.random.default_rng(45))
        self.assertTrue(torch.equal(single, expected))
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        self.assertTrue(np.array_equal(after_numpy[1], numpy_state[1]))
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(np.array_equal(image, before_image))
        for invalid in (None, np.random, np.random.RandomState(0)):
            with self.assertRaises(TypeError):
                augmix.augmix_view(image, invalid)


if __name__ == "__main__":
    unittest.main()
