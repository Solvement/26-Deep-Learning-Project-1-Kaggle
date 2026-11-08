"""CPU ResizeMix contracts: full donor, side ratio, author full-fit boxes and RNG."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import runtime
import numpy as np
import torch
from torch.nn import functional as F
from research.resizemix import resizemix, validate_resizemix


class ResearchResizeMixTests(unittest.TestCase):
    def forced(self, images, tau, centers_y, centers_x, donors=None):
        donors = torch.tensor([1, 0]) if donors is None else donors
        with patch("random.random", return_value=0), patch("random.uniform", return_value=tau) as uniform, \
                patch("numpy.random.randint", side_effect=[np.array(centers_y), np.array(centers_x)]) as centers, \
                patch("torch.randperm", return_value=donors):
            result = resizemix(images, probability=.5)
        uniform.assert_called_once_with(.1, .8)
        self.assertEqual(centers.call_count, 2)
        return result

    def test_uniform_tau_scales_sides_and_actual_area_weights_labels(self):
        images = torch.stack([torch.zeros(3, 32, 20), torch.ones(3, 32, 20)])
        labels = torch.tensor([2, 7])
        saved_labels = labels.clone()
        mixed, permutation, retained, boxes = self.forced(images, .5, [16, 16], [10, 10])
        self.assertEqual(retained, .75)  # tau=.5 => area=.25, not area=.5
        self.assertEqual(boxes, ((8, 24, 5, 15), (8, 24, 5, 15)))
        self.assertEqual(mixed[0].sum().item(), 3*16*10)
        self.assertTrue(torch.equal(labels[permutation], torch.tensor([7, 2])))
        self.assertTrue(torch.equal(labels, saved_labels))

    def test_full_donor_is_resized_not_cropped_and_original_batch_is_unchanged(self):
        images = torch.zeros(2, 1, 8, 8)
        images[1, :, :4, :4] = 1
        images[1, :, :4, 4:] = 3
        images[1, :, 4:, :4] = 5
        images[1, :, 4:, 4:] = 7
        original = images.clone()
        mixed, permutation, retained, boxes = self.forced(images, .5, [0, 7], [0, 7])
        self.assertEqual(boxes, ((0, 4, 0, 4), (4, 8, 4, 8)))
        full_donor = F.interpolate(original[1:2], (4, 4), mode="bilinear", align_corners=False)[0]
        self.assertTrue(torch.equal(mixed[0, :, :4, :4], full_donor))
        self.assertEqual(set(mixed[0, :, :4, :4].unique().tolist()), {1., 3., 5., 7.})
        self.assertTrue(torch.equal(mixed[1, :, 4:, 4:], torch.zeros(1, 4, 4)))
        self.assertTrue(torch.equal(images, original))

    def test_author_box_semantics_keep_entire_odd_patch_at_both_boundaries(self):
        images = torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)])
        for tau, side in ((.1, 3), (.8, 25)):
            with self.subTest(tau=tau):
                mixed, _, retained, boxes = self.forced(images, tau, [0, 31], [31, 0])
                self.assertEqual(boxes, ((0, side, 32-side, 32), (32-side, 32, 0, side)))
                self.assertEqual(retained, 1-side*side/1024)
                self.assertEqual(mixed[0].sum().item(), 3*side*side)
                for top, bottom, left, right in boxes:
                    self.assertEqual((bottom-top, right-left), (side, side))
                    self.assertTrue(0 <= top < bottom <= 32 and 0 <= left < right <= 32)

    def test_independent_centers_allow_different_boxes_with_equal_area(self):
        images = torch.arange(3*2*16*20, dtype=torch.float32).reshape(3, 2, 16, 20)
        _, permutation, retained, boxes = self.forced(images, .5, [0, 8, 15], [0, 10, 19], torch.tensor([2, 0, 1]))
        self.assertEqual(boxes, ((0, 8, 0, 10), (4, 12, 5, 15), (8, 16, 10, 20)))
        self.assertEqual(retained, .75)
        self.assertTrue(torch.equal(permutation, torch.tensor([2, 0, 1])))

    def test_bilinear_interpolation_and_channels_last_are_preserved(self):
        images = torch.randn(2, 3, 16, 20).contiguous(memory_format=torch.channels_last)
        mixed, donors, _, boxes = self.forced(images, .5, [8, 8], [10, 10])
        self.assertTrue(mixed.is_contiguous(memory_format=torch.channels_last))
        expected = F.interpolate(images[donors], (8, 10), mode="bilinear", align_corners=False)
        for sample, (top, bottom, left, right) in enumerate(boxes):
            self.assertTrue(torch.equal(mixed[sample, :, top:bottom, left:right], expected[sample]))

    def test_probability_zero_consumes_no_rng_and_returns_original_object(self):
        images = torch.randn(2, 3, 16, 20)
        random.seed(61); np.random.seed(61); torch.manual_seed(61)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        mixed, donors, retained, boxes = resizemix(images, probability=0)
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.assertIs(mixed, images)
        self.assertEqual((donors, retained, boxes), (None, 1., None))
        self.assertEqual(before[0], after[0])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(before[1], after[1])))
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_python_numpy_torch_seed_replay_reproduces_mixing_exactly(self):
        images = torch.arange(4*3*32*32, dtype=torch.float32).reshape(4, 3, 32, 32)
        def sequence():
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
            return [resizemix(images) for _ in range(8)]
        first, second = sequence(), sequence()
        self.assertTrue(any(item[1] is not None for item in first))
        self.assertTrue(any(item[1] is None for item in first))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a[0], b[0]))
            if a[1] is None:
                self.assertIsNone(b[1])
            else:
                self.assertTrue(torch.equal(a[1], b[1]))
            self.assertEqual(a[2:], b[2:])

    def test_invalid_parameters_inputs_and_empty_sampled_patch_fail(self):
        for params in ((0, .8, .5), (.8, .1, .5), (.1, 1.1, .5), (.1, .8, -1),
                       (.1, .8, True), (float("nan"), .8, .5)):
            with self.assertRaises(ValueError):
                validate_resizemix(*params)
        with patch("random.random", return_value=0):
            for images in (torch.zeros(2, 3, 32), torch.zeros(0, 3, 32, 32), torch.zeros(2, 3, 32, 32, dtype=torch.uint8)):
                with self.assertRaisesRegex(ValueError, "floating NCHW"):
                    resizemix(images)
            with patch("random.uniform", return_value=.1), self.assertRaisesRegex(ValueError, "empty patch"):
                resizemix(torch.zeros(2, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
