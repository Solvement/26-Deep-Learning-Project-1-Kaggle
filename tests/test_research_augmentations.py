"""CutMix pixel/label-area contracts on synthetic, non-square CPU batches."""
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
from research.augmentations import apply_cutmix, cutmix, validate_cutmix


class ResearchCutMixTests(unittest.TestCase):
    def test_full_patch_donors_come_from_unmodified_batch(self):
        images = torch.stack([torch.zeros(3, 4, 6), torch.ones(3, 4, 6)])
        saved = images.clone()
        permutation = torch.tensor([1, 0])
        mixed, weight = apply_cutmix(images, permutation, (0, 4, 0, 6))
        self.assertEqual(weight, 0.0)
        self.assertTrue(torch.equal(mixed[0], saved[1]))
        self.assertTrue(torch.equal(mixed[1], saved[0]))
        self.assertTrue(torch.equal(images, saved))

    def test_non_square_odd_patch_and_clipping_recompute_exact_area(self):
        images = torch.stack([torch.zeros(3, 6, 10), torch.ones(3, 6, 10)])
        # beta=.75 gives integer patch3x5 at center(0,0), clipped to2x3.
        with patch("random.random", return_value=0), patch("numpy.random.beta", return_value=.75), \
                patch("numpy.random.randint", side_effect=[0, 0]), \
                patch("torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, permutation, weight, rectangle = cutmix(images, alpha=1, probability=.5)
        self.assertEqual(rectangle, (0, 2, 0, 3))
        top, bottom, left, right = rectangle
        actual = (bottom-top)*(right-left)
        self.assertEqual(weight, 1-actual/60)
        self.assertEqual(mixed[0].sum().item(), 3*actual)
        self.assertTrue(torch.equal(permutation, torch.tensor([1, 0])))
        self.assertNotEqual(weight, .75)
        # Explicit odd-size center coordinates retain their entire3x3 area.
        _, weight = apply_cutmix(images, torch.tensor([1, 0]), (1, 4, 2, 5))
        self.assertEqual(weight, 1-9/60)

    def test_zero_probability_preserves_input_and_all_random_streams(self):
        images = torch.randn(2, 3, 4, 6)
        random.seed(13); np.random.seed(13); torch.manual_seed(13)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        mixed, permutation, weight, rectangle = cutmix(images, alpha=1, probability=0)
        self.assertIs(mixed, images)
        self.assertEqual((permutation, weight, rectangle), (None, 1.0, None))
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_donor_labels_follow_returned_permutation_and_weight(self):
        images = torch.stack([torch.full((3, 4, 6), float(i)) for i in range(3)])
        labels = torch.tensor([2, 5, 8])
        original_labels = labels.clone()
        donors = torch.tensor([2, 0, 1])
        with patch("random.random", return_value=0), patch("numpy.random.beta", return_value=.5), \
                patch("numpy.random.randint", side_effect=[2, 3]), patch("torch.randperm", return_value=donors):
            mixed, permutation, weight, rectangle = cutmix(images, alpha=1, probability=1)
        self.assertTrue(torch.equal(labels, original_labels))
        self.assertTrue(torch.equal(labels[permutation], torch.tensor([8, 2, 5])))
        for sample in range(3):
            expected_mean = weight*sample+(1-weight)*int(donors[sample])
            self.assertAlmostEqual(mixed[sample].mean().item(), expected_mean, places=6)

    def test_invalid_parameters_and_unclipped_rectangles_rejected(self):
        for alpha, probability in ((-1, .5), (0, .5), (1, -1), (1, 1.1), (float("nan"), .5), (True, .5)):
            with self.assertRaises(ValueError):
                validate_cutmix(alpha, probability)
        with self.assertRaisesRegex(ValueError, "clipped"):
            apply_cutmix(torch.zeros(2, 3, 4, 6), torch.tensor([1, 0]), (-1, 2, 0, 2))


if __name__ == "__main__":
    unittest.main()
