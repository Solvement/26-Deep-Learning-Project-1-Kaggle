import unittest

import numpy as np

from research.development_shift_stats import RADII, offsets, summarize_shift_predictions


class DevelopmentShiftStatsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(10), 500)
        self.clean = {border: self.labels.copy() for border in RADII}
        self.shifted = {
            border: {offset: self.labels.copy() for distance in distances for offset in offsets(distance)}
            for border, distances in RADII.items()
        }

    def summary(self):
        return summarize_shift_predictions(self.labels, self.clean, self.shifted, replicates=1000)

    def test_counts_use_border_specific_baseline_and_true_labels(self):
        self.clean["canvas"][0] = 2
        self.shifted["canvas"][(3, 0)][1] = 7
        row = self.summary()["by_border"]["canvas"]["by_offset"]["dx+3_dy+0"]
        self.assertEqual(row["correct"], 4999)
        self.assertEqual(row["clean_correct_opportunities"], 4999)
        self.assertEqual(row["clean_correct_to_wrong"], 1)
        self.assertEqual(row["clean_wrong_to_correct"], 1)
        self.assertEqual(row["prediction_flips"], 2)
        airplane = row["per_true_class"]["airplane"]
        self.assertEqual((airplane["images"], airplane["correct"], airplane["prediction_flips"]), (500, 499, 2))
        self.assertEqual(row["per_true_class"]["horse"]["prediction_flips"], 0)

    def test_joint_image_bootstrap_keeps_border_and_direction_aligned(self):
        for offset in offsets(3):
            self.shifted["reflect"][offset][:117] = 1
            self.shifted["canvas"][offset][:117] = 1
        result = self.summary()
        name = "distance3_excess_error_vs_distance2_and4"
        a = result["by_border"]["reflect"]["paired_contrasts"][name]
        b = result["by_border"]["canvas"]["paired_contrasts"][name]
        self.assertEqual(a, b)
        self.assertEqual(a["delta"], 117 / 5000)
        self.assertGreater(a["paired_bootstrap_ci95"][0], 0)
        self.assertEqual(result["paired_border_controls"][name], {"delta": 0, "paired_bootstrap_ci95": [0, 0]})
        self.assertEqual(result["by_border"]["reflect"]["by_distance"]["3"]["observations"], 20000)
        self.assertEqual(result, self.summary())

    def test_secondary_and_primary_use_requested_distances(self):
        self.shifted["reflect"][(1, 0)][:200] = 1
        self.shifted["reflect"][(2, 0)][:100] = 1
        self.shifted["reflect"][(3, 0)][:200] = 1
        self.shifted["reflect"][(4, 0)][:80] = 1
        rows = self.summary()["by_border"]["reflect"]["paired_contrasts"]
        self.assertEqual(rows["distance3_excess_error_vs_distance2_and4"]["delta"], 110 / 20000)
        self.assertEqual(rows["distance4_minus_distance3_accuracy"]["delta"], 120 / 20000)

    def test_rejects_missing_direction_or_noninteger_labels(self):
        with self.assertRaises(ValueError):
            summarize_shift_predictions(self.labels.astype(float), self.clean, self.shifted)
        del self.shifted["canvas"][(4, 0)]
        with self.assertRaises(ValueError):
            self.summary()


class DevelopmentShiftGeometryTests(unittest.TestCase):
    def test_canvas_preserves_patch_and_fill_at_all_offsets(self):
        import torch
        from research.run_development_shift_probe import CANVAS_FILL, shift

        batch = (torch.arange(32 * 32 * 3) % 251).to(torch.uint8).reshape(1, 32, 32, 3)
        for dx, dy in ((0, 0), *(o for radius in RADII["canvas"] for o in offsets(radius))):
            view = shift(batch, "canvas", dx, dy)
            expected = torch.empty_like(batch)
            expected[:] = torch.tensor(CANVAS_FILL, dtype=batch.dtype)
            expected[:, 5 + dy:27 + dy, 5 + dx:27 + dx] = batch[:, 5:27, 5:27]
            self.assertTrue(torch.equal(view, expected))

    def test_reflect_shift_sign_and_border(self):
        import torch
        from research.run_development_shift_probe import shift

        batch = torch.arange(32, dtype=torch.uint8)[None, None, :, None].expand(1, 32, 32, 3)
        right = shift(batch, "reflect", 3, 0)
        left = shift(batch, "reflect", -3, 0)
        self.assertEqual(right[0, 0, :6, 0].tolist(), [3, 2, 1, 0, 1, 2])
        self.assertEqual(left[0, 0, -6:, 0].tolist(), [29, 30, 31, 30, 29, 28])


if __name__ == "__main__":
    unittest.main()
