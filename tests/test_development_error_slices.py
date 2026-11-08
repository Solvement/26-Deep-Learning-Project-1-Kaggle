import unittest

import numpy as np

from research.development_error_slices import CASES, compare_directions, summarize_slices


class DevelopmentErrorSlicesTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(10), 500)
        self.candidate = np.zeros((5000, 6), dtype=bool)
        self.incumbent = np.zeros_like(self.candidate)

    def summarize(self, candidate, incumbent):
        return summarize_slices(self.labels, candidate, incumbent, replicates=1000)

    def test_matched_counts_and_adjusted_signal(self):
        self.candidate[:40, 0] = True
        self.incumbent[30:50, 0] = True
        result = self.summarize(self.candidate, self.incumbent)
        row = result["per_class"]["airplane"]["clean"]
        self.assertEqual((row["both_right"], row["both_wrong"], row["candidate_only"], row["incumbent_only"]),
                         (10, 450, 30, 10))
        self.assertEqual(row["candidate_accuracy"], 0.08)
        self.assertEqual(row["incumbent_accuracy"], 0.04)
        self.assertEqual(row["candidate_minus_incumbent"], 0.04)
        self.assertGreater(row["paired_bootstrap_ci95"][0], 0)
        self.assertEqual(row["bonferroni_adjusted_p"], 66 * row["mcnemar_exact_p"])
        self.assertFalse(row["passes_familywise_0_05"])
        self.assertEqual(result["overall"]["clean"]["candidate_only"], 30)
        self.assertEqual(result, self.summarize(self.candidate, self.incumbent))

    def test_zero_discordance_and_repeat_direction(self):
        self.candidate[:80, 0] = True
        primary = self.summarize(self.candidate, self.incumbent)
        repeat = self.summarize(self.incumbent, self.candidate)
        row = primary["overall"]["noise_004"]
        self.assertEqual(row["paired_bootstrap_ci95"], [0, 0])
        self.assertEqual(row["mcnemar_exact_p"], 1)
        result = compare_directions(primary, repeat)
        row = result["per_class"]["airplane"]["clean"]
        self.assertFalse(row["same_nonzero_direction"])
        self.assertTrue(row["passes_familywise_in_both"])
        self.assertFalse(result["seed_results_pooled"])
        same_direction = compare_directions(primary, primary)
        self.assertTrue(same_direction["per_class"]["airplane"]["clean"]["same_nonzero_direction"])

    def test_contract_rejects_wrong_input(self):
        with self.assertRaises(ValueError):
            self.summarize(self.candidate.astype(np.uint8), self.incumbent)
        with self.assertRaises(ValueError):
            self.summarize(self.candidate[:-1], self.incumbent)
        with self.assertRaises(ValueError):
            summarize_slices(self.labels, self.candidate, self.incumbent, reversed(CASES))
        self.labels[:500] = 10
        with self.assertRaises(ValueError):
            self.summarize(self.candidate, self.incumbent)

    def test_cases_share_each_original_image_bootstrap_draw(self):
        self.candidate[:57, 0:2] = True
        self.incumbent[:57, 2] = True
        self.incumbent[57:66, 0:2] = True
        self.candidate[57:66, 2] = True
        result = self.summarize(self.candidate, self.incumbent)
        for group in (result["overall"], result["per_class"]["airplane"]):
            interval = group["clean"]["paired_bootstrap_ci95"]
            self.assertEqual(interval, group["noise_004"]["paired_bootstrap_ci95"])
            np.testing.assert_allclose(group["boxblur3"]["paired_bootstrap_ci95"], [-interval[1], -interval[0]])


if __name__ == "__main__":
    unittest.main()
