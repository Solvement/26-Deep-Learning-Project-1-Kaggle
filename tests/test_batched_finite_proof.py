"""CPU equivalence checks only; no model, official data or CUDA capability probe."""
import unittest

import torch

from research.batched_finite_proof import (
    FLOAT_DTYPES, all_finite, prove_bn_increment, snapshot_bn_counters,
)


def reference(values):
    return all(bool(torch.isfinite(t).all().item()) for t in values)


class BatchedFiniteProofTests(unittest.TestCase):
    def assert_equivalent(self, values):
        originals = [t.detach().clone() for t in values]
        result = all_finite(values)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.requires_grad)
        self.assertEqual(bool(result.item()), reference(values))
        for value, original in zip(values, originals):
            torch.testing.assert_close(value, original, rtol=0, atol=0, equal_nan=True)

    def test_each_supported_float_and_real_state_shapes(self):
        for dtype in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                self.assert_equivalent([torch.zeros(16, dtype=dtype), torch.ones(3, 4, dtype=dtype),
                                        torch.tensor(-2.5, dtype=dtype)])

    def test_nan_and_both_infinities_in_each_position_and_dtype(self):
        for dtype in FLOAT_DTYPES:
            for bad in (float("nan"), float("inf"), -float("inf")):
                for position in range(3):
                    with self.subTest(dtype=dtype, bad=bad, position=position):
                        values = [torch.ones(2, dtype=dtype) for _ in range(3)]
                        values[position][1] = bad
                        self.assert_equivalent(values)

    def test_maximum_finite_values_are_not_norm_overflow(self):
        values = []
        for dtype in FLOAT_DTYPES:
            maximum = torch.finfo(dtype).max
            pair = torch.tensor([maximum, -maximum], dtype=dtype)
            self.assert_equivalent([pair])
            values.append(pair)
        self.assert_equivalent(values)
        self.assertTrue(bool(all_finite(values).item()))

    def test_mixed_float_formats_preserve_finite_and_nonfinite(self):
        values = [torch.tensor([0., 1.], dtype=dtype) for dtype in FLOAT_DTYPES]
        values += [torch.tensor([torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max]),
                   torch.tensor([True, False])]
        self.assert_equivalent(values)
        for index in range(len(FLOAT_DTYPES)):
            changed = [t.clone() for t in values]
            changed[index][0] = float("inf")
            self.assert_equivalent(changed)
            self.assertFalse(bool(all_finite(changed).item()))

    def test_int64_and_bool_are_finite_without_float_cast(self):
        self.assert_equivalent([torch.tensor([torch.iinfo(torch.int64).min,
                                             torch.iinfo(torch.int64).max]),
                                torch.tensor([False, True])])

    def test_empty_collection_empty_tensors_and_empty_before_bad(self):
        self.assert_equivalent([])
        self.assert_equivalent([torch.empty(0, 4), torch.empty(3, 0, dtype=torch.float64)])
        self.assert_equivalent([torch.empty(0, dtype=torch.int64)])
        self.assert_equivalent([torch.empty(0), torch.tensor([float("nan")])])
        self.assertEqual(all_finite([]).device.type, "cpu")

    def test_noncontiguous_and_channels_last_values(self):
        transposed = torch.arange(30, dtype=torch.float32).view(5, 6).t()
        channels_last = torch.ones(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        self.assertFalse(transposed.is_contiguous())
        self.assert_equivalent([transposed, channels_last, transposed[:, ::2]])
        transposed[1, 2] = float("nan")
        self.assert_equivalent([transposed, channels_last])

    def test_gradient_inventory_is_read_only_and_not_attached(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
        parameter.grad = torch.tensor([3., 4.])
        self.assert_equivalent([parameter, parameter.grad])
        self.assertIsNone(all_finite([parameter]).grad_fn)
        torch.testing.assert_close(parameter.grad, torch.tensor([3., 4.]), rtol=0, atol=0)

    def test_unsupported_dtypes_layouts_devices_and_non_tensors_fail(self):
        for dtype in (torch.complex64, torch.complex128, torch.int32, torch.uint8):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                all_finite([torch.ones(2, dtype=dtype)])
        with self.assertRaises(ValueError):
            all_finite([torch.ones(2).to_sparse()])
        with self.assertRaises(ValueError):
            all_finite([torch.empty(1, device="meta")])
        with self.assertRaises(TypeError):
            all_finite([1.0])

    def test_seeded_mixed_inventory_matches_elementwise_reference(self):
        generator = torch.Generator(device="cpu").manual_seed(12026)
        for iteration in range(30):
            values = [torch.randn(3, 4, generator=generator).to(dtype).t() for dtype in FLOAT_DTYPES]
            if iteration % 2:
                values[iteration % 4][0, 0] = float("nan")
            self.assert_equivalent(values)


class BNCounterProofTests(unittest.TestCase):
    def counters(self, numbers):
        return [torch.tensor(number, dtype=torch.int64) for number in numbers]

    def test_snapshot_clone_does_not_alias_live_counters(self):
        counters = self.counters([0, 4, 9])
        before = snapshot_bn_counters(counters)
        for counter in counters:
            counter.add_(1)
        report = prove_bn_increment(before, counters)
        self.assertEqual(report, {"all_incremented": True, "before": [0, 4, 9],
                                  "after": [1, 5, 10], "expected_increment": 1})
        self.assertEqual(before.tolist(), [0, 4, 9])
        before[0] = 100
        self.assertEqual(counters[0].item(), 1)

    def test_one_missed_or_double_forward_is_rejected_per_layer(self):
        before = snapshot_bn_counters(self.counters([1, 2, 3]))
        for after in ([2, 2, 4], [2, 4, 4], [2, 3, 3]):
            report = prove_bn_increment(before, self.counters(after))
            self.assertFalse(report["all_incremented"])
            self.assertEqual(report["after"], after)

    def test_int64_overflow_cannot_falsely_prove_increment(self):
        info = torch.iinfo(torch.int64)
        before = snapshot_bn_counters(self.counters([info.max, info.min, info.max-1]))
        after = self.counters([info.min, info.min+1, info.max])
        report = prove_bn_increment(before, after)
        self.assertFalse(report["all_incremented"])
        self.assertEqual(report["before"], [info.max, info.min, info.max-1])
        self.assertEqual(report["after"], [info.min, info.min+1, info.max])
        self.assertTrue(prove_bn_increment(snapshot_bn_counters(self.counters([info.max-1])),
                                           self.counters([info.max]))["all_incremented"])

    def test_empty_snapshot_and_counter_inventory_identity_is_explicit(self):
        before = snapshot_bn_counters([])
        self.assertEqual(before.shape, torch.Size([0]))
        self.assertEqual(prove_bn_increment(before, []),
                         {"all_incremented": True, "before": [], "after": [], "expected_increment": 1})

    def test_counter_dtype_shape_count_device_and_increment_are_checked(self):
        for bad in (torch.tensor(0.), torch.tensor(False), torch.tensor([0], dtype=torch.int64)):
            with self.assertRaises(ValueError):
                snapshot_bn_counters([bad])
        with self.assertRaises(ValueError):
            snapshot_bn_counters([torch.tensor(0, device="meta", dtype=torch.int64)])
        before = snapshot_bn_counters(self.counters([0]))
        for after in ([], self.counters([1, 1])):
            with self.assertRaises(ValueError):
                prove_bn_increment(before, after)
        with self.assertRaises(ValueError):
            prove_bn_increment(torch.tensor(0, dtype=torch.int64), self.counters([1]))
        for increment in (True, -1, 1.0):
            with self.assertRaises(ValueError):
                prove_bn_increment(before, self.counters([1]), increment)

    def test_zero_and_multiple_expected_increments_use_exact_python_arithmetic(self):
        before = snapshot_bn_counters(self.counters([2, 7]))
        self.assertTrue(prove_bn_increment(before, self.counters([2, 7]), 0)["all_incremented"])
        self.assertTrue(prove_bn_increment(before, self.counters([4, 9]), 2)["all_incremented"])
        self.assertEqual(before.tolist(), [2, 7])


if __name__ == "__main__":
    unittest.main()
