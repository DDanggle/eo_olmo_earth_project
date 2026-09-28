"""Contract tests for E3 token interventions; no model or dataset required."""

import unittest

import numpy as np

from e3_pair_transforms_v0 import ARMS, PAIR_SHAPE, TOKEN_SHAPE, transform_pair


class TransformPairTests(unittest.TestCase):
    def setUp(self):
        # Nonconstant frames prevent accidental broadcasting from looking correct.
        a = np.arange(64 * 768, dtype=np.float32).reshape(64, 768) / 1024
        b = np.flip(a, axis=0).copy() + 7
        self.pair = np.stack([a, b])

    def blocks(self, arm, pair=None):
        tokens, _ = transform_pair(self.pair if pair is None else pair, arm)
        return tokens[:64], tokens[64:128], tokens[128:]

    def test_real_preserves_observations_and_recomputes_difference(self):
        a, b, delta = self.blocks("real")
        np.testing.assert_array_equal(a, self.pair[0])
        np.testing.assert_array_equal(b, self.pair[1])
        np.testing.assert_array_equal(delta, self.pair[1] - self.pair[0])

    def test_earlier_only_removes_later_and_difference(self):
        a, b, delta = self.blocks("earlier_only")
        np.testing.assert_array_equal(a, self.pair[0])
        self.assertFalse(b.any())
        self.assertFalse(delta.any())

    def test_later_only_removes_earlier_and_difference(self):
        a, b, delta = self.blocks("later_only")
        self.assertFalse(a.any())
        np.testing.assert_array_equal(b, self.pair[1])
        self.assertFalse(delta.any())

    def test_repeat_earlier_has_identical_frames_and_zero_difference(self):
        a, b, delta = self.blocks("repeat_earlier")
        np.testing.assert_array_equal(a, self.pair[0])
        np.testing.assert_array_equal(b, self.pair[0])
        self.assertFalse(delta.any())

    def test_repeat_later_has_identical_frames_and_zero_difference(self):
        a, b, delta = self.blocks("repeat_later")
        np.testing.assert_array_equal(a, self.pair[1])
        np.testing.assert_array_equal(b, self.pair[1])
        self.assertFalse(delta.any())

    def test_no_delta_preserves_both_observations(self):
        a, b, delta = self.blocks("no_delta")
        np.testing.assert_array_equal(a, self.pair[0])
        np.testing.assert_array_equal(b, self.pair[1])
        self.assertFalse(delta.any())

    def test_reverse_reorders_frames_and_negates_difference(self):
        a, b, delta = self.blocks("reverse")
        np.testing.assert_array_equal(a, self.pair[1])
        np.testing.assert_array_equal(b, self.pair[0])
        np.testing.assert_array_equal(delta, -self.blocks("real")[2])

    def test_reverse_equals_real_of_swapped_pair(self):
        reverse, types = transform_pair(self.pair, "reverse")
        swapped, swapped_types = transform_pair(self.pair[::-1], "real")
        np.testing.assert_array_equal(reverse, swapped)
        np.testing.assert_array_equal(types, swapped_types)

    def test_reverse_twice_recovers_real(self):
        reverse, _ = transform_pair(self.pair, "reverse")
        recovered, _ = transform_pair(reverse[:128].reshape(PAIR_SHAPE), "reverse")
        original, _ = transform_pair(self.pair, "real")
        np.testing.assert_array_equal(recovered, original)

    def test_each_arm_has_fixed_shape_types_and_finite_values(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                tokens, types = transform_pair(self.pair, arm)
                self.assertEqual(tokens.shape, TOKEN_SHAPE)
                self.assertTrue(tokens.flags.c_contiguous)
                self.assertTrue(np.isfinite(tokens).all())
                self.assertEqual(types.shape, (192,))
                self.assertEqual(types.dtype, np.int64)
                np.testing.assert_array_equal(types, [0] * 64 + [1] * 64 + [3] * 64)

    def test_input_is_unchanged_for_every_arm(self):
        expected = self.pair.copy()
        for arm in ARMS:
            transform_pair(self.pair, arm)
            np.testing.assert_array_equal(self.pair, expected)

    def test_output_does_not_alias_input_or_other_calls(self):
        original = self.pair.copy()
        for arm in ARMS:
            with self.subTest(arm=arm):
                tokens, types = transform_pair(self.pair, arm)
                again, again_types = transform_pair(self.pair, arm)
                expected = again.copy()
                self.assertFalse(np.shares_memory(tokens, self.pair))
                self.assertFalse(np.shares_memory(tokens, again))
                tokens[:] = -991
                types[:] = 77
                np.testing.assert_array_equal(self.pair, original)
                np.testing.assert_array_equal(again, expected)
                np.testing.assert_array_equal(again_types, [0] * 64 + [1] * 64 + [3] * 64)

    def test_readonly_noncontiguous_input_is_supported(self):
        pair = self.pair[:, :, ::-1]
        self.assertFalse(pair.flags.c_contiguous)
        pair.setflags(write=False)
        actual, _ = transform_pair(pair, "real")
        np.testing.assert_array_equal(actual[:64], pair[0])

    def test_dtype_is_preserved(self):
        for dtype in (np.float16, np.float32, np.float64):
            for arm in ARMS:
                with self.subTest(dtype=dtype, arm=arm):
                    tokens, _ = transform_pair(self.pair.astype(dtype), arm)
                    self.assertEqual(tokens.dtype, dtype)

    def test_removed_later_cannot_leak_through_difference(self):
        changed = self.pair.copy()
        changed[1] = changed[1] * -17 + 31
        for arm in ("earlier_only", "repeat_earlier"):
            np.testing.assert_array_equal(transform_pair(self.pair, arm)[0], transform_pair(changed, arm)[0])

    def test_removed_earlier_cannot_leak_through_difference(self):
        changed = self.pair.copy()
        changed[0] = changed[0] * -17 + 31
        for arm in ("later_only", "repeat_later"):
            np.testing.assert_array_equal(transform_pair(self.pair, arm)[0], transform_pair(changed, arm)[0])

    def test_identical_frames_have_zero_delta_and_reverse_is_identical(self):
        pair = np.stack([self.pair[0], self.pair[0]])
        real, _ = transform_pair(pair, "real")
        reverse, _ = transform_pair(pair, "reverse")
        self.assertFalse(real[128:].any())
        np.testing.assert_array_equal(real, reverse)
        np.testing.assert_array_equal(real, transform_pair(pair, "no_delta")[0])

    def test_constant_frames_do_not_lose_slot_identity(self):
        pair = np.stack([np.full((64, 768), 2.), np.full((64, 768), 5.)])
        a, b, delta = self.blocks("real", pair)
        self.assertTrue((a == 2).all())
        self.assertTrue((b == 5).all())
        self.assertTrue((delta == 3).all())

    def test_unknown_arm_is_rejected(self):
        for arm in ("REAL", "zero", "", None, ["real"]):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                transform_pair(self.pair, arm)

    def test_bad_shapes_are_rejected_without_resizing(self):
        for shape in ((64, 768), (3, 64, 768), (2, 63, 768), (2, 64, 769), (2, 768, 64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                transform_pair(np.zeros(shape, dtype=np.float32), "real")

    def test_nonfloating_and_complex_dtypes_are_rejected(self):
        for dtype in (np.int32, np.uint8, np.bool_, np.complex64, object, "U1"):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                transform_pair(np.zeros(PAIR_SHAPE, dtype=dtype), "real")

    def test_nonarray_is_rejected(self):
        with self.assertRaises(TypeError):
            transform_pair(self.pair.tolist(), "real")

    def test_nonfinite_in_removed_frame_still_invalidates_input(self):
        for arm in ARMS:
            for frame in (0, 1):
                for bad in (np.nan, np.inf, -np.inf):
                    with self.subTest(arm=arm, frame=frame, bad=bad):
                        pair = self.pair.copy()
                        pair[frame, 3, 17] = bad
                        with self.assertRaises(ValueError):
                            transform_pair(pair, arm)

    def test_finite_input_difference_overflow_is_rejected(self):
        pair = np.empty(PAIR_SHAPE, dtype=np.float32)
        pair[0] = -np.finfo(np.float32).max
        pair[1] = np.finfo(np.float32).max
        for arm in ("real", "reverse"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                transform_pair(pair, arm)

    def test_deleted_difference_is_not_unnecessarily_computed(self):
        pair = np.empty(PAIR_SHAPE, dtype=np.float32)
        pair[0] = -np.finfo(np.float32).max
        pair[1] = np.finfo(np.float32).max
        for arm in set(ARMS) - {"real", "reverse"}:
            with self.subTest(arm=arm):
                tokens, _ = transform_pair(pair, arm)
                self.assertTrue(np.isfinite(tokens).all())
                self.assertFalse(tokens[128:].any())


if __name__ == "__main__":
    unittest.main()
