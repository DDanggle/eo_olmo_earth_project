import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from cache_grid_controls import expected_olmo_shape, transform_embedding


class CacheGridControlsTest(unittest.TestCase):
    def test_patch_shape_tracks_patch_size(self):
        self.assertEqual(expected_olmo_shape("base", 4), (768, 32, 32))
        self.assertEqual(expected_olmo_shape("base", 2), (768, 64, 64))

    def test_avgpool_exact(self):
        x = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        got = transform_embedding(x, "avgpool2")
        want = np.array([[[2.5, 4.5], [10.5, 12.5]]], dtype=np.float32)
        np.testing.assert_allclose(got, want)

    def test_upsample_adds_no_new_range(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("local lightweight Python has no torch; server smoke covers this branch")
        x = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
        got = transform_embedding(x, "upsample2")
        self.assertEqual(got.shape, (1, 4, 4))
        self.assertGreaterEqual(float(got.min()), float(x.min()))
        self.assertLessEqual(float(got.max()), float(x.max()))

    def test_bad_shape_and_patch_fail(self):
        with self.assertRaises(ValueError):
            transform_embedding(np.zeros((2, 3, 4), dtype=np.float32), "native")
        with self.assertRaises(ValueError):
            expected_olmo_shape("base", 3)


if __name__ == "__main__":
    unittest.main()
