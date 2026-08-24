from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from olmo_release_raster_contract import expected_grid, inspect_open_raster  # noqa: E402


METADATA = {
    "projection": {"crs": "EPSG:32652", "x_resolution": 10.0, "y_resolution": -10.0},
    "bounds": [22528, -367616, 23552, -366592],
}


class _FakeCRS:
    def to_string(self) -> str:
        return "EPSG:32652"


class _FakeRaster:
    height = 256
    width = 256
    count = 768
    dtypes = ("float32",) * 768
    crs = _FakeCRS()
    transform = (40.0, 0.0, 225280.0, 0.0, -40.0, 3676160.0)
    bounds = (225280.0, 3665920.0, 235520.0, 3676160.0)
    nodata = None

    def __init__(self, *, zero: bool = False, nan: bool = False) -> None:
        self.zero = zero
        self.nan = nan

    def read(
        self, indexes: list[int], window: tuple[tuple[int, int], tuple[int, int]], out_dtype: str
    ) -> np.ndarray:
        (row_start, row_stop), (column_start, column_stop) = window
        values = np.ones(
            (len(indexes), row_stop - row_start, column_stop - column_start),
            dtype=np.float32,
        )
        if self.zero:
            values.fill(0.0)
        if self.nan:
            values.fill(np.nan)
        return values

    def read_masks(
        self, indexes: list[int], window: tuple[tuple[int, int], tuple[int, int]]
    ) -> np.ndarray:
        (row_start, row_stop), (column_start, column_stop) = window
        return np.full(
            (len(indexes), row_stop - row_start, column_stop - column_start),
            255,
            dtype=np.uint8,
        )


class RasterContractTests(unittest.TestCase):
    def test_expected_grid_is_exact_256_at_40_meters(self) -> None:
        expected = expected_grid(METADATA)
        self.assertEqual(expected["height"], 256)
        self.assertEqual(expected["width"], 256)
        self.assertEqual(
            expected["transform"],
            [40.0, 0.0, 225280.0, 0.0, -40.0, 3676160.0],
        )

    def test_finite_nonzero_raster_passes(self) -> None:
        contract, validity = inspect_open_raster(
            _FakeRaster(), METADATA, spatial_row_chunk=32, return_validity_mask=True
        )
        self.assertEqual(contract["usable_tokens"], 256 * 256)
        self.assertEqual(contract["nonzero_usable_tokens"], 256 * 256)
        self.assertTrue(contract["all_values_finite"])
        self.assertIsNotNone(validity)

    def test_nan_only_and_zero_only_rasters_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "fully finite"):
            inspect_open_raster(_FakeRaster(nan=True), METADATA, spatial_row_chunk=64)
        with self.assertRaisesRegex(ValueError, "non-zero"):
            inspect_open_raster(_FakeRaster(zero=True), METADATA, spatial_row_chunk=64)

    def test_wrong_grid_fails(self) -> None:
        raster = _FakeRaster()
        raster.width = 1
        with self.assertRaisesRegex(ValueError, "shape drift"):
            inspect_open_raster(raster, METADATA)


if __name__ == "__main__":
    unittest.main()
