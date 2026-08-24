#!/usr/bin/env python3
"""Fail-closed raster health checks for the OlmoEarth release audit.

The writer produces a 4x downsampled embedding grid.  A file existing with the
right number of bands is not sufficient evidence: NaN-only, zero-only, or
geometrically shifted rasters would make the downstream release comparison
meaningless.  This module keeps the geometry derivation and value checks shared
by the full runner and the paired-evidence finalizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FEATURES = 768
DOWNSAMPLE_FACTOR = 4
EXPECTED_DTYPE = "float32"
MINIMUM_USABLE_TOKENS = 2


def expected_grid(window_metadata: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact embedding grid from an rslearn window metadata record."""

    projection = window_metadata["projection"]
    bounds = window_metadata["bounds"]
    if len(bounds) != 4:
        raise ValueError("window bounds must contain four coordinates")
    pixel_width = int(bounds[2]) - int(bounds[0])
    pixel_height = int(bounds[3]) - int(bounds[1])
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError("window bounds do not define a positive grid")
    if pixel_width % DOWNSAMPLE_FACTOR or pixel_height % DOWNSAMPLE_FACTOR:
        raise ValueError("window grid is not divisible by the output downsample factor")

    source_x = float(projection["x_resolution"])
    source_y = float(projection["y_resolution"])
    if source_x <= 0 or source_y >= 0:
        raise ValueError("expected north-up source resolution with x>0 and y<0")
    x_resolution = source_x * DOWNSAMPLE_FACTOR
    y_resolution = source_y * DOWNSAMPLE_FACTOR
    x_values = (float(bounds[0]) * source_x, float(bounds[2]) * source_x)
    y_values = (float(bounds[1]) * source_y, float(bounds[3]) * source_y)
    left, right = min(x_values), max(x_values)
    bottom, top = min(y_values), max(y_values)
    return {
        "height": pixel_height // DOWNSAMPLE_FACTOR,
        "width": pixel_width // DOWNSAMPLE_FACTOR,
        "count": FEATURES,
        "dtype": EXPECTED_DTYPE,
        "crs": projection["crs"],
        "transform": [x_resolution, 0.0, left, 0.0, y_resolution, top],
        "bounds": [left, bottom, right, top],
    }


def inspect_open_raster(
    dataset: Any,
    window_metadata: dict[str, Any],
    *,
    spatial_row_chunk: int = 8,
    return_validity_mask: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Validate geometry and scan every value without a 200 MiB allocation.

    A usable token must be unmasked and finite in all 768 feature dimensions.
    At least two usable, non-zero tokens are required because all downstream
    geometry statistics need more than a single observation.
    """

    if spatial_row_chunk < 1:
        raise ValueError("spatial_row_chunk must be positive")
    expected = expected_grid(window_metadata)
    actual_crs = dataset.crs.to_string() if dataset.crs else None
    actual_transform = list(dataset.transform)[:6]
    actual_bounds = list(dataset.bounds)
    actual_dtypes = sorted(set(dataset.dtypes))
    if dataset.height != expected["height"] or dataset.width != expected["width"]:
        raise ValueError(
            f"output grid shape drift: {(dataset.height, dataset.width)} != "
            f"{(expected['height'], expected['width'])}"
        )
    if dataset.count != expected["count"]:
        raise ValueError(f"output feature count drift: {dataset.count}")
    if actual_dtypes != [expected["dtype"]]:
        raise ValueError(f"output dtype drift: {actual_dtypes}")
    if actual_crs != expected["crs"]:
        raise ValueError(f"output CRS drift: {actual_crs} != {expected['crs']}")
    if not np.allclose(actual_transform, expected["transform"], rtol=0.0, atol=1e-6):
        raise ValueError(f"output transform drift: {actual_transform}")
    if not np.allclose(actual_bounds, expected["bounds"], rtol=0.0, atol=1e-6):
        raise ValueError(f"output bounds drift: {actual_bounds}")

    usable = np.ones((dataset.height, dataset.width), dtype=bool)
    any_nonzero = np.zeros_like(usable)
    finite_values = 0
    total_values = dataset.count * dataset.height * dataset.width
    # These GeoTIFFs are pixel-interleaved.  Reading feature chunks over the
    # full image would decompress the same strips repeatedly.  Read every band
    # once for a small row window instead: bounded memory, one logical pass.
    indexes = list(range(1, dataset.count + 1))
    for row_start in range(0, dataset.height, spatial_row_chunk):
        row_stop = min(row_start + spatial_row_chunk, dataset.height)
        window = ((row_start, row_stop), (0, dataset.width))
        values = dataset.read(indexes=indexes, window=window, out_dtype="float32")
        masks = dataset.read_masks(indexes=indexes, window=window)
        finite = np.isfinite(values)
        row_slice = slice(row_start, row_stop)
        usable[row_slice] &= finite.all(axis=0) & (masks > 0).all(axis=0)
        any_nonzero[row_slice] |= (finite & (values != 0.0)).any(axis=0)
        finite_values += int(finite.sum())
    nonzero_usable = usable & any_nonzero
    usable_tokens = int(usable.sum())
    nonzero_usable_tokens = int(nonzero_usable.sum())
    if usable_tokens < MINIMUM_USABLE_TOKENS:
        raise ValueError(f"only {usable_tokens} fully finite, unmasked output tokens")
    if nonzero_usable_tokens < MINIMUM_USABLE_TOKENS:
        raise ValueError(f"only {nonzero_usable_tokens} non-zero usable output tokens")

    contract = {
        "height": dataset.height,
        "width": dataset.width,
        "count": dataset.count,
        "dtypes": actual_dtypes,
        "crs": actual_crs,
        "transform": actual_transform,
        "bounds": actual_bounds,
        "nodata": dataset.nodata,
        "usable_tokens": usable_tokens,
        "nonzero_usable_tokens": nonzero_usable_tokens,
        "finite_values": finite_values,
        "total_values": total_values,
        "all_values_finite": finite_values == total_values,
    }
    return contract, usable if return_validity_mask else None


def inspect_raster(
    path: Path,
    window_metadata: dict[str, Any],
    *,
    spatial_row_chunk: int = 8,
    return_validity_mask: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Open and validate one embedding GeoTIFF."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - server integration dependency
        raise RuntimeError("rasterio is required for output validation") from exc
    with rasterio.open(path) as dataset:
        return inspect_open_raster(
            dataset,
            window_metadata,
            spatial_row_chunk=spatial_row_chunk,
            return_validity_mask=return_validity_mask,
        )
