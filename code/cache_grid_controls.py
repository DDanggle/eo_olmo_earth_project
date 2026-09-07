"""Pure helpers for the token-resolution contract experiment."""
from __future__ import annotations

import numpy as np


OLMO_CHANNELS = {"nano": 128, "tiny": 192, "base": 768}


def expected_olmo_shape(size: str, patch: int, tile_pixels: int = 128) -> tuple[int, int, int]:
    if size not in OLMO_CHANNELS:
        raise ValueError(f"unknown OlmoEarth size: {size}")
    if patch <= 0 or tile_pixels % patch:
        raise ValueError(f"patch must be a positive divisor of {tile_pixels}, got {patch}")
    grid = tile_pixels // patch
    return OLMO_CHANNELS[size], grid, grid


def transform_embedding(x: np.ndarray, mode: str) -> np.ndarray:
    """Return a CxGxG token-grid control as float32.

    ``upsample2`` changes only decoder input sampling; it adds no representation
    information. ``avgpool2`` removes the extra spatial detail from a dense cache.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != x.shape[2]:
        raise ValueError(f"expected CxGxG embedding, got {x.shape}")
    if mode == "native":
        return x
    if mode == "avgpool2":
        c, g, _ = x.shape
        if g % 2:
            raise ValueError(f"avgpool2 needs an even grid, got {x.shape}")
        return x.reshape(c, g // 2, 2, g // 2, 2).mean(axis=(2, 4), dtype=np.float32)
    if mode == "upsample2":
        import torch
        import torch.nn.functional as F

        return F.interpolate(
            torch.from_numpy(x).unsqueeze(0),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).numpy()
    raise ValueError(f"unknown grid transform: {mode}")
