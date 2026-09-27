#!/usr/bin/env python3
"""A2b validity check: are the AlphaEarth windows registered to the Sen12 tiles?

AlphaEarth T2 (token AUC) in A2b stays ~.52-.60 even with labels. That is also what a misregistered window would give.
Test: per positive tile, correlate the AlphaEarth change map 1-cos(Y, Y-1) with (a) the S2 vegetation-loss map
NDVI_pre - NDVI_post and (b) the landslide mask, at integer shifts of -12..12 px (10 m) and with the row order flipped.
Registered windows peak at shift (0, 0) without flip. Same test for the OlmoEarth change map (40 m, as a control).

  .venv-master/bin/python -B code/a2b_aef_alignment_check_v0.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import cos_change, dequant  # noqa: E402
from a2b_fewshot_landslide_v0 import AEF_TILES, B04, B08, DEFAULT_SRC, ROOT, SRC  # noqa: E402

S = 12


def corr_shift(a, b, dy, dx):
    h, w = a.shape
    ya, yb = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xa, xb = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    u, v = a[ya, xa].ravel(), b[yb, xb].ravel()
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 100 or u[ok].std() == 0 or v[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(u[ok], v[ok])[0, 1])


def main():
    tiles = [t for t in json.loads((ROOT / "a2/tiles.json").read_text()) if t["kind"] == "pos"]
    out = {}
    for flip in (False, True):
        for ref in ("dndvi", "mask"):
            grid = np.zeros((2 * S + 1, 2 * S + 1))
            cnt = np.zeros_like(grid)
            for t in tiles:
                fY, fB = (AEF_TILES / f"{t['id']}_{t['year'] + d}.npy" for d in (0, -1))
                if not (fY.exists() and fB.exists()):
                    continue
                emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
                a = cos_change(dequant(np.load(fY)), dequant(np.load(fB)))
                if flip:
                    a = a[::-1]
                if ref == "mask":
                    b = (np.load(src / "mask_u8" / f"{t['id']}.npy") > 0).astype(np.float32)
                else:
                    raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
                    _, p0, q = t["idx"]
                    nd = lambda i: (raw[B08, i] - raw[B04, i]) / np.where(raw[B08, i] + raw[B04, i] > 0, raw[B08, i] + raw[B04, i], np.nan)
                    b = nd(p0) - nd(q)
                for i, dy in enumerate(range(-S, S + 1)):
                    for j, dx in enumerate(range(-S, S + 1)):
                        c = corr_shift(a, b, dy, dx)
                        if np.isfinite(c):
                            grid[i, j] += c
                            cnt[i, j] += 1
            g = grid / np.maximum(cnt, 1)
            iy, ix = np.unravel_index(np.nanargmax(g), g.shape)
            out[f"aef_vs_{ref}{'_flipped' if flip else ''}"] = {"corr_at_0": round(float(g[S, S]), 4), "best": round(float(g[iy, ix]), 4),
                                                                "best_shift_px": [int(iy - S), int(ix - S)], "n": int(cnt[S, S])}
    # control: OlmoEarth change (40 m) vs mask pooled to 40 m, shifts in tokens
    g, c = np.zeros((7, 7)), np.zeros((7, 7))
    for t in tiles:
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        _, p0, q = t["idx"]
        E = np.load(emb_dir / f"{t['id']}.npy").astype(np.float32)
        a = cos_change(E[q], E[p0])
        m = (np.load(src / "mask_u8" / f"{t['id']}.npy") > 0).astype(np.float32).reshape(32, 4, 32, 4).mean((1, 3))
        for i, dy in enumerate(range(-3, 4)):
            for j, dx in enumerate(range(-3, 4)):
                v = corr_shift(a, m, dy, dx)
                if np.isfinite(v):
                    g[i, j] += v
                    c[i, j] += 1
    g = g / np.maximum(c, 1)
    iy, ix = np.unravel_index(np.nanargmax(g), g.shape)
    out["olmo_vs_mask_40m"] = {"corr_at_0": round(float(g[3, 3]), 4), "best": round(float(g[iy, ix]), 4), "best_shift_tokens": [int(iy - 3), int(ix - 3)]}
    (ROOT / "a2b/aef_alignment_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
