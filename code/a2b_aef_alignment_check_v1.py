#!/usr/bin/env python3
"""A2b validity check v1: registration of AlphaEarth windows to Sen12 tiles by content, not change.

v0 (change map vs mask/dNDVI) was flat everywhere (~.02), which cannot separate "no landslide signal" from "misregistered
by > 120 m". Here: per tile, ridge-regress the S2 pre-event NDVI (10 m) on the AlphaEarth Y-1 embedding (64-d) with a
2-fold spatial split (left/right halves) and report held-out R^2 at shifts of the AEF window (-48..48 px, step 8, axes and diagonals) and with
rows flipped. AlphaEarth encodes vegetation strongly, so a registered window gives a clear R^2 peak at (0, 0).
64 tiles (8 per region, deterministic).

  .venv-master/bin/python -B code/a2b_aef_alignment_check_v1.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import dequant  # noqa: E402
from a2b_fewshot_landslide_v0 import AEF_TILES, B04, B08, DEFAULT_SRC, ROOT, SRC  # noqa: E402

SHIFTS = range(-48, 49, 8)


def heldout_r2(A, y):
    """A (64,H,W), y (H,W); train on left half, test on right half and vice versa."""
    h, w = y.shape
    r2 = []
    for tr, te in ((slice(0, w // 2), slice(w // 2, w)), (slice(w // 2, w), slice(0, w // 2))):
        Xa, ya = A[:, :, tr].reshape(64, -1).T, y[:, tr].ravel()
        Xb, yb = A[:, :, te].reshape(64, -1).T, y[:, te].ravel()
        oa, ob = np.isfinite(Xa).all(1) & np.isfinite(ya), np.isfinite(Xb).all(1) & np.isfinite(yb)
        if oa.sum() < 200 or ob.sum() < 200 or yb[ob].std() == 0:
            return np.nan
        Xa1 = np.c_[Xa[oa], np.ones(oa.sum())]
        coef = np.linalg.solve(Xa1.T @ Xa1 + 1e-2 * np.eye(65), Xa1.T @ ya[oa])
        pred = np.c_[Xb[ob], np.ones(ob.sum())] @ coef
        r2.append(1 - ((yb[ob] - pred) ** 2).sum() / ((yb[ob] - yb[ob].mean()) ** 2).sum())
    return float(np.mean(r2))


def shifted(A, dy, dx):
    """Shift AEF content by (dy, dx) px, NaN-padded."""
    out = np.full_like(A, np.nan)
    c, h, w = A.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = A[:, ys, xs]
    return out


def main():
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    by_reg = defaultdict(list)
    for t in sorted(tiles, key=lambda t: t["id"]):
        if (AEF_TILES / f"{t['id']}_{t['year'] - 1}.npy").exists() and len(by_reg[t["region"]]) < 8:
            by_reg[t["region"]].append(t)
    res = {}
    for flip in (False, True):
        grid = defaultdict(list)
        per_region0 = defaultdict(list)
        for reg, ts in by_reg.items():
            for t in ts:
                emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
                A = dequant(np.load(AEF_TILES / f"{t['id']}_{t['year'] - 1}.npy"))
                if flip:
                    A = A[:, ::-1]
                raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
                p0 = t["idx"][1]
                y = (raw[B08, p0] - raw[B04, p0]) / np.where(raw[B08, p0] + raw[B04, p0] > 0, raw[B08, p0] + raw[B04, p0], np.nan)
                for dy in SHIFTS:
                    for dx in SHIFTS:
                        if dy and dx and abs(dy) != abs(dx):
                            continue      # axes and diagonals only (runtime)
                        v = heldout_r2(shifted(A, dy, dx), y)
                        if np.isfinite(v):
                            grid[(dy, dx)].append(v)
                            if (dy, dx) == (0, 0):
                                per_region0[reg].append(v)
        med = {k: float(np.median(v)) for k, v in grid.items()}
        best = max(med, key=med.get)
        res["flipped" if flip else "as_used"] = {"median_r2_at_0": round(med[(0, 0)], 3), "best_shift_px": list(best), "median_r2_best": round(med[best], 3),
                                                 "median_r2_shift_24_0": round(med.get((24, 0), np.nan), 3), "median_r2_shift_8_0": round(med.get((8, 0), np.nan), 3),
                                                 "per_region_r2_at_0": {r: round(float(np.median(v)), 3) for r, v in per_region0.items()}}
    (ROOT / "a2b/aef_alignment_check_v1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
