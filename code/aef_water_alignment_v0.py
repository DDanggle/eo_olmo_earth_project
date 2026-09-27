#!/usr/bin/env python3
"""AlphaEarth registration, sharp test: permanent water in KuroSiwo tiles (A1/A1b windows as saved in a1_aef/tiles).

Water is sharp and distinctive in any embedding. For A1b tiles of kind 'perm' (permanent-water tiles) and flood tiles with
permanent water, build a label from the KuroSiwo mask (1 = permanent water, 0 = land; flood pixels ignored), and score
how well the AEF Y-1 embedding separates them with an in-sample nearest-centroid rule (2 x 64 means, little overfit):
AUC of cos(to water centroid) - cos(to land centroid). Repeat with the AEF window shifted by dy, dx in -48..48 px (step 4)
against the mask. Registered windows peak at (0, 0). Also the same with the rows of the AEF window flipped back.

  .venv-master/bin/python -B code/aef_water_alignment_v0.py [kuro|sen12]   (sen12: water from S2 NDWI, A2 windows)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import auc, dequant  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
KURO = ROOT / "kurosiwo_s1_cache"
AEF = ROOT / "a1_aef/tiles"
SH = range(-48, 49, 4)
MIN_WATER = int(__import__("os").environ.get("MIN_WATER", "200"))   # sen12 tiles are mountainous: run with MIN_WATER=50
PERM = 2          # KuroSiwo mask classes in the cache: 0 nodata, 1 no water, 2 permanent water, 3 flood (A1 uses m == 3 flood)


def score(A, lab):
    """A (64,H,W) embedding, lab (H,W) in {1 water, 0 land, -1 ignore}."""
    X = A.reshape(64, -1).T
    y = lab.ravel()
    ok = np.isfinite(X).all(1) & (y >= 0)
    if (y[ok] == 1).sum() < 50 or (y[ok] == 0).sum() < 50:
        return np.nan
    Xn = X[ok] / np.linalg.norm(X[ok], axis=1, keepdims=True)
    cw, cl = Xn[y[ok] == 1].mean(0), Xn[y[ok] == 0].mean(0)
    s = Xn @ cw - Xn @ cl
    return auc(s[y[ok] == 1], s[y[ok] == 0])


def overlap(A, lab, dy, dx):
    h, w = lab.shape
    ya, yl = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xa, xl = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    return A[:, ya, xa], lab[yl, xl]


def sen12_labels():
    """A2 (Sen12) tiles: water from S2 pre-event NDWI (B03, B08) > .2, land < 0; yields (id, year-1 AEF path, label)."""
    from a2b_fewshot_landslide_v0 import DEFAULT_SRC, SRC
    for t in json.loads((ROOT / "a2/tiles.json").read_text()):
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
        g, n = raw[1, t["idx"][1]], raw[6, t["idx"][1]]
        ndwi = (g - n) / np.where(g + n > 0, g + n, np.nan)
        yield t, AEF / f"{t['id']}_{t['year'] - 1}.npy", np.where(ndwi > .2, 1, np.where(ndwi < 0, 0, -1)).astype(np.int8)


def kuro_labels():
    for t in json.loads((ROOT / "a1b/tiles.json").read_text()):
        m = np.load(KURO / "mask_u8" / f"{t['id']}.npy")
        yield t, AEF / f"{t['id']}_{t['year'] - 1}.npy", np.where(m == PERM, 1, np.where(m == 1, 0, -1)).astype(np.int8)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "kuro"
    gen = sen12_labels if source == "sen12" else kuro_labels
    res = {}
    for flip in (False, True):
        grid = defaultdict(list)
        n = 0
        for t, f, lab in gen():
            if not f.exists() or (lab == 1).sum() < MIN_WATER:
                continue
            A = dequant(np.load(f))
            if A.shape[1:] != lab.shape:
                continue
            if flip:
                A = A[:, ::-1]
            n += 1
            pairs = [(dy, dx) for dy in SH for dx in SH]
            if __import__("os").environ.get("PROFILE"):
                pairs = sorted({(0, d) for d in SH} | {(d, 0) for d in SH} | {(d, -24) for d in SH})
            for dy, dx in pairs:
                    v = score(*overlap(A, lab, dy, dx))
                    if np.isfinite(v):
                        grid[(dy, dx)].append(v)
            if n >= 150:
                break
        med = {k: float(np.median(v)) for k, v in grid.items() if len(v) >= 0.8 * n}
        if not med:
            res["rows_flipped_back" if flip else "as_saved"] = {"n_tiles": n}
            continue
        best = max(med, key=med.get)
        res["rows_flipped_back" if flip else "as_saved"] = {"n_tiles": n, "median_auc_at_0": round(med[(0, 0)], 3), "best_shift": list(best),
                                                            "median_auc_best": round(med[best], 3),
                                                            "median_auc_at_(8,0)": round(med.get((8, 0), np.nan), 3), "median_auc_at_(0,8)": round(med.get((0, 8), np.nan), 3),
                                                            "median_auc_at_(24,24)": round(med.get((24, 24), np.nan), 3), "median_auc_at_(48,48)": round(med.get((48, 48), np.nan), 3)}
        if __import__("os").environ.get("PROFILE"):
            res[("rows_flipped_back" if flip else "as_saved") + "_profile"] = {
                "dx_at_dy0": {d: round(med.get((0, d), np.nan), 3) for d in SH}, "dy_at_dx0": {d: round(med.get((d, 0), np.nan), 3) for d in SH},
                "dy_at_dx-24": {d: round(med.get((d, -24), np.nan), 3) for d in SH}}
    (ROOT / f"a2b/aef_water_alignment_{source}{'_profile' if __import__('os').environ.get('PROFILE') else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
