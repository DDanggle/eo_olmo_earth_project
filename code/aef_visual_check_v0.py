#!/usr/bin/env python3
"""L5 visual check: AlphaEarth windows (as saved in a1_aef/tiles, Y-1) vs the Sen12 S2 true colour of the same tile.

Row per tile: S2 RGB (pre date) | AEF PCA-3 RGB (per-tile PCA) | AEF PCA-3 rows flipped | AEF with values NOT dequantised.
2 tiles per region. Writes a2b/aef_visual_check.png.

  .venv-geobench/bin/python -B code/aef_visual_check_v0.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import dequant  # noqa: E402
from a2b_fewshot_landslide_v0 import AEF_TILES, DEFAULT_SRC, ROOT, SRC  # noqa: E402


def pca3(A):
    c, h, w = A.shape
    X = A.reshape(c, -1).T.astype(np.float64)
    ok = np.isfinite(X).all(1)
    Z = np.zeros((h * w, 3))
    if ok.sum() > 10:
        Xo = X[ok] - X[ok].mean(0)
        _, _, vt = np.linalg.svd(Xo, full_matrices=False)
        Z[ok] = Xo @ vt[:3].T
    lo, hi = np.percentile(Z[ok], 2, 0), np.percentile(Z[ok], 98, 0)
    return np.clip((Z - lo) / np.where(hi > lo, hi - lo, 1), 0, 1).reshape(h, w, 3)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    by_reg = defaultdict(list)
    for t in sorted(tiles, key=lambda t: t["id"]):
        if (AEF_TILES / f"{t['id']}_{t['year'] - 1}.npy").exists() and len(by_reg[t["region"]]) < 2:
            by_reg[t["region"]].append(t)
    sel = [t for ts in by_reg.values() for t in ts]
    fig, ax = plt.subplots(len(sel), 4, figsize=(10, 2.5 * len(sel)))
    for i, t in enumerate(sel):
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
        p0 = t["idx"][1]
        rgb = np.clip(np.stack([raw[2, p0], raw[1, p0], raw[0, p0]], -1) / 3000.0, 0, 1)
        q = np.load(AEF_TILES / f"{t['id']}_{t['year'] - 1}.npy")
        A = dequant(q)
        for j, (img, title) in enumerate(((rgb, f"{t['region']} S2 {t['dates'][1]}"), (pca3(A), "AEF Y-1 PCA (as used)"),
                                          (pca3(A[:, ::-1]), "AEF rows flipped"), (pca3(q.astype(np.float32)), "AEF raw int8"))):
            ax[i, j].imshow(img)
            ax[i, j].set_title(title, fontsize=7)
            ax[i, j].axis("off")
    plt.tight_layout()
    out = ROOT / "a2b/aef_visual_check.png"
    plt.savefig(out, dpi=80)
    q = np.load(AEF_TILES / f"{sel[0]['id']}_{sel[0]['year'] - 1}.npy")
    print(out, q.dtype, q.shape, "int8 range", int(q.min()), int(q.max()), "frac -128", float((q == -128).mean()),
          "dequant norm median", float(np.nanmedian(np.sqrt(np.nansum(dequant(q) ** 2, 0)))))


if __name__ == "__main__":
    main()
