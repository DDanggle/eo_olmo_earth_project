#!/usr/bin/env python3
"""AlphaEarth placement v1: boundary cross-correlation (no fitting), exhaustive at 1 px over +-R px.

v0 (in-sample ridge R^2) was ~.3-.5 at every position, so it cannot localise. Here: edge map of the S2 tile (gradient
magnitude of 4 bands B02 B03 B04 B08, pre date, z-scored) vs edge map of the AEF block (gradient magnitude summed over the
64 dims), normalised cross-correlation via FFT. A true match gives a sharp peak; report peak NCC, its offset from the
computed window, and peak-to-median ratio. Anchors: computed window, rows mirrored; row orders as stored and flipped.

  .venv-geobench/bin/python -B code/aef_offset_search_v1.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import MIRROR_FROM, MIRROR_TO, dequant  # noqa: E402
from a2b_fewshot_landslide_v0 import DEFAULT_SRC, ROOT, SRC  # noqa: E402

W, R = 128, 768


def edges(img):
    """img (C,H,W) -> gradient magnitude summed over channels, z-scored."""
    gy, gx = np.gradient(np.nan_to_num(img.astype(np.float64)), axis=(1, 2))
    e = np.sqrt(gx ** 2 + gy ** 2).sum(0)
    return (e - e.mean()) / (e.std() + 1e-9)


def ncc_map(block, tmpl):
    """Normalised cross-correlation of tmpl (h,w) over every full position in block (H,W)."""
    from numpy.fft import irfft2, rfft2
    H, Wd = block.shape
    h, w = tmpl.shape
    t = (tmpl - tmpl.mean()) / (tmpl.std() * tmpl.size)
    F = rfft2(block, (H, Wd))
    num = irfft2(F * np.conj(rfft2(t, (H, Wd))), (H, Wd))[:H - h + 1, :Wd - w + 1]
    c1 = np.cumsum(np.cumsum(np.pad(block, ((1, 0), (1, 0))), 0), 1)
    c2 = np.cumsum(np.cumsum(np.pad(block ** 2, ((1, 0), (1, 0))), 0), 1)
    box = lambda c: c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    s1, s2 = box(c1), box(c2)
    sd = np.sqrt(np.maximum(s2 / (h * w) - (s1 / (h * w)) ** 2, 1e-12))
    return num / sd


def main():
    import h5py
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    pick = []
    for reg in ("hiroshima", "hokkaido", "chimanimani", "thrissur", "kyrgyzstan2", "newzealand"):
        pick += [t for t in sorted(tiles, key=lambda t: t["id"]) if t["region"] == reg][:2]
    years = {t["year"] - 1 for t in pick}
    idx = [r for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")) if r.get("year") and int(r["year"]) in years]
    out = []
    for t in pick:
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
        tm = edges(raw[[0, 1, 2, 6], t["idx"][1]])
        with h5py.File(t["path"], "r") as f:
            cx, cy = float(f["x"][:].mean()), float(f["y"][:].mean())
        lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
        zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
        c = [r for r in idx if int(r["year"]) == t["year"] - 1 and float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"])
             and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
        row = ([r for r in c if r["utm_zone"] == f"{zone}{hemi}"] or c)[0]
        rec = {"id": t["id"], "region": t["region"], "results": {}}
        with rasterio.open("/vsicurl/" + row["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
            ax, ay = Transformer.from_crs(t["crs"], ds.crs, always_xy=True).transform(cx, cy)
            col, rr = ~ds.transform * (ax, ay)
            c0, r0 = int(round(col)) - W // 2, int(round(rr)) - W // 2
            H, Wd = ds.height, ds.width
            for name, (ar, ac) in {"computed": (r0, c0), "rows_mirrored": (H - r0 - W, c0)}.items():
                br0, bc0 = max(ar - R, 0), max(ac - R, 0)
                br1, bc1 = min(ar + W + R, H), min(ac + W + R, Wd)
                if br1 - br0 < W or bc1 - bc0 < W:
                    continue
                A = dequant(ds.read(window=Window(bc0, br0, bc1 - bc0, br1 - br0)))
                for order in ("rows_as_stored", "rows_flipped"):
                    B = A[:, ::-1] if order == "rows_flipped" else A
                    m = ncc_map(edges(B), tm)
                    i, j = np.unravel_index(np.argmax(m), m.shape)
                    # offset of the best window from the anchor, in stored-row coordinates
                    top = (B.shape[1] - W - i) if order == "rows_flipped" else i
                    at = (ar - br0, ac - bc0)
                    ia = (B.shape[1] - W - at[0]) if order == "rows_flipped" else at[0]
                    rec["results"][f"{name}|{order}"] = {"peak_ncc": round(float(m[i, j]), 3), "ncc_at_anchor": round(float(m[ia, at[1]]), 3),
                                                         "offset_rows_cols": [int(br0 + top - ar), int(bc0 + j - ac)],
                                                         "peak_over_median": round(float(m[i, j] / (np.median(np.abs(m)) + 1e-9)), 1)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    (ROOT / "a2b/aef_offset_search_v1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
