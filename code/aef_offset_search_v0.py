#!/usr/bin/env python3
"""Find where the Sen12 tile really is inside the AlphaEarth GeoTIFF (the A1/A2 windows do not match by content).

For a few A2 tiles: read a large block of the AEF Y-1 file around several anchors (as computed; rows mirrored; row/col
swapped), in both row orders, and slide a 128x128 window with stride 16 px (pixels subsampled 2x in the fit). Score = in-sample R^2 of a ridge fit from the 64-d
embedding to the tile's S2 pre-event NDVI (65 parameters, ~16k pixels, so chance R^2 is small). Report the best position
per anchor/order as an offset in pixels from the computed window.

  .venv-geobench/bin/python -B code/aef_offset_search_v0.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import MIRROR_FROM, MIRROR_TO, dequant  # noqa: E402
from a2b_fewshot_landslide_v0 import B04, B08, DEFAULT_SRC, ROOT, SRC  # noqa: E402

W, R, STRIDE = 128, 512, 16
IDS = 4


def r2_fit(A, y):
    X = A[:, ::2, ::2].reshape(A.shape[0], -1).T
    yy = y[::2, ::2].ravel()
    ok = np.isfinite(X).all(1) & np.isfinite(yy)
    X1 = np.c_[X[ok], np.ones(ok.sum())]
    coef = np.linalg.solve(X1.T @ X1 + 1e-3 * np.eye(X1.shape[1]), X1.T @ yy[ok])
    res = yy[ok] - X1 @ coef
    return 1 - (res ** 2).sum() / ((yy[ok] - yy[ok].mean()) ** 2).sum()


def main():
    import h5py
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    pick = []
    for reg in ("hiroshima", "hokkaido", "chimanimani", "thrissur"):
        pick += [t for t in sorted(tiles, key=lambda t: t["id"]) if t["region"] == reg][:1]
    years = {t["year"] - 1 for t in pick}
    idx = [r for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")) if r.get("year") and int(r["year"]) in years]
    out = []
    for t in pick:
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
        p0 = t["idx"][1]
        y = (raw[B08, p0] - raw[B04, p0]) / np.where(raw[B08, p0] + raw[B04, p0] > 0, raw[B08, p0] + raw[B04, p0], np.nan)
        with h5py.File(t["path"], "r") as f:
            cx, cy = float(f["x"][:].mean()), float(f["y"][:].mean())
        lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
        zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
        c = [r for r in idx if int(r["year"]) == t["year"] - 1 and float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"])
             and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
        row = ([r for r in c if r["utm_zone"] == f"{zone}{hemi}"] or c)[0]
        rec = {"id": t["id"], "region": t["region"], "file": row["path"].split("/")[-1], "results": {}}
        with rasterio.open("/vsicurl/" + row["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
            ax, ay = Transformer.from_crs(t["crs"], ds.crs, always_xy=True).transform(cx, cy)
            col, rr = ~ds.transform * (ax, ay)
            c0, r0 = int(round(col)) - W // 2, int(round(rr)) - W // 2
            H, Wd = ds.height, ds.width
            anchors = {"computed": (r0, c0), "rows_mirrored": (H - r0 - W, c0), "row_col_swapped": (c0, r0),
                       "cols_mirrored": (r0, Wd - c0 - W)}
            for name, (ar, ac) in anchors.items():
                br0, bc0 = max(ar - R, 0), max(ac - R, 0)
                br1, bc1 = min(ar + W + R, H), min(ac + W + R, Wd)
                if br1 - br0 < W or bc1 - bc0 < W:
                    continue
                block = dequant(ds.read(window=Window(bc0, br0, bc1 - bc0, br1 - br0)))
                for order in ("rows_as_stored", "rows_flipped"):
                    best = (-9, None)
                    for i in range(0, block.shape[1] - W + 1, STRIDE):
                        for j in range(0, block.shape[2] - W + 1, STRIDE):
                            A = block[:, i:i + W, j:j + W]
                            if order == "rows_flipped":
                                A = A[:, ::-1]
                            s = r2_fit(A, y)
                            if s > best[0]:
                                best = (s, (br0 + i - ar, bc0 + j - ac))
                    at0 = block[:, ar - br0:ar - br0 + W, ac - bc0:ac - bc0 + W]
                    rec["results"][f"{name}|{order}"] = {"best_r2": round(float(best[0]), 3), "offset_rows_cols": best[1],
                                                         "r2_at_anchor": round(float(r2_fit(at0[:, ::-1] if order == "rows_flipped" else at0, y)), 3)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    (ROOT / "a2b/aef_offset_search.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
