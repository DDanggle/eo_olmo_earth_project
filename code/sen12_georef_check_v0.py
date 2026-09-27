#!/usr/bin/env python3
"""Are the Sen12Landslides tile coordinates right? S2 (Sen12 raw) vs S2 (Planetary Computer) at the tile's own x/y.

The AEF windows for A2/A2b are placed from the .nc x/y (EPSG from the contract). KuroSiwo windows passed a sharp water test
(a2b/aef_water_alignment_kuro.json); Sen12 tiles have too little water for that test. Here: for 2 tiles per region, read the
PC sentinel-2-l2a scene of the same date (or nearest within 5 days) on the tile grid +-64 px (step 2), for the Sen12
image as is, flipped up-down, left-right, transposed and rotated 180, and find the shift that
maximises NCC of B08 (NIR) between Sen12 and PC. A correct georeference peaks at (0, 0) with high NCC.

  .venv-geobench/bin/python -B code/sen12_georef_check_v0.py
"""
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2b_fewshot_landslide_v0 import DEFAULT_SRC, ROOT, SRC  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
P = 64


def ncc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    u, v = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float((u * v).sum() / (np.sqrt((u * u).sum() * (v * v).sum()) + 1e-9))


def main():
    import h5py
    import rasterio
    import requests
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    tok = requests.get(SAS, timeout=60).json()["token"]
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    by = defaultdict(list)
    for t in sorted(tiles, key=lambda t: t["id"]):
        if len(by[t["region"]]) < 2:
            by[t["region"]].append(t)
    out = []
    for reg, ts in by.items():
        for t in ts:
            emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
            raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
            with h5py.File(t["path"], "r") as f:
                x, y = f["x"][:], f["y"][:]
            d = date.fromisoformat(t["dates"][1])
            ti = t["idx"][1]
            lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(float(x.mean()), float(y.mean()))
            q = {"collections": ["sentinel-2-l2a"], "limit": 20, "intersects": {"type": "Point", "coordinates": [lon, lat]},
                 "datetime": f"{d - timedelta(days=5)}/{d + timedelta(days=5)}"}
            feats = requests.post(STAC, json=q, timeout=120).json().get("features", [])
            if not feats:
                out.append({"id": t["id"], "status": "no_scene"})
                continue
            it = sorted(feats, key=lambda f: (abs((date.fromisoformat(f["properties"]["datetime"][:10]) - d).days), f["properties"]["eo:cloud_cover"]))[0]
            # tile grid: x/y are pixel centres at 10 m, y descending
            x0, y0 = float(x.min()) - 5 - P * 10, float(y.max()) + 5 + P * 10
            tr = from_origin(x0, y0, 10, 10)
            with rasterio.open(it["assets"]["B08"]["href"] + "?" + tok) as s, WarpedVRT(s, crs=t["crs"], transform=tr, width=128 + 2 * P,
                                                                                      height=128 + 2 * P, resampling=Resampling.nearest) as v:
                pc = v.read(1).astype(np.float32)
            pc[pc == 0] = np.nan
            rec = {"id": t["id"], "region": reg, "sen12_date": t["dates"][1], "pc_date": it["properties"]["datetime"][:10], "y_desc": bool(y[0] > y[-1])}
            base = raw[6, ti]
            for vname, a in (("as_is", base), ("flipud", base[::-1]), ("fliplr", base[:, ::-1]), ("transpose", base.T), ("rot180", base[::-1, ::-1])):
                best, c0 = (-2, None), None
                for dy in range(-P, P + 1, 2):
                    for dx in range(-P, P + 1, 2):
                        c = ncc(a, pc[P + dy:P + dy + 128, P + dx:P + dx + 128])
                        if (dy, dx) == (0, 0):
                            c0 = c
                        if c > best[0]:
                            best = (c, (dy, dx))
                rec[vname] = {"ncc_at_0": round(c0, 3), "best_ncc": round(best[0], 3), "best_shift": best[1]}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    (ROOT / "a2b/sen12_georef_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
