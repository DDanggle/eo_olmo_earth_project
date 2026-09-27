#!/usr/bin/env python3
"""A2b validity: where do the A2 AlphaEarth windows actually sit relative to the Sen12 tiles?

1) positive control for the v1 R^2 test: predict pre-event NDVI from the S2 bands of the acquisition BEFORE it (pre1), same
   left/right split. If this is high while AEF is negative, the test works and AEF is not registered.
2) coordinates of 3 tiles per region: .nc crs/x/y ranges, centre lon/lat, the AEF index row chosen (zone, bounds) and the
   window pixel offsets, so a wrong CRS / axis order / zone can be seen.

  .venv-master/bin/python -B code/a2b_aef_location_probe_v0.py      (needs h5py, pyproj, rasterio -> .venv-geobench)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import MIRROR_FROM, MIRROR_TO  # noqa: E402
from a2b_aef_alignment_check_v1 import heldout_r2  # noqa: E402
from a2b_fewshot_landslide_v0 import B04, B08, DEFAULT_SRC, ROOT, SRC  # noqa: E402


def main():
    import h5py
    import rasterio
    from pyproj import Transformer
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    by_reg = defaultdict(list)
    for t in sorted(tiles, key=lambda t: t["id"]):
        if len(by_reg[t["region"]]) < 8:
            by_reg[t["region"]].append(t)
    ctrl = []
    for ts in by_reg.values():
        for t in ts:
            emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
            raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
            p1, p0, _ = t["idx"]
            y = (raw[B08, p0] - raw[B04, p0]) / np.where(raw[B08, p0] + raw[B04, p0] > 0, raw[B08, p0] + raw[B04, p0], np.nan)
            A = np.zeros((64,) + y.shape, np.float32)
            A[:10] = raw[:, p1] / 10000.0          # 10 bands of the earlier date, rest zero (same 64-d solver)
            ctrl.append(heldout_r2(A, y))
    out = {"control_s2_pre1_to_ndvi_pre_median_r2": float(np.nanmedian(ctrl)), "tiles": []}
    idx = defaultdict(list)
    years = {t["year"] - 1 for ts in by_reg.values() for t in ts[:3]}
    for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")):
        if r.get("year") and int(r["year"]) in years:
            idx[int(r["year"])].append(r)
    for reg, ts in by_reg.items():
        for t in ts[:3]:
            with h5py.File(t["path"], "r") as f:
                x, y = f["x"][:], f["y"][:]
                keys = list(f.keys())
            cx, cy = float(x.mean()), float(y.mean())
            lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
            zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
            c = [r for r in idx[t["year"] - 1] if float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"]) and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
            e = [r for r in c if r["utm_zone"] == f"{zone}{hemi}"]
            row = (e or c or [None])[0]
            rec = {"id": t["id"], "region": reg, "crs": t["crs"], "nc_keys": keys, "x_range": [float(x.min()), float(x.max())], "y_range": [float(y.min()), float(y.max())],
                   "x_len": int(len(x)), "y_first_last": [float(y[0]), float(y[-1])], "lonlat": [round(lon, 5), round(lat, 5)], "n_candidates": len(c)}
            if row:
                with rasterio.open("/vsicurl/" + row["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
                    ax, ay = Transformer.from_crs(t["crs"], ds.crs, always_xy=True).transform(cx, cy)
                    col, rr = ~ds.transform * (ax, ay)
                    rec.update({"aef_crs": str(ds.crs), "aef_transform": list(ds.transform)[:6], "aef_size": [ds.width, ds.height],
                                "aef_zone": row["utm_zone"], "col_row": [round(col, 1), round(rr, 1)]})
            out["tiles"].append(rec)
    (ROOT / "a2b/aef_location_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
