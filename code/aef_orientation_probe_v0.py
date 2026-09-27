#!/usr/bin/env python3
"""AlphaEarth window placement hypotheses, tested by content (affects A1, A1b, A2, A2b).

The AEF GeoTIFFs report transform.e > 0 (bottom-up). A1/A2 fetch: row = inverse transform of the tile centre, read rows
r0..r0+W, then flip rows. If the pixels are in fact stored top-down while the transform claims bottom-up, the right rows are
the mirrored ones: r0' = H - r0 - W. For A2 tiles (8 per region) re-read the Y-1 window under
  H1 as_used      rows r0.., flipped      (what A1/A2 did)
  H2 mirrored     rows H-r0-W.., not flipped
  H3 mirrored     rows H-r0-W.., flipped
  H4 as_read      rows r0.., not flipped
and score held-out R^2 of pre-event NDVI from the 64-d embedding (left/right split). Positive control (S2 earlier date)
was R^2 ~.40 (code/a2b_aef_location_probe_v0.py).

  .venv-geobench/bin/python -B code/aef_orientation_probe_v0.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import MIRROR_FROM, MIRROR_TO, dequant  # noqa: E402
from a2b_aef_alignment_check_v1 import heldout_r2  # noqa: E402
from a2b_fewshot_landslide_v0 import B04, B08, DEFAULT_SRC, ROOT, SRC  # noqa: E402

W = 128


def main():
    import h5py
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    by_reg = defaultdict(list)
    for t in sorted(tiles, key=lambda t: t["id"]):
        if len(by_reg[t["region"]]) < 8:
            by_reg[t["region"]].append(t)
    years = {t["year"] - 1 for ts in by_reg.values() for t in ts}
    idx = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")):
        if r.get("year") and int(r["year"]) in years:
            idx[int(r["year"])].append(r)
    scores = defaultdict(list)
    per_region = defaultdict(lambda: defaultdict(list))
    for reg, ts in by_reg.items():
        for t in ts:
            emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
            raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32)
            p0 = t["idx"][1]
            y = (raw[B08, p0] - raw[B04, p0]) / np.where(raw[B08, p0] + raw[B04, p0] > 0, raw[B08, p0] + raw[B04, p0], np.nan)
            with h5py.File(t["path"], "r") as f:
                cx, cy = float(f["x"][:].mean()), float(f["y"][:].mean())
            lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
            zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
            c = [r for r in idx[t["year"] - 1] if float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"]) and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
            row = ([r for r in c if r["utm_zone"] == f"{zone}{hemi}"] or c or [None])[0]
            if row is None:
                continue
            with rasterio.open("/vsicurl/" + row["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
                ax, ay = Transformer.from_crs(t["crs"], ds.crs, always_xy=True).transform(cx, cy)
                col, rr = ~ds.transform * (ax, ay)
                c0, r0 = int(round(col)) - W // 2, int(round(rr)) - W // 2
                rm = ds.height - r0 - W
                reads = {}
                for name, r_, flip in (("H1_as_used", r0, True), ("H2_mirrored", rm, False), ("H3_mirrored_flipped", rm, True), ("H4_as_read", r0, False)):
                    if 0 <= r_ and r_ + W <= ds.height and 0 <= c0 and c0 + W <= ds.width:
                        if (r_, 0) not in reads:
                            reads[(r_, 0)] = ds.read(window=Window(c0, r_, W, W))
                        a = reads[(r_, 0)]
                        A = dequant(a[:, ::-1, :] if flip else a)
                        v = heldout_r2(A, y)
                        if np.isfinite(v):
                            scores[name].append(v)
                            per_region[name][reg].append(v)
    out = {k: {"median_r2": round(float(np.median(v)), 3), "n": len(v),
               "per_region": {r: round(float(np.median(x)), 3) for r, x in per_region[k].items()}} for k, v in scores.items()}
    (ROOT / "a2b/aef_orientation_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
