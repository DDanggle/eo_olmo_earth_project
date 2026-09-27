#!/usr/bin/env python3
"""K0 step 2 (server): L5 visual check - do the Korean landslide points sit on visible scars in Sentinel-2?

Takes data_kr/k0_points.json (pushed to /home/work/data/olmoearth/k0/), picks the largest recorded events
(gapyeong 2025-07 by damage ha, gapyeong 2020-08, namwon 2020-08), and for each point renders S2 L2A true colour
pre (least cloudy in the 3 months before) and post (least cloudy within 3 months after) on a 1.92 km window
(192 px at 10 m, UTM), with the point marked. Planetary Computer via plain HTTP (same as n1_fetch).

  .venv-geobench/bin/python -B code/k0_kr_visual_check_v0.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/home/work/data/olmoearth")
K0 = ROOT / "k0"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
HALF = 960


def pick(points):
    by = lambda src, ym: [p for p in points if p["src"] == src and p["date"].startswith(ym) and p["lon"] is not None]
    g25 = sorted(by("gapyeong", "2025-07"), key=lambda p: -(p["ha"] or 0))[:8]
    g20 = sorted(by("gapyeong", "2020-08"), key=lambda p: -(p["ha"] or 0))[:4]
    nw = by("namwon", "2020-08")[::50][:4]
    return g25 + g20 + nw


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import requests
    from pyproj import CRS, Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    tok = requests.get(SAS, timeout=60).json()["token"]
    pts = pick(json.loads((K0 / "k0_points.json").read_text()))

    def best(lon, lat, d0, d1):
        q = {"collections": ["sentinel-2-l2a"], "limit": 100, "intersects": {"type": "Point", "coordinates": [lon, lat]},
             "datetime": f"{d0}/{d1}", "query": {"eo:cloud_cover": {"lt": 60}}}
        f = requests.post(STAC, json=q, timeout=120).json().get("features", [])
        return sorted(f, key=lambda x: x["properties"]["eo:cloud_cover"])[0] if f else None

    fig, ax = plt.subplots(len(pts), 2, figsize=(6, 3 * len(pts)))
    rec = []
    for i, p in enumerate(pts):
        ev = date.fromisoformat(p["date"][:10] if len(p["date"]) >= 10 else p["date"] + "-15")
        zone = int((p["lon"] + 180) // 6) + 1
        dst = CRS.from_epsg(32600 + zone)
        cx, cy = Transformer.from_crs("EPSG:4326", dst, always_xy=True).transform(p["lon"], p["lat"])
        tr = from_origin(cx - HALF, cy + HALF, 10, 10)
        row = {"src": p["src"], "date": p["date"], "addr": p["addr"], "ha": p["ha"]}
        for j, (name, d0, d1) in enumerate((("pre", ev - timedelta(days=90), ev - timedelta(days=3)), ("post", ev + timedelta(days=3), ev + timedelta(days=90)))):
            it = best(p["lon"], p["lat"], d0, d1)
            if it is None:
                ax[i, j].set_title(f"{name}: no scene", fontsize=7)
                ax[i, j].axis("off")
                continue
            bands = []
            for b in ("B04", "B03", "B02"):
                with rasterio.open(it["assets"][b]["href"] + "?" + tok) as src, WarpedVRT(src, crs=dst, transform=tr, width=192, height=192,
                                                                                         resampling=Resampling.bilinear) as v:
                    bands.append(v.read(1).astype(np.float32))
            rgb = np.clip(np.stack(bands, -1) / 2500.0, 0, 1)
            ax[i, j].imshow(rgb)
            ax[i, j].plot(96, 96, "o", mfc="none", mec="yellow", ms=14, mew=1.5)
            ax[i, j].set_title(f"{p['src']} {p['date']} {name} {it['properties']['datetime'][:10]} cc{it['properties']['eo:cloud_cover']:.0f} ha={p['ha']}", fontsize=6)
            ax[i, j].axis("off")
            row[name] = it["properties"]["datetime"][:10]
        rec.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    plt.tight_layout()
    plt.savefig(K0 / "k0_visual_check.png", dpi=90)
    (K0 / "k0_visual_check.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
