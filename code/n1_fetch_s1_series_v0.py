#!/usr/bin/env python3
"""N1 data: multi-year monthly Sentinel-1 RTC series for KuroSiwo tiles (Planetary Computer), same relative orbit per tile.

Prereg: config/n1_learned_normal_prereg_v0.json. For every tile in a1b/tiles.json:
  - query sentinel-1-rtc items within +-0.01 deg of the tile centroid from event-36 months to event+20 days
  - keep the relative orbit with the most acquisitions (geometry changes would otherwise look like 'change')
  - per calendar month before the event keep the first acquisition (history); around the event keep the last acquisition
    <= event-5 d ('pre') and the first acquisition in [event, event+12 d] ('post')
  - read a 1,920 m window (192 x 192 at 10 m) on the tile's UTM grid (WarpedVRT), VV/VH -> dB, zeros -> -30 dB (as the
    KuroSiwo cache), save float16 (2,192,192) chips + a per-tile manifest (dates, orbit, item ids)
Runs in .venv-geobench (rasterio + requests); STAC and SAS token via plain HTTP.

  python3 -B code/n1_fetch_s1_series_v0.py --selftest
  .venv-geobench/bin/python -B code/n1_fetch_s1_series_v0.py --workers 24
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/home/work/data/olmoearth")
OUT = ROOT / "n1_s1"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
HALF = 960
HIST_MONTHS = 36


def plan_dates(items, event):
    """items: [(date_str, orbit, id)]. Returns (orbit, [(role, date, id)]) with one history item per month, pre, post."""
    if not items:
        return None, []
    orbit = Counter(o for _, o, _ in items).most_common(1)[0][0]
    its = sorted((d, i) for d, o, i in items if o == orbit)
    ev = date.fromisoformat(event)
    start = date(ev.year - HIST_MONTHS // 12, ev.month, 1)
    hist, seen = [], set()
    for d, i in its:
        dd = date.fromisoformat(d)
        if start <= dd < ev - timedelta(days=5):
            ym = d[:7]
            if ym not in seen:
                seen.add(ym)
                hist.append(("hist", d, i))
    pre = [(d, i) for d, i in its if date.fromisoformat(d) <= ev - timedelta(days=5)]
    post = [(d, i) for d, i in its if ev <= date.fromisoformat(d) <= ev + timedelta(days=12)]
    out = hist + ([("pre", *pre[-1])] if pre else []) + ([("post", *post[0])] if post else [])
    return orbit, out


def selftest():
    items = [("2020-01-05", 10, "a"), ("2020-01-17", 10, "b"), ("2020-02-10", 10, "c"), ("2020-02-11", 99, "x"),
             ("2020-03-01", 10, "d"), ("2020-03-10", 10, "e")]
    orbit, plan = plan_dates(items, "2020-03-08")
    assert orbit == 10
    roles = [(r, d) for r, d, _ in plan]
    assert ("hist", "2020-01-05") in roles and ("hist", "2020-02-10") in roles and ("pre", "2020-03-01") in roles and ("post", "2020-03-10") in roles
    assert not any(d == "2020-02-11" for _, d in roles), "other orbit excluded"
    print("selftest ok: dominant orbit, one history item per month, pre/post around the event")


def run(a):
    import requests
    import rasterio
    from pyproj import CRS, Transformer
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_origin

    OUT.mkdir(exist_ok=True)
    (OUT / "chips").mkdir(exist_ok=True)
    tiles = json.loads((ROOT / "a1b/tiles.json").read_text())
    meta = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "kurosiwo_s1_cache/meta.jsonl").read_text().splitlines() if l.strip()}
    tok = {"t": None, "at": 0}

    def token():
        if time.time() - tok["at"] > 1800:
            tok["t"], tok["at"] = requests.get(SAS, timeout=60).json()["token"], time.time()
        return tok["t"]

    def search(t, event):
        ev = date.fromisoformat(event)
        q = {"collections": ["sentinel-1-rtc"], "limit": 1000,
             "intersects": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
             "datetime": f"{date(ev.year - HIST_MONTHS // 12, ev.month, 1)}/{ev + timedelta(days=20)}"}
        feats, url, body = [], STAC, q
        for _ in range(10):
            r = requests.post(url, json=body, timeout=120).json()
            feats += r.get("features", [])
            nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
            if not nxt:
                break
            url, body = nxt[0]["href"], nxt[0].get("body", body)
        return {f["id"]: f for f in feats}

    def one(t):
        mf = OUT / "chips" / f"{t['id']}.json"
        if mf.exists():
            return json.loads(mf.read_text())
        event = meta[t["id"]]["flood_date"][:10]
        try:
            feats = search(t, event)
            items = [(f["properties"]["datetime"][:10], f["properties"].get("sat:relative_orbit"), fid) for fid, f in feats.items()]
            orbit, plan = plan_dates(items, event)
            zone = int((t["lon"] + 180) // 6) + 1
            dst = CRS.from_epsg((32600 if t["lat"] >= 0 else 32700) + zone)
            cx, cy = Transformer.from_crs("EPSG:4326", dst, always_xy=True).transform(t["lon"], t["lat"])
            transform = from_origin(cx - HALF, cy + HALF, 10, 10)
            done = []
            for role, d, fid in plan:
                arrs = []
                for band in ("vv", "vh"):
                    href = feats[fid]["assets"][band]["href"] + "?" + token()
                    with rasterio.open(href) as src, WarpedVRT(src, crs=dst, transform=transform, width=192, height=192,
                                                               resampling=Resampling.bilinear) as vrt:
                        arrs.append(vrt.read(1).astype(np.float64))
                x = np.stack(arrs)
                db = 10 * np.log10(np.clip(x, 1e-6, None))
                db[x <= 0] = -30.0
                np.save(OUT / "chips" / f"{t['id']}_{d}.npy", db.astype(np.float16))
                done.append({"role": role, "date": d, "item": fid})
            rec = {"id": t["id"], "event": event, "orbit": orbit, "n": len(done), "chips": done, "status": "ok"}
        except Exception as e:
            rec = {"id": t["id"], "event": event, "status": "error", "err": str(e)[:200]}
        mf.write_text(json.dumps(rec))
        return rec

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, tiles[:a.limit] if a.limit else tiles))
    summ = {"tiles": len(res), "status": Counter(r["status"] for r in res), "chips": sum(r.get("n", 0) for r in res),
            "with_pre_post": sum(1 for r in res if r.get("status") == "ok" and {c["role"] for c in r["chips"]} >= {"pre", "post"}),
            "median_hist": float(np.median([sum(c["role"] == "hist" for c in r["chips"]) for r in res if r.get("status") == "ok"] or [0])),
            "seconds": round(time.perf_counter() - t0)}
    (OUT / f"fetch_summary_{int(time.time())}.json").write_text(json.dumps(summ, default=str))
    print(json.dumps(summ, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
