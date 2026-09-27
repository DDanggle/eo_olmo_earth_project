#!/usr/bin/env python3
"""A2: the rapid-change scenario on landslides (Sen12Landslides, Sentinel-2), label-free.

Prereg: config/a2_landslide_prereg_v0.json. Same tool as A1b, second hazard:
  olmo_raw  1-cos(post, pre) on 40 m tokens (existing per-date OlmoEarth v1 Base caches)
  olmo_norm olmo_raw / (1-cos(pre, pre1) + eps)                         <- full method (no 20 m zoom for S2 in v0)
  dndvi_raw NDVI_pre - NDVI_post (vegetation loss, classic landslide index), 10 m pooled to 40 m
  dndvi_norm dndvi_raw / (|NDVI_pre - NDVI_pre1| + eps)
  aef_raw   1-cos(AEF_Y, AEF_Y-1); aef_norm aef_raw / (1-cos(AEF_Y-1, AEF_Y-2)+eps) (events >= 2019 only)
Dates: per region, event date = modal first post-event acquisition of positive tiles; per tile pre = last kept (12 clearest of 15)
acquisition before the event, post = first kept at/after it, pre1 = kept acquisition before pre.
Tiles: positive = landslide mask >= 200 px, negative = empty mask; <= 100 each per region (deterministic hash).

  python3 -B code/a2_landslide_v0.py --selftest
  .venv-geobench/bin/python -B code/a2_landslide_v0.py select|fetch|score
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import MIRROR_FROM, MIRROR_TO, auc, cos_change, dequant, event_bootstrap, pool4  # noqa: E402
from a1b_scenario_v0 import normalise  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
OUT = ROOT / "a2"
AEF_TILES = ROOT / "a1_aef/tiles"
SEED = 20260927
REGIONS = ("hiroshima", "hokkaido", "thrissur", "itogon", "chimanimani", "italy", "kyrgyzstan2", "newzealand")
SRC = {"italy": (ROOT / "x1_italy_stream/emb_time_fp16", ROOT / "sen12_pilot/holdout_italy")}
DEFAULT_SRC = (ROOT / "olmo_streaming_dev/single_fp16", ROOT / "olmo_streaming_dev")
WIN = 128
B04, B08 = 2, 6          # raw_u16 band order B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 (checked in the .nc files)
PRIMARY_BASELINES = ("aef_raw", "dndvi_raw", "dndvi_norm")


def kept12(r):
    q = r["scl_clear_fraction"]
    return sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])


def choose_dates(times_kept, event):
    """Indices into the kept list: (pre1, pre, post) around the event date string; None if unavailable."""
    before = [i for i, t in enumerate(times_kept) if t < event]
    after = [i for i, t in enumerate(times_kept) if t >= event]
    if len(before) < 2 or not after:
        return None
    return before[-2], before[-1], after[0]


def ndvi(raw, t):
    r, n = raw[B04, t].astype(np.float32), raw[B08, t].astype(np.float32)
    return (n - r) / np.where(n + r > 0, n + r, np.nan)


def selftest():
    assert choose_dates(["2018-01", "2018-03", "2018-05", "2018-09"], "2018-06") == (1, 2, 3)
    assert choose_dates(["2018-05", "2018-09"], "2018-06") is None
    raw = np.zeros((10, 2, 2, 2), np.uint16)
    raw[B04, 0], raw[B08, 0] = 100, 300
    assert np.allclose(ndvi(raw, 0), 0.5)
    print("selftest ok: date selection around the event, NDVI band indices")


def select(a):
    OUT.mkdir(exist_ok=True)
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}
    h = lambda s: hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest()
    tiles = []
    for reg in REGIONS:
        emb_dir, src = SRC.get(reg, DEFAULT_SRC)
        ids = [s for s, r in rec.items() if r["region"] == reg and (emb_dir / f"{s}.npy").exists() and (src / "mask_u8" / f"{s}.npy").exists()]
        post_dates = [str(rec[s]["times"][rec[s]["post_index"]])[:10] for s in ids if rec[s].get("post_index") is not None]
        if not post_dates:
            continue
        event = Counter(d[:7] for d in post_dates).most_common(1)[0][0]           # modal event month
        event_day = min(d for d in post_dates if d.startswith(event))               # earliest post acquisition that month
        pos, neg = [], []
        for s in ids:
            px = int(np.load(src / "mask_u8" / f"{s}.npy").sum())
            r = rec[s]
            if px >= 200 and (r.get("post_index") is None or not str(r["times"][r["post_index"]]).startswith(event)):
                continue    # prereg: positives need a first post-event acquisition in the modal event month (fixed 2026-09-27 before any fetch/score)
            kept_times = [str(r["times"][i])[:10] for i in kept12(r)]
            d = choose_dates(kept_times, event_day)
            if d is None:
                continue
            row = {"id": s, "region": reg, "event": reg, "event_day": event_day, "year": int(event_day[:4]), "crs": r["crs"],
                   "path": r["path"], "idx": d, "dates": [kept_times[i] for i in d]}
            (pos if px >= 200 else neg if px == 0 else []).append(row)
        pos, neg = sorted(pos, key=lambda x: h(x["id"]))[:100], sorted(neg, key=lambda x: h(x["id"]))[:100]
        tiles += [dict(x, kind="pos") for x in pos] + [dict(x, kind="neg") for x in neg]
    (OUT / "tiles.json").write_text(json.dumps(tiles, indent=0))
    print(json.dumps({"tiles": len(tiles), "per_region": {r: dict(Counter(t["kind"] for t in tiles if t["region"] == r)) for r in REGIONS},
                      "event_days": {t["region"]: t["event_day"] for t in tiles}}, indent=1))


def fetch(a):
    import h5py
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    tiles = json.loads((OUT / "tiles.json").read_text())
    years = sorted({t["year"] + d for t in tiles for d in (-2, -1, 0) if t["year"] + d >= 2017})
    by_year = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")):
        if r.get("year") and int(r["year"]) in years:
            by_year[int(r["year"])].append(r)

    def one(job):
        t, year = job
        p = AEF_TILES / f"{t['id']}_{year}.npy"
        if p.exists():
            return {"id": t["id"], "year": year, "status": "cached"}
        try:
            with h5py.File(t["path"], "r") as f:
                x, y = f["x"][:], f["y"][:]
            cx, cy = float(x.mean()), float(y.mean())
            lon, lat = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
            zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
            c = [r for r in by_year[year] if float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"]) and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
            e = [r for r in c if r["utm_zone"] == f"{zone}{hemi}"]
            r = (e or c or [None])[0]
            if r is None:
                return {"id": t["id"], "year": year, "status": "no_file"}
            with rasterio.open("/vsicurl/" + r["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
                ax, ay = Transformer.from_crs(t["crs"], ds.crs, always_xy=True).transform(cx, cy)
                col, row = ~ds.transform * (ax, ay)
                c0, r0 = int(round(col)) - WIN // 2, int(round(row)) - WIN // 2
                if c0 < 0 or r0 < 0 or c0 + WIN > ds.width or r0 + WIN > ds.height:
                    return {"id": t["id"], "year": year, "status": "edge"}
                arr = ds.read(window=Window(c0, r0, WIN, WIN))
                if ds.transform.e > 0:
                    arr = arr[:, ::-1, :]
            np.save(p, np.ascontiguousarray(arr))
            return {"id": t["id"], "year": year, "status": "ok"}
        except Exception as e:
            return {"id": t["id"], "year": year, "status": "error", "err": str(e)[:120]}

    t0 = time.perf_counter()
    jobs = [(t, t["year"] + d) for t in tiles for d in (-2, -1, 0) if t["year"] + d >= 2017]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    (OUT / "fetch_log.json").write_text(json.dumps(res, indent=0))
    print(json.dumps({"status": Counter(r["status"] for r in res), "seconds": round(time.perf_counter() - t0)}))


def score(a):
    tiles = json.loads((OUT / "tiles.json").read_text())
    per = []
    for t in tiles:
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        fY, fB, fB2 = (AEF_TILES / f"{t['id']}_{t['year'] + d}.npy" for d in (0, -1, -2))
        if not (fY.exists() and fB.exists()):
            continue
        p1, p0, q = t["idx"]
        S = np.load(emb_dir / f"{t['id']}.npy").astype(np.float32)
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy")
        m = np.load(src / "mask_u8" / f"{t['id']}.npy") > 0
        eY, eB = dequant(np.load(fY)), dequant(np.load(fB))
        aef_post = cos_change(eY, eB)
        aef_pre = cos_change(eB, dequant(np.load(fB2))) if fB2.exists() else None
        n1, n0, nq = ndvi(raw, p1), ndvi(raw, p0), ndvi(raw, q)
        per.append({**t, "o_post": cos_change(S[q], S[p0]), "o_pre": cos_change(S[p0], S[p1]),
                    "aef_post": aef_post, "aef_pre": aef_pre, "nd_post": n0 - nq, "nd_pre": np.abs(n0 - n1), "mf": pool4(m.astype(np.float32))})
    eps = {"o": float(np.nanmedian(np.concatenate([p["o_pre"].ravel() for p in per]))),
           "nd": float(np.nanmedian(np.concatenate([p["nd_pre"].ravel() for p in per]))),
           "aef": float(np.nanmedian(np.concatenate([p["aef_pre"].ravel() for p in per if p["aef_pre"] is not None])))}
    G = {}
    for p in per:
        G[p["id"]] = {"olmo_raw": p["o_post"], "olmo_norm": normalise(p["o_post"], p["o_pre"], eps["o"]),
                      "dndvi_raw": pool4(p["nd_post"]), "dndvi_norm": pool4(normalise(p["nd_post"], p["nd_pre"], eps["nd"])),
                      "aef_raw": pool4(p["aef_post"]),
                      "aef_norm": pool4(normalise(p["aef_post"], p["aef_pre"], eps["aef"])) if p["aef_pre"] is not None else None}
    tmean = lambda i, k: (lambda x: None if x is None or not np.isfinite(x).any() else float(np.nanmean(x)))(G[i][k])

    def T1(ps, k):
        pos = [tmean(p["id"], k) for p in ps if p["kind"] == "pos"]
        neg = [tmean(p["id"], k) for p in ps if p["kind"] == "neg"]
        return auc([x for x in pos if x is not None], [x for x in neg if x is not None])

    def T2(ps, k):
        pos, neg = [], []
        for p in ps:
            x = G[p["id"]][k]
            if p["kind"] != "pos" or x is None:
                continue
            sel = np.isfinite(x)
            pos += list(x[sel & (p["mf"] >= 0.5)])
            neg += list(x[sel & (p["mf"] == 0)])
        return auc(pos, neg)

    def analyse(ps, baselines):
        by_ev = defaultdict(list)
        for p in ps:
            by_ev[p["event"]].append(p)
        flat = lambda gs: [p for g in gs for p in g]
        out = {"n_tiles": len(ps), "kinds": dict(Counter(p["kind"] for p in ps)), "events": {e: len(v) for e, v in by_ev.items()}}
        for name, fn in (("T1", T1), ("T2", T2)):
            r = {k: fn(ps, k) for k in ("olmo_raw", "olmo_norm") + tuple(baselines)}
            best = max(baselines, key=lambda k: r[k] if r[k] is not None else -1)
            diff = lambda gs, b=best: (lambda x, y: None if x is None or y is None else x - y)(fn(flat(gs), "olmo_norm"), fn(flat(gs), b))
            r.update({"strongest_baseline": best, "full_minus_strongest": (r["olmo_norm"] - r[best]) if r["olmo_norm"] is not None and r[best] is not None else None,
                      "ci95": event_bootstrap(by_ev, diff),
                      "per_event": {e: {k: fn(v, k) for k in ("olmo_norm", "olmo_raw") + tuple(baselines)} for e, v in by_ev.items()}})
            out[name] = r
        ok = lambda k: out[k]["full_minus_strongest"] is not None and out[k]["full_minus_strongest"] >= 0.05 and out[k]["ci95"] and out[k]["ci95"][0] > 0
        out["verdict"] = ("invalid" if len(by_ev) < 4 or out["kinds"].get("pos", 0) < 150 else
                          "meaningful" if ok("T1") and ok("T2") else "t1_only" if ok("T1") else "t2_only" if ok("T2") else "not_meaningful")
        return out

    res = {"eps": eps, "primary_all_events_2018plus": analyse(per, PRIMARY_BASELINES),
           "secondary_2019plus_with_aef_norm": analyse([p for p in per if p["aef_pre"] is not None], PRIMARY_BASELINES + ("aef_norm",)),
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))
    strip = lambda d: {k: ({kk: vv for kk, vv in v.items() if kk != "per_event"} if isinstance(v, dict) and "per_event" in v else v) for k, v in d.items()}
    print(json.dumps({k: (strip(v) if isinstance(v, dict) and "T1" in v else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["select", "fetch", "score"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    selftest() if a.selftest else {"select": select, "fetch": fetch, "score": score}[a.stage](a)
