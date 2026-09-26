#!/usr/bin/env python3
"""A1: AlphaEarth annual embeddings vs per-observation OlmoEarth embeddings for label-free flood detection.

Prereg: config/a1_aef_vs_olmoearth_floods_prereg_v0.json.
  fetch : read AlphaEarth V1 annual windows (192x192 px, 10 m) for KuroSiwo test tiles, years Y and Y-1 (and Y+1),
          from the anonymous Source Cooperative mirror (CC-BY 4.0, produced by Google and Google DeepMind).
  score : label-free change scores (1 - cos) for AEF (year pair) and OlmoEarth (post vs pre_2, S1), AUCs for
          tile detection (T1) and within-tile localisation (T2), event-clustered bootstrap of the differences.

  python3 -B code/a1_aef_vs_olmoearth_floods_v0.py --selftest
  .venv-geobench/bin/python -B code/a1_aef_vs_olmoearth_floods_v0.py fetch
  .venv-geobench/bin/python -B code/a1_aef_vs_olmoearth_floods_v0.py score
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/work/data/olmoearth")
KURO = ROOT / "kurosiwo_s1_cache"
OUT = ROOT / "a1_aef"
MIRROR_FROM, MIRROR_TO = "s3://us-west-2.opendata.source.coop/", "https://data.source.coop/"
WIN = 192
SEED = 20260927


# ---------------------------------------------------------------- pure logic

def auc(pos, neg):
    """Mann-Whitney AUC with average ranks for ties."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dequant(v):
    v = v.astype(np.float32)
    x = np.sign(v) * (v / 127.5) ** 2
    x[v == -128] = np.nan
    return x


def cos_change(a, b):
    """a, b: (C, H, W) float; returns 1 - cosine per pixel (NaN where either is missing)."""
    num = np.nansum(a * b, 0)
    den = np.sqrt(np.nansum(a * a, 0) * np.nansum(b * b, 0))
    out = 1 - num / np.where(den > 0, den, np.nan)
    out[np.isnan(a).any(0) | np.isnan(b).any(0)] = np.nan
    return out


def pool4(x):
    h, w = x.shape
    return np.nanmean(x.reshape(h // 4, 4, w // 4, 4), axis=(1, 3))


def event_bootstrap(per_event, fn, n=2000, seed=SEED):
    """per_event: {event: data}; fn(list of data) -> value. Percentile CI over resampled events."""
    evs = sorted(per_event)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = fn([per_event[rng.choice(evs)] for _ in evs])
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]] if vals else None


def selftest():
    assert abs(auc([2, 3, 4], [0, 1]) - 1.0) < 1e-9 and abs(auc([1], [1]) - 0.5) < 1e-9
    v = np.array([127, -127, 0, -128], dtype=np.int8)
    d = dequant(v)
    assert abs(d[0] - (127 / 127.5) ** 2) < 1e-6 and d[1] < 0 and d[2] == 0 and np.isnan(d[3])
    a = np.random.default_rng(0).normal(size=(64, 8, 8)).astype(np.float32)
    assert np.allclose(cos_change(a, a), 0, atol=1e-5) and np.allclose(cos_change(a, -a), 2, atol=1e-5)
    assert pool4(np.ones((8, 8))).shape == (2, 2)
    ci = event_bootstrap({"e1": 1.0, "e2": 2.0, "e3": 3.0}, lambda xs: float(np.mean(xs)))
    assert 1.0 <= ci[0] <= ci[1] <= 3.0
    print("selftest ok: AUC (ties), AEF dequantisation, cosine change, pooling, event bootstrap")


# ---------------------------------------------------------------- tile selection

def tile_table():
    meta = [json.loads(l) for l in (KURO / "meta.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for r in meta:
        y = int(r["flood_date"][:4])
        if r["split"] != "test" or y < 2018:
            continue
        m = np.load(KURO / "mask_u8" / f"{r['id']}.npy")
        v = (np.load(KURO / "valid_u8" / f"{r['id']}.npy") == 1) & (m > 0)
        n = max(int(v.sum()), 1)
        flood, perm = float(((m == 3) & v).sum()) / n, float(((m == 2) & v).sum()) / n
        kind = ("flood" if flood >= 0.02 else "perm" if (flood == 0 and perm >= 0.05)
                else "dry" if (flood == 0 and perm == 0 and n / m.size >= 0.9) else None)
        if kind:
            lon, lat = map(float, re.findall(r"[-0-9.]+", r["centroid"]))
            rows.append({"id": r["id"], "event": r["actid"], "year": y, "kind": kind, "lon": lon, "lat": lat, "flood_frac": flood})
    # cap dry tiles per event at the number of flood tiles (download budget), deterministic
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[r["event"]].append(r)
    keep = []
    for ev, rs in by_ev.items():
        nflood = sum(r["kind"] == "flood" for r in rs)
        dry = sorted((r for r in rs if r["kind"] == "dry"), key=lambda r: hashlib.sha256(f"{SEED}|{r['id']}".encode()).hexdigest())
        keep += [r for r in rs if r["kind"] != "dry"] + dry[:max(nflood, 10)]
    return keep


# ---------------------------------------------------------------- fetch

def fetch(a):
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    OUT.mkdir(exist_ok=True)
    (OUT / "tiles").mkdir(exist_ok=True)
    idx = [r for r in csv.DictReader(open(OUT / "aef_index.csv")) if r.get("year")]
    tiles = tile_table()
    need_years = sorted({t["year"] + d for t in tiles for d in (-1, 0, 1)})
    by_year = defaultdict(list)
    for r in idx:
        if int(r["year"]) in need_years:
            by_year[int(r["year"])].append(r)

    def find(lon, lat, year):
        zone = int((lon + 180) // 6) + 1
        hemi = "N" if lat >= 0 else "S"
        cands = [r for r in by_year[year] if float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"])
                 and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
        exact = [r for r in cands if r["utm_zone"] == f"{zone}{hemi}"]
        return (exact or cands or [None])[0]

    def one(job):
        t, year = job
        path = OUT / "tiles" / f"{t['id']}_{year}.npy"
        if path.exists():
            return {"id": t["id"], "year": year, "status": "cached"}
        r = find(t["lon"], t["lat"], year)
        if r is None:
            return {"id": t["id"], "year": year, "status": "no_file"}
        url = "/vsicurl/" + r["path"].replace(MIRROR_FROM, MIRROR_TO)
        try:
            with rasterio.open(url) as ds:
                x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(t["lon"], t["lat"])
                col, row = ~ds.transform * (x, y)
                c0, r0 = int(round(col)) - WIN // 2, int(round(row)) - WIN // 2
                if c0 < 0 or r0 < 0 or c0 + WIN > ds.width or r0 + WIN > ds.height:
                    return {"id": t["id"], "year": year, "status": "edge"}
                arr = ds.read(window=Window(c0, r0, WIN, WIN))
                if ds.transform.e > 0:          # bottom-up file: flip rows to north-up like KuroSiwo tiles
                    arr = arr[:, ::-1, :]
            np.save(path, np.ascontiguousarray(arr))
            return {"id": t["id"], "year": year, "status": "ok", "masked_frac": float((arr == -128).all(0).mean()), "file": r["path"]}
        except Exception as e:
            return {"id": t["id"], "year": year, "status": "error", "err": str(e)[:160]}

    jobs = [(t, t["year"] + d) for t in tiles for d in (-1, 0, 1)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    (OUT / "tiles.json").write_text(json.dumps(tiles, indent=0))
    (OUT / "fetch_log.json").write_text(json.dumps(res, indent=0))
    from collections import Counter
    print(json.dumps({"tiles": len(tiles), "kinds": Counter(t["kind"] for t in tiles), "events": Counter(t["event"] for t in tiles),
                      "status": Counter(r["status"] for r in res), "seconds": round(time.perf_counter() - t0)}, default=str))


# ---------------------------------------------------------------- score

def score(a):
    tiles = json.loads((OUT / "tiles.json").read_text())
    log = {(r["id"], r["year"]): r for r in json.loads((OUT / "fetch_log.json").read_text())}
    per = []
    for t in tiles:
        yp = OUT / "tiles" / f"{t['id']}_{t['year']}.npy"
        yb = OUT / "tiles" / f"{t['id']}_{t['year'] - 1}.npy"
        if not (yp.exists() and yb.exists()):
            continue
        aY, aB = dequant(np.load(yp)), dequant(np.load(yb))
        aef = pool4(cos_change(aY, aB))
        ya = OUT / "tiles" / f"{t['id']}_{t['year'] + 1}.npy"
        aef_next = pool4(cos_change(dequant(np.load(ya)), aB)) if ya.exists() else None
        S = np.load(KURO / "single_fp16" / f"{t['id']}.npy").astype(np.float32)
        olmo = cos_change(S[2], S[1])
        # classic label-free SAR flood change (amendment 2026-09-27): backscatter decrease post vs pre_2, dB, VV+VH mean
        raw = np.load(ROOT / "kurosiwo_npy/raw_f32" / f"{t['id']}.npy")[:, :, 16:208, 16:208].astype(np.float64)
        db = 10 * np.log10(np.clip(raw, 1e-6, None))
        db[raw == 0] = np.nan
        sar = pool4(np.nanmean(db[:, 1] - db[:, 2], 0).astype(np.float32))
        m = np.load(KURO / "mask_u8" / f"{t['id']}.npy")
        v = (np.load(KURO / "valid_u8" / f"{t['id']}.npy") == 1) & (m > 0)
        flood_frac = pool4(((m == 3) & v).astype(np.float32))
        valid_frac = pool4(v.astype(np.float32))
        per.append({**t, "aef": aef, "aef_next": aef_next, "olmo": olmo, "sar": sar, "ff": flood_frac, "vf": valid_frac})
    ok = lambda x: np.isfinite(x)

    def tile_mean(p, key):
        x = p[key]
        if x is None:
            return None
        sel = ok(x) & (p["vf"] > 0.5)
        return float(np.nanmean(x[sel])) if sel.any() else None

    def t1(ps, key, neg_kinds):
        pos = [tile_mean(p, key) for p in ps if p["kind"] == "flood"]
        neg = [tile_mean(p, key) for p in ps if p["kind"] in neg_kinds]
        pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
        return auc(pos, neg)

    def t2(ps, key):
        pos, neg = [], []
        for p in ps:
            if p["kind"] != "flood" or p[key] is None:
                continue
            x, ff, vf = p[key], p["ff"], p["vf"]
            sel = ok(x) & (vf > 0.5)
            pos += list(x[sel & (ff >= 0.5)])
            neg += list(x[sel & (ff == 0)])
        return auc(pos, neg)

    by_ev = defaultdict(list)
    for p in per:
        by_ev[p["event"]].append(p)
    flat = lambda groups: [p for g in groups for p in g]
    res = {"n_tiles": len(per), "kinds": {k: sum(p["kind"] == k for p in per) for k in ("flood", "perm", "dry")},
           "events": {str(e): len(v) for e, v in by_ev.items()},
           "masked_frac_mean": float(np.mean([r.get("masked_frac", 0) for r in log.values() if r["status"] == "ok"] or [0]))}
    for name, fn in (("T1_flood_vs_dry_perm", lambda ps, k: t1(ps, k, ("dry", "perm"))),
                     ("T1_flood_vs_perm", lambda ps, k: t1(ps, k, ("perm",))),
                     ("T2_token_localisation", t2)):
        r = {k: fn(per, k) for k in ("olmo", "aef", "aef_next", "sar")}
        diff_fn = lambda groups: (lambda o, a_: None if o is None or a_ is None else o - a_)(fn(flat(groups), "olmo"), fn(flat(groups), "aef"))
        r["olmo_minus_aef"] = r["olmo"] - r["aef"] if r["olmo"] is not None and r["aef"] is not None else None
        r["ci95"] = event_bootstrap(by_ev, diff_fn)
        r["per_event"] = {str(e): {k: fn(ps, k) for k in ("olmo", "aef", "sar")} for e, ps in by_ev.items()}
        sar_fn = lambda groups: (lambda o, s_: None if o is None or s_ is None else o - s_)(fn(flat(groups), "olmo"), fn(flat(groups), "sar"))
        r["olmo_minus_sar"] = r["olmo"] - r["sar"] if r["olmo"] is not None and r["sar"] is not None else None
        r["ci95_vs_sar"] = event_bootstrap(by_ev, sar_fn)
        res[name] = r
    enough = len(by_ev) >= 3 and res["kinds"]["flood"] >= 100 and res["masked_frac_mean"] <= 0.10
    win = lambda k: res[k]["olmo_minus_aef"] is not None and res[k]["olmo_minus_aef"] >= 0.05 and res[k]["ci95"] and res[k]["ci95"][0] > 0
    lose = lambda k: res[k]["olmo_minus_aef"] is not None and res[k]["olmo_minus_aef"] <= -0.05 and res[k]["ci95"] and res[k]["ci95"][1] < 0
    res["verdict"] = ("invalid" if not enough else
                      "olmoearth_beats_aef" if win("T1_flood_vs_dry_perm") and win("T2_token_localisation") else
                      "aef_beats_olmoearth" if lose("T1_flood_vs_dry_perm") and lose("T2_token_localisation") else "tie_or_mixed")
    # amendment: meaningful only if OlmoEarth beats the STRONGEST of {AEF, SAR log-ratio} by >= .05 on T1 and T2 (CI > 0)
    beats = lambda k: all(res[k][d] is not None and res[k][d] >= 0.05 and res[k][c] and res[k][c][0] > 0
                          for d, c in (("olmo_minus_aef", "ci95"), ("olmo_minus_sar", "ci95_vs_sar")))
    res["meaningful_vs_strongest"] = res["verdict"] != "invalid" and beats("T1_flood_vs_dry_perm") and beats("T2_token_localisation")
    res["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or "per_event" not in v else {kk: vv for kk, vv in v.items() if kk != "per_event"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["fetch", "score"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.stage == "fetch":
        fetch(a)
    else:
        score(a)
