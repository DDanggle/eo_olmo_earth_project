#!/usr/bin/env python3
"""A1b: the rapid-change scenario (docs/SCENARIO_RAPID_CHANGE_20260927.md) on ALL KuroSiwo flood events (label-free).

Prereg: config/a1b_scenario_prereg_v0.json. Stages:
  select : tiles from every split (method is label-free, no training), flood year >= 2019 so AlphaEarth Y-2..Y exist;
           per event up to 80 flood tiles, all permanent-water-only tiles, dry tiles = max(#flood, 10) (deterministic).
  fetch  : AlphaEarth annual windows for Y-2, Y-1, Y (shares a1_aef/tiles with A1).
  score  : methods (all label-free):
             olmo4_raw  d(post, pre_2) on 40 m tokens (existing cache)      olmo4_norm  d_post / (d_pre + eps)
             olmo2_raw / olmo2_norm on 20 m tokens (a1b/olmo_p2)              olmo_adapt  T2 uses olmo2_norm only for the
             top-K% tiles by olmo4_norm tile score, olmo4_norm (upsampled) elsewhere (K = 30)
             aef_raw 1-cos(Y, Y-1)   aef_norm aef_raw / (1-cos(Y-1, Y-2) + eps)
             sar_raw dB decrease pre_2->post (VV+VH)   sar_norm sar_raw / (|dB change pre_1->pre_2| + eps)
           eps = median of that method's pre-event change over all selected tiles (label-free constant).
           T1 = tile detection AUC (flood vs dry+perm; flood vs perm) at each method's native grid;
           T2 = token localisation AUC at a common 20 m grid (96x96).

  python3 -B code/a1b_scenario_v0.py --selftest
  .venv-geobench/bin/python -B code/a1b_scenario_v0.py select|fetch|score
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import (MIRROR_FROM, MIRROR_TO, WIN, auc, cos_change, dequant,  # noqa: E402
                                           event_bootstrap, pool4)

ROOT = Path("/home/work/data/olmoearth")
KURO = ROOT / "kurosiwo_s1_cache"
AEF_TILES = ROOT / "a1_aef/tiles"
OUT = ROOT / "a1b"
SEED = 20260927
TOPK = 0.30
BASELINES = ("aef_raw", "aef_norm", "sar_raw", "sar_norm")


def pool2(x):
    h, w = x.shape
    return np.nanmean(x.reshape(h // 2, 2, w // 2, 2), axis=(1, 3))


def up2(x):
    return np.repeat(np.repeat(x, 2, 0), 2, 1)


def normalise(d_post, d_pre, eps):
    return d_post / (d_pre + eps)


def adaptive(tile_scores, fine, coarse_up, k=TOPK):
    """Use the fine map only for the top-k fraction of tiles by coarse tile score."""
    ids = sorted(tile_scores, key=lambda i: -tile_scores[i])
    chosen = set(ids[:max(1, int(round(k * len(ids))))])
    return {i: (fine[i] if i in chosen else coarse_up[i]) for i in tile_scores}, chosen


def selftest():
    d = normalise(np.array([1.0, 1.0]), np.array([0.0, 1.0]), 1.0)
    assert np.allclose(d, [1.0, 0.5])
    assert pool2(np.ones((4, 4))).shape == (2, 2) and up2(np.ones((2, 2))).shape == (4, 4)
    maps, chosen = adaptive({"a": 3, "b": 1, "c": 2}, {k: "F" for k in "abc"}, {k: "C" for k in "abc"}, k=0.34)
    assert chosen == {"a"} and maps["a"] == "F" and maps["b"] == "C"
    print("selftest ok: normalisation, 2x pooling/upsampling, adaptive top-k selection")


def select(a):
    OUT.mkdir(exist_ok=True)
    meta = [json.loads(l) for l in (KURO / "meta.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for r in meta:
        y = int(r["flood_date"][:4])
        if y < 2019:
            continue
        m = np.load(KURO / "mask_u8" / f"{r['id']}.npy")
        v = (np.load(KURO / "valid_u8" / f"{r['id']}.npy") == 1) & (m > 0)
        n = max(int(v.sum()), 1)
        flood, perm = float(((m == 3) & v).sum()) / n, float(((m == 2) & v).sum()) / n
        kind = ("flood" if flood >= 0.02 else "perm" if (flood == 0 and perm >= 0.05)
                else "dry" if (flood == 0 and perm == 0 and n / m.size >= 0.9) else None)
        if kind:
            lon, lat = map(float, re.findall(r"[-0-9.]+", r["centroid"]))
            rows.append({"id": r["id"], "event": r["actid"], "split": r["split"], "year": y, "kind": kind, "lon": lon, "lat": lat})
    h = lambda r: hashlib.sha256(f"{SEED}|{r['id']}".encode()).hexdigest()
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[r["event"]].append(r)
    keep = []
    for ev, rs in by_ev.items():
        fl = sorted((r for r in rs if r["kind"] == "flood"), key=h)[:80]
        if not fl:
            continue
        dry = sorted((r for r in rs if r["kind"] == "dry"), key=h)[:max(len(fl), 10)]
        keep += fl + [r for r in rs if r["kind"] == "perm"] + dry
    (OUT / "tiles.json").write_text(json.dumps(keep, indent=0))
    (OUT / "tiles_ids.txt").write_text("\n".join(r["id"] for r in keep) + "\n")
    print(json.dumps({"tiles": len(keep), "kinds": Counter(r["kind"] for r in keep), "events": len({r['event'] for r in keep}),
                      "splits": Counter(r["split"] for r in keep)}))


def fetch(a):
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    AEF_TILES.mkdir(parents=True, exist_ok=True)
    tiles = json.loads((OUT / "tiles.json").read_text())
    years = sorted({t["year"] + d for t in tiles for d in (-2, -1, 0)})
    by_year = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "a1_aef/aef_index.csv")):
        if r.get("year") and int(r["year"]) in years:
            by_year[int(r["year"])].append(r)

    def find(lon, lat, year):
        zone, hemi = int((lon + 180) // 6) + 1, ("N" if lat >= 0 else "S")
        c = [r for r in by_year[year] if float(r["wgs84_west"]) <= lon <= float(r["wgs84_east"]) and float(r["wgs84_south"]) <= lat <= float(r["wgs84_north"])]
        e = [r for r in c if r["utm_zone"] == f"{zone}{hemi}"]
        return (e or c or [None])[0]

    def one(job):
        t, year = job
        p = AEF_TILES / f"{t['id']}_{year}.npy"
        if p.exists():
            return {"id": t["id"], "year": year, "status": "cached"}
        r = find(t["lon"], t["lat"], year)
        if r is None:
            return {"id": t["id"], "year": year, "status": "no_file"}
        try:
            with rasterio.open("/vsicurl/" + r["path"].replace(MIRROR_FROM, MIRROR_TO)) as ds:
                x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(t["lon"], t["lat"])
                col, row = ~ds.transform * (x, y)
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
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(t, t["year"] + d) for t in tiles for d in (-2, -1, 0)]))
    (OUT / "fetch_log.json").write_text(json.dumps(res, indent=0))
    print(json.dumps({"status": Counter(r["status"] for r in res), "seconds": round(time.perf_counter() - t0)}))


def score(a):
    tiles = json.loads((OUT / "tiles.json").read_text())
    P2 = OUT / "olmo_p2"
    per = []
    for t in tiles:
        f = [AEF_TILES / f"{t['id']}_{t['year'] + d}.npy" for d in (-2, -1, 0)]
        p2 = P2 / f"{t['id']}.npz"
        if not all(x.exists() for x in f) or not p2.exists():
            continue
        e2, e1, e0 = (dequant(np.load(x)) for x in f)
        S = np.load(KURO / "single_fp16" / f"{t['id']}.npy").astype(np.float32)
        z = np.load(p2)
        raw = np.load(ROOT / "kurosiwo_npy/raw_f32" / f"{t['id']}.npy")[:, :, 16:208, 16:208].astype(np.float64)
        db = 10 * np.log10(np.clip(raw, 1e-6, None))
        db[raw == 0] = np.nan
        m = np.load(KURO / "mask_u8" / f"{t['id']}.npy")
        v = (np.load(KURO / "valid_u8" / f"{t['id']}.npy") == 1) & (m > 0)
        per.append({**t,
                    "aef_post": cos_change(e0, e1), "aef_pre": cos_change(e1, e2),                 # 192x192 (10 m)
                    "sar_post": np.nanmean(db[:, 1] - db[:, 2], 0), "sar_pre": np.nanmean(np.abs(db[:, 0] - db[:, 1]), 0),
                    "o4_post": cos_change(S[2], S[1]), "o4_pre": cos_change(S[1], S[0]),           # 48x48 (40 m)
                    "o2_post": z["d_post"].astype(np.float32), "o2_pre": z["d_pre"].astype(np.float32),  # 96x96 (20 m)
                    "ff": ((m == 3) & v).astype(np.float32), "vv": v.astype(np.float32)})
    eps = {k: float(np.nanmedian(np.concatenate([p[f"{k}_pre"].ravel() for p in per]))) for k in ("aef", "sar", "o4", "o2")}

    def maps(p):
        """All method maps at the 20 m grid (96x96) plus tile scores at native grids."""
        g = {}
        g["aef_raw"] = pool2(p["aef_post"])
        g["aef_norm"] = pool2(normalise(p["aef_post"], p["aef_pre"], eps["aef"]))
        g["sar_raw"] = pool2(p["sar_post"])
        g["sar_norm"] = pool2(normalise(p["sar_post"], p["sar_pre"], eps["sar"]))
        g["olmo4_raw"] = up2(p["o4_post"])
        g["olmo4_norm"] = up2(normalise(p["o4_post"], p["o4_pre"], eps["o4"]))
        g["olmo2_raw"] = p["o2_post"]
        g["olmo2_norm"] = normalise(p["o2_post"], p["o2_pre"], eps["o2"])
        return g

    ff20 = {p["id"]: pool2(p["ff"]) for p in per}
    vv20 = {p["id"]: pool2(p["vv"]) for p in per}
    G = {p["id"]: maps(p) for p in per}
    tile_score = lambda i, k: (lambda x, vf: float(np.nanmean(x[np.isfinite(x) & (vf > 0.5)])) if (np.isfinite(x) & (vf > 0.5)).any() else None)(G[i][k], vv20[i])
    t_o4 = {p["id"]: tile_score(p["id"], "olmo4_norm") for p in per}
    adapt, chosen = adaptive({i: s for i, s in t_o4.items() if s is not None},
                             {i: G[i]["olmo2_norm"] for i in G}, {i: G[i]["olmo4_norm"] for i in G})
    for i in G:
        G[i]["olmo_adapt"] = adapt.get(i, G[i]["olmo4_norm"])
    METHODS = ["olmo4_raw", "olmo4_norm", "olmo2_raw", "olmo2_norm", "olmo_adapt"] + list(BASELINES)

    def T1(ps, k, negs):
        pos = [tile_score(p["id"], k) for p in ps if p["kind"] == "flood"]
        neg = [tile_score(p["id"], k) for p in ps if p["kind"] in negs]
        return auc([x for x in pos if x is not None], [x for x in neg if x is not None])

    def T2(ps, k):
        pos, neg = [], []
        for p in ps:
            if p["kind"] != "flood":
                continue
            x, ff, vf = G[p["id"]][k], ff20[p["id"]], vv20[p["id"]]
            sel = np.isfinite(x) & (vf > 0.5)
            pos += list(x[sel & (ff >= 0.5)])
            neg += list(x[sel & (ff == 0)])
        return auc(pos, neg)

    by_ev = defaultdict(list)
    for p in per:
        by_ev[p["event"]].append(p)
    flat = lambda gs: [p for g in gs for p in g]
    metrics = {"T1_flood_vs_dry_perm": lambda ps, k: T1(ps, k, ("dry", "perm")), "T1_flood_vs_perm": lambda ps, k: T1(ps, k, ("perm",)),
               "T2_token_20m": T2}
    res = {"n_tiles": len(per), "kinds": dict(Counter(p["kind"] for p in per)), "n_events": len(by_ev),
           "events": {str(e): len(v) for e, v in by_ev.items()}, "eps": eps, "adaptive_fraction_fine": len(chosen) / max(len(per), 1)}
    FULL = {"T1_flood_vs_dry_perm": "olmo4_norm", "T1_flood_vs_perm": "olmo4_norm", "T2_token_20m": "olmo_adapt"}
    for name, fn in metrics.items():
        r = {k: fn(per, k) for k in METHODS}
        best_b = max(BASELINES, key=lambda k: r[k] if r[k] is not None else -1)
        full = FULL[name]
        diff = lambda gs, b=best_b, f=full: (lambda x, y: None if x is None or y is None else x - y)(fn(flat(gs), f), fn(flat(gs), b))
        r.update({"full_method": full, "strongest_baseline": best_b, "full_minus_strongest": (r[full] - r[best_b]) if r[full] is not None and r[best_b] is not None else None,
                  "ci95": event_bootstrap(by_ev, diff),
                  "per_event": {str(e): {k: fn(ps, k) for k in (full, "olmo4_raw", "aef_raw", "sar_raw")} for e, ps in by_ev.items()}})
        res[name] = r
    ok = lambda k: res[k]["full_minus_strongest"] is not None and res[k]["full_minus_strongest"] >= 0.05 and res[k]["ci95"] and res[k]["ci95"][0] > 0
    enough = res["n_events"] >= 8 and res["kinds"].get("flood", 0) >= 300
    res["verdict"] = "invalid" if not enough else ("meaningful" if ok("T1_flood_vs_dry_perm") and ok("T2_token_20m") else
                                                   "t1_only" if ok("T1_flood_vs_dry_perm") else "not_meaningful")
    res["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_event"} if isinstance(v, dict) and "per_event" in v else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["select", "fetch", "score"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    selftest() if a.selftest else {"select": select, "fetch": fetch, "score": score}[a.stage](a)
