#!/usr/bin/env python3
"""E1 (flood): audit the KuroSiwo S1 OlmoEarth cache and build landslide-style Q1 pairs.

Direction: docs/DIRECTION_EO_EMBEDDING_READER_2026_09_25.md. CPU only, reads meta/masks, never a model.
Cache contract (code/extract_kurosiwo_s1_cache.py): single_fp16/<id>.npy is (3,768,48,48) for
pre_1 (-24 d), pre_2 (-12 d), post (= flood_date), OlmoEarth v1 Base, S1 vv/vh; mask_u8 (192,192)
0 no-data, 1 no water, 2 permanent water, 3 flood; valid_u8 1 = valid.

Items mirror sentinel_qa Q1 so the EarthTalk reader/controls apply unchanged:
  pos  (pre_2, post) on a flood tile             -> "yes"
  neg  (pre_1, pre_2) on the same flood tile     -> "no"   (within-tile pair for the swap control)
  hard (pre_2, post) on a tile with no flood     -> "no"   (blocks a "second image is the post one" shortcut)
Flood tile  : flood pixels >= FLOOD_MIN of valid pixels (from mask_u8, not the meta pflood field alone).
No-flood tile: zero flood pixels and valid fraction >= VALID_MIN.

  python3 -B code/e1_flood_qa_v0.py audit            # counts, split/event overlap, thresholds
  python3 -B code/e1_flood_qa_v0.py build --out flood_qa_v0
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/home/work/data/olmoearth")
CACHE = ROOT / "kurosiwo_s1_cache"
FLOOD_MIN = 0.02
VALID_MIN = 0.90
SEED = "e1-flood-v0-20260925"


def tile_stats(sid):
    mask = np.load(CACHE / "mask_u8" / f"{sid}.npy")
    valid = np.load(CACHE / "valid_u8" / f"{sid}.npy") == 1
    labelled = valid & (mask > 0)
    n = int(labelled.sum())
    flood = int(((mask == 3) & labelled).sum())
    return {"valid_frac": n / mask.size, "flood_frac": flood / max(n, 1), "flood_px": flood}


def load_meta():
    rows = [json.loads(l) for l in (CACHE / "meta.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if (CACHE / "single_fp16" / f"{r['id']}.npy").exists()]


def dates_for(r):
    post = date.fromisoformat(r["flood_date"])
    return {"pre_1": str(post - timedelta(days=24)), "pre_2": str(post - timedelta(days=12)), "post": str(post)}


def audit(rows):
    stats = {r["id"]: tile_stats(r["id"]) for r in rows}
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    events = {s: {x["actid"] for x in v} for s, v in by_split.items()}
    overlap = {f"{a}&{b}": len(events[a] & events[b]) for a in events for b in events if a < b}
    q = lambda xs: [round(float(np.quantile(xs, p)), 4) for p in (0.5, 0.75, 0.9, 0.99)] if xs else None
    out = {"n_tiles": len(rows), "per_split": {}, "event_overlap_between_splits": overlap,
           "thresholds": {"FLOOD_MIN": FLOOD_MIN, "VALID_MIN": VALID_MIN}}
    for s, v in sorted(by_split.items()):
        ff = [stats[x["id"]]["flood_frac"] for x in v]
        flood_tiles = [x for x in v if stats[x["id"]]["flood_frac"] >= FLOOD_MIN]
        dry_tiles = [x for x in v if stats[x["id"]]["flood_px"] == 0 and stats[x["id"]]["valid_frac"] >= VALID_MIN]
        out["per_split"][s] = {"tiles": len(v), "events": len(events[s]),
                               "flood_frac_quantiles_p50_p75_p90_p99": q(ff),
                               "flood_tiles": len(flood_tiles), "dry_tiles": len(dry_tiles),
                               "flood_events": len({x["actid"] for x in flood_tiles}),
                               "meta_pflood_agrees_nonzero": sum((x["pflood"] > 0) == (stats[x["id"]]["flood_px"] > 0) for x in v) / len(v)}
    return out, stats


def build(rows, stats, out_name, hard_per_pos=1):
    items = []
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    for split, v in sorted(by_split.items()):
        flood = [r for r in v if stats[r["id"]]["flood_frac"] >= FLOOD_MIN]
        dry = sorted((r for r in v if stats[r["id"]]["flood_px"] == 0 and stats[r["id"]]["valid_frac"] >= VALID_MIN),
                     key=lambda r: hashlib.sha256(f"{SEED}|{r['id']}".encode()).hexdigest())
        for r in flood:
            d = dates_for(r)
            base = {"fold": split, "tile": r["id"], "event": r["actid"], "type": "Q1", "sensor": "sentinel1",
                    "flood_frac": round(stats[r["id"]]["flood_frac"], 4)}
            items.append({**base, "id": f"{r['id']}_q1_pos", "dates": [d["pre_2"], d["post"]], "slots": ["pre_2", "post"], "answer": "yes", "kind": "pos"})
            items.append({**base, "id": f"{r['id']}_q1_neg", "dates": [d["pre_1"], d["pre_2"]], "slots": ["pre_1", "pre_2"], "answer": "no", "kind": "neg"})
        for r in dry[:len(flood) * hard_per_pos]:
            d = dates_for(r)
            items.append({"fold": split, "tile": r["id"], "event": r["actid"], "type": "Q1", "sensor": "sentinel1",
                          "flood_frac": 0.0, "id": f"{r['id']}_q1_hard", "dates": [d["pre_2"], d["post"]],
                          "slots": ["pre_2", "post"], "answer": "no", "kind": "hard_neg"})
    out = ROOT / out_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    summary = {"n_items": len(items), "by_split_kind": {f"{s}|{k}": c for (s, k), c in
               sorted(Counter((i["fold"], i["kind"]) for i in items).items())},
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "thresholds": {"FLOOD_MIN": FLOOD_MIN, "VALID_MIN": VALID_MIN}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "build"])
    ap.add_argument("--out", default="flood_qa_v0")
    a = ap.parse_args()
    rows = load_meta()
    report, stats = audit(rows)
    if a.cmd == "audit":
        print(json.dumps(report, indent=1))
    else:
        print(json.dumps({"audit": report, "build": build(rows, stats, a.out)}, indent=1))
