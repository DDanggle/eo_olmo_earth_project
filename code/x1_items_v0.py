#!/usr/bin/env python3
"""X1: landslide Q1 items for an untouched Sen12 region (default italy), for external-region evaluation.

Generalises code/e7c_china_items_v0.py (region and source folder as arguments; embeddings not required,
they are extracted afterwards for exactly the tiles written here).

Q1 rules copied from code/sentinel_qa_gen.py lines 26-44 and 60-65 (same thresholds, same gap-matched
negative, same zero-mask negatives, rng seed 0, RGB PNG renders); only the source folder differs:
olmo_streaming_dev/{raw_u16,mask_u8} holds the china tiles, whose streaming embeddings
(olmo_streaming_dev/single_fp16) already exist. china was in no train or test set of MS-131/156/157 or E7.

  python3 -B code/x1_items_v0.py --region italy --src sen12_pilot/holdout_italy --out x1_italy_items_v0
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/work/data/olmoearth")


def kept12(r):
    q = r["scl_clear_fraction"]
    return sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])


def rgb_png(raw, t, path):
    if not path.exists():
        x = np.clip(np.stack([raw[2, t], raw[1, t], raw[0, t]], -1) / 3000.0, 0, 1)
        Image.fromarray((x * 255).astype("uint8")).resize((512, 512), Image.BICUBIC).save(path)
    return str(path)


def main(out_name, REGION, SRC):
    FOLD = f"holdout_{REGION}"
    out = ROOT / out_name
    (out / "img").mkdir(parents=True, exist_ok=False)
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}
    rng = np.random.default_rng(0)
    ids = [s for s in sorted(p.stem for p in (SRC / "mask_u8").glob("*.npy"))
           if s in rec and rec[s]["region"] == REGION and rec[s].get("post_index") is not None
           ]
    pos = [s for s in ids if np.load(SRC / "mask_u8" / f"{s}.npy").sum() >= 200]
    neg = [s for s in ids if np.load(SRC / "mask_u8" / f"{s}.npy").sum() == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    items = []

    def slots(s):
        r = rec[s]
        k = kept12(r)
        return r, k, [j for j, i in enumerate(k) if i < r["post_index"]], [j for j, i in enumerate(k) if i >= r["post_index"]]

    for s in pos:
        r, k, pre, post = slots(s)
        if len(pre) < 2 or len(post) < 2:
            continue
        raw = np.load(SRC / "raw_u16" / f"{s}.npy").astype("float32")
        d = lambda j: str(r["times"][k[j]])[:10]
        dd = lambda j: date.fromisoformat(d(j))
        a, b = pre[-1], post[0]
        pos_gap = (dd(b) - dd(a)).days
        cands = [(x, y) for grp in (pre, post) for x in grp for y in grp if y > x]
        cands = [(x, y) for x, y in cands if abs((dd(y) - dd(x)).days - pos_gap) <= 0.5 * pos_gap]
        if not cands:
            continue
        x, y = min(cands, key=lambda p: abs((dd(p[1]) - dd(p[0])).days - pos_gap))
        img = lambda j: rgb_png(raw, j, out / "img" / f"{s}_{j}.png")
        items.append({"id": f"{s}_q1_pos", "fold": FOLD, "tile": s, "type": "Q1", "dates": [d(a), d(b)], "gap_days": pos_gap,
                      "images": [img(a), img(b)], "answer": "yes"})
        items.append({"id": f"{s}_q1_neg", "fold": FOLD, "tile": s, "type": "Q1", "dates": [d(x), d(y)],
                      "gap_days": (dd(y) - dd(x)).days, "images": [img(x), img(y)], "answer": "no"})
    for s in neg:
        r, k, pre, post = slots(s)
        if not pre or not post:
            continue
        raw = np.load(SRC / "raw_u16" / f"{s}.npy").astype("float32")
        d = lambda j: str(r["times"][k[j]])[:10]
        a, b = pre[-1], post[0]
        items.append({"id": f"{s}_q1_zero", "fold": FOLD, "tile": s, "type": "Q1", "dates": [d(a), d(b)],
                      "gap_days": (date.fromisoformat(d(b)) - date.fromisoformat(d(a))).days,
                      "images": [rgb_png(raw, a, out / "img" / f"{s}_{a}.png"), rgb_png(raw, b, out / "img" / f"{s}_{b}.png")],
                      "answer": "no"})
    (out / "tiles.txt").write_text("\n".join(sorted({x["tile"] for x in items})) + "\n")
    (out / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    summary = {"region": REGION, "n_candidate_tiles": len(ids), "n_pos_tiles_mask200": len(pos), "n_zero_tiles": len(neg),
               "counts": dict(Counter(x["id"].rsplit("_", 1)[1] for x in items)),
               "items_sha256": hashlib.sha256((out / "items.jsonl").read_bytes()).hexdigest(),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="italy")
    ap.add_argument("--src", default="sen12_pilot/holdout_italy")
    ap.add_argument("--out", default="x1_italy_items_v0")
    a = ap.parse_args()
    main(a.out, a.region, ROOT / a.src)
