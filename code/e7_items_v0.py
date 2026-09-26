#!/usr/bin/env python3
"""E7 items: solar-farm presence (Sentinel-2, a second S2 phenomenon) + cross-question items.

Plan: docs/PAPER_EO_VLM_SKELETON_20260926.md (E7). CPU only, never loads a model.
Solar source: task2_cache (OlmoEarth v1 Base, emb (768,32,32) from a 4-timestep S2 window;
mask_u8 label>0 = solar), region folds from task2_contract/loco_folds.json.
  test  = task2_fold0 + task2_fold1 (v1; v0 used fold0 only -> 47 positives), train = task2_fold2..7.
  yes   : solar pixels >= SOLAR_MIN of the chip;  no : zero solar pixels;  in between dropped.
Cross items: the solar question asked on landslide tiles (single S2 date), gold "no".
  per landslide tile: the pos item's first date (pre-event) and second date (post-event),
  so an "anything anomalous -> yes" reader shows higher FPR on post than on pre.

  python3 -B code/e7_items_v0.py --out e7_items_v0
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/work/data/olmoearth")
SOLAR_MIN = 0.01
TEST_REGIONS = {"task2_fold0", "task2_fold1"}  # v1: fold1 added to double test positives (v0 had 47)


def solar_items():
    contract = [json.loads(l) for l in (ROOT / "task2_contract/sample_contract.jsonl").read_text().splitlines() if l.strip()]
    items, fracs = [], []
    for r in contract:
        sid, region = r["sample_id"], r["region"]
        emb = ROOT / "task2_cache/emb_fp16" / f"{sid}.npy"
        if r.get("error") or not emb.exists():
            continue
        m = np.load(ROOT / "task2_cache/mask_u8" / f"{sid}.npy")
        frac = float((m > 0).mean())
        fracs.append(frac)
        if frac >= SOLAR_MIN:
            ans = "yes"
        elif frac == 0:
            ans = "no"
        else:
            continue
        items.append({"id": f"{sid}_solar", "tile": sid, "region": region, "fold": "test" if region in TEST_REGIONS else "train",
                      "phen": "solar", "type": "P", "answer": ans, "kind": "pos" if ans == "yes" else "neg",
                      "solar_frac": round(frac, 4)})
    return items, fracs


def cross_items():
    out = []
    for name, fold in (("sentinel_qa_train_v0", "train"), ("sentinel_qa_v0_1", "test")):
        for l in (ROOT / name / "items.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            x = json.loads(l)
            if x["type"] != "Q1" or x["answer"] != "yes" or not (ROOT / "olmo_streaming_dev/single_fp16" / f"{x['tile']}.npy").exists():
                continue
            for which, d in (("pre", x["dates"][0]), ("post", x["dates"][1])):
                out.append({"id": f"{x['tile']}_cross_{which}", "tile": x["tile"], "fold": fold, "region": x["fold"],
                            "phen": "cross_solar_on_landslide", "type": "P", "dates": [d], "answer": "no",
                            "kind": f"cross_{which}"})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="e7_items_v0")
    a = ap.parse_args()
    solar, fracs = solar_items()
    cross = cross_items()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in solar + cross))
    q = lambda p: round(float(np.quantile(fracs, p)), 4)
    summary = {"solar_frac_quantiles_p10_p25_p50_p75": [q(.1), q(.25), q(.5), q(.75)],
               "solar_zero_fraction_chips": round(sum(f == 0 for f in fracs) / len(fracs), 4),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "SOLAR_MIN": SOLAR_MIN}
    summary["counts"] = {f"{p}|{f}|{k}": c for (p, f, k), c in sorted(Counter((i["phen"], i["fold"], i["kind"]) for i in solar + cross).items())}
    summary["items_sha256"] = hashlib.sha256((out / "items.jsonl").read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
