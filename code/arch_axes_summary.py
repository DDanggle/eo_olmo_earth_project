"""architecture-axes(깊이·규모·목적) 진단 집계.

addendum_v1b. bv1_summary.py와 동일 지표·동일 기준(sealed pilot OlmoEarth P4, raw P2).
완료된 캐시만 집계하고, 미완 캐시는 n<8로 표시해 부분 결과임을 드러낸다.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
ref = {x["fold"]: x["primary_mean"]
       for x in json.load(open(ROOT / "artifacts/confirmatory_8region_summary.json"))["regions"]}
FOLDS = ["holdout_hiroshima", "holdout_hokkaido", "holdout_indonesia", "holdout_itogon",
         "holdout_kyrgyzstan1", "holdout_kyrgyzstan2", "holdout_newzealand", "holdout_thrissur"]
# 축: (캐시, family, 축 이름, 축 값)
AXES = [
    ("olmo_nano",         "OlmoEarth", "scale",  "nano"),
    ("olmo_tiny",         "OlmoEarth", "scale",  "tiny"),
    ("olmo_base_half",    "OlmoEarth", "depth",  "base@50%"),
    ("galileo_nano",      "Galileo",   "scale",  "nano"),
    ("galileo_tiny",      "Galileo",   "scale",  "tiny"),
    ("galileo_base_half", "Galileo",   "depth",  "base@50%"),
    ("clay_in256_half",   "Clay",      "depth",  "in256@50%"),
]
# 기존 full-size 기준선 (MS-102 / 봉인 pilot)
BASE = {"OlmoEarth": ("olmo (full, sealed P4)", 0.272),
        "Galileo":   ("galileo_cache (base, full)", 0.153),
        "Clay":      ("clay_cache_in256 (full)", 0.195)}

out = {"schema": "arch-axes-summary-v1",
       "status": "single seed diagnostics (addendum_v1b); not headline",
       "rows": {}, "verdict": {}}
for cache, fam, axis, val in AXES:
    r = {}
    for f in FOLDS:
        p = ROOT / "bv1_runs" / cache / f"{f}_seed1.json"
        if p.exists():
            d = json.load(open(p))
            r[f] = {"iou": d["test"]["positive_patch_macro_iou"],
                    "ap": d["test"]["auprc_exact"], "shape": d["emb_shape"]}
    out["rows"][cache] = r
    if not r:
        continue
    done = list(r)
    ious = [r[f]["iou"] for f in done]
    out["verdict"][cache] = {
        "family": fam, "axis": axis, "axis_value": val, "n": len(done),
        "macro": float(np.mean(ious)),
        "raw_same_folds": float(np.mean([ref[f]["raw_strong"] for f in done])),
        "olmo_same_folds": float(np.mean([ref[f]["reuse"] for f in done])),
        "beats_raw": int(sum(r[f]["iou"] > ref[f]["raw_strong"] for f in done)),
        "beats_olmo": int(sum(r[f]["iou"] > ref[f]["reuse"] for f in done)),
        "family_full_baseline": BASE[fam][1], "complete": len(done) == 8,
    }
(ROOT / "artifacts/arch_axes_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

print(f"{'cache':20s} {'fam':10s} {'축':11s} {'n':>2s} {'macro':>6s} "
      f"{'full대비':>8s} {'raw':>6s} {'>raw':>4s} {'>olmo':>5s}")
for c, v in out["verdict"].items():
    delta = v["macro"] - v["family_full_baseline"]
    flag = "" if v["complete"] else "  (부분)"
    print(f"{c:20s} {v['family']:10s} {v['axis']+'='+v['axis_value']:11s} {v['n']:2d} "
          f"{v['macro']:6.3f} {delta:+8.3f} {v['raw_same_folds']:6.3f} "
          f"{v['beats_raw']:4d} {v['beats_olmo']:5d}{flag}")
print("\n기준: raw P2 .197 / OlmoEarth 봉인 P4 .272 (동일 8폴드)")
