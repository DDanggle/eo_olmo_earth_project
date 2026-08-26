#!/usr/bin/env python3
"""P2-P4 격차의 신뢰구간을 **가능한 단위로** 낸다. GPU 불필요.

region 단위는 불가능하다 — test region이 chimanimani 하나뿐이라 n=1이다(실측).
가능한 것은 `ann_id` cluster bootstrap(실측 422개)과 tile 단위이며,
tile은 공간 상관 때문에 구간을 과소추정한다. 둘 다 내고 차이를 남긴다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

B = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/per_sample")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/arm_gap_ci.json")
N_BOOT, SEED = 10000, 20260826


def load(arm):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in (B / f"{arm}_test.jsonl").read_text().splitlines() if l)}


def micro_iou(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows); d = tp + fp + fn
    return tp / d if d else float("nan")


def boot(groups, a, b, rng):
    keys = list(groups)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        ra, rb = [], []
        for j in pick:
            for sid in groups[keys[j]]:
                ra.append(a[sid]); rb.append(b[sid])
        out[i] = micro_iou(ra) - micro_iou(rb)
    return out


def main():
    a, b = load("P2"), load("P4")
    sids = sorted(a)
    obs = micro_iou([a[s] for s in sids]) - micro_iou([b[s] for s in sids])

    res = {"schema": "arm-gap-ci-v1", "comparison": "P2(공식 UNet3D) - P4(frozen)",
           "metric": "micro IoU", "observed_gap": round(obs, 6),
           "n_bootstrap": N_BOOT, "seed": SEED, "units": {}}

    for name, keyfn in (("ann_id_cluster", lambda r: r.get("ann_id") or f"__blank__{r['sample_id']}"),
                        ("tile", lambda r: r["sample_id"])):
        groups = {}
        for s in sids:
            groups.setdefault(keyfn(a[s]), []).append(s)
        rng = np.random.default_rng(SEED)
        d = boot(groups, a, b, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        res["units"][name] = {
            "n_units": len(groups), "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "p_gap_le_0": round(float((d <= 0).mean()), 6),
            "excludes_zero": bool(lo > 0 or hi < 0)}

    res["caveat"] = ("region 단위 CI는 원리상 불가능(test region n=1). "
                     "tile 단위는 공간 상관을 무시하므로 구간을 과소추정한다 — "
                     "ann_id cluster 쪽을 우선 본다.")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
