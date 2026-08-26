#!/usr/bin/env python3
"""P2-P4 격차의 **공간 블록** 부트스트랩. GPU 불필요.

왜 다시 하는가: 앞선 `ann_id` cluster 부트스트랩은 무효였다. 실측 결과 ann_id는
양성 타일 423개에만 있고 423개 중 422개가 고유해서 **타일과 1:1**이다.
즉 그건 타일 i.i.d. 부트스트랩이고, 인접 타일의 공간 상관을 무시해 구간을 좁게 만든다.

여기서는 타일 중심 좌표를 NetCDF에서 읽어 **격자 블록**으로 묶고 블록 단위로 재표집한다.
블록 크기를 여러 개 두고 구간이 어떻게 넓어지는지 함께 보고한다 —
하나만 고르면 그 선택이 결론을 만든다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

B = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/per_sample")
ROOT = pathlib.Path("/home/work/data/sen12landslides/extracted")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/arm_gap_ci_spatial.json")
N_BOOT, SEED = 10000, 20260826
BLOCK_KM = [2.56, 5.12, 10.24, 20.48]     # 타일 1.28 km의 2·4·8·16배


def load(arm):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in (B / f"{arm}_test.jsonl").read_text().splitlines() if l)}


def micro_iou_arrays(tp, fp, fn):
    d = tp.sum() + fp.sum() + fn.sum()
    return tp.sum() / d if d else np.nan


def main():
    import xarray as xr
    a, b = load("P2"), load("P4")
    sids = sorted(a)

    # 타일 중심 좌표 (EPSG:32736, m)
    cx, cy = [], []
    for sid in sids:
        with xr.open_dataset(ROOT / f"{sid}.nc", decode_times=False, cache=False) as ds:
            cx.append(float(ds["x"].values.mean())); cy.append(float(ds["y"].values.mean()))
    cx, cy = np.array(cx), np.array(cy)

    tp2 = np.array([a[s]["tp"] for s in sids], float)
    fp2 = np.array([a[s]["fp"] for s in sids], float)
    fn2 = np.array([a[s]["fn"] for s in sids], float)
    tp4 = np.array([b[s]["tp"] for s in sids], float)
    fp4 = np.array([b[s]["fp"] for s in sids], float)
    fn4 = np.array([b[s]["fn"] for s in sids], float)

    obs = micro_iou_arrays(tp2, fp2, fn2) - micro_iou_arrays(tp4, fp4, fn4)
    res = {"schema": "arm-gap-ci-spatial-v1",
           "comparison": "P2(공식 UNet3D) - P4(frozen)", "metric": "micro IoU",
           "observed_gap": round(float(obs), 6), "n_tiles": len(sids),
           "n_bootstrap": N_BOOT, "seed": SEED,
           "why_not_ann_id": "ann_id는 양성 423타일에만 있고 422개가 고유 — 타일과 1:1이라 군집이 아님",
           "why_not_region": "test region이 chimanimani 하나 (n=1)",
           "blocks": {}}

    for km in BLOCK_KM:
        m = km * 1000.0
        key = (np.floor(cx / m).astype(np.int64) * 1_000_003
               + np.floor(cy / m).astype(np.int64))
        uniq, inv = np.unique(key, return_inverse=True)
        idx_by_block = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(SEED)
        d = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([idx_by_block[j] for j in pick])
            d[i] = (micro_iou_arrays(tp2[sel], fp2[sel], fn2[sel])
                    - micro_iou_arrays(tp4[sel], fp4[sel], fn4[sel]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        sizes = np.array([len(x) for x in idx_by_block])
        res["blocks"][f"{km}km"] = {
            "n_blocks": int(len(uniq)),
            "tiles_per_block_median": int(np.median(sizes)),
            "tiles_per_block_max": int(sizes.max()),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "ci_width": round(float(hi - lo), 6),
            "p_gap_le_0": round(float((d <= 0).mean()), 6),
            "excludes_zero": bool(lo > 0 or hi < 0)}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
