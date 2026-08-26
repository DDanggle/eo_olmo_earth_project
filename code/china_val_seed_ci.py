#!/usr/bin/env python3
"""M50 확장 — china val 격차를 3 seed 전부로 내고, **seed 평균 격차**의 공간 블록 CI. 로컬.

M50은 seed 1 하나의 CI였다. M43·M47이 seed 변동의 크기를 보여줬으므로
seed 평균 격차에 대한 CI가 있어야 주장의 단위가 맞는다.

부트스트랩 단위: 공간 블록(타일 재표집). seed는 고정된 3개를 평균하는 대상으로 두고,
각 부트스트랩 표본에서 3 seed 평균 격차를 계산한다 — 즉 seed는 모집단이 아니라
프로토콜의 일부로 취급한다(3개뿐이라 seed 재표집은 검정력이 없음).
"""
from __future__ import annotations
import json, pathlib
import numpy as np

E = pathlib.Path("evidence")
FILES = {
    ("P2", 1): E / "gp_official_bundle/per_sample/P2_val.jsonl",
    ("P2", 2): E / "seed_spread_val/P2_s2/P2_val.jsonl",
    ("P2", 3): E / "seed_spread_val/P2_s3/P2_val.jsonl",
    ("P4", 1): E / "gp_official_bundle/per_sample/P4_val.jsonl",
    ("P4", 2): E / "seed_spread_val/P4_s2/P4_val.jsonl",
    ("P4", 3): E / "seed_spread_val/P4_s3/P4_val.jsonl",
}
OUT = E / "china_val_seed_ci.json"
N_BOOT, SEED = 10000, 20260826


def load(p):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l)}


def micro(d, ss):
    tp = sum(d[s]["tp"] for s in ss); fp = sum(d[s]["fp"] for s in ss)
    fn = sum(d[s]["fn"] for s in ss)
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0


def macro_pos(d, ss):
    v = [d[s]["tp"] / t for s in ss
         if d[s]["mask_positive_pixels"] > 0
         and (t := d[s]["tp"] + d[s]["fp"] + d[s]["fn"]) > 0]
    return float(np.mean(v)) if v else np.nan


def main():
    data = {k: load(v) for k, v in FILES.items()}
    coords = json.loads((E / "tile_coords_val.json").read_text())
    sids = sorted(set.intersection(*(set(d) for d in data.values())) & set(coords))
    res = {"schema": "china-val-seed-ci-v1",
           "evidence_status": "development_only_not_confirmatory",
           "region": "china (val) — test는 chimanimani로 다른 지역",
           "caveat": "val은 epoch 선택에 쓰였으므로 완전한 held-out이 아니다",
           "n_tiles": len(sids), "n_bootstrap": N_BOOT, "seed": SEED,
           "per_seed": {}, "seed_mean": {}}

    for metric, fn in (("micro_iou", micro), ("macro_pos_iou", macro_pos)):
        per = {}
        for s in (1, 2, 3):
            a, b = fn(data[("P2", s)], sids), fn(data[("P4", s)], sids)
            per[f"seed{s}"] = {"P2": round(a, 6), "P4": round(b, 6),
                               "gap": round(b - a, 6)}
        gaps = [per[f"seed{s}"]["gap"] for s in (1, 2, 3)]
        res["per_seed"][metric] = per
        res["per_seed"][metric]["all_positive"] = bool(all(g > 0 for g in gaps))
        res["per_seed"][metric]["mean_gap"] = round(float(np.mean(gaps)), 6)

        rng = np.random.default_rng(SEED)
        arr = np.array(sids, dtype=object)
        blocks = {}
        for km in (2.56, 5.12, 10.24):
            m = km * 1000
            key = np.array([int(np.floor(coords[s][0] / m)) * 1_000_003
                            + int(np.floor(coords[s][1] / m)) for s in sids])
            uq, inv = np.unique(key, return_inverse=True)
            idx = [np.where(inv == i)[0] for i in range(len(uq))]
            d = np.empty(N_BOOT)
            for i in range(N_BOOT):
                pick = rng.integers(0, len(uq), len(uq))
                sel = list(arr[np.concatenate([idx[j] for j in pick])])
                d[i] = float(np.mean([fn(data[("P4", s)], sel) - fn(data[("P2", s)], sel)
                                      for s in (1, 2, 3)]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            blocks[f"{km}km"] = {"n_blocks": int(len(uq)),
                                 "ci95": [round(float(lo), 6), round(float(hi), 6)],
                                 "p_le_0": round(float((d <= 0).mean()), 6),
                                 "excludes_zero": bool(lo > 0)}
        res["seed_mean"][metric] = {"mean_gap": round(float(np.mean(gaps)), 6),
                                    "blocks": blocks,
                                    "all_block_sizes_exclude_zero":
                                        all(v["excludes_zero"] for v in blocks.values())}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
