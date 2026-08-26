#!/usr/bin/env python3
"""arm 쌍의 **paired 공간 블록 CI**. 감사 지적: P4>P3를 CI 없이 우열로 쓰면 안 된다.

M52: P4 macro 0.180305 vs P3 0.167113, 격차 0.0132. 그런데 P4 폭 0.0397, P3 폭 0.0484.
격차가 양쪽 seed 폭보다 작으므로 CI 없이는 순위를 주장할 수 없다.

두 층의 불확실성을 분리해 보고한다.
  (a) 공간 표집   블록 부트스트랩 — seed 평균 격차에 대한 CI
  (b) seed 변동   seed별 격차의 부호 일치 여부 (M54 기준: 3/3 일치를 요구)
둘 다 통과해야 "우세"로 쓴다. 하나만 통과하면 "관측 평균 우위"까지만 쓴다.
"""
from __future__ import annotations
import argparse, itertools, json, pathlib
import numpy as np

E = pathlib.Path("evidence")
ARM_FILES = {
    "P4_reuse": [E / "gp_official_bundle/per_sample/P4_test.jsonl",
                 E / "seed_spread/P4_seed2_test.jsonl/P4_test.jsonl",
                 E / "seed_spread/P4_seed3_test.jsonl/P4_test.jsonl"],
    "P2_unet3d": [E / "gp_official_bundle/per_sample/P2_test.jsonl",
                  E / "seed_spread/P2_seed2_test.jsonl/P2_test.jsonl",
                  E / "seed_spread/P2_seed3_test.jsonl/P2_test.jsonl"],
    "P3_utae": [E / "gp_official_bundle/per_sample/P3_test.jsonl",
                E / "matrix_fill/P3_seed2/P3_test.jsonl",
                E / "matrix_fill/P3_seed3/P3_test.jsonl"],
    "P4c_bigdec": [E / "e1_factorial_v2/tiled_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
                   E / "noise_floor/seed2_P4c_test.jsonl",
                   E / "noise_floor/seed3_P4c_test.jsonl"],
}
N_BOOT, SEED = 10000, 20260826


def load(p):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l)}


def macro_pos(d, ss):
    v = [d[s]["tp"] / t for s in ss
         if d[s]["mask_positive_pixels"] > 0
         and (t := d[s]["tp"] + d[s]["fp"] + d[s]["fn"]) > 0]
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=pathlib.Path, default=E / "tile_coords.json")
    ap.add_argument("--out", type=pathlib.Path, default=E / "paired_arm_ci.json")
    a = ap.parse_args()

    data = {k: [load(p) for p in v] for k, v in ARM_FILES.items()}
    coords = json.loads(a.coords.read_text())
    sids = sorted(set.intersection(*(set(d) for lst in data.values() for d in lst))
                  & set(coords))
    rng = np.random.default_rng(SEED)
    arr = np.array(sids, dtype=object)

    res = {"schema": "paired-arm-ci-v1",
           "evidence_status": "development_only_not_confirmatory",
           "metric": "positive-tile macro IoU", "n_tiles": len(sids),
           "n_bootstrap": N_BOOT, "seed": SEED,
           "decision_rule": ("'우세'는 (a) 세 블록 크기 전부 CI가 0 제외 **그리고** "
                             "(b) 3 seed 전부 부호 일치일 때만. 하나만이면 '관측 평균 우위'"),
           "pairs": {}}

    for A, B in itertools.combinations(ARM_FILES, 2):
        per_seed = [macro_pos(data[A][i], sids) - macro_pos(data[B][i], sids)
                    for i in range(3)]
        sign_ok = all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed)
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
                d[i] = float(np.mean([macro_pos(data[A][k], sel) - macro_pos(data[B][k], sel)
                                      for k in range(3)]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            blocks[f"{km}km"] = {"n_blocks": int(len(uq)),
                                 "ci95": [round(float(lo), 6), round(float(hi), 6)],
                                 "excludes_zero": bool(lo > 0 or hi < 0)}
        ci_ok = all(v["excludes_zero"] for v in blocks.values())
        res["pairs"][f"{A}_vs_{B}"] = {
            "per_seed_gap": [round(x, 6) for x in per_seed],
            "mean_gap": round(float(np.mean(per_seed)), 6),
            "sign_consistent_3of3": sign_ok,
            "blocks": blocks, "all_ci_exclude_zero": ci_ok,
            "verdict": ("우세" if (ci_ok and sign_ok)
                        else ("관측 평균 우위(미확정)" if abs(np.mean(per_seed)) > 0 else "동률"))}
        print(f"{A} vs {B}: 평균 {np.mean(per_seed):+.6f} · 부호 3/3 {sign_ok} · "
              f"CI 전부 0제외 {ci_ok} → {res['pairs'][f'{A}_vs_{B}']['verdict']}", flush=True)

    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
