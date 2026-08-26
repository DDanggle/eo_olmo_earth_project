#!/usr/bin/env python3
"""통제 1 완결 — seed 분산과 잡음 바닥 oracle. 로컬 실행, GPU 불필요.

두 가지를 확정한다.

1. **seed 분산**: 같은 구성(P4c·tiled·큰 decoder)의 seed 1/2/3 결과가 얼마나 퍼지는가.
   이 폭 안에 있는 arm 간 차이는 "모델 차이"라고 부를 수 없다.
2. **잡음 바닥 oracle**: seed만 다른 두 실행 사이의 per-tile oracle gain.
   표현이 같으므로 이 gain은 100% 선택 잡음이다. M40의 관측 gain이
   이 바닥을 유의하게 넘지 못하면 routing 여유 주장은 죽는다.

M40의 지표 불일치 결함도 여기서 고친다: 선택과 보고를 **같은 지표**(양성 타일의
tile-IoU 평균, macro)로 통일하고, micro-IoU oracle은 탐욕 좌표상승으로 따로 잰다.
"""
from __future__ import annotations
import json, itertools, pathlib
import numpy as np

E = pathlib.Path("evidence")
SEEDS = {
    "seed1": E / "e1_factorial_v2/tiled_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
    "seed2": E / "noise_floor/seed2_P4c_test.jsonl",
    "seed3": E / "noise_floor/seed3_P4c_test.jsonl",
}
ARMS = {
    "P2_unet3d": E / "gp_official_bundle/per_sample/P2_test.jsonl",
    "P3_utae": E / "gp_official_bundle/per_sample/P3_test.jsonl",
    "P4_tiled_small": E / "gp_official_bundle/per_sample/P4_test.jsonl",
    "P4c_tiled_big_seed1": SEEDS["seed1"],
}
OUT = E / "noise_floor_analysis.json"
RNG = np.random.default_rng(20260826)


def load(p):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l)}


def tile_iou(r):
    d = r["tp"] + r["fp"] + r["fn"]
    return r["tp"] / d if d else None


def micro(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0


def macro_pos(d, sids):
    """양성 타일의 tile-IoU 평균 — 선택 지표와 보고 지표가 일치하는 macro 축."""
    v = [tile_iou(d[s]) for s in sids]
    return float(np.mean([x for x in v if x is not None]))


def aligned_oracle(arms_d, sids):
    """지표 정렬 oracle: per-tile max tile-IoU 선택 → 같은 지표(macro 평균)로 보고.
    macro 평균은 타일별로 분해되므로 per-tile max가 정확히 최적이다."""
    vals = []
    for s in sids:
        c = [tile_iou(d[s]) for d in arms_d.values()]
        c = [x for x in c if x is not None]
        if c:
            vals.append(max(c))
    return float(np.mean(vals))


def greedy_micro_oracle(arms_d, sids, iters=6):
    """micro-IoU 좌표상승 oracle(하한). best single에서 시작해 타일 하나씩
    전역 micro-IoU를 올리는 arm으로 바꾼다. 수렴까지 반복."""
    names = list(arms_d)
    singles = {k: micro([arms_d[k][s] for s in sids]) for k in names}
    assign = {s: max(singles, key=singles.get) for s in sids}

    def totals():
        tp = sum(arms_d[assign[s]][s]["tp"] for s in sids)
        fp = sum(arms_d[assign[s]][s]["fp"] for s in sids)
        fn = sum(arms_d[assign[s]][s]["fn"] for s in sids)
        return tp, fp, fn
    tp, fp, fn = totals()
    for _ in range(iters):
        changed = 0
        for s in sids:
            cur = arms_d[assign[s]][s]
            base_tp, base_fp, base_fn = tp - cur["tp"], fp - cur["fp"], fn - cur["fn"]
            best_k, best_v = assign[s], tp / (tp + fp + fn)
            for k in names:
                r = arms_d[k][s]
                t2, f2, n2 = base_tp + r["tp"], base_fp + r["fp"], base_fn + r["fn"]
                v = t2 / (t2 + f2 + n2) if (t2 + f2 + n2) else 0.0
                if v > best_v + 1e-15:
                    best_v, best_k = v, k
            if best_k != assign[s]:
                r = arms_d[best_k][s]
                tp, fp, fn = base_tp + r["tp"], base_fp + r["fp"], base_fn + r["fn"]
                assign[s] = best_k; changed += 1
        if not changed:
            break
    return tp / (tp + fp + fn), max(singles.values())


def block_ci(fn_stat, sids, coords, n_boot=5000, block_m=5120.0):
    key = (np.floor(coords[:, 0] / block_m).astype(np.int64) * 1_000_003
           + np.floor(coords[:, 1] / block_m).astype(np.int64))
    uniq, inv = np.unique(key, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    sa = np.array(sids, dtype=object)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        out[i] = fn_stat(list(sa[sel]))
    return [round(float(np.percentile(out, 2.5)), 6),
            round(float(np.percentile(out, 97.5)), 6)]


def main():
    seeds = {k: load(v) for k, v in SEEDS.items()}
    sids = sorted(set.intersection(*(set(d) for d in seeds.values())))
    pos = [s for s in sids if seeds["seed1"][s]["mask_positive_pixels"] > 0]

    res = {"schema": "noise-floor-analysis-v1",
           "evidence_status": "development_only_not_confirmatory",
           "n_tiles": len(sids), "n_positive_tiles": len(pos)}

    # 1. seed 분산
    res["seed_spread_P4c_tiled"] = {
        k: {"micro_iou_all": round(micro([d[s] for s in sids]), 6),
            "macro_pos_iou": round(macro_pos(d, pos), 6)}
        for k, d in seeds.items()}
    mi = [v["micro_iou_all"] for v in res["seed_spread_P4c_tiled"].values()]
    res["seed_spread_summary"] = {
        "micro_iou_mean": round(float(np.mean(mi)), 6),
        "micro_iou_range": round(float(max(mi) - min(mi)), 6),
        "micro_iou_std": round(float(np.std(mi, ddof=1)), 6),
        "P2_single_seed_micro_iou": 0.159254,
        "P2_inside_seed_range": bool(min(mi) <= 0.159254 <= max(mi))}

    # 2. 잡음 바닥 oracle — seed쌍별, 지표 정렬(macro, 양성 타일)
    floors = {}
    for a, b in itertools.combinations(seeds, 2):
        pair = {a: seeds[a], b: seeds[b]}
        base = max(macro_pos(seeds[a], pos), macro_pos(seeds[b], pos))
        orc = aligned_oracle(pair, pos)
        floors[f"{a}|{b}"] = {"best_single_macro": round(base, 6),
                              "oracle_macro": round(orc, 6),
                              "noise_floor_gain": round(orc - base, 6)}
    res["noise_floor_pairs"] = floors
    nf = max(v["noise_floor_gain"] for v in floors.values())
    res["noise_floor_max"] = round(nf, 6)

    # 3. 관측 oracle — 서로 다른 arm 4개, 같은 정렬 지표
    arms = {k: load(v) for k, v in ARMS.items()}
    base = max(macro_pos(d, pos) for d in arms.values())
    orc = aligned_oracle(arms, pos)
    obs = orc - base
    res["observed_oracle_4arms_macro_pos"] = {
        "best_single": round(base, 6), "oracle": round(orc, 6),
        "gain": round(obs, 6),
        "gain_minus_noise_floor": round(obs - nf, 6),
        "gain_over_noise_floor_ratio": round(obs / nf, 3) if nf > 0 else None}

    # 4. micro 좌표상승 oracle (참고 — M40의 per-tile max 방식은 micro와 정렬되지 않았음)
    gm, gs = greedy_micro_oracle(arms, sids)
    res["greedy_micro_oracle_4arms"] = {
        "best_single_micro": round(gs, 6), "greedy_oracle_micro": round(gm, 6),
        "gain": round(gm - gs, 6),
        "note": "좌표상승 하한. M40의 per-tile-max 방식은 micro 지표와 목적이 어긋났음"}

    # 5. 공간 블록 CI — 관측 gain과 잡음 바닥 gain의 차이
    coords_p = E / "tile_coords.json"
    if coords_p.exists():
        cm = json.loads(coords_p.read_text())
        coords = np.array([cm[s] for s in pos])
        s2 = {a: seeds[a] for a in ("seed2", "seed3")}

        def stat(sub):
            o1 = aligned_oracle(arms, sub) - max(macro_pos(d, sub) for d in arms.values())
            o2 = aligned_oracle(s2, sub) - max(macro_pos(seeds["seed2"], sub),
                                               macro_pos(seeds["seed3"], sub))
            return o1 - o2
        res["gain_minus_floor_ci95_5.12km"] = block_ci(stat, pos, coords)
    else:
        res["gain_minus_floor_ci95_5.12km"] = "tile_coords.json 없음 — 서버에서 좌표 추출 필요"

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
