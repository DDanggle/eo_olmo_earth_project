#!/usr/bin/env python3
"""kill gate 2 — **타일 단위로 골라 쓸 이득이 있는가**. GPU 불필요, 새 학습 불필요.

router를 만들기 전에 천장부터 잰다. 없으면 방향 자체를 접는다.

  oracle       타일마다 정답을 미리 알고 가장 좋은 arm을 고른 경우
  best-single  전체에서 가장 좋은 arm 하나만 쓴 경우
  gain         oracle - best_single

**중요**: gain이 커도 승자를 **라벨 없이** 맞힐 수 없으면 method가 아니라 분석일 뿐이다.
그래서 라벨 없이 얻을 수 있는 값(`mean_probability`, `prediction_positive_pixels`)만으로
승자를 맞히는 단순 규칙의 상한도 같이 잰다.

주의: 이 test region(chimanimani)은 이미 여러 번 노출됐다. 전부 development-only다.
"""
from __future__ import annotations
import json, itertools, pathlib
import numpy as np

E = pathlib.Path("evidence")
ARMS = {
    "P1_shallow":      E / "gp_official_bundle/per_sample/P1_test.jsonl",
    "P2_unet3d":       E / "gp_official_bundle/per_sample/P2_test.jsonl",
    "P3_utae":         E / "gp_official_bundle/per_sample/P3_test.jsonl",
    "P4_tiled_small":  E / "gp_official_bundle/per_sample/P4_test.jsonl",
    "P4c_tiled_big":   E / "e1_factorial_v2/tiled_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
    "P4_full_small":   E / "e1_factorial_v2/full_small/per_sample/holdout_chimanimani/P4_test.jsonl",
    "P4c_full_big":    E / "e1_factorial_v2/full_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
}
OUT = E / "oracle_routing_headroom.json"


def load(p):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l)}


def micro(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows); d = tp + fp + fn
    return tp / d if d else 0.0


def tile_iou(r):
    d = r["tp"] + r["fp"] + r["fn"]
    return r["tp"] / d if d else None      # 예측·정답 모두 비었으면 정의되지 않음


def main() -> None:
    data = {k: load(v) for k, v in ARMS.items()}
    sids = sorted(set.intersection(*(set(d) for d in data.values())))
    names = list(ARMS)

    singles = {k: micro([data[k][s] for s in sids]) for k in names}
    best_single = max(singles, key=singles.get)

    # ── 전체 arm 오라클 ──
    def oracle_over(subset):
        picked, winners = [], []
        for s in sids:
            cand = [(tile_iou(data[k][s]), k) for k in subset]
            scored = [(v, k) for v, k in cand if v is not None]
            if not scored:                       # 전부 정의되지 않음 = 정답·예측 모두 없음
                picked.append(data[subset[0]][s]); winners.append(None); continue
            best = max(scored)[1]
            picked.append(data[best][s]); winners.append(best)
        return micro(picked), winners

    orc_all, win_all = oracle_over(names)
    from collections import Counter
    wc = Counter(w for w in win_all if w)

    res = {"schema": "oracle-routing-headroom-v1",
           "evidence_status": "development_only_not_confirmatory",
           "warning": "chimanimani test는 이미 다회 노출됨. 확증이 아님",
           "n_tiles": len(sids),
           "single_arm_micro_iou": {k: round(v, 6) for k, v in sorted(
               singles.items(), key=lambda x: -x[1])},
           "best_single_arm": best_single,
           "best_single_micro_iou": round(singles[best_single], 6),
           "oracle_all_arms": {
               "micro_iou": round(orc_all, 6),
               "gain_over_best_single": round(orc_all - singles[best_single], 6),
               "relative_gain_pct": round(100 * (orc_all / singles[best_single] - 1), 2),
               "winner_share": {k: round(v / len(sids), 4) for k, v in wc.most_common()}},
           "pairwise": {}}

    # ── 쌍별 오라클: 어떤 두 arm 조합이 실제로 상보적인가 ──
    for a, b in itertools.combinations(names, 2):
        o, w = oracle_over([a, b])
        base = max(singles[a], singles[b])
        c = Counter(x for x in w if x)
        res["pairwise"][f"{a}|{b}"] = {
            "oracle": round(o, 6), "best_of_two": round(base, 6),
            "gain": round(o - base, 6),
            "minority_share": round(min(c.get(a, 0), c.get(b, 0)) / len(sids), 4)}

    # ── 라벨 없이 승자를 맞힐 수 있는가 (핵심 2번 게이트) ──
    # 후보 특징은 전부 라벨을 쓰지 않는 값이다.
    pair = ("P2_unet3d", "P4c_tiled_big")
    a, b = pair
    feats, labels = [], []
    for s in sids:
        ia, ib = tile_iou(data[a][s]), tile_iou(data[b][s])
        if ia is None or ib is None or ia == ib:
            continue
        ra, rb = data[a][s], data[b][s]
        feats.append([ra.get("mean_probability", 0.0), rb.get("mean_probability", 0.0),
                      ra.get("prediction_positive_pixels", 0),
                      rb.get("prediction_positive_pixels", 0)])
        labels.append(1 if ib > ia else 0)
    X, y = np.array(feats, dtype=float), np.array(labels)
    prevalence = float(y.mean()) if len(y) else None
    single_feature_best = None
    if len(y) > 20:
        best = (0.0, None, None)
        for j in range(X.shape[1]):
            for thr in np.percentile(X[:, j], np.arange(5, 100, 5)):
                for sign in (1, -1):
                    pred = ((X[:, j] * sign) > (thr * sign)).astype(int)
                    acc = float((pred == y).mean())
                    if acc > best[0]:
                        best = (acc, j, float(thr))
        single_feature_best = {"accuracy": round(best[0], 4), "feature_index": best[1],
                               "threshold": best[2],
                               "features": ["A_mean_prob", "B_mean_prob",
                                            "A_pred_pos_px", "B_pred_pos_px"]}
    res["label_free_winner_prediction"] = {
        "pair": list(pair), "n_decidable_tiles": int(len(y)),
        "majority_class_rate": round(max(prevalence, 1 - prevalence), 4) if prevalence is not None else None,
        "best_single_feature_rule": single_feature_best,
        "note": ("규칙 정확도가 다수결과 비슷하면 라벨 없는 예측이 안 된다는 뜻. "
                 "임계값을 같은 데이터에서 고른 in-sample 상한이므로 낙관적이다.")}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "pairwise"},
                     ensure_ascii=False, indent=2))
    print("\n상위 쌍별 gain:")
    for k, v in sorted(res["pairwise"].items(), key=lambda x: -x[1]["gain"])[:6]:
        print(f"  {k:38s} gain {v['gain']:+.6f}  소수편 비율 {v['minority_share']:.3f}")


if __name__ == "__main__":
    main()
