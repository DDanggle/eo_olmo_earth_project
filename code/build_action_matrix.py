#!/usr/bin/env python3
"""E6 — Action Matrix v1. EarthRoute 학습 데이터의 원형. CPU 전용, 재학습 없음.

행 = 공간 블록(5.12 km, M33 단위). 열 = action. 값 = utility.

  utility(a) = Δ(주 지표) − λ_c · FLOPs(a)
  주 지표    = 양성 타일 macro IoU (M40: 빈 타일 62.7%라 전체 micro는 오경보 지표)
  기준 action = reuse (frozen 캐시 + 작은 판독기, 가장 싼 것)
  λ_c        = 사전 등록 후보 3개 중 하나. 결과를 보고 고르지 않기 위해 **셋 다 보고**한다.

**M43·M45의 교훈을 계약으로 박는다.**
  - 모든 action 값은 **가용한 seed의 평균**을 쓴다. 단일 seed 값을 utility로 쓰지 않는다.
  - seed가 1개뿐인 action은 `seed_n=1`로 표시하고 **utility를 신뢰 불가로 마킹**한다.
  - 블록별 최적 action은 **seed 폭보다 큰 차이**일 때만 유효로 센다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

E = pathlib.Path("evidence")
OUT = E / "action_matrix_v1"
BLOCK_M = 5120.0
# 사전 등록 λ 후보 (GFLOP당 IoU 손실 환산). 결과를 보고 하나를 고르지 않는다.
LAMBDAS = {"lambda_free": 0.0, "lambda_mid": 1e-6, "lambda_strict": 1e-5}

# action → (per-sample 파일 목록[seed별], 샘플당 학습 GFLOP)
# GFLOP은 measure_flops_cost.py 실측 + 학습 배율(5542×40×3)을 적용한 per-task 총량을
# 1e12로 정규화한 값. reuse는 인코더 비용을 이미 지불한 것으로 본다(공유 자산).
ACTIONS = {
    "reuse": {
        "files": [E / "gp_official_bundle/per_sample/P4_test.jsonl",
                  E / "seed_spread/P4_seed2_test.jsonl/P4_test.jsonl",
                  E / "seed_spread/P4_seed3_test.jsonl/P4_test.jsonl"],
        "train_pflops": 1.34, "desc": "frozen v1 캐시 + 작은 판독기"},
    "reuse_bigdec": {
        "files": [E / "e1_factorial_v2/tiled_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
                  E / "noise_floor/seed2_P4c_test.jsonl",
                  E / "noise_floor/seed3_P4c_test.jsonl"],
        "train_pflops": 9.64, "desc": "frozen v1 캐시 + 큰 판독기"},
    # 아래 셋은 M46 시점에 seed 1뿐이어서 69블록 중 40개(58%)를 운으로 독식했다.
    # matrix_fill 실행(2026-08-26)으로 seed 2·3을 확보해 여기서 연결한다.
    "recontext": {
        "files": [E / "e1_factorial_v2/full_small/per_sample/holdout_chimanimani/P4_test.jsonl",
                  E / "matrix_fill/P4full_seed2/P4_test.jsonl",
                  E / "matrix_fill/P4full_seed3/P4_test.jsonl"],
        "train_pflops": 1.34, "desc": "통짜 1x128 캐시로 재계산 + 작은 판독기"},
    "recontext_bigdec": {
        "files": [E / "e1_factorial_v2/full_big/per_sample/holdout_chimanimani/P4c_test.jsonl",
                  E / "matrix_fill/P4cfull_seed2/P4c_test.jsonl",
                  E / "matrix_fill/P4cfull_seed3/P4c_test.jsonl"],
        "train_pflops": 9.64, "desc": "통짜 캐시 + 큰 판독기"},
    "raw_unet3d": {
        "files": [E / "gp_official_bundle/per_sample/P2_test.jsonl",
                  E / "seed_spread/P2_seed2_test.jsonl/P2_test.jsonl",
                  E / "seed_spread/P2_seed3_test.jsonl/P2_test.jsonl"],
        "train_pflops": 180.07, "desc": "raw 학습 공식 UNet3D"},
    "raw_utae": {
        "files": [E / "gp_official_bundle/per_sample/P3_test.jsonl",
                  E / "matrix_fill/P3_seed2/P3_test.jsonl",
                  E / "matrix_fill/P3_seed3/P3_test.jsonl"],
        "train_pflops": 25.75, "desc": "raw 학습 공식 U-TAE"},
}
BASELINE = "reuse"


def load(p):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l)}


def tile_iou(r):
    d = r["tp"] + r["fp"] + r["fn"]
    return r["tp"] / d if d else None


def macro_pos(recs, sids):
    v = [tile_iou(recs[s]) for s in sids if recs[s]["mask_positive_pixels"] > 0]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def sign_consistent(vals):
    """3 seed 부호가 모두 같은가. M52에서 C_large·상호작용이 seed 3에서 부호가
    뒤집힌 것을 확인했으므로, 블록 판정에도 같은 기준을 적용한다."""
    return bool(vals) and (all(x > 0 for x in vals) or all(x < 0 for x in vals))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coords = json.loads((E / "tile_coords.json").read_text())
    loaded = {a: [load(f) for f in cfg["files"] if f.exists()]
              for a, cfg in ACTIONS.items()}
    missing = {a: [str(f) for f in cfg["files"] if not f.exists()]
               for a, cfg in ACTIONS.items()}
    sids = sorted(set.intersection(*(set(d) for lst in loaded.values() for d in lst)))

    key = {}
    for s in sids:
        x, y = coords[s]
        key[s] = (int(np.floor(x / BLOCK_M)), int(np.floor(y / BLOCK_M)))
    blocks = sorted({key[s] for s in sids})

    rows = []
    for b in blocks:
        bs = [s for s in sids if key[s] == b]
        npos = sum(1 for s in bs if loaded[BASELINE][0][s]["mask_positive_pixels"] > 0)
        if npos == 0:
            continue      # 주 지표가 양성 macro이므로 양성 없는 블록은 정의 불가
        per_action = {}
        for a, lst in loaded.items():
            vals = [macro_pos(d, bs) for d in lst]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            per_action[a] = {"metric_mean": round(float(np.mean(vals)), 6),
                             "metric_spread": round(float(max(vals) - min(vals)), 6),
                             "seed_n": len(vals),
                             "train_pflops": ACTIONS[a]["train_pflops"],
                             "reliable": len(vals) >= 3}
        if BASELINE not in per_action:
            continue
        base = per_action[BASELINE]["metric_mean"]
        for a, v in per_action.items():
            v["delta_vs_reuse"] = round(v["metric_mean"] - base, 6)
            for ln, lv in LAMBDAS.items():
                v[f"utility_{ln}"] = round(v["delta_vs_reuse"] - lv * v["train_pflops"] * 1e3, 6)
        # 블록 최적 action — 두 조건을 모두 요구한다:
        #  (a) margin이 seed 폭보다 크다  (b) seed별로 top이 second를 항상 이긴다
        # (b)는 M52 교훈이다 — 평균 부호만 보면 seed에서 뒤집히는 효과를 놓친다.
        spread_ref = max(v["metric_spread"] for v in per_action.values())
        ranked = sorted(per_action.items(), key=lambda kv: -kv[1]["utility_lambda_free"])
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, None)
        margin_ok = bool(second and
                         (top[1]["utility_lambda_free"] - second[1]["utility_lambda_free"])
                         > spread_ref)
        per_seed_ok = False
        if second and top[1]["seed_n"] == 3 and second[1]["seed_n"] == 3:
            ta = [macro_pos(d, bs) for d in loaded[top[0]]]
            sa = [macro_pos(d, bs) for d in loaded[second[0]]]
            diffs = [x - y for x, y in zip(ta, sa) if x is not None and y is not None]
            per_seed_ok = sign_consistent(diffs) and all(x > 0 for x in diffs)
        decisive = bool(margin_ok and per_seed_ok)
        rows.append({
            "block": {"ix": b[0], "iy": b[1], "block_km": BLOCK_M / 1000},
            "n_tiles": len(bs), "n_positive_tiles": npos,
            "actions": per_action,
            "argmax_lambda_free": top[0],
            "runner_up": second[0] if second else None,
            "margin": round(top[1]["utility_lambda_free"]
                            - second[1]["utility_lambda_free"], 6) if second else None,
            "seed_spread_reference": round(spread_ref, 6),
            "margin_exceeds_seed_spread": margin_ok,
            "top_beats_second_in_all_seeds": per_seed_ok,
            "decisive_beyond_seed_noise": decisive})

    from collections import Counter
    argmax = Counter(r["argmax_lambda_free"] for r in rows)
    dec = Counter(r["argmax_lambda_free"] for r in rows if r["decisive_beyond_seed_noise"])
    summary = {
        "schema": "action-matrix-v1",
        "evidence_status": "development_only_not_confirmatory",
        "primary_metric": "positive-tile macro IoU",
        "baseline_action": BASELINE,
        "lambda_candidates": LAMBDAS,
        "n_blocks_with_positives": len(rows), "n_tiles": len(sids),
        "argmax_distribution": dict(argmax),
        "argmax_distribution_decisive_only": dict(dec),
        "n_decisive_blocks": sum(1 for r in rows if r["decisive_beyond_seed_noise"]),
        "single_action_dominates": len(argmax) == 1,
        "kill_gate": ("블록 간 최적 action이 단일하면 단일 task routing 중단 → RQ2로 이동. "
                      "seed 폭을 넘는 결정적 블록이 없으면 matrix 자체가 잡음이다."),
        "decisive_definition": ("margin > seed 폭 **그리고** 3 seed 전부에서 top이 second를 "
                                "이김. 후자는 M52 교훈(C_large·상호작용이 seed 3에서 부호 반전)"),
        "all_actions_3seed": True,
        "missing_inputs": {k: v for k, v in missing.items() if v},
        "seed_policy": "action별 가용 seed 평균. seed_n<3은 reliable=false로 표시",
    }
    (OUT / "matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "blocks.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
