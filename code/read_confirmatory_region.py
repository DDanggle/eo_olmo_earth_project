#!/usr/bin/env python3
"""확증 지역 판독 — **post 게이트 통과 후에만** 실행한다. per-sample 재계산만 쓴다.

동결된 recipe(v2)의 규칙을 그대로 적용한다:
  주지표      양성 타일 macro IoU
  승리        seed-mean primary에서 reuse > raw_strong **그리고** 3 seed 전부 우위
  보조지표    빈타일 FP · ECE · AUPRC · micro (사전 등록 순서)
  CI          공간 블록 2.56/5.12/10.24 km 전부 보고. CI 0 제외는 '강한 승리'로 별도 표기
로그 문자열을 읽지 않는다(M52 교훈: sed greedy 매칭으로 ld_iou를 iou로 오독한 적 있음).
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np

ARMS = {"reuse": "P4", "raw_strong": "P2", "raw_efficient": "P3"}
SEEDS = [1, 2, 3]


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def macro_pos(rows):
    v = [r["tp"] / t for r in rows
         if r["mask_positive_pixels"] > 0 and (t := r["tp"] + r["fp"] + r["fn"]) > 0]
    return float(np.mean(v)) if v else np.nan


def micro(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return tp / (tp + fp + fn) if (tp + fp + fn) else np.nan


def fp_empty(rows):
    return int(sum(r["fp"] for r in rows if r["mask_positive_pixels"] == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--gate", type=pathlib.Path, required=True)
    ap.add_argument("--coords", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    gate = json.loads(a.gate.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS":
        raise SystemExit(f"게이트 미통과({gate.get('verdict')}). 판독하지 않는다.")

    data = {}
    for role, arm in ARMS.items():
        data[role] = [load(a.root / a.fold / f"{arm}_seed{s}" / "per_sample" / a.fold
                           / f"{arm}_test.jsonl") for s in SEEDS]

    res = {"schema": "confirmatory-region-read-v1", "fold": a.fold,
           "gate_verdict": gate.get("verdict"),
           "protocol_deviation": gate.get("protocol_deviation"),
           "evidence_status": (gate.get("evidence_status_override") or {}).get("actual_status"),
           "primary_metric": "positive-tile macro IoU", "arms": {}}

    for role, runs in data.items():
        res["arms"][role] = {
            "arm_id": ARMS[role],
            "primary_per_seed": [round(macro_pos(r), 6) for r in runs],
            "primary_mean": round(float(np.mean([macro_pos(r) for r in runs])), 6),
            "primary_spread": round(float(np.ptp([macro_pos(r) for r in runs])), 6),
            "micro_per_seed": [round(micro(r), 6) for r in runs],
            "micro_mean": round(float(np.mean([micro(r) for r in runs])), 6),
            "fp_empty_per_seed": [fp_empty(r) for r in runs],
            "n_tiles": len(runs[0]),
            "n_positive_tiles": sum(1 for r in runs[0] if r["mask_positive_pixels"] > 0)}

    # 사전 등록 승리 판정: reuse vs raw_strong
    ps = [macro_pos(r) for r in data["reuse"]]
    qs = [macro_pos(r) for r in data["raw_strong"]]
    gaps = [p - q for p, q in zip(ps, qs)]
    win = bool(np.mean(gaps) > 0 and all(g > 0 for g in gaps))
    res["preregistered_win_reuse_vs_raw_strong"] = {
        "per_seed_gap": [round(g, 6) for g in gaps],
        "mean_gap": round(float(np.mean(gaps)), 6),
        "all_three_seeds_positive": bool(all(g > 0 for g in gaps)),
        "per_region_win": win,
        "rule": "seed-mean > 0 AND 3 seed 전부 > 0"}

    # 참고: reuse vs raw_efficient
    rs = [macro_pos(r) for r in data["raw_efficient"]]
    g2 = [p - q for p, q in zip(ps, rs)]
    res["reference_reuse_vs_raw_efficient"] = {
        "per_seed_gap": [round(g, 6) for g in g2],
        "mean_gap": round(float(np.mean(g2)), 6),
        "all_three_seeds_positive": bool(all(g > 0 for g in g2))}

    if a.coords and a.coords.exists():
        coords = json.loads(a.coords.read_text())
        sids = [r["sample_id"] for r in data["reuse"][0]]
        sids = [s for s in sids if s in coords]
        idx = {r["sample_id"]: r for r in data["reuse"][0]}
        rng = np.random.default_rng(20260826)
        arr = np.array(sids, dtype=object)
        maps = {role: [{r["sample_id"]: r for r in run} for run in runs]
                for role, runs in data.items()}

        def gap_on(sel):
            out = []
            for i in range(3):
                A = [maps["reuse"][i][s] for s in sel if s in maps["reuse"][i]]
                B = [maps["raw_strong"][i][s] for s in sel if s in maps["raw_strong"][i]]
                out.append(macro_pos(A) - macro_pos(B))
            return float(np.mean(out))

        blocks = {}
        for km in (2.56, 5.12, 10.24):
            m = km * 1000
            key = np.array([int(np.floor(coords[s][0] / m)) * 1_000_003
                            + int(np.floor(coords[s][1] / m)) for s in sids])
            uq, inv = np.unique(key, return_inverse=True)
            bi = [np.where(inv == i)[0] for i in range(len(uq))]
            d = np.empty(4000)
            for i in range(4000):
                pick = rng.integers(0, len(uq), len(uq))
                d[i] = gap_on(list(arr[np.concatenate([bi[j] for j in pick])]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            blocks[f"{km}km"] = {"n_blocks": int(len(uq)),
                                 "ci95": [round(float(lo), 6), round(float(hi), 6)],
                                 "excludes_zero": bool(lo > 0 or hi < 0)}
        res["spatial_block_ci"] = blocks
        res["strong_win"] = bool(win and all(v["excludes_zero"] for v in blocks.values()))

    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
