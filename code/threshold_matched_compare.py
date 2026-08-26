#!/usr/bin/env python3
"""E5c — FP율 정합 비교. P4c 우위가 임계값 인공물인지 판정한다. CPU 전용.

M40: P4c(frozen+큰 decoder)의 전체 micro 우위는 빈 타일 오경보를 P2의 34%로 줄인
데서 나왔고, 양성 타일 IoU는 P2가 위였다. 그런데 두 모델은 같은 임계값 0.5를 쓰므로
서로 다른 작동점에 있다. **P2의 임계값을 P4c와 같은 빈 타일 FP율에 맞추면** 어떻게 되는가.

사전 등록 (실행 전 고정):
  정합 기준   빈 타일(양성 0) 총 FP 화소 수를 P4c@0.5와 같게 만드는 P2 임계값 t*
  보고 지표   그 t*에서의 (1) 양성 타일 macro IoU  (2) 전체 micro IoU
  판정        정합 후에도 P2의 양성 macro가 위면 → P4c 우위는 임계값 인공물로 확정,
              "frozen 경쟁력" 서술 전면 철회.
              정합 후 P4c가 위로 올라오면 → 진짜 표현 차이. 서술 유지(단 seed 폭 병기).
  대칭 검사   반대 방향(P4c를 P2@0.5의 FP율로)도 같이 보고한다. 한 방향만 보면
              방향 선택 자체가 결론을 만든다.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

BASE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/home/work/data/olmoearth/probmaps_eval")
MASK = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/mask_u8")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/threshold_matched.json")
ARMS = {"P2": "P2", "P4c": "P4c"}   # 디렉터리 이름 규약: BASE/<arm>/prob_maps/...


def load(arm):
    d = BASE / arm / "prob_maps" / "holdout_chimanimani"
    idx = json.loads((d / f"{ARMS[arm]}_test_probs_index.json").read_text())
    arr = np.load(d / f"{ARMS[arm]}_test_probs_u8.npy", mmap_mode="r")
    return idx["sample_ids"], arr


def masks_for(sids):
    ys, pos = [], []
    for s in sids:
        m = np.load(MASK / f"{s}.npy") > 0
        ys.append(m); pos.append(int(m.sum()))
    return np.stack(ys), np.array(pos)


def stats_at(probs_u8, ys, pos, thr_u8):
    pred = probs_u8 >= thr_u8
    tp = (pred & ys).sum(axis=(1, 2)).astype("int64")
    fp = (pred & ~ys).sum(axis=(1, 2)).astype("int64")
    fn = (~pred & ys).sum(axis=(1, 2)).astype("int64")
    empty = pos == 0
    fp_empty = int(fp[empty].sum())
    micro = float(tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1))
    den = tp + fp + fn
    posm = den > 0
    tile_iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    macro_pos = float(np.nanmean(tile_iou[pos > 0]))
    return {"fp_empty": fp_empty, "micro_iou": round(micro, 6),
            "macro_pos_iou": round(macro_pos, 6)}


def match_threshold(probs_u8, ys, pos, target_fp_empty):
    """빈 타일 FP 합이 target 이하가 되는 최소 임계값(uint8)을 이분 탐색."""
    lo, hi = 0, 255
    while lo < hi:
        mid = (lo + hi) // 2
        s = stats_at(probs_u8, ys, pos, mid)
        if s["fp_empty"] > target_fp_empty:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    sA, pA = load("P2")
    sB, pB = load("P4c")
    assert sA == sB, "sample 순서 불일치"
    ys, pos = masks_for(sA)
    pA = np.asarray(pA); pB = np.asarray(pB)
    base = 128  # p>0.5 == u8>=128

    a05, b05 = stats_at(pA, ys, pos, base), stats_at(pB, ys, pos, base)
    res = {"schema": "threshold-matched-compare-v1",
           "evidence_status": "development_only_not_confirmatory",
           "at_0.5": {"P2": a05, "P4c": b05}, "directions": {}}

    # 방향 1: P2를 P4c의 빈타일 FP율로
    tA = match_threshold(pA, ys, pos, b05["fp_empty"])
    aM = stats_at(pA, ys, pos, tA)
    res["directions"]["P2_matched_to_P4c_fp"] = {
        "p2_threshold_u8": int(tA), "p2_threshold_p": round(tA / 255, 4),
        "P2": aM, "P4c_at_0.5": b05,
        "p2_macro_still_higher": bool(aM["macro_pos_iou"] > b05["macro_pos_iou"]),
        "p2_micro_now_higher": bool(aM["micro_iou"] > b05["micro_iou"])}
    # 방향 2: P4c를 P2의 빈타일 FP율로
    tB = match_threshold(pB, ys, pos, a05["fp_empty"])
    bM = stats_at(pB, ys, pos, tB)
    res["directions"]["P4c_matched_to_P2_fp"] = {
        "p4c_threshold_u8": int(tB), "p4c_threshold_p": round(tB / 255, 4),
        "P4c": bM, "P2_at_0.5": a05,
        "p4c_macro_now_higher": bool(bM["macro_pos_iou"] > a05["macro_pos_iou"]),
        "p4c_micro_still_higher": bool(bM["micro_iou"] > a05["micro_iou"])}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
