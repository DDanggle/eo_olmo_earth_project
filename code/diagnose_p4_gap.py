#!/usr/bin/env python3
"""E2 — 학습 없이 P4 열세의 원인을 분해한다. GPU 불필요.

두 가지를 잰다.

A. **토큰 격자 천장 (token-grid mask oracle)**
   P4의 캐시는 32x32 토큰(= 40 m)이다. 128x128(10 m) 라벨을 40 m 격자로 내렸다가
   다시 올리면 IoU가 얼마나 남는가? 이게 **어떤 decoder를 붙여도 넘을 수 없는 상한**이다.
   이 상한이 P2의 0.1593보다 낮으면 원인은 decoder가 아니라 **캐시 해상도**다.

B. **산사태 크기별 P2-P4 격차**
   per-sample tp/fp/fn으로 샘플별 IoU를 내고, 라벨 면적 구간별로 두 방식을 비교한다.
   P4가 **작고 가는 산사태에서만** 크게 진다면 high-frequency residual의 직접 근거가 된다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

MASK_DIR = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/mask_u8")
BUNDLE = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/per_sample")
PATCH = 4          # 128 / 32
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/p4_gap_diagnosis.json")


def iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else None


def main() -> None:
    # ── A. 토큰 격자 천장 ──
    test_ids = [json.loads(l)["sample_id"]
                for l in (BUNDLE / "P2_test.jsonl").read_text().splitlines() if l]
    ceilings = {}
    for rule, thr in (("any_positive", 0.0), ("majority", 0.5)):
        TP = FP = FN = 0
        for sid in test_ids:
            m = np.load(MASK_DIR / f"{sid}.npy") > 0
            # 4x4 블록 평균 -> 규칙 적용 -> 최근접 복원
            blocks = m.reshape(32, PATCH, 32, PATCH).mean(axis=(1, 3))
            coarse = blocks > thr
            recon = np.repeat(np.repeat(coarse, PATCH, axis=0), PATCH, axis=1)
            TP += int((recon & m).sum()); FP += int((recon & ~m).sum())
            FN += int((~recon & m).sum())
        ceilings[rule] = {"tp": TP, "fp": FP, "fn": FN, "iou_ceiling": round(iou(TP, FP, FN), 6)}

    # ── B. 크기별 격차 ──
    per_arm = {}
    for arm in ("P1", "P2", "P3", "P4"):
        rows = [json.loads(l) for l in
                (BUNDLE / f"{arm}_test.jsonl").read_text().splitlines() if l]
        per_arm[arm] = {r["sample_id"]: r for r in rows}

    pos = [(sid, per_arm["P2"][sid]["mask_positive_pixels"]) for sid in test_ids
           if per_arm["P2"][sid]["mask_positive_pixels"] > 0]
    pos.sort(key=lambda x: x[1])
    n = len(pos)
    bins = [("아주 작음", pos[:n // 4]), ("작음", pos[n // 4:n // 2]),
            ("중간", pos[n // 2:3 * n // 4]), ("큼", pos[3 * n // 4:])]
    size_gap = []
    for name, group in bins:
        row = {"bin": name, "n_tiles": len(group),
               "positive_px_range": [group[0][1], group[-1][1]]}
        for arm in ("P2", "P4"):
            TP = sum(per_arm[arm][s]["tp"] for s, _ in group)
            FP = sum(per_arm[arm][s]["fp"] for s, _ in group)
            FN = sum(per_arm[arm][s]["fn"] for s, _ in group)
            row[f"{arm}_iou"] = round(iou(TP, FP, FN), 6)
        row["gap_P2_minus_P4"] = round(row["P2_iou"] - row["P4_iou"], 6)
        size_gap.append(row)

    out = {
        "schema": "p4-gap-diagnosis-v1",
        "n_test_tiles": len(test_ids),
        "n_positive_tiles": n,
        "token_grid_ceiling": {
            "note": "캐시 토큰 32x32 (=40m). 어떤 decoder도 이 상한을 넘을 수 없다.",
            "compare_P2_actual_iou": 0.159254,
            "compare_P4_actual_iou": 0.130582,
            **ceilings},
        "size_stratified_gap": size_gap,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
