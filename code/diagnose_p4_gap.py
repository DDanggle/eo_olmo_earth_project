#!/usr/bin/env python3
"""E2 — 학습 없이 P4 열세의 원인 후보를 진단한다. GPU 불필요.

두 가지를 잰다.

A. **블록-상수 라벨 oracle 참고값 (block-constant label oracle)**
   P4의 캐시는 32x32 토큰(= 40 m)이다. 128x128(10 m) 라벨을 40 m 격자로 내렸다가
   블록마다 하나의 이진값으로 다시 올리면 IoU가 얼마나 남는가? 이는 라벨만 본
   **비배포 가능 geometric reference**다. 학습 decoder는 토큰에서 4x4 비상수 출력을
   만들 수 있으므로 이 값은 모델의 성능 상한이 아니다. 따라서 이 진단만으로 실제
   embedding이 세부 정보를 보존하는지, 또는 40 m support가 병목인지 판정할 수 없다.

B. **산사태 면적별 P2-P4 격차**
   per-sample tp/fp/fn으로 샘플별 IoU를 내고, 라벨 면적 구간별로 두 방식을 비교한다.
   이는 면적 효과만 보며 폭·가늘기·경계복잡도 효과는 판정하지 않는다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

MASK_DIR = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/mask_u8")
BUNDLE = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/per_sample")
PATCH = 4          # 128 / 32
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/p4_gap_diagnosis_v2.json")


def iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else None


def main() -> None:
    # ── A. 블록-상수 라벨 oracle 참고값 ──
    test_ids = [json.loads(l)["sample_id"]
                for l in (BUNDLE / "P2_test.jsonl").read_text().splitlines() if l]
    references = {}
    # k/16 이상이 양성이면 블록 전체를 양성으로 복원한다. k=17은 전부 음성이다.
    # 라벨을 사용해 k를 고르므로 best_k는 배포 가능한 모델이 아니라 oracle 참고값이다.
    for rule, min_positive_pixels in (("any_positive", 1), ("majority", 9)):
        TP = FP = FN = 0
        for sid in test_ids:
            m = np.load(MASK_DIR / f"{sid}.npy") > 0
            # 4x4 블록 평균 -> 규칙 적용 -> 최근접 복원
            blocks = m.reshape(32, PATCH, 32, PATCH).mean(axis=(1, 3))
            coarse = (blocks * (PATCH * PATCH)) >= min_positive_pixels
            recon = np.repeat(np.repeat(coarse, PATCH, axis=0), PATCH, axis=1)
            TP += int((recon & m).sum()); FP += int((recon & ~m).sum())
            FN += int((~recon & m).sum())
        references[rule] = {
            "min_positive_pixels_in_4x4": min_positive_pixels,
            "tp": TP, "fp": FP, "fn": FN,
            "iou_reference": round(iou(TP, FP, FN), 6),
        }

    threshold_sweep = []
    for k in range(1, PATCH * PATCH + 2):
        TP = FP = FN = 0
        for sid in test_ids:
            m = np.load(MASK_DIR / f"{sid}.npy") > 0
            counts = m.reshape(32, PATCH, 32, PATCH).sum(axis=(1, 3))
            coarse = counts >= k
            recon = np.repeat(np.repeat(coarse, PATCH, axis=0), PATCH, axis=1)
            TP += int((recon & m).sum()); FP += int((recon & ~m).sum())
            FN += int((~recon & m).sum())
        threshold_sweep.append({
            "min_positive_pixels_in_4x4": k,
            "tp": TP, "fp": FP, "fn": FN,
            "iou_reference": round(iou(TP, FP, FN), 6),
        })
    best_block_constant = max(threshold_sweep, key=lambda x: x["iou_reference"])

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
        "schema": "p4-gap-diagnosis-v2",
        "n_test_tiles": len(test_ids),
        "n_positive_tiles": n,
        "block_constant_label_oracle": {
            "note": (
                "라벨을 4x4 블록마다 단일 이진값으로 복원한 비배포 가능 geometric reference. "
                "학습 decoder의 성능 상한이 아니며 embedding의 정보 보존을 증명하지 않는다."
            ),
            "compare_P2_actual_iou": 0.159254,
            "compare_P4_actual_iou": 0.130582,
            "fixed_rules": references,
            "best_threshold_oracle": best_block_constant,
            "threshold_sweep": threshold_sweep,
        },
        "size_stratified_gap": size_gap,
        "limitations": [
            "면적 구간만 사용해 객체 폭·가늘기·경계복잡도를 판정하지 못한다.",
            "라벨 oracle은 실제 embedding이 해당 정보를 보존한다는 증거가 아니다.",
            "test를 사용한 사후 진단이므로 후속 모델 선택은 exploratory다.",
        ],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
