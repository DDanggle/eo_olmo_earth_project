#!/usr/bin/env python3
"""ep33 체크포인트의 인코더가 HF 원본과 다른지 확인 — frozen-head 실험 설계의 전제 검증.

frozen_head_release_swap.py의 타당성 게이트가 실패했다(v1 병합 1006.91 vs 원본 558.79).
가설: lfmc 레시피가 `FreezeUnfreeze(unfreeze_at_epoch=20)`로 백본을 해동하므로 ep33의 인코더는
HF 원본이 아니라 **미세조정된** 가중치다. 병합이 그것을 원본으로 덮어써 성능이 떨어진 것.

이 스크립트는 ep33 인코더 텐서와 갓 구성한 v1 모델(HF 원본 로드) 인코더 텐서를 비교한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = "/home/work/data/olmoearth/frozen_head_swap"
EP33 = "/home/work/data/olmoearth/scratch/lfmc/trainer_checkpoints/epoch=33-step=22270.ckpt"
FRESH = f"{OUT}/model_v1.sd.pt"  # frozen_head_release_swap.py가 저장한 원본-로드 state_dict

trained = torch.load(EP33, map_location="cpu")["state_dict"]
fresh = torch.load(FRESH, map_location="cpu")

rows, n_diff, n_same = [], 0, 0
for key, ref in fresh.items():
    if not key.startswith("model.encoder."):
        continue
    got = trained.get(key)
    if got is None or got.shape != ref.shape:
        rows.append({"key": key, "status": "missing_or_shape_mismatch"})
        continue
    a, b = got.float(), ref.float()
    denom = b.norm().item() or 1.0
    rel = (a - b).norm().item() / denom
    identical = torch.equal(got, ref)
    n_same += int(identical)
    n_diff += int(not identical)
    rows.append({"key": key, "identical": identical, "rel_l2_diff": round(rel, 6)})

diffs = sorted((r for r in rows if "rel_l2_diff" in r), key=lambda r: -r["rel_l2_diff"])
result = {
    "schema": "encoder-finetune-diagnosis-v1",
    "encoder_tensors": len(rows),
    "identical_to_hf_release": n_same,
    "differ_from_hf_release": n_diff,
    "verdict": (
        "encoder WAS fine-tuned — merging pristine release weights discards it"
        if n_diff > 0
        else "encoder is pristine — merge should have been lossless"
    ),
    "largest_relative_differences": diffs[:8],
    "smallest_relative_differences": diffs[-4:],
}
Path(OUT, "encoder_finetune_diagnosis.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
