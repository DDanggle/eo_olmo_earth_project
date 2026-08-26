#!/usr/bin/env bash
# M43 관문 1 — P2의 seed 분산(0.0760)이 튜닝 인공물인지 검정한다.
#
# 사전 등록 (실행 전 고정):
#   처치      가장 표준적인 안정화 2개를 동시 적용 — grad clip 1.0 + LR 절반(5e-4).
#             다른 축(pos_weight, scheduler, warmup)은 건드리지 않는다.
#   측정      seed 1·2·3의 test micro IoU 폭 S_stab 과 3-seed 평균.
#   판정      S_stab <= 0.021 (P4 폭의 2배) 이고 평균 >= 0.1334 (P4 평균)
#               -> "raw 불안정은 튜닝 인공물" 확정. 안정성 주장 사망.
#             S_stab >= 0.031 (P4 폭의 3배)
#               -> 표준 안정화로 안 고쳐짐. 주장 생존(단 '이 프로토콜군에서'로 한정).
#             그 사이 -> 미확정. seed 5개로 확장 후 재판정.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
for S in 1 2 3; do
  echo "=== P2-stab seed $S (lr 5e-4, clip 1.0) ==="
  $PY code/pilot_sen12_gp_heads.py --arms P2 --epochs 40 --seed "$S" \
      --lr 5e-4 --grad-clip 1.0 --save-probs \
      --cache "$TILED" --out "/home/work/data/olmoearth/p2_stab/seed$S"
done
echo "P2 STAB DONE"
