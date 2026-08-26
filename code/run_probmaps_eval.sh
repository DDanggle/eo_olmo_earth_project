#!/usr/bin/env bash
# E5b/E5c — 봉인된 seed1 체크포인트에서 확률맵을 소급 생성 (재학습 없음).
# 내장 검증: eval-only가 재현한 test 지표가 봉인값과 다르면 프로토콜 위반이다.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani

$PY code/pilot_sen12_gp_heads.py --arms P2 --save-probs \
  --eval-only-ckpt /home/work/data/olmoearth/sen12_gp_official/checkpoints/holdout_chimanimani/P2_best.pt \
  --cache "$TILED" --out /home/work/data/olmoearth/probmaps_eval/P2

$PY code/pilot_sen12_gp_heads.py --arms P4c --save-probs \
  --eval-only-ckpt /home/work/data/olmoearth/e1_tiled_big/checkpoints/holdout_chimanimani/P4c_best.pt \
  --cache "$TILED" --out /home/work/data/olmoearth/probmaps_eval/P4c

echo "PROBMAPS DONE"
