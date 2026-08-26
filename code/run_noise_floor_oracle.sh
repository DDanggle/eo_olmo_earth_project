#!/usr/bin/env bash
# 통제 1 — 잡음 바닥 oracle.
# 같은 구성(P4c · tiled 캐시 · 큰 decoder)을 **seed만 바꿔** 두 번 돌린다.
# 두 실행 사이의 per-tile oracle gain은 표현/모델 차이가 아니라 순수 선택 잡음이다.
# M40의 +0.069403이 이 바닥을 넘지 못하면 routing 여유 주장은 죽는다.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
for S in 2 3; do
  echo "=== seed $S ==="
  $PY code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 --seed "$S" \
      --cache "$TILED" --out "/home/work/data/olmoearth/noise_floor/seed$S"
done
echo "NOISE FLOOR DONE"
