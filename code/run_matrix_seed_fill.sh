#!/usr/bin/env bash
# M46 후속 — action matrix의 단일 seed action 3개를 3-seed로 채운다.
# M46: seed 1개뿐인 action(raw_utae, recontext, recontext_bigdec)이 69블록 중 40개(58%)
# 에서 최적으로 뽑혔다. 운과 실력을 구분할 수 없으므로 matrix 자체가 신뢰 불가였다.
# 이 실행 후 E6를 재계산하면 블록 이질성의 진짜 크기를 판정할 수 있다.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
FULL=/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani
for S in 2 3; do
  echo "=== raw_utae (P3) seed $S ==="
  $PY code/pilot_sen12_gp_heads.py --arms P3 --epochs 40 --seed "$S" --save-probs \
      --cache "$TILED" --out "/home/work/data/olmoearth/matrix_fill/P3_seed$S"
  echo "=== recontext (P4 on full) seed $S ==="
  $PY code/pilot_sen12_gp_heads.py --arms P4 --epochs 40 --seed "$S" --save-probs \
      --cache "$FULL" --out "/home/work/data/olmoearth/matrix_fill/P4full_seed$S"
  echo "=== recontext_bigdec (P4c on full) seed $S ==="
  $PY code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 --seed "$S" --save-probs \
      --cache "$FULL" --out "/home/work/data/olmoearth/matrix_fill/P4cfull_seed$S"
done
echo "MATRIX FILL DONE"
