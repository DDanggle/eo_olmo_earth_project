#!/usr/bin/env bash
# 다음 순서 1 — 비교 상대의 seed 재현 폭.
# M41에서 P4c(frozen+큰)만 seed 3회를 쟀고 폭 0.0332가 나왔다. 그런데 M30/M33의
# 모든 arm 간 비교는 여전히 단일 seed 대 단일 seed다. P2가 같은 폭으로 흔들리면
# "P2 > P4, CI 0 제외"(M33)의 해석이 다시 열린다.
# 그래서 격차의 양쪽(P2와 P4)을 모두 seed 2·3으로 반복한다.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
for S in 2 3; do
  for ARM in P2 P4; do
    echo "=== $ARM seed $S ==="
    $PY code/pilot_sen12_gp_heads.py --arms "$ARM" --epochs 40 --seed "$S" \
        --cache "$TILED" --out "/home/work/data/olmoearth/seed_spread/${ARM}_seed${S}"
  done
done
echo "SEED SPREAD DONE"
