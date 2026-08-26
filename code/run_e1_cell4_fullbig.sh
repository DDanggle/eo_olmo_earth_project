#!/usr/bin/env bash
# E1 칸4만 — 통짜 캐시 + 큰 decoder. 파일명을 고유하게 둔다(다른 세션이 같은 이름의
# run_e1_factorial.sh를 덮어쓰고 있어 충돌을 피한다).
set -euo pipefail
cd /home/work/data
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python \
  code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 \
  --cache /home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani \
  --out /home/work/data/olmoearth/e1_full_big
echo "CELL4 DONE"
