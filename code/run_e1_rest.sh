#!/usr/bin/env bash
# E1 요인설계 나머지 두 칸. 칸2(타일+큰)는 이미 끝났고, 칸3에서 캐시 봉인 게이트에
# 걸려 멈췄었다. audit_sen12_fold_cache.py를 통과시킨 뒤 재개한다.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
FULL=/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani

echo "=== 칸3: 통짜 캐시 + 작은 decoder (문맥 효과) ==="
$PY code/pilot_sen12_gp_heads.py --arms P4 --epochs 40 \
    --cache "$FULL" --out /home/work/data/olmoearth/e1_full_small

echo "=== 칸4: 통짜 캐시 + 큰 decoder (문맥 + 용량) ==="
$PY code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 \
    --cache "$FULL" --out /home/work/data/olmoearth/e1_full_big

echo "E1 FACTORIAL DONE"
