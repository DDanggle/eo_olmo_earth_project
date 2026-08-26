#!/usr/bin/env bash
# 확증 실행 — 동결된 recipe(evidence/recipe_frozen_v1.json)로 미열람 지역 하나를 연다.
# 사용법: bash code/run_confirmatory_region.sh <fold_name>
#
# 동결 내용 (변경 금지):
#   arms   P4(reuse) · P2(raw_strong) · P3(raw_efficient)
#   seeds  1 2 3 · epochs 40 · lr 1e-3 · clip 없음 · threshold 0.5
#   주지표 양성 타일 macro IoU
# 이 스크립트는 설정을 인자로 받지 않는다. 지역만 받는다.
set -euo pipefail
FOLD="${1:?fold 이름이 필요하다 (예: holdout_thrissur)}"
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
CACHE="/home/work/data/olmoearth/sen12_pilot/$FOLD"

if [ ! -d "$CACHE/emb_fp16" ]; then
  echo "=== [1/3] 캐시 추출: $FOLD ==="
  $PY code/extract_sen12_fold_cache.py --fold "$FOLD" \
      --out /home/work/data/olmoearth/sen12_pilot
fi

echo "=== [2/3] 캐시 감사 ==="
$PY code/audit_sen12_fold_cache.py --cache "$CACHE" --fold "$FOLD" | tail -20

echo "=== [3/3] 3 arm x 3 seed ==="
for S in 1 2 3; do
  for ARM in P4 P2 P3; do
    echo "--- $FOLD / $ARM / seed $S ---"
    $PY code/pilot_sen12_gp_heads.py --arms "$ARM" --epochs 40 --seed "$S" \
        --save-probs --fold "$FOLD" --cache "$CACHE" \
        --out "/home/work/data/olmoearth/confirmatory/$FOLD/${ARM}_seed${S}"
  done
done
echo "CONFIRMATORY $FOLD DONE"
