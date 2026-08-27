#!/usr/bin/env bash
# 남은 확증 지역의 캐시를 미리 추출함 — hokkaido 학습과 병행.
# 캐시 추출은 GPU ~1.2GB·결정적이며, runner는 캐시가 있으면 건너뛰고 감사만 수행함.
# 실행 경로 코드는 건드리지 않음(규칙 4c) — live extract 스크립트는 hokkaido snapshot과
# 해시가 같음(ebe1ee88... 등 4개 불변 확인됨).
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
for F in holdout_indonesia holdout_itogon holdout_kyrgyzstan1 holdout_kyrgyzstan2 holdout_newzealand; do
  if [ ! -d "olmoearth/sen12_pilot/$F/emb_fp16" ]; then
    echo "=== prefetch $F ==="
    $PY code/extract_sen12_fold_cache.py --fold "$F" --out /home/work/data/olmoearth/sen12_pilot
  fi
done
echo "PREFETCH DONE"
