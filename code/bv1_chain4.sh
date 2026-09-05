#!/usr/bin/env bash
# chain 4: Galileo group-concat readout (3840x32x32) -> 8 folds seed 1. Registered readout sensitivity.
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/bv1_chain.log; PY=./.venv-master/bin/python
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_galileo_cache.py --size base --patch 4 --readout groupcat --out galileo_cache_groupcat > logs/galileo_cache_groupcat.log 2>&1; echo "$(date -u +%FT%TZ) galileo groupcat extract rc=$?" >> $LOG
for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
  out="bv1_runs/galileo_cache_groupcat"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache galileo_cache_groupcat --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_galgc_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) galileo_cache_groupcat $fold rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) BV1_CHAIN4_DONE" >> $LOG
