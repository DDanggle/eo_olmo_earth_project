#!/usr/bin/env bash
# chain 3: after v1.1 pairs -> v1.1 bridge screen (8 folds, R0..R5); after Clay in256 cache + BV1_CHAIN2_DONE -> decoder runs for clay_cache_in256 (8 folds, seed 1).
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/bv1_chain.log; PY=./.venv-master/bin/python
for i in $(seq 1 300); do grep -q "CACHE DONE" logs/task2_cache_v11.log 2>/dev/null && break; sleep 30; done
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/release_bridge_screen.py --new-cache task2_cache_v11 --r5 --folds task2_fold0,task2_fold1,task2_fold2,task2_fold3,task2_fold4,task2_fold5,task2_fold6,task2_fold7 --out artifacts/release_migration/v11_8fold > logs/release_v11.log 2>&1; echo "$(date -u +%FT%TZ) v11 screen rc=$?" >> $LOG
for i in $(seq 1 600); do grep -q "CLAY CACHE DONE" logs/clay_cache_in256.log 2>/dev/null && grep -q BV1_CHAIN2_DONE $LOG && break; sleep 60; done
for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
  out="bv1_runs/clay_cache_in256"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache clay_cache_in256 --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_clayin256_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) clay_cache_in256 $fold rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) BV1_CHAIN3_DONE" >> $LOG
