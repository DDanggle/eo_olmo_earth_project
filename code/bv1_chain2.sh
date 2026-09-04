#!/usr/bin/env bash
# B-v1 chain 2: Prithvi-EO-2.0 cache extraction now; decoder runs (8 folds, seed 1) after BV1_CHAIN_DONE.
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/bv1_chain.log; PY=./.venv-master/bin/python
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_prithvi_cache.py --out prithvi_cache > logs/prithvi_cache.log 2>&1; echo "$(date -u +%FT%TZ) prithvi extract rc=$?" >> $LOG
for i in $(seq 1 2000); do grep -q BV1_CHAIN_DONE $LOG 2>/dev/null && break; sleep 60; done
for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
  out="bv1_runs/prithvi_cache"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache prithvi_cache --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_prithvi_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) prithvi_cache $fold rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) BV1_CHAIN2_DONE" >> $LOG
