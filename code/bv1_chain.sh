#!/usr/bin/env bash
# B-v1 diagnostics chain (GPU1): Galileo-base cache extraction -> cache-decoder training (pilot recipe re-impl) for 4 caches x 8 confirmatory folds x seed 1.
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/bv1_chain.log; PY=./.venv-master/bin/python
for i in $(seq 1 200); do grep -q "^DONE" logs/bv1_calib.log 2>/dev/null && break; sleep 30; done
for i in $(seq 1 200); do grep -q CLAY_NATIVE_CACHES_DONE logs/clay_native_chain.log 2>/dev/null && break; sleep 30; done
for cache in olmo_cache_pool16 clay_cache_native16 clay_cache_native16_last galileo_cache; do
  [[ $cache == galileo_cache ]] && for i in $(seq 1 2000); do grep -q "GALILEO CACHE DONE" logs/galileo_cache.log 2>/dev/null && break; sleep 60; done
  for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
    out="bv1_runs/$cache"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache "$cache" --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_${cache}_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) $cache $fold rc=$?" >> $LOG
  done
done
echo "$(date -u +%FT%TZ) BV1_CHAIN_DONE" >> $LOG
