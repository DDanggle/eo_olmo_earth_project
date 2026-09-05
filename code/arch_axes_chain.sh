#!/usr/bin/env bash
# addendum_v1b architecture axes: 7 caches -> decoders (8 folds, seed 1). GPU1.
set -uo pipefail; cd /home/work/data/olmoearth; PY=./.venv-master/bin/python; LOG=logs/arch_axes.log; run(){ env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY "$@"; }
run code/extract_olmo_variants.py --size nano --out olmo_nano > logs/x_olmo_nano.log 2>&1; echo "$(date -u +%FT%TZ) olmo_nano rc=$?" >> $LOG
run code/extract_olmo_variants.py --size tiny --out olmo_tiny > logs/x_olmo_tiny.log 2>&1; echo "$(date -u +%FT%TZ) olmo_tiny rc=$?" >> $LOG
run code/extract_olmo_variants.py --size base --depth-frac 0.5 --out olmo_base_half > logs/x_olmo_base_half.log 2>&1; echo "$(date -u +%FT%TZ) olmo_base_half rc=$?" >> $LOG
run code/extract_galileo_cache.py --size nano --out galileo_nano > logs/x_galileo_nano.log 2>&1; echo "$(date -u +%FT%TZ) galileo_nano rc=$?" >> $LOG
run code/extract_galileo_cache.py --size tiny --out galileo_tiny > logs/x_galileo_tiny.log 2>&1; echo "$(date -u +%FT%TZ) galileo_tiny rc=$?" >> $LOG
run code/extract_clay_cache.py --grid in256 --temporal mean --depth-frac 0.5 --out clay_in256_half > logs/x_clay_in256_half.log 2>&1; echo "$(date -u +%FT%TZ) clay_in256_half rc=$?" >> $LOG
run code/extract_galileo_cache.py --size base --exit-after 6 --out galileo_base_half > logs/x_galileo_base_half.log 2>&1; echo "$(date -u +%FT%TZ) galileo_base_half rc=$?" >> $LOG
for cache in olmo_nano olmo_tiny olmo_base_half galileo_nano galileo_tiny clay_in256_half galileo_base_half; do
  for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
    out="bv1_runs/$cache"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
    run code/cache_decoder_train.py --cache "$cache" --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_${cache}_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) $cache $fold rc=$?" >> $LOG
  done
done
echo "$(date -u +%FT%TZ) ARCH_AXES_DONE" >> $LOG
