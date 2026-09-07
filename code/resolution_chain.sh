#!/usr/bin/env bash
# cache-contract lever: OlmoEarth base patch 2 (20 m tokens, 64x64) on Sen12 -> decoders 8 folds seed 1 (then seeds 2,3 if registered rule passes is decided later, not here).
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/resolution_chain.log; PY=./.venv-master/bin/python
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_variants.py --size base --patch 2 --out olmo_base_p2 > logs/x_olmo_base_p2.log 2>&1; echo "$(date -u +%FT%TZ) extract p2 rc=$?" >> $LOG
for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
  out="bv1_runs/olmo_base_p2"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache olmo_base_p2 --fold "holdout_$fold" --seed 1 --out "$out" > "logs/bv1_olmo_base_p2_${fold}.log" 2>&1; echo "$(date -u +%FT%TZ) olmo_base_p2 $fold rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) RESOLUTION_DONE" >> $LOG
