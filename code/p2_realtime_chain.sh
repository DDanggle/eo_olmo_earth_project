#!/usr/bin/env bash
# Re-extract the 20 m (patch-2) cache with the sealed contract's REAL acquisition dates (the running olmo_base_p2 used synthetic month dates:
# cos .989 vs sealed on a probe tile), then the same 8-fold seed-1 decoders (native + avgpool2) so the resolution comparison is timestamp-matched.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/p2_rt_chain.log"; }
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')
foreign(){ local p f=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}'); do ps -o cmd= -p "$p" | grep -Eq "extract_olmo_|temporal_readout_train|cache_decoder_train|extract_sen12_fold_cache|streaming_update_train" || f=1; done; [[ $f -eq 1 ]]; }
wait_free(){ while foreign; do log "foreign GPU1 use, wait"; sleep 120; done; }
log start; wait_free
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_variants.py --size base --patch 2 --real-times --out olmo_base_p2_rt > "$L/x_olmo_base_p2_rt.log" 2>&1; log "extraction rc=$?"
$PY -c 'import json,sys; a=json.load(open("olmo_base_p2_rt/olmo_variant_audit.json")); sys.exit(0 if a.get("all_gates_pass") else 1)' || { log "audit FAILED"; touch "$L/p2_rt_FAILED.txt"; exit 5; }
until [[ -e resolution_contract_v2/COMPLETED_AT_UTC.txt ]]; do sleep 300; done   # do not compete with the sealed 4-arm decoders
for tr in native avgpool2; do for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
  [[ -f artifacts/resolution_rt/p2_$tr/holdout_${fold}_seed1.json ]] && continue; wait_free
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache olmo_base_p2_rt --grid-transform $tr --fold holdout_$fold --seed 1 --out artifacts/resolution_rt/p2_$tr > "$L/p2rt_${tr}_${fold}.log" 2>&1; log "p2rt $tr $fold rc=$?"
done; done; log "P2_RT DONE"; touch "$L/p2_rt_DONE.txt"
