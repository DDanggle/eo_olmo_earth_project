#!/usr/bin/env bash
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/t1_chain.log"; }
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')
foreign(){ local p f=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}'); do ps -o cmd= -p "$p" | grep -Eq "extract_olmo_|temporal_readout_train|cache_decoder_train|extract_sen12_fold_cache|streaming_update_train" || f=1; done; [[ $f -eq 1 ]]; }
wait_free(){ while foreign; do log "foreign GPU1 use, wait"; sleep 120; done; }
log start; $PY code/build_t1_manifest.py | tee -a "$L/t1_chain.log"
wait_free; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_streaming.py --out olmo_streaming_dev --ids-file sen12_gp_contract/t1_ids.txt > "$L/x_olmo_streaming.log" 2>&1; log "extraction rc=$?"
$PY -c 'import json,sys; a=json.load(open("olmo_streaming_dev/olmo_streaming_audit.json")); sys.exit(0 if a["all_gates_pass"] and (a["audit_max"] or 1)<0.05 else 1)' || { log "streaming audit FAILED"; touch "$L/t1_FAILED.txt"; exit 5; }
until [[ -f resolution_contract_v2/p4_native_control/holdout_hiroshima_seed1_best.pt && -f resolution_contract_v2/p4_native_control/holdout_chimanimani_seed1_best.pt ]]; do sleep 60; done
for fold in holdout_chimanimani holdout_hiroshima; do for mod in ema gru residual; do for s in 1 2 3; do
  [[ -f artifacts/streaming_t1/${fold}_${mod}_seed${s}.json ]] && continue; wait_free
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/streaming_update_train.py --fold $fold --module $mod --seed $s --out artifacts/streaming_t1 > "$L/t1_${fold}_${mod}_s${s}.log" 2>&1; log "t1 $fold $mod s$s rc=$?"
done; done; done
log "T1 DONE"; touch "$L/t1_DONE.txt"
