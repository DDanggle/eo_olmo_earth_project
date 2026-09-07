#!/usr/bin/env bash
# 2026-09-07 evening GPU1 queue: (1) wait for resolution_contract_v2 completion, (2) temporal-evidence cache + 24 T0 runs, (3) Italy sealed-region chain.
# Every stage refuses to start while another user's process is on GPU1 (waits, does not kill).
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; LOG="$ROOT/logs/evening_queue_0907.log"
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')
ours(){ pgrep -f "resolution_chain.sh|resolution_resume_loop|cache_decoder_train|extract_olmo|temporal_readout_train|extract_sen12_fold_cache" >/dev/null; }
foreign_busy(){ nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID" && ! ours; }
wait_free(){ while foreign_busy; do echo "$(date -u +%FT%TZ) GPU1 foreign busy, wait" >> "$LOG"; sleep 120; done; }
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "queue start"
until [[ -e resolution_contract_v2/COMPLETED_AT_UTC.txt ]]; do sleep 300; done; log "resolution chain complete"
# stage 2: temporal cache
wait_free; log "temporal extraction start"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_temporal.py --out olmo_temporal_p4 > logs/x_olmo_temporal.log 2>&1; log "temporal extraction rc=$?"
$PY -c 'import json,sys; a=json.load(open("olmo_temporal_p4/olmo_temporal_audit.json")); sys.exit(0 if a["all_gates_pass"] and (a["audit_max"] or 1)<0.05 else 1)' || { log "temporal audit FAILED (gates or mean mismatch); T0 skipped"; touch logs/temporal_T0_FAILED.txt; }
if [[ ! -e logs/temporal_T0_FAILED.txt ]]; then
  for fold in holdout_chimanimani holdout_hiroshima; do for ro in mean diffpca sketch full; do for s in 1 2 3; do
    [[ -f artifacts/temporal_t0/${fold}_${ro}_seed${s}.json ]] && continue; wait_free
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/temporal_readout_train.py --cache olmo_temporal_p4 --fold $fold --readout $ro --seed $s --out artifacts/temporal_t0 > logs/t0_${fold}_${ro}_s${s}.log 2>&1; log "t0 $fold $ro s$s rc=$?"
  done; done; done; log "T0 DONE"; touch logs/temporal_T0_DONE.txt
fi
# stage 3: Italy
wait_free; log "italy chain start"; bash code/italy_chain.sh >> logs/italy_chain.log 2>&1; log "italy rc=$?"
log "queue end"
