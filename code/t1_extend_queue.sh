#!/usr/bin/env bash
# 7-hour GPU0 queue (2026-09-08 01:40 KST): (a) control decoders seeds 2/3 for the two dev folds; (b) T1 GRU/noobs/EMA with checkpoints on dev folds (t1v);
# (c) extend T1 to holdout_thrissur and holdout_newzealand: manifest -> streaming extraction -> t1v; (d) re-evaluate every saved GRU under decoder seeds 1-3.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/t1_extend.log"; }
log start
setsid nohup bash code/run_locked.sh 0 ctrl holdout_hiroshima holdout_chimanimani holdout_thrissur holdout_newzealand > /dev/null 2>&1 &
$PY code/build_t1_manifest.py holdout_thrissur holdout_newzealand | tee -a "$L/t1_extend.log"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/extract_olmo_streaming.py --out olmo_streaming_dev --ids-file sen12_gp_contract/t1_ids_new.txt > "$L/x_olmo_streaming_ext.log" 2>&1; log "ext extraction rc=$?"
setsid nohup bash code/run_locked.sh 0 t1v holdout_hiroshima holdout_chimanimani holdout_thrissur holdout_newzealand > /dev/null 2>&1 &
setsid nohup bash code/run_locked.sh 1 t1v holdout_thrissur holdout_newzealand holdout_hiroshima holdout_chimanimani > /dev/null 2>&1 &
until [[ -f artifacts/control_seeds/holdout_newzealand_seed3.json ]]; do sleep 300; done
while true; do for f in holdout_hiroshima holdout_chimanimani holdout_thrissur holdout_newzealand; do for s in 1 2 3; do ck=artifacts/streaming_t1v/${f}_gru_seed${s}_best.pt; [[ -f $ck && ! -f artifacts/streaming_t1_reeval/${f}_${f}_gru_seed${s}_best.json ]] && env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/t1_reeval_decoders.py $f $ck >> "$L/t1_reeval.log" 2>&1; done; done
  n=$(ls artifacts/streaming_t1_reeval 2>/dev/null | wc -l); [[ $n -ge 12 ]] && break; sleep 600; done; log "REEVAL DONE"; touch "$L/t1_extend_DONE.txt"
