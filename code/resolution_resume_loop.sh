#!/usr/bin/env bash
# Re-run resolution_chain.sh --resume whenever it self-stops because GPU1 was momentarily occupied (exit 4).
# Never runs alongside another user: waits until GPU1 has no compute apps, then resumes. Stops on completion or any other exit code.
ROOT=/home/work/data/olmoearth; cd "$ROOT"; OUT="$ROOT/resolution_contract_v2"; LOG="$OUT/logs/resume_loop.log"
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F", " "\$1==1 {print \$2}")
for i in $(seq 1 200); do
  [[ -e "$OUT/COMPLETED_AT_UTC.txt" ]] && { echo "$(date -u +%FT%TZ) completed" >> "$LOG"; exit 0; }
  foreign=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}'); do ps -o cmd= -p "$p" | grep -Eq "extract_olmo_|temporal_readout_train|cache_decoder_train|extract_sen12_fold_cache|streaming_update_train" || foreign=1; done
  if [[ $foreign -eq 1 ]]; then echo "$(date -u +%FT%TZ) GPU1 foreign busy, wait 120s" >> "$LOG"; sleep 120; continue; fi
  echo "$(date -u +%FT%TZ) resume attempt $i" >> "$LOG"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 bash code/resolution_chain.sh --resume >> "$OUT/logs/runner_resume_loop.log" 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) chain exited rc=$rc" >> "$LOG"
  [[ $rc -eq 0 ]] && exit 0
  [[ $rc -eq 4 ]] || { echo "non-occupancy failure rc=$rc; stopping loop" >> "$LOG"; exit $rc; }
  sleep 120
done
