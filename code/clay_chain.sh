#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth; LOG=logs/clay_chain.log
ready=0
for i in $(seq 1 240); do
  if grep -q "CLAY CACHE DONE" logs/clay_cache.log 2>/dev/null; then ready=1; break; fi
  sleep 30
done
[[ "$ready" -eq 1 ]] || { echo "$(date -u +%FT%TZ) clay cache marker timeout" >> "$LOG"; exit 4; }
python3 -c "import json,sys; a=json.load(open('clay_cache/cache_audit.json')); print(a); sys.exit(0 if a['all_gates_pass'] else 1)" >> $LOG 2>&1 || { echo "$(date -u +%FT%TZ) clay cache audit FAILED" >> $LOG; exit 1; }
gpu_ready=0
for i in $(seq 1 144); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  if [[ "$free" -ge 12000 ]]; then gpu_ready=1; break; fi
  sleep 600
done
[[ "$gpu_ready" -eq 1 ]] || { echo "$(date -u +%FT%TZ) GPU1 availability timeout" >> "$LOG"; exit 5; }
if env -u PYTHONPATH bash code/run_clay_source.sh > logs/clay_source.log 2>&1; then
  echo "$(date -u +%FT%TZ) clay source rc=0" >> "$LOG"
else
  rc=$?; echo "$(date -u +%FT%TZ) clay source rc=$rc" >> "$LOG"; exit "$rc"
fi
count=$(find clay_source_v1 -name P4_best.pt -type f | wc -l | tr -d ' ')
[[ "$count" -eq 24 ]] || { echo "$(date -u +%FT%TZ) source checkpoint count=$count expected=24" >> "$LOG"; exit 6; }
echo "$(date -u +%FT%TZ) CLAY_SOURCE_DONE" >> $LOG
