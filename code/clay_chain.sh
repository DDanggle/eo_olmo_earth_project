#!/usr/bin/env bash
cd /home/work/data/olmoearth; LOG=logs/clay_chain.log
for i in $(seq 1 240); do grep -q "CLAY CACHE DONE" logs/clay_cache.log 2>/dev/null && break; sleep 30; done
python3 -c "import json,sys; a=json.load(open('clay_cache/cache_audit.json')); print(a); sys.exit(0 if a['all_gates_pass'] else 1)" >> $LOG 2>&1 || { echo "$(date -u +%FT%TZ) clay cache audit FAILED" >> $LOG; exit 1; }
for i in $(seq 1 144); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1); [ "$free" -ge 12000 ] && break; sleep 600; done
env -u PYTHONPATH bash code/run_clay_source.sh > logs/clay_source.log 2>&1; echo "$(date -u +%FT%TZ) clay source rc=$?" >> $LOG
echo "$(date -u +%FT%TZ) CLAY_SOURCE_DONE" >> $LOG
