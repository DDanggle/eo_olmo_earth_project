#!/usr/bin/env bash
cd /home/work/data/olmoearth; LOG=logs/task2_chain.log
for i in $(seq 1 240); do grep -q "CACHE DONE" logs/task2_cache.log 2>/dev/null && break; sleep 30; done
env -u PYTHONPATH ./.venv-master/bin/python code/build_task2_pilot_contract.py > logs/task2_contract.log 2>&1; echo "$(date -u +%FT%TZ) contract rc=$?" >> $LOG
python3 -c "import json,sys; a=json.load(open('task2_cache/cache_audit.json')); sys.exit(0 if a['all_gates_pass'] else 1)" || { echo "$(date -u +%FT%TZ) cache audit FAILED — not launching source training" >> $LOG; exit 1; }
for i in $(seq 1 144); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1); [ "$free" -ge 12000 ] && break; sleep 600; done
env -u PYTHONPATH bash code/run_task2_source.sh > logs/task2_source.log 2>&1; echo "$(date -u +%FT%TZ) source rc=$?" >> $LOG
echo "$(date -u +%FT%TZ) TASK2_SOURCE_DONE" >> $LOG
