#!/usr/bin/env bash
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/field_dev_chain.log; PY=./.venv-master/bin/python
run(){ env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY code/fewshot_a1_a4.py "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
for i in $(seq 1 1500); do grep -q FIELD_DEV_DONE $LOG 2>/dev/null && break; sleep 60; done
run --arms A0,A1,A1R2,A1NR2 --support stratified --exposure fixed_update --out artifacts/field_adapt/dev_sen12_r2 > logs/field_dev_sen12_r2.log 2>&1
run --task2 --arms A0,A1,A1R2,A1NR2 --support stratified --exposure fixed_update --out artifacts/field_adapt/dev_solar_r2 > logs/field_dev_solar_r2.log 2>&1
echo "$(date -u +%FT%TZ) FIELD_DEV2_DONE" >> $LOG
