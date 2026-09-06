#!/usr/bin/env bash
# field-adaptation development: after the running Sen12 dev finishes -> Sen12 N arms -> Solar dev (all arms). GPU1.
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/field_dev_chain.log; PY=./.venv-master/bin/python
run(){ env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY code/fewshot_a1_a4.py "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
for i in $(seq 1 400); do pgrep -f "fewshot_a1_a4.py --arms A0,A0T,A1,A1T,A1R,A1TR,A1P" >/dev/null || break; sleep 30; done
run --arms A0,A1,A1N,A1NR --support stratified --exposure fixed_update --out artifacts/field_adapt/dev_sen12_n > logs/field_dev_sen12_n.log 2>&1
run --task2 --arms A0,A0T,A1,A1T,A1R,A1TR,A1P,A1N,A1NR --support stratified --exposure fixed_update --out artifacts/field_adapt/dev_solar > logs/field_dev_solar.log 2>&1
echo "$(date -u +%FT%TZ) FIELD_DEV_DONE" >> $LOG
