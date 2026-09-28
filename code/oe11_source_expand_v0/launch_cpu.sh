#!/usr/bin/env bash
set -euo pipefail
TASK_BASE=/home/work/data/olmoearth/oe11_source_expand_v0
TASK_CODE="$TASK_BASE/code_snapshot/oe11_source_expand_v0"
test ! -e "$TASK_BASE/prepared_v0"
test ! -e "$TASK_BASE/prepare.log"
setsid nohup timeout 1800 env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/work/data/olmoearth/.venv-geobench/bin/python -u "$TASK_CODE/prepare_regions.py" \
 --root /home/work/data/olmoearth/geobench2/pastis \
 --audit /home/work/data/olmoearth/oe6_pastis_input_v0/run_v0/audit.json \
 --legacy "$TASK_CODE/legacy80.jsonl" --computed "$TASK_CODE/computed.json" \
 --transform "$TASK_CODE/oe8_transform.py" --policy "$TASK_CODE/selection_policy.json" \
 --out "$TASK_BASE/prepared_v0" --extract > "$TASK_BASE/prepare.log" 2>&1 < /dev/null &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$TASK_BASE/prepare.pid"
printf 'Started bounded CPU source preparation pid=%s\n' "$TASK_PID"
