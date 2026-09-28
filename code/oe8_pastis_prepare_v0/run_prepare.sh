#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
TASK_BASE=/home/work/data/olmoearth/oe8_pastis_prepare_v0
TASK_CODE="$TASK_BASE/code_snapshot/oe8_pastis_prepare_v0"
TASK_PY=/home/work/data/olmoearth/.venv-geobench/bin/python
env -u PYTHONPATH "$TASK_PY" "$TASK_CODE/prepare_inputs.py" \
  --root /home/work/data/olmoearth/geobench2/pastis \
  --audit /home/work/data/olmoearth/oe6_pastis_input_v0/run_v0/audit.json \
  --candidates "$TASK_CODE/candidates.jsonl" --computed "$TASK_CODE/computed.json" \
  --out "$TASK_BASE/prepared_v0"
