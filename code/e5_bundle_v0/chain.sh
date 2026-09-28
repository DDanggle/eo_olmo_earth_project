#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
task_bundle=/home/work/data/olmoearth/code/e5_bundle_v0
task_python=/home/work/data/olmoearth/.venv-master/bin/python
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$task_python" -B "$task_bundle/e5_prepare_v0.py" \
  --config "$task_bundle/e5_equal_budget_prereg_v0.json" --inputs "$task_bundle/frozen_inputs" \
  --source-dir "$task_bundle"
exec env -u PYTHONPATH "$task_python" -B /home/work/data/olmoearth/e5_equal_budget_v0/code_snapshot/run_e5_when_idle_v0.py
