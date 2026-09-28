#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
bundle=code/e3_gpu0_amendment_v1
.venv-master/bin/python -B "$bundle/handoff_e3_waiter_gpu0_v1.py" --out /home/work/data/olmoearth/e3_gpu0_handoff_20260925.json
.venv-master/bin/python -B "$bundle/prepare_e3_gpu0_v1.py" \
  --runner "$bundle/e3_pair_dependence_v0.py" \
  --config "$bundle/e3_pair_dependence_prereg_v1.json" \
  --launcher "$bundle/run_e3_pair_when_idle_v1.py" \
  --handoff /home/work/data/olmoearth/e3_gpu0_handoff_20260925.json
exec .venv-master/bin/python -B e3_pair_dependence_v1/launcher_snapshot/run_e3_pair_when_idle_v1.py
