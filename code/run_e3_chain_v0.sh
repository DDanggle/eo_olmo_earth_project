#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
.venv-master/bin/python -B code/e3_pair_dependence_v0.py prepare --out e3_pair_dependence_v0 --config config/e3_pair_dependence_prereg_v0.json
.venv-master/bin/python -B code/run_e3_pair_when_idle_v0.py
