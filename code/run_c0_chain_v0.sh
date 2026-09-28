#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
unset PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
.venv-master/bin/python -B code/c0_linear_view_probe_v0.py prepare --config config/c0_linear_view_prereg_v0.json
.venv-master/bin/python -B c0_linear_view_probe_v0/source.py run
