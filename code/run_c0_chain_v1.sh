#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
unset PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
.venv-master/bin/python -B code/c0_linear_view_probe_v1.py prepare --config config/c0_linear_view_prereg_v1.json
.venv-master/bin/python -B c0_linear_view_probe_v1/source.py run
