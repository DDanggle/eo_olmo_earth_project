#!/usr/bin/env bash
set -euo pipefail
cd /home/work/data/olmoearth
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-master/bin/python -B code/e4_delta_bundle_v0/e4_delta_runner_v0.py prepare
exec env -u PYTHONPATH .venv-master/bin/python -B e4_delta_probe_v0/code_snapshot/run_e4_when_idle_v0.py
