#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/work/data/olmoearth/oe9_review_p1_v0
CODE_DIR=$RUN_ROOT/code_snapshot/oe9_human_pilot_v2
PY=/home/work/data/olmoearth/.venv-geobench/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
env -u PYTHONPATH "$PY" "$CODE_DIR/build_human_pilot.py" --prepared-root /home/work/data/olmoearth/oe8_pastis_prepare_v0/prepared_v0 --episodes-root /home/work/data/olmoearth/oe8_pastis_prepare_v0/episodes_v0 --out "$RUN_ROOT/human_pilot_v2"
env -u PYTHONPATH "$PY" "$CODE_DIR/score_human_pilot.py" --package "$RUN_ROOT/human_pilot_v2/reviewer_package" --private-reference "$RUN_ROOT/human_pilot_v2/private/reference_mapping.json" --out "$RUN_ROOT/human_pilot_v2/pending_status.json"
