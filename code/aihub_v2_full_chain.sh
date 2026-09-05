#!/usr/bin/env bash
set -uo pipefail; cd /home/work/data/olmoearth; PY=./.venv-master/bin/python; LOG=logs/aihub_v2_full.log
env -u PYTHONPATH $PY code/materialize_aihub_s2_12band_v2.py >> $LOG 2>&1; echo "$(date -u +%FT%TZ) materialize rc=$?" >> $LOG
env -u PYTHONPATH $PY code/audit_aihub_v2.py >> $LOG 2>&1; echo "$(date -u +%FT%TZ) AIHUB_V2_FULL_DONE" >> $LOG
