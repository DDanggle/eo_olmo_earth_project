#!/usr/bin/env bash
set -uo pipefail; cd /home/work/data/olmoearth; PY=./.venv-master/bin/python; LOG=logs/aihub_v2_pilot.log
env -u PYTHONPATH $PY code/aihub_v2_pilot_keys.py >> $LOG 2>&1
env -u PYTHONPATH $PY code/materialize_aihub_s2_12band_v2.py --keys-file aihub/s2_12band_v2/pilot40_keys.txt >> $LOG 2>&1; echo "materialize rc=$?" >> $LOG
env -u PYTHONPATH $PY code/audit_aihub_v2.py aihub/s2_12band_v2/pilot40_keys.txt >> $LOG 2>&1; echo "PILOT40_DONE" >> $LOG
