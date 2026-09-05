#!/usr/bin/env bash
# parallel v2 materialization: 4 shards (disjoint key files, separate out dirs) -> merge -> full audit
set -uo pipefail; cd /home/work/data/olmoearth; PY=./.venv-master/bin/python; LOG=logs/aihub_v2_full.log
env -u PYTHONPATH $PY code/aihub_v2_shards.py >> $LOG 2>&1
for i in 0 1 2 3; do (env -u PYTHONPATH $PY code/materialize_aihub_s2_12band_v2.py --keys-file aihub/s2_12band_v2/shard${i}_keys.txt --out aihub/s2_12band_v2_shard${i} > logs/aihub_v2_shard${i}.log 2>&1; echo "$(date -u +%FT%TZ) shard$i rc=$?" >> $LOG) & done; wait
env -u PYTHONPATH $PY code/aihub_v2_merge.py >> $LOG 2>&1
env -u PYTHONPATH $PY code/audit_aihub_v2.py >> $LOG 2>&1; echo "$(date -u +%FT%TZ) AIHUB_V2_FULL_DONE" >> $LOG
