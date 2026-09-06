#!/usr/bin/env bash
set -uo pipefail
cd /home/work/data/olmoearth
LOG=logs/dl_cls_reg.log; : > $LOG
note(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
for ds in benv2 biomassters; do
  note "BEGIN $ds"
  env -u PYTHONPATH ./.venv-geobench/bin/python code/geobench_hf_download.py $ds --root /home/work/data/olmoearth/geobench2 >> $LOG 2>&1
  rc=$?; note "END $ds rc=$rc"
  if [ $rc -ne 0 ]; then printf "{\"failed\":\"%s\",\"rc\":%s}\n" "$ds" "$rc" > logs/${ds}_FAILED.json; continue; fi
  env -u PYTHONPATH ./.venv-geobench/bin/python code/geobench_verify_one.py $ds --root /home/work/data/olmoearth/geobench2 >> $LOG 2>&1
  vrc=$?; note "verify $ds rc=$vrc"
  [ $vrc -eq 0 ] && printf "{\"status\":\"ok\"}\n" > logs/${ds}_DONE.json || printf "{\"failed\":\"verify\",\"rc\":%s}\n" "$vrc" > logs/${ds}_FAILED.json
done
note "ALL_DONE"
