#!/usr/bin/env bash
# pastis 다운로드(PID 인자)가 끝나면 적재 검증을 돌리고 마커를 남긴다. 다운로드와 검증을 분리한 설계.
set -uo pipefail
cd /home/work/data/olmoearth
PID="${1:?downloader pid}"; LOG=logs/pastis_followup.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
note "waiting for downloader pid=$PID"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
note "downloader exited"
if grep -q "pastis: ALL FILES OK" logs/dl_pastis2.log; then
  note "download OK → verify"
  env -u PYTHONPATH ./.venv-geobench/bin/python code/geobench_verify_one.py pastis --root /home/work/data/olmoearth/geobench2 >> "$LOG" 2>&1
  rc=$?; note "verify rc=$rc"
  if [ "$rc" -eq 0 ]; then printf '{"status":"ok","utc":"%s"}\n' "$(date -u +%FT%TZ)" > logs/pastis_DONE.json
  else printf '{"failed_step":"verify_pastis","rc":%s}\n' "$rc" > logs/pastis_FAILED.json; fi
else
  note "download did not finish OK"; tail -3 logs/dl_pastis2.log >> "$LOG"
  printf '{"failed_step":"download_pastis"}\n' > logs/pastis_FAILED.json
fi
