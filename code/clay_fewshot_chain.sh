#!/usr/bin/env bash
# Clay few-shot: CLAY_SOURCE_DONE → A0,A1 층화 fu → random → fe. 8 확증 지역, 기존 manifest. raw arm(A4w/A4h)을 같은 report 안에서 실행 → FP budget(Clay A0) 공유. 2026-09-04 검토 반영.
set -euo pipefail
cd /home/work/data/olmoearth; LOG=logs/clay_fewshot_chain.log
ready=0
for i in $(seq 1 2000); do
  if grep -q CLAY_SOURCE_DONE logs/clay_chain.log 2>/dev/null; then ready=1; break; fi
  sleep 60
done
[[ "$ready" -eq 1 ]] || { echo "$(date -u +%FT%TZ) source marker timeout" >> "$LOG"; exit 4; }
run(){
  echo "$(date -u +%FT%TZ) start $*" >> "$LOG"
  if env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv-master/bin/python code/fewshot_a1_a4.py --clay "$@"; then
    echo "$(date -u +%FT%TZ) rc=0 $*" >> "$LOG"
  else
    rc=$?; echo "$(date -u +%FT%TZ) rc=$rc $*" >> "$LOG"; return "$rc"
  fi
}
run --arms A0,A1,A4w,A4h --exposure fixed_update --support stratified --out artifacts/clay_fewshot/fu > logs/clay_fs_fu.log 2>&1
run --arms A0,A1,A4w --exposure fixed_update --support random --out artifacts/clay_fewshot/fu_random > logs/clay_fs_random.log 2>&1
run --arms A1 --exposure fixed_exposure --support stratified --out artifacts/clay_fewshot/fe > logs/clay_fs_fe.log 2>&1
python3 - <<'PY'
import json
from pathlib import Path
expected = {
    Path("artifacts/clay_fewshot/fu/report.json"): 168,
    Path("artifacts/clay_fewshot/fu_random/report.json"): 120,
    Path("artifacts/clay_fewshot/fe/report.json"): 48,
}
for path, n in expected.items():
    obj = json.loads(path.read_text())
    got = len(obj.get("runs", []))
    if got != n:
        raise SystemExit(f"{path}: runs={got}, expected={n}")
PY
echo "$(date -u +%FT%TZ) CLAY_FEWSHOT_DONE" >> $LOG
