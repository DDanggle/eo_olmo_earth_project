#!/usr/bin/env bash
# After the singles extraction: audit, precompute (CPU), then T0b runners on GPU0 (chimanimani) and GPU1 (hiroshima).
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/t0b_chain.log"; }
until grep -q "OLMO SINGLE CACHE DONE" "$L/x_olmo_single.log" 2>/dev/null; do sleep 60; done; log "singles extraction done"
$PY -c 'import json,sys; a=json.load(open("olmo_single_p4/olmo_single_audit.json")); sys.exit(0 if a["all_gates_pass"] else 1)' || { log "singles audit FAILED"; touch "$L/t0b_FAILED.txt"; exit 5; }
env -u PYTHONPATH $PY code/t0_precompute.py olmo_single_p4 holdout_hiroshima holdout_chimanimani > "$L/t0b_precompute.log" 2>&1; log "precompute rc=$?"
setsid nohup bash code/run_locked.sh 0 t0b holdout_chimanimani holdout_hiroshima > /dev/null 2>&1 &
setsid nohup bash code/run_locked.sh 1 t0b holdout_hiroshima holdout_chimanimani > /dev/null 2>&1 &
log "T0b runners launched on GPU0 and GPU1"
