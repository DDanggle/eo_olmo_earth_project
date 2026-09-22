#!/usr/bin/env bash
# v1 extension arms (gaps, cloud, year) on GPU0; reuses v0 ctrl/s*/t* arms in frozen_sensitivity_v0. Snapshot v1 kept separately.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUTROOT="$ROOT/frozen_sensitivity_v0"; LOCK="$ROOT/.frozen_sensitivity_v4.lock"
cd "$ROOT"; exec 9>"$LOCK"; flock -n 9 || { echo "locked" >&2; exit 9; }
S="$OUTROOT/code_snapshot_v4"; mkdir -p "$S" "$OUTROOT/logs"
if [[ ! -f "$S/SHA256SUMS" ]]; then cp code/extract_olmo_perturb.py code/eval_frozen_sensitivity.py code/cache_decoder_train_lib.py config/frozen_sensitivity_prereg_v4.json "$S/"; ( cd "$S" && sha256sum ./* > SHA256SUMS ); date -u +%FT%TZ > "$S/started_at_utc.txt"; fi
( cd "$S" && sha256sum -c --quiet SHA256SUMS ) || { echo "snapshot changed" >&2; exit 10; }
export CUDA_VISIBLE_DEVICES=0
run () { local fold=$1 arm=$2; shift 2
  [[ -f "$OUTROOT/$fold/$arm/extract_audit.json" ]] && return 0
  env -u PYTHONPATH "$PY" "$S/extract_olmo_perturb.py" --fold "$fold" --arm "$arm" "$@" > "$OUTROOT/logs/${fold}_${arm}.log" 2>&1
  "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["all_gates_pass"] else 1)' "$OUTROOT/$fold/$arm/extract_audit.json" || { echo "arm $fold/$arm failed" >&2; exit 5; }
}
for fold in holdout_hiroshima holdout_indonesia; do
  run $fold sm_dropY --drop sm_dropY; run $fold sm_swap --drop sm_swap
  env -u PYTHONPATH "$PY" "$S/eval_frozen_sensitivity.py" --fold "$fold" --prereg "$S/frozen_sensitivity_prereg_v4.json" > "$OUTROOT/logs/${fold}_eval_v4.log" 2>&1
  echo "FOLD_DONE_V4 $fold"
done
date -u +%FT%TZ > "$OUTROOT/COMPLETED_V4_AT_UTC.txt"; echo "FROZEN_SENSITIVITY_V4_DONE"
