#!/usr/bin/env bash
# Frozen-sensitivity diagnostic, GPU0 (user allowed both GPUs 2026-09-17). Snapshot + lock + 8 arms x 2 folds + eval.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUTROOT="$ROOT/frozen_sensitivity_v0"; LOCK="$ROOT/.frozen_sensitivity_v0.lock"
cd "$ROOT"; exec 9>"$LOCK"; flock -n 9 || { echo "locked" >&2; exit 9; }
mkdir -p "$OUTROOT/code_snapshot" "$OUTROOT/logs" "$ROOT/config"
if [[ ! -f "$OUTROOT/code_snapshot/SHA256SUMS" ]]; then
  cp code/extract_olmo_perturb.py code/eval_frozen_sensitivity.py code/cache_decoder_train_lib.py config/frozen_sensitivity_prereg_v0.json "$OUTROOT/code_snapshot/"
  ( cd "$OUTROOT/code_snapshot" && sha256sum ./* > SHA256SUMS ); date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
fi
( cd "$OUTROOT/code_snapshot" && sha256sum -c --quiet SHA256SUMS ) || { echo "snapshot changed" >&2; exit 10; }
export CUDA_VISIBLE_DEVICES=0
run () { local fold=$1 arm=$2 off=$3 tm=$4
  [[ -f "$OUTROOT/$fold/$arm/extract_audit.json" ]] && return 0
  env -u PYTHONPATH "$PY" "$OUTROOT/code_snapshot/extract_olmo_perturb.py" --fold "$fold" --arm "$arm" --offset "$off" --time "$tm" > "$OUTROOT/logs/${fold}_${arm}.log" 2>&1
  "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["all_gates_pass"] else 1)' "$OUTROOT/$fold/$arm/extract_audit.json" || { echo "arm $fold/$arm failed" >&2; exit 5; }
}
for fold in holdout_hiroshima holdout_indonesia; do
  run $fold ctrl 0,0 real; run $fold s20_y 2,0 real; run $fold s20_x 0,2 real; run $fold s20_xy 2,2 real; run $fold s40_xy 4,4 real
  run $fold t_synth 0,0 synth_month; run $fold t_plus1 0,0 plus1; run $fold t_shuffle 0,0 shuffle
  env -u PYTHONPATH "$PY" "$OUTROOT/code_snapshot/eval_frozen_sensitivity.py" --fold "$fold" --prereg "$OUTROOT/code_snapshot/frozen_sensitivity_prereg_v0.json" > "$OUTROOT/logs/${fold}_eval.log" 2>&1
  echo "FOLD_DONE $fold"
done
date -u +%FT%TZ > "$OUTROOT/COMPLETED_AT_UTC.txt"; echo "FROZEN_SENSITIVITY_DONE"
