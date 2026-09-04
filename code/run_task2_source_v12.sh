#!/usr/bin/env bash
# A / R6: new v1.2-native P4 heads on the paired v1.2 cache — screen folds 0,1 x 3 seeds (sealed pilot snapshot, same recipe as task2_source_v1). GPU1.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUTROOT="$ROOT/task2_source_v12"
[[ -e "$OUTROOT" ]] && { echo "refusing to overwrite $OUTROOT" >&2; exit 2; }
mkdir -p "$OUTROOT/code_snapshot"; cp "$ROOT/pilot_sen12_gp_heads.py" "$ROOT/sen12_official_baselines.py" "$OUTROOT/code_snapshot/"
sha256sum "$OUTROOT/code_snapshot/"*.py > "$OUTROOT/code_snapshot/SHA256SUMS"; date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
cp "$ROOT/task2_cache_v12/cache_audit.json" "$OUTROOT/cache_v12_audit.json"
for k in ${FOLDS:-0 1}; do for seed in 1 2 3; do
  out="$OUTROOT/holdout_task2_fold${k}_seed${seed}_P4"; [[ -e "$out" ]] && { echo "refusing to reuse $out" >&2; exit 3; }
  echo "=== v1.2 P4 fold$k seed=$seed $(date -u +%H:%M:%S) ==="
  env PYTHONPATH="$OUTROOT/code_snapshot" CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
    --cache "$ROOT/task2_cache" --emb-cache "$ROOT/task2_cache_v12" --folds "$ROOT/task2_contract/loco_folds.json" --contract "$ROOT/task2_contract/sample_contract.jsonl" \
    --fold "holdout_task2_fold${k}" --arms P4 --seed "$seed" --save-probs --out "$out"
done; done
n=$(find "$OUTROOT" -name P4_best.pt | wc -l); [[ "$n" -eq 6 ]] || { echo "checkpoint count $n != 6" >&2; exit 6; }
echo ALL_DONE
