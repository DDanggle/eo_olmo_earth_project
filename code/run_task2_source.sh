#!/usr/bin/env bash
# Task-2 source decoders: P4 + P2, 8 folds x 3 seeds (48 runs), sealed pilot snapshot, GPU1. M65-style step; headline is the few-shot replication.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUTROOT="$ROOT/task2_source_v1"
[[ -e "$OUTROOT" ]] && { echo "refusing to overwrite $OUTROOT" >&2; exit 2; }
mkdir -p "$OUTROOT/code_snapshot"; cp "$ROOT/pilot_sen12_gp_heads.py" "$ROOT/sen12_official_baselines.py" "$OUTROOT/code_snapshot/"
sha256sum "$OUTROOT/code_snapshot/"*.py > "$OUTROOT/code_snapshot/SHA256SUMS"; date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
for k in 0 1 2 3 4 5 6 7; do for seed in 1 2 3; do for arm in P4 P2; do
  out="$OUTROOT/holdout_task2_fold${k}_seed${seed}_${arm}"; [[ -e "$out" ]] && { echo "refusing to reuse $out" >&2; exit 3; }
  echo "=== task2 $arm fold$k seed=$seed $(date -u +%H:%M:%S) ==="
  env PYTHONPATH="$OUTROOT/code_snapshot" CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
    --cache "$ROOT/task2_cache" --emb-cache "$ROOT/task2_cache" --folds "$ROOT/task2_contract/loco_folds.json" --contract "$ROOT/task2_contract/sample_contract.jsonl" \
    --fold "holdout_task2_fold${k}" --arms "$arm" --seed "$seed" --save-probs --out "$out"
done; done; done
echo ALL_DONE
