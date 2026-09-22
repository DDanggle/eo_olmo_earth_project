#!/usr/bin/env bash
# G-T4 chain on GPU1: 2 models x 3 seeds, snapshot + lock.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUT="$ROOT/latent_interp_train_v0"; LOCK="$ROOT/.latent_interp_train_v0.lock"
cd "$ROOT"; exec 9>"$LOCK"; flock -n 9 || { echo locked >&2; exit 9; }
S="$OUT/code_snapshot"; mkdir -p "$S" "$OUT/logs"
if [[ ! -f "$S/SHA256SUMS" ]]; then cp code/latent_interp_train.py config/latent_interp_train_prereg_v0.json "$S/"; ( cd "$S" && sha256sum ./* > SHA256SUMS ); date -u +%FT%TZ > "$S/started_at_utc.txt"; fi
( cd "$S" && sha256sum -c --quiet SHA256SUMS ) || { echo "snapshot changed" >&2; exit 10; }
export CUDA_VISIBLE_DEVICES=1
for seed in 1 2 3; do for model in resmlp ctxattn; do
  [[ -f "$OUT/${model}_seed${seed}.json" ]] && continue
  env -u PYTHONPATH "$PY" "$S/latent_interp_train.py" --model $model --seed $seed --out latent_interp_train_v0 > "$OUT/logs/${model}_seed${seed}.log" 2>&1 || { echo "run $model $seed failed" >&2; exit 5; }
  echo "RUN_DONE $model $seed"
done; done
date -u +%FT%TZ > "$OUT/COMPLETED_AT_UTC.txt"; echo "LATENT_INTERP_TRAIN_DONE"
