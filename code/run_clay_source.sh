#!/usr/bin/env bash
# 두 번째 FM(Clay v1.5) 캐시 위 P4-style decoder: 8 확증 폴드 x 3 seed, 봉인 pilot 스냅샷. raw/mask/audit = Sen12 캐시, emb = clay_cache (cin 1024 자동 추론).
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"; OUTROOT="$ROOT/clay_source_v1"; CACHE="$ROOT/sen12_pilot/holdout_chimanimani"
[[ -e "$OUTROOT" ]] && { echo "refusing to overwrite $OUTROOT" >&2; exit 2; }
mkdir -p "$OUTROOT/code_snapshot"
cp "$ROOT/pilot_sen12_gp_heads.py" "$ROOT/sen12_official_baselines.py" \
  "$ROOT/code/extract_clay_cache.py" "$ROOT/code/run_clay_source.sh" \
  "$ROOT/config/second_fm_cache_prereg_v0.json" "$OUTROOT/code_snapshot/"
sha256sum "$OUTROOT/code_snapshot/"* > "$OUTROOT/code_snapshot/SHA256SUMS"
date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
cp "$ROOT/clay_cache/cache_audit.json" "$OUTROOT/clay_cache_audit.json"
for fold in holdout_hiroshima holdout_hokkaido holdout_indonesia holdout_itogon holdout_kyrgyzstan1 holdout_kyrgyzstan2 holdout_newzealand holdout_thrissur; do for seed in 1 2 3; do
  out="$OUTROOT/${fold}_seed${seed}"; [[ -e "$out" ]] && { echo "refusing to reuse $out" >&2; exit 3; }
  echo "=== clay P4 $fold seed=$seed $(date -u +%H:%M:%S) ==="
  env PYTHONPATH="$OUTROOT/code_snapshot" CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
    --cache "$CACHE" --emb-cache "$ROOT/clay_cache" --fold "$fold" --arms P4 --seed "$seed" --save-probs --out "$out"
done; done
echo ALL_DONE
