#!/usr/bin/env bash
# CacheTune PT-1 준비 — 개발 폴드(holdout_china, holdout_chimanimani)의 source P4 decoder를 봉인 pilot 스냅샷으로 학습.
# 스냅샷을 만들고 **그 사본을 실행**한다(run_c1b 와 같은 규약). GPU1 only.
set -euo pipefail
ROOT=/home/work/data/olmoearth; PY="$ROOT/.venv-master/bin/python"
OUTROOT="$ROOT/cachetune_source_p4_v1"
[[ -e "$OUTROOT" ]] && { echo "refusing to overwrite $OUTROOT" >&2; exit 2; }
mkdir -p "$OUTROOT/code_snapshot"
cp "$ROOT/pilot_sen12_gp_heads.py" "$ROOT/sen12_official_baselines.py" "$OUTROOT/code_snapshot/"
sha256sum "$OUTROOT/code_snapshot/"*.py > "$OUTROOT/code_snapshot/SHA256SUMS"
date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
# fold 캐시는 6,834타일 전체를 담고 있어 폴드 간 동일함(감사 seal은 holdout_chimanimani 것을 사용; pilot 은 cache 경로·audit sha 를 산출물에 기록함).
CACHE="$ROOT/sen12_pilot/holdout_chimanimani"
for fold in holdout_china holdout_chimanimani; do
  for seed in 1 2 3; do
    out="$OUTROOT/${fold}_seed${seed}"; [[ -e "$out" ]] && { echo "refusing to reuse $out" >&2; exit 3; }
    echo "=== source P4 $fold seed=$seed $(date -u +%H:%M:%S) ==="
    env PYTHONPATH="$OUTROOT/code_snapshot" CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
      --cache "$CACHE" --emb-cache "$CACHE" --fold "$fold" --arms P4 --seed "$seed" --save-probs --out "$out"
  done
done
echo ALL_DONE
