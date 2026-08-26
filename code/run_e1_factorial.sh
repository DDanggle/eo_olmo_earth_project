#!/usr/bin/env bash
# E1 2x2 요인설계 — {타일 캐시, 통짜 캐시} x {작은 decoder, 큰 decoder}
# M30 타일+작은 값은 참고 가능하지만 최종 factorial은 동일 code SHA를 위해 네 칸 모두 다시 돈다.
# GPU1 전용. 각 칸은 같은 fold·같은 12 timestep·같은 40 epoch·best val IoU 선택.
set -euo pipefail
cd /home/work/data
PY="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 olmoearth/.venv-master/bin/python"
TILED=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
FULL=/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani
ROOT=/home/work/data/olmoearth/e1_factorial_v2

# full cache는 embedding만 있으므로 base raw/mask audit에 결합한 별도 seal을 먼저 만든다.
$PY code/audit_sen12_embedding_cache.py --base-cache "$TILED" --emb-cache "$FULL"

echo "=== 칸1: 타일 캐시 + 작은 decoder ==="
$PY code/pilot_sen12_gp_heads.py --arms P4 --epochs 40 \
    --cache "$TILED" --out "$ROOT/tiled_small"

echo "=== 칸2: 타일 캐시 + 큰 decoder (decoder 용량 효과) ==="
$PY code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 \
    --cache "$TILED" --out "$ROOT/tiled_big"

echo "=== 칸3: 통짜 캐시 + 작은 decoder (문맥 효과) ==="
$PY code/pilot_sen12_gp_heads.py --arms P4 --epochs 40 \
    --cache "$TILED" --emb-cache "$FULL" --cache-audit "$FULL/cache_audit.json" \
    --out "$ROOT/full_small"

echo "=== 칸4: 통짜 캐시 + 큰 decoder (둘 다) ==="
$PY code/pilot_sen12_gp_heads.py --arms P4c --epochs 40 \
    --cache "$TILED" --emb-cache "$FULL" --cache-audit "$FULL/cache_audit.json" \
    --out "$ROOT/full_big"

echo "E1 FACTORIAL DONE"
