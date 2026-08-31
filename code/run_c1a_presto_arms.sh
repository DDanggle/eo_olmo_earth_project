#!/usr/bin/env bash
# C1a — Presto common-grid(128x32x32) retrospective matched control.
# 8개 확증 fold x 3 seed, arm=P4(EmbDecoder, cin은 캐시에서 추론), --save-probs.
# 주의: 이것은 retrospective control이다. 새 confirmatory 주장을 만들지 않는다.
set -euo pipefail
cd /home/work/data/olmoearth
PY=./.venv-master/bin/python
EMB=/home/work/data/olmoearth/presto_c1_common32
OUTROOT=/home/work/data/olmoearth/presto_c1a_runs

# 선행조건: pooled 캐시 seal 존재
$PY - <<'PYEOF'
import json, pathlib, sys
seal = pathlib.Path("/home/work/data/olmoearth/presto_c1_common32/seal_manifest.json")
if not seal.exists():
    sys.exit("pooled seal 없음 — pool_presto_common_grid.py 먼저")
d = json.loads(seal.read_text())
assert d["n_files"] == 6834, d["n_files"]
print("pooled seal ok:", d["content_sha256"][:16])
PYEOF

sha256sum pilot_sen12_gp_heads.py | tee "$OUTROOT.code_sha" || true
mkdir -p "$OUTROOT"
FOLDS="holdout_thrissur holdout_hiroshima holdout_hokkaido holdout_indonesia holdout_itogon holdout_kyrgyzstan1 holdout_kyrgyzstan2 holdout_newzealand"
for fold in $FOLDS; do
  for seed in 1 2 3; do
    out="$OUTROOT/${fold}_seed${seed}"
    if ls "$out"/*_pilot.json >/dev/null 2>&1; then echo "skip $out"; continue; fi
    echo "=== $fold seed=$seed $(date -u +%H:%M:%S) ==="
    env PYTHONPATH=/home/work/data/olmoearth/code CUDA_VISIBLE_DEVICES=1 $PY pilot_sen12_gp_heads.py \
      --cache /home/work/data/olmoearth/sen12_pilot/$fold \
      --emb-cache "$EMB" \
      --fold "$fold" --arms P4 --seed "$seed" --save-probs \
      --out "$out"
  done
done
echo ALL_DONE
