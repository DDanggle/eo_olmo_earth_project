#!/usr/bin/env bash
# C1b — Presto native-grid(128x128x128) retrospective product sensitivity.
# GPU1 only. C1a remains the primary representation-family comparison.
set -euo pipefail

ROOT=/home/work/data/olmoearth
PY="$ROOT/.venv-master/bin/python"
RAW_CACHE="$ROOT/sen12_pilot"
EMB="$ROOT/presto_c1/holdout_chimanimani"
OUTROOT="$ROOT/presto_c1b_native_runs_v1"

if [[ -e "$OUTROOT" ]]; then
  echo "refusing to overwrite existing OUTROOT: $OUTROOT" >&2
  exit 2
fi

mkdir -p "$OUTROOT/code_snapshot"
cp "$ROOT/pilot_sen12_gp_heads.py" "$OUTROOT/code_snapshot/"
cp "$ROOT/sen12_official_baselines.py" "$OUTROOT/code_snapshot/"
sha256sum "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
          "$OUTROOT/code_snapshot/sen12_official_baselines.py" \
  > "$OUTROOT/code_snapshot/SHA256SUMS"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTROOT/code_snapshot/started_at_utc.txt"

"$PY" - "$EMB" <<'PYEOF'
import json
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
seal = root / "seal_manifest.json"
if not seal.is_file():
    raise SystemExit(f"native cache seal missing: {seal}")
payload = json.loads(seal.read_text())
aggregate = payload.get("aggregate", {})
if aggregate.get("n_files") != 6834:
    raise SystemExit(f"native cache count mismatch: {aggregate.get('n_files')}")
expected_manifest = "aad49d14f94f36dacaedd6abba3b136cada0ca14c61649afd8ea632a05da162b"
if aggregate.get("manifest_sha256") != expected_manifest:
    raise SystemExit("native cache manifest seal mismatch")
if aggregate.get("spot_check_40", {}).get("pass") is not True:
    raise SystemExit("native cache finite/shape spot-check seal failed")
if aggregate.get("expected_shape") != [128, 128, 128] or aggregate.get("dtype") != "float16":
    raise SystemExit("native cache declared shape/dtype mismatch")
files = sorted((root / "emb_fp16").glob("*.npy"))
if len(files) != 6834:
    raise SystemExit(f"native cache file count mismatch: {len(files)}")
shape = np.load(files[0], mmap_mode="r").shape
if tuple(shape) != (128, 128, 128):
    raise SystemExit(f"native cache shape mismatch: {shape}")
print("C1b native preflight ok", aggregate["manifest_sha256"][:16], shape)
PYEOF

FOLDS="holdout_thrissur holdout_hiroshima holdout_hokkaido holdout_indonesia holdout_itogon holdout_kyrgyzstan1 holdout_kyrgyzstan2 holdout_newzealand"
for fold in $FOLDS; do
  for seed in 1 2 3; do
    out="$OUTROOT/${fold}_seed${seed}"
    if [[ -e "$out" ]]; then
      echo "refusing to reuse output directory: $out" >&2
      exit 3
    fi
    echo "=== C1b $fold seed=$seed $(date -u +%H:%M:%S) ==="
    env PYTHONPATH="$OUTROOT/code_snapshot" CUDA_VISIBLE_DEVICES=1 \
      "$PY" "$OUTROOT/code_snapshot/pilot_sen12_gp_heads.py" \
      --cache "$RAW_CACHE/$fold" \
      --emb-cache "$EMB" \
      --fold "$fold" --arms P4native --seed "$seed" --save-probs \
      --out "$out"
  done
done
echo ALL_DONE
