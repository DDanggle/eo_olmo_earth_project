#!/usr/bin/env bash
# Extract primary OLMoEarth v1 Base 768-d embeddings after materialization gates pass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"
MODE="${1:-baseline}"
RUN_ROOT="$REPO_DIR/artifacts/external_data/nepal_olmo_live_v1/materialized/$MODE"
DATASET_PATH="$RUN_ROOT/dataset"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO_DIR/code/model.yaml}"
RSLEARN_BIN="${RSLEARN_BIN:-$WORKSPACE_DIR/.venv/bin/rslearn}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/.venv/bin/python}"

if [[ ! -f "$RUN_ROOT/materialization_manifest.json" ]]; then
  echo "missing materialization manifest: $RUN_ROOT/materialization_manifest.json" >&2
  exit 2
fi
"$PYTHON_BIN" - "$RUN_ROOT/materialization_manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if (not manifest.get("valid") or not manifest.get("exact_four_periods_per_modality")
        or not manifest.get("required_scene_present")):
    raise SystemExit("materialization gate failed")
PY

STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT="$RUN_ROOT/code_snapshot/$STARTED_AT"
mkdir -p "$SNAPSHOT"
cp "$MODEL_CONFIG" "$SNAPSHOT/model.yaml"
cp "$DATASET_PATH/config.json" "$SNAPSHOT/dataset_config.json"
cp "$SCRIPT_DIR/run_nepal_olmo_embeddings.sh" "$SNAPSHOT/runner.sh"
shasum -a 256 "$SNAPSHOT/model.yaml" "$SNAPSHOT/dataset_config.json" "$SNAPSHOT/runner.sh" > "$SNAPSHOT/SHA256SUMS"

export DATASET_PATH
export CUDA_VISIBLE_DEVICES="${OLMO_GPU:-0}"
"$RSLEARN_BIN" model predict --config "$MODEL_CONFIG" \
  --data.init_args.num_workers="${OLMO_WORKERS:-2}" \
  --data.init_args.batch_size="${OLMO_BATCH_SIZE:-4}"

"$PYTHON_BIN" "$SCRIPT_DIR/seal_nepal_olmo_embeddings.py" \
  --dataset "$DATASET_PATH" --mode "$MODE" --code-snapshot "$SNAPSHOT"
