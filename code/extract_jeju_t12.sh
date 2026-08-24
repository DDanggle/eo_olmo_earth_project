#!/usr/bin/env bash
# Reproducible second pass over the completed Jeju v5 dataset using all 12
# materialized periods. This does not re-download imagery.
set -euo pipefail

if [ "${ALLOW_HISTORICAL_INVALID_JEJU_TIME_WINDOWS:-0}" != "1" ]; then
  echo "REFUSED: the existing 216-window dataset fails overlap/month-alignment gates; t12 extraction is historical replay only." >&2
  echo "Set ALLOW_HISTORICAL_INVALID_JEJU_TIME_WINDOWS=1 only for an explicit replay." >&2
  exit 2
fi

E=/home/work/data/olmoearth/embed_jeju_v2
export PATH=/home/work/data/olmoearth/.venv-master/bin:$PATH
export DATASET_PATH=$E/dataset
export HF_HOME=/home/work/data/.cache/huggingface
unset PYTHONPATH

if [ ! -d "$DATASET_PATH/windows/default" ]; then
  echo "missing dataset: $DATASET_PATH" >&2
  exit 1
fi

window_count=$(find "$DATASET_PATH/windows/default" -mindepth 1 -maxdepth 1 -type d -name 'jeju*' | wc -l)
if [ "$window_count" -ne 216 ]; then
  echo "expected 216 Jeju windows, got $window_count" >&2
  exit 1
fi

cp /home/work/data/olmoearth/code/model_s2_t12.yaml "$E/model_s2_t12.yaml"
cd "$E"
CUDA_VISIBLE_DEVICES=0 rslearn model predict --config model_s2_t12.yaml \
  --data.init_args.num_workers=6 --data.init_args.batch_size=4

completed=$(find "$DATASET_PATH/windows/default" -path '*/layers/embeddings_t12/completed' -type f | wc -l)
if [ "$completed" -ne 216 ]; then
  echo "expected 216 embeddings_t12 completion markers, got $completed" >&2
  exit 1
fi

echo "PIPELINE_T12_DONE"
