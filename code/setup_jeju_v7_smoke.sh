#!/usr/bin/env bash
# One-window intervention test for genuinely different Sentinel-2 compositing.
set -euo pipefail

E=/home/work/data/olmoearth/embed_jeju_v7_smoke
DATASET_PATH=$E/dataset
SOURCE_WINDOW=/home/work/data/olmoearth/embed_jeju_v2/dataset/windows/default/jeju25_30720_-372736
TARGET_WINDOW=$DATASET_PATH/windows/default/smoke25_30720_-372736
export PATH=/home/work/data/olmoearth/.venv-master/bin:$PATH
export DATASET_PATH
unset PYTHONPATH

# The installed ``rslearn`` console script places the venv's bin directory at
# sys.path[0], so jsonargparse cannot import our compositor from the current
# experiment directory. Running the same entry point through ``python -c``
# keeps the current directory importable without setting PYTHONPATH.
run_rslearn() {
  /home/work/data/olmoearth/.venv-master/bin/python -c \
    'from rslearn.main import main; main()' "$@"
}

mkdir -p "$DATASET_PATH" "$TARGET_WINDOW"
cp /home/work/data/olmoearth/code/scl_compositor.py "$E/scl_compositor.py"
cp "$SOURCE_WINDOW/metadata.json" "$TARGET_WINDOW/metadata.json"

cat > "$DATASET_PATH/config.json" <<'JSON'
{
  "layers": {
    "sentinel2_l2a": {
      "band_sets": [
        {
          "bands": ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
          "dtype": "uint16",
          "nodata_value": 0
        },
        {
          "bands": ["SCL"],
          "dtype": "uint8",
          "nodata_value": 0
        }
      ],
      "resampling_method": "bilinear",
      "compositing_method": {
        "class_path": "scl_compositor.Sentinel2SCLBestClearNearest",
        "init_args": {
          "clear_values": [4, 5, 6],
          "min_clear_fraction": 0.0,
          "min_valid_cover": 0.5
        }
      },
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {"harmonize": true, "sort_by": "eo:cloud_cover"},
        "ingest": false,
        "query_config": {
          "max_matches": 4,
          "period_duration": "30d",
          "space_mode": "MOSAIC",
          "mosaic_compositing_overlaps": 3,
          "per_period_mosaic_reverse_time_order": true
        }
      },
      "type": "raster"
    }
  }
}
JSON

cd "$E"
run_rslearn dataset prepare --root "$DATASET_PATH" --workers 4 \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5
run_rslearn dataset materialize --root "$DATASET_PATH" --workers 4 --no-use-initial-job \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5

completed=$(find "$TARGET_WINDOW/layers" -path '*/completed' -type f | wc -l)
if [ "$completed" -ne 4 ]; then
  echo "expected 4 completed period layers, got $completed" >&2
  exit 1
fi
echo "V7_SMOKE_MATERIALIZE_DONE"
