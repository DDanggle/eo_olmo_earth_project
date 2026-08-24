#!/usr/bin/env bash
# 제주 임베딩 데이터셋 v5 — 실패 재현용. 구름 개선 경로로 사용하지 말 것.
#
# 2026-08-22 전수 감사 결과, rslearn 0.1.13에서 MOSAIC+period_duration과
# PER_PERIOD_MOSAIC+period_duration은 같은 handler를 사용한다. v1/v5의 2,592 source group,
# B02 품질 지표, 고정 원본/임베딩 표본이 모두 동일했다. 이 스크립트는 실패 계보 재현용이며,
# 실제 구름 개선은 SCL/cloud mask를 pixel validity에 연결한 별도 합성기가 필요하다.
set -euo pipefail

if [ "${ALLOW_HISTORICAL_INVALID_JEJU_TIME_WINDOWS:-0}" != "1" ]; then
  echo "REFUSED: jeju25 and rolling-jeju26 overlap by 184 days; this setup is historical-failure reproduction only." >&2
  echo "Set ALLOW_HISTORICAL_INVALID_JEJU_TIME_WINDOWS=1 only for an explicit replay." >&2
  exit 2
fi

E=/home/work/data/olmoearth/embed_jeju_v2
export PATH=/home/work/data/olmoearth/.venv-master/bin:$PATH
export DATASET_PATH=$E/dataset
export HF_HOME=/home/work/data/.cache/huggingface
unset PYTHONPATH

mkdir -p "$DATASET_PATH"
cat > "$DATASET_PATH/config.json" <<'JSON'
{
  "layers": {
    "sentinel2_l2a": {
      "band_sets": [{
        "bands": ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
        "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {"harmonize": true, "sort_by": "eo:cloud_cover"},
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    },
    "embeddings": {
      "band_sets": [{"dtype": "float32", "num_bands": 768}],
      "type": "raster"
    },
    "embeddings_t12": {
      "band_sets": [{"dtype": "float32", "num_bands": 768}],
      "type": "raster"
    }
  }
}
JSON

BOX="126.10,33.15,127.00,33.60"
add() {  # name start end
  rslearn dataset add_windows --root "$DATASET_PATH" --group default --name "$1" \
    --utm --resolution 10 --src_crs EPSG:4326 --grid_size 1024 --box="$BOX" \
    --start "$2" --end "$3" 2>&1 | tail -1
}
add jeju23  2023-01-01T00:00:00+00:00 2024-01-01T00:00:00+00:00
add jeju24  2024-01-01T00:00:00+00:00 2025-01-01T00:00:00+00:00
add jeju25  2025-01-01T00:00:00+00:00 2026-01-01T00:00:00+00:00
add jeju26r 2025-07-01T00:00:00+00:00 2026-07-01T00:00:00+00:00
echo "windows: $(ls "$DATASET_PATH/windows/default" | wc -l)"

echo "=== prepare ==="
rslearn dataset prepare --root "$DATASET_PATH" --workers 16 \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5

echo "=== materialize (historical v5 alias experiment; not cloud-aware) ==="
rslearn dataset materialize --root "$DATASET_PATH" --workers 16 --no-use-initial-job \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5

echo "=== embeddings (OlmoEarth-v1-Base, patch 4 → 40m) ==="
cp /home/work/data/olmoearth/embed_search/model_s2.yaml "$E/model_s2.yaml"
cd "$E"
CUDA_VISIBLE_DEVICES=0 rslearn model predict --config model_s2.yaml \
  --data.init_args.num_workers=6 --data.init_args.batch_size=4

echo "PIPELINE_DONE"
