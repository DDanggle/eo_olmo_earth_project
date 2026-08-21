#!/usr/bin/env bash
# 제주 임베딩 데이터셋 v2 — 구름에 강한 합성 방식으로 재수집.
#
# v1의 결함(2026-08-21 실측): rslearn 임베딩 가이드의 기본 설정은
#   space_mode: MOSAIC       → 한 달에 장면 1개만 사용
# 이라서, 그 달의 최선 장면이 흐리면 그대로 오염된다. 제주는 거의 모든 픽셀이
# 매년 12장 중 최소 1장이 절반 이상 구름에 덮여(최악-모자이크 평균 0.53~0.84)
# 사후 마스킹으로는 1.2%만 남는다 → 원리적으로 사후 처리가 불가능.
#
# v2: Ai2가 실전 모델(lfmc)에서 쓰는 방식으로 교체
#   space_mode: PER_PERIOD_MOSAIC  → 한 기간의 여러 장면을 겹쳐 합성해 구름 구멍을 메움
#
# v1 데이터셋은 지우지 않는다 → "합성 방식이 임베딩을 얼마나 바꾸는가" 통제 실험용.
set -euo pipefail

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

echo "=== materialize (PER_PERIOD_MOSAIC — 장면 수가 많아 오래 걸림) ==="
rslearn dataset materialize --root "$DATASET_PATH" --workers 16 --no-use-initial-job \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5

echo "=== embeddings (OlmoEarth-v1-Base, patch 4 → 40m) ==="
cp /home/work/data/olmoearth/embed_search/model_s2.yaml "$E/model_s2.yaml"
cd "$E"
CUDA_VISIBLE_DEVICES=0 rslearn model predict --config model_s2.yaml \
  --data.init_args.num_workers=6 --data.init_args.batch_size=4

echo "PIPELINE_DONE"
