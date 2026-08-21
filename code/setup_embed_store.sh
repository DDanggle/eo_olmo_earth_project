#!/usr/bin/env bash
# Korea Earth Search 임베딩 스토어(v1) 구축 — 완도·제주·지리산 + 제주 다개년.
#
# 주의: 이 v1 스토어는 space_mode=MOSAIC(기간당 장면 1개)을 쓴다. 검색에는 충분하지만
# 변화탐지에는 구름 오염이 치명적이었다(GOAL.md v1~v4 실패 계보 참고).
# 변화탐지용은 setup_jeju_v2.sh(PER_PERIOD_MOSAIC)를 쓸 것.
#
# 왜 S2-only인가: Planetary Computer의 Sentinel-1이 제주 2024년에 0장이었다(완도는 정상).
# 모달리티 구성이 지역마다 다르면 임베딩 공간이 갈라져 교차지역 검색이 무의미해진다.
set -euo pipefail

E=/home/work/data/olmoearth/embed_search
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
        "query_config": {"max_matches": 12, "period_duration": "30d", "space_mode": "MOSAIC"}
      },
      "type": "raster"
    },
    "embeddings": {"band_sets": [{"dtype": "float32", "num_bands": 768}], "type": "raster"}
  }
}
JSON

add() {  # name box start end
  rslearn dataset add_windows --root "$DATASET_PATH" --group default --name "$1" \
    --utm --resolution 10 --src_crs EPSG:4326 --grid_size 1024 --box="$2" \
    --start "$3" --end "$4" 2>&1 | tail -1
}
# 검색용 3지역 (2024년)
add wando  126.60,34.20,126.90,34.45 2024-01-01T00:00:00+00:00 2025-01-01T00:00:00+00:00
add jeju   126.10,33.15,127.00,33.60 2024-01-01T00:00:00+00:00 2025-01-01T00:00:00+00:00
add jiri   127.55,35.25,127.85,35.45 2024-01-01T00:00:00+00:00 2025-01-01T00:00:00+00:00
# 제주 다개년 (변화탐지 실험용)
add jeju23  126.10,33.15,127.00,33.60 2023-01-01T00:00:00+00:00 2024-01-01T00:00:00+00:00
add jeju25  126.10,33.15,127.00,33.60 2025-01-01T00:00:00+00:00 2026-01-01T00:00:00+00:00
add jeju26r 126.10,33.15,127.00,33.60 2025-07-01T00:00:00+00:00 2026-07-01T00:00:00+00:00
echo "windows: $(ls "$DATASET_PATH/windows/default" | wc -l)"

rslearn dataset prepare --root "$DATASET_PATH" --workers 16 \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root "$DATASET_PATH" --workers 16 --no-use-initial-job \
  --enabled-layers sentinel2_l2a --retry-max-attempts 5 --retry-backoff-seconds 5

# 임베딩 추출: workers 6 / batch 4 (기본값 16은 대형 윈도우에서 조용히 OOM으로 죽는다)
cd "$E"
CUDA_VISIBLE_DEVICES=0 rslearn model predict --config model_s2.yaml \
  --data.init_args.num_workers=6 --data.init_args.batch_size=4
echo "STORE_DONE"
