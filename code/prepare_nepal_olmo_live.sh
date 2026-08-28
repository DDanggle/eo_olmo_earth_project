#!/usr/bin/env bash
# Materialize five 2.56 km OLMoEarth anchors as four 14-day S1+S2 periods.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"
MODE="${1:-baseline}"
DATASET_ROOT="${2:-$REPO_DIR/artifacts/external_data/nepal_olmo_live_v1/materialized/$MODE/dataset}"
RSLEARN_BIN="${RSLEARN_BIN:-$WORKSPACE_DIR/.venv/bin/rslearn}"

case "$MODE" in
  baseline)
    START="2026-07-01T00:00:00+00:00"
    END="2026-08-26T00:00:00+00:00"
    PREPARE_FORCE_ARGS=()
    ;;
  s2_live)
    START="2026-07-03T00:00:00+00:00"
    END="2026-08-28T00:00:00+00:00"
    PREPARE_FORCE_ARGS=(--force)
    ;;
  s1_live)
    START="2026-07-05T00:00:00+00:00"
    END="2026-08-30T00:00:00+00:00"
    PREPARE_FORCE_ARGS=(--force)
    ;;
  placebo_a)
    # 사건 전 구간만 담는 shifted window — RQ-N1의 "일상 rolling delta" 표본.
    # baseline과 같은 계약(4x14d), END만 2026-08-12로 이동. START=END-56d.
    START="2026-06-17T00:00:00+00:00"
    END="2026-08-12T00:00:00+00:00"
    PREPARE_FORCE_ARGS=()
    ;;
  placebo_b)
    START="2026-06-24T00:00:00+00:00"
    END="2026-08-19T00:00:00+00:00"
    PREPARE_FORCE_ARGS=()
    ;;
  *)
    echo "mode must be baseline, s2_live, s1_live, placebo_a, or placebo_b" >&2
    exit 2
    ;;
esac

if [[ ! -x "$RSLEARN_BIN" ]]; then
  echo "rslearn not found: $RSLEARN_BIN" >&2
  exit 2
fi

mkdir -p "$DATASET_ROOT"
if [[ ! -e "$DATASET_ROOT/config.json" ]]; then
  cp "$REPO_DIR/config/nepal_olmo_dataset.json" "$DATASET_ROOT/config.json"
elif ! cmp -s "$REPO_DIR/config/nepal_olmo_dataset.json" "$DATASET_ROOT/config.json"; then
  if [[ -d "$DATASET_ROOT/windows" ]]; then
    echo "dataset config differs after windows were created; refusing mixed contract" >&2
    exit 3
  fi
  cp "$REPO_DIR/config/nepal_olmo_dataset.json" "$DATASET_ROOT/config.json"
fi

add_anchor() {
  local name="$1"
  local lon="$2"
  local lat="$3"
  if [[ -d "$DATASET_ROOT/windows/nepal/$name" ]]; then
    return
  fi
  "$RSLEARN_BIN" dataset add_windows \
    --root "$DATASET_ROOT" --group nepal --name "$name" \
    --utm --resolution 10 --src_crs EPSG:4326 --window_size 256 \
    --box="$lon,$lat,$lon,$lat" --start "$START" --end "$END"
}

add_anchor source_provisional 85.5194 28.2765
add_anchor rasuwagadhi 85.3780 28.2760
add_anchor timure 85.3630 28.2350
add_anchor syabrubesi 85.3470 28.1640
add_anchor dhunche 85.2960 28.1020

"$RSLEARN_BIN" dataset prepare \
  --root "$DATASET_ROOT" --group nepal --workers 2 \
  ${PREPARE_FORCE_ARGS[@]+"${PREPARE_FORCE_ARGS[@]}"} \
  --disabled-layers embeddings --retry-max-attempts 5 --retry-backoff-seconds 5

# Live selection is cheap to inspect and expensive to materialize.  A provider
# catalogue can lag the official Copernicus catalogue, so stop before any pixel
# download unless every anchor actually selected the required post-event scene.
# preflight는 "특정 post-event 장면이 5/5 앵커에 선택됐는가" 검사이므로 live 모드 전용임.
# placebo는 사건 전 창이라 그런 장면이 없어야 정상 — 대신 seal의 cutoff 규칙이 검증함.
if [[ "$MODE" == "s1_live" || "$MODE" == "s2_live" ]]; then
  "$WORKSPACE_DIR/.venv/bin/python" "$SCRIPT_DIR/check_nepal_live_selection.py" \
    --dataset "$DATASET_ROOT" --mode "$MODE"
fi

"$RSLEARN_BIN" dataset materialize \
  --root "$DATASET_ROOT" --group nepal --workers 2 --no-use-initial-job \
  --disabled-layers embeddings --retry-max-attempts 5 --retry-backoff-seconds 5

"$WORKSPACE_DIR/.venv/bin/python" "$SCRIPT_DIR/seal_nepal_olmo_dataset.py" \
  --dataset "$DATASET_ROOT" --mode "$MODE" --start "$START" --end "$END"
