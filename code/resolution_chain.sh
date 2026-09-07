#!/usr/bin/env bash
# Cache-contract development screen, seed 1.  Four-way interpretation:
# p4/native is the sealed reference; p4/upsample2 controls decoder input grid;
# p2/native tests dense FlexiViT readout; p2/avgpool2 removes its fine grid.
set -euo pipefail
ROOT=/home/work/data/olmoearth
PY="$ROOT/.venv-master/bin/python"
OUTROOT="$ROOT/resolution_contract_v2"
CACHE_P2="$ROOT/olmo_base_p2"
CACHE_P4="$ROOT/sen12_pilot/holdout_chimanimani"
LOCK="$ROOT/.resolution_contract_v2.lock"
MODE="${1:-new}"
cd "$ROOT"

exec 9>"$LOCK"
flock -n 9 || { echo "another resolution contract run holds $LOCK" >&2; exit 9; }
if [[ -e "$OUTROOT" ]]; then
  [[ "$MODE" == "--resume" && ! -e "$OUTROOT/COMPLETED_AT_UTC.txt" ]] || {
    echo "refusing existing/completed OUTROOT $OUTROOT (use --resume only for an incomplete run)" >&2; exit 2;
  }
  ( cd "$OUTROOT/code_snapshot" && sha256sum -c SHA256SUMS ) || {
    echo "snapshot changed; refusing resume" >&2; exit 10;
  }
else
  [[ "$MODE" == "new" ]] || { echo "cannot resume missing $OUTROOT" >&2; exit 2; }
fi

# GPU1 only.  Refuse to queue behind or interfere with another user's process.
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')
[[ -n "$GPU1_UUID" ]] || { echo "GPU1 UUID not found" >&2; exit 3; }
require_gpu1_free () {
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID"; then
    echo "GPU1 is occupied; stopping before the next stage" >&2
    exit 4
  fi
}
require_gpu1_free

if [[ "$MODE" == "new" ]]; then
  mkdir -p "$OUTROOT/code_snapshot" "$OUTROOT/logs"
  cp "$ROOT/code/extract_olmo_variants.py" "$ROOT/code/cache_decoder_train.py" "$ROOT/code/cache_grid_controls.py" \
     "$ROOT/code/verify_resolution_contract.py" \
     "$ROOT/config/second_fm_cache_prereg_v1_draft.json" "$OUTROOT/code_snapshot/"
  ( cd "$OUTROOT/code_snapshot" && sha256sum ./* > SHA256SUMS )
  date -u +%FT%TZ > "$OUTROOT/code_snapshot/started_at_utc.txt"
  chmod -w "$OUTROOT/code_snapshot"/* || true
fi

require_gpu1_free
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/extract_olmo_variants.py" \
  --size base --patch 2 --out olmo_base_p2 > "$OUTROOT/logs/extract_p2.log" 2>&1
"$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); sys.exit(0 if a.get("all_gates_pass") is True and a.get("expected_shape")==[768,64,64] else 1)' \
  "$CACHE_P2/olmo_variant_audit.json" || { echo "patch-2 cache audit failed" >&2; exit 5; }
cp "$CACHE_P2/olmo_variant_audit.json" "$OUTROOT/"
cp "$ROOT/artifacts/confirmatory_8region_summary.json" "$OUTROOT/"

run_arm () {
  local cache="$1" transform="$2" label="$3" fold out
  out="$OUTROOT/$label"
  mkdir -p "$out"
  for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
    if [[ -f "$out/holdout_${fold}_seed1.json" ]]; then
      continue
    fi
    require_gpu1_free
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$PY" "$OUTROOT/code_snapshot/cache_decoder_train.py" \
      --cache "$cache" --grid-transform "$transform" --fold "holdout_$fold" --seed 1 --out "$out" \
      > "$OUTROOT/logs/${label}_${fold}.log" 2>&1
  done
}

# The same-trainer P4 native arm is a calibration control.  The sealed
# three-seed P4 (.272) remains the promotion reference.
run_arm "${CACHE_P4#$ROOT/}" native p4_native_control
run_arm "${CACHE_P4#$ROOT/}" upsample2 p4_upsample2
run_arm "${CACHE_P2#$ROOT/}" native p2_native
run_arm "${CACHE_P2#$ROOT/}" avgpool2 p2_avgpool2

for arm in p4_native_control p4_upsample2 p2_native p2_avgpool2; do
  [[ $(find "$OUTROOT/$arm" -maxdepth 1 -name 'holdout_*_seed1.json' | wc -l | tr -d ' ') == 8 ]] || exit 6
done
env -u PYTHONPATH "$PY" "$OUTROOT/code_snapshot/verify_resolution_contract.py" \
  --root "$OUTROOT" --p4-summary "$OUTROOT/confirmatory_8region_summary.json" --out "$OUTROOT/summary.json"
date -u +%FT%TZ > "$OUTROOT/COMPLETED_AT_UTC.txt"
echo "RESOLUTION_CONTRACT_V2_DONE"
