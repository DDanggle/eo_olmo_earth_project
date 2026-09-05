#!/usr/bin/env bash
# addendum_v1b architecture axes: 7 caches -> decoders (8 folds, seed 1). GPU1.
set -euo pipefail
cd /home/work/data/olmoearth
PY=./.venv-master/bin/python
LOG=logs/arch_axes.log

run() { env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$PY" "$@"; }

step() {
  local label=$1 logfile=$2 rc
  shift 2
  set +e
  run "$@" > "$logfile" 2>&1
  rc=$?
  set -e
  printf '%s %s rc=%s\n' "$(date -u +%FT%TZ)" "$label" "$rc" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    printf '%s ARCH_AXES_FAILED step=%s rc=%s\n' "$(date -u +%FT%TZ)" "$label" "$rc" >> "$LOG"
    exit "$rc"
  fi
}

step olmo_nano logs/x_olmo_nano.log code/extract_olmo_variants.py --size nano --out olmo_nano
step olmo_tiny logs/x_olmo_tiny.log code/extract_olmo_variants.py --size tiny --out olmo_tiny
step olmo_base_half logs/x_olmo_base_half.log code/extract_olmo_variants.py --size base --depth-frac 0.5 --out olmo_base_half
step galileo_nano logs/x_galileo_nano.log code/extract_galileo_cache.py --size nano --out galileo_nano
step galileo_tiny logs/x_galileo_tiny.log code/extract_galileo_cache.py --size tiny --out galileo_tiny
step clay_in256_half logs/x_clay_in256_half.log code/extract_clay_cache.py --grid in256 --temporal mean --depth-frac 0.5 --out clay_in256_half
step galileo_base_half logs/x_galileo_base_half.log code/extract_galileo_cache.py --size base --exit-after 6 --out galileo_base_half
for cache in olmo_nano olmo_tiny olmo_base_half galileo_nano galileo_tiny clay_in256_half galileo_base_half; do
  for fold in hiroshima hokkaido indonesia itogon kyrgyzstan1 kyrgyzstan2 newzealand thrissur; do
    out="bv1_runs/$cache"; [[ -f "$out/holdout_${fold}_seed1.json" ]] && continue
    step "$cache $fold" "logs/bv1_${cache}_${fold}.log" code/cache_decoder_train.py \
      --cache "$cache" --fold "holdout_$fold" --seed 1 --out "$out"
  done
done
printf '%s ARCH_AXES_DONE\n' "$(date -u +%FT%TZ)" >> "$LOG"
