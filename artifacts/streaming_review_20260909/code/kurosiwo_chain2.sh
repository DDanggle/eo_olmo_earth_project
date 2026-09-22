#!/usr/bin/env bash
# KuroSiwo stages 2-3 only (decoders done): updaters with grad clipping + eval. GPU0.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/kurosiwo_chain.log"; }
log "chain2 start (updaters rerun: lr 5e-4, clip 1.0, nan-skip; first attempt diverged to NaN)"
for mod in gru gru_noobs ema; do for s in 1 2 3; do [[ -f artifacts/kurosiwo/updater_${mod}_seed${s}.pt ]] && continue; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/kurosiwo_pipeline.py --stage update --module $mod --seed $s > "$L/ks_update_${mod}_s$s.log" 2>&1; log "update $mod s$s rc=$?"; done; done
for d in 1 2 3; do env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/kurosiwo_pipeline.py --stage eval --dec-seed $d > "$L/ks_eval_dec$d.log" 2>&1; log "eval dec$d rc=$?"; done
log "KUROSIWO DONE"; touch "$L/kurosiwo_DONE.txt"
