#!/usr/bin/env bash
# KuroSiwo streaming chain on GPU0: wait export -> S1 cache extraction -> decoders x3 -> updaters (gru, gru_noobs, ema) x3 -> eval x3 decoders.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/kurosiwo_chain.log"; }
until grep -q "EXPORT DONE" "$L/kurosiwo_export.log" 2>/dev/null; do sleep 60; done; log "export done"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/extract_kurosiwo_s1_cache.py > "$L/x_kurosiwo_s1.log" 2>&1; log "extraction rc=$?"
grep -q "KUROSIWO S1 CACHE DONE" "$L/x_kurosiwo_s1.log" || { log "extraction FAILED"; exit 5; }
for s in 1 2 3; do env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/kurosiwo_pipeline.py --stage decoder --seed $s > "$L/ks_decoder_s$s.log" 2>&1; log "decoder s$s rc=$?"; done
for mod in gru gru_noobs ema; do for s in 1 2 3; do env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/kurosiwo_pipeline.py --stage update --module $mod --seed $s > "$L/ks_update_${mod}_s$s.log" 2>&1; log "update $mod s$s rc=$?"; done; done
for d in 1 2 3; do env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $PY code/kurosiwo_pipeline.py --stage eval --dec-seed $d > "$L/ks_eval_dec$d.log" 2>&1; log "eval dec$d rc=$?"; done
log "KUROSIWO DONE"; touch "$L/kurosiwo_DONE.txt"
