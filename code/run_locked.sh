#!/usr/bin/env bash
# Generic locked runner: run_locked.sh <gpu> <kind> <fold...>   kind = t1 | t0b | p2rt
# Each (kind,fold,arm,seed) is claimed with an atomic mkdir lock so runners on GPU0 and GPU1 never duplicate work.
set -uo pipefail; GPU=$1; KIND=$2; shift 2; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; LK="$ROOT/.locks"; mkdir -p "$LK"
log(){ echo "$(date -u +%FT%TZ) gpu$GPU $*" >> "$L/run_locked_$KIND.log"; }
run(){ local key=$1; shift; local out=$1; shift; [[ -f "$out" ]] && return 0; mkdir "$LK/$key" 2>/dev/null || return 0; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU "$@" > "$L/${key}.log" 2>&1; local rc=$?; log "$key rc=$rc"; [[ $rc -eq 0 ]] || rmdir "$LK/$key"; }
case $KIND in
 t1) for fold in "$@"; do until [[ -f resolution_contract_v2/p4_native_control/${fold}_seed1_best.pt ]]; do sleep 60; done
     for mod in residual gru ema; do for s in 1 2 3; do run t1_${fold}_${mod}_s${s} artifacts/streaming_t1/${fold}_${mod}_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module $mod --seed $s --out artifacts/streaming_t1; done; done; done ;;
 t0b) for fold in "$@"; do for ro in diffpca sketch full; do for s in 1 2 3; do run t0b_${fold}_${ro}_s${s} artifacts/temporal_t0b/${fold}_${ro}_seed${s}.json $PY code/temporal_readout_train.py --cache olmo_single_p4 --fold $fold --readout $ro --seed $s --out artifacts/temporal_t0b; done; done; done ;;
 t1x) for fold in "$@"; do until [[ -f resolution_contract_v2/p4_native_control/${fold}_seed1_best.pt ]]; do sleep 60; done
     for s in 1 2 3; do
       run t1x_${fold}_gru_noobs_s${s} artifacts/streaming_t1x/${fold}_gru_noobs_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module gru_noobs --seed $s --out artifacts/streaming_t1x
       run t1x_${fold}_calib_s${s} artifacts/streaming_t1x/${fold}_calib_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module calib --seed $s --out artifacts/streaming_t1x
       run t1x_${fold}_gru_aux_s${s} artifacts/streaming_t1x/${fold}_gru_aux_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module gru --aux-decoder-loss 1.0 --tag _aux --seed $s --out artifacts/streaming_t1x
     done; done ;;
 ctrl) for fold in "$@"; do for s in 2 3; do run ctrl_${fold}_s${s} artifacts/control_seeds/${fold}_seed${s}.json $PY code/cache_decoder_train.py --cache sen12_pilot/holdout_chimanimani --fold $fold --seed $s --out artifacts/control_seeds; done; done ;;
 t1v) for fold in "$@"; do until [[ -f resolution_contract_v2/p4_native_control/${fold}_seed1_best.pt ]]; do sleep 60; done
     for s in 1 2 3; do run t1v_${fold}_gru_s${s} artifacts/streaming_t1v/${fold}_gru_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module gru --seed $s --out artifacts/streaming_t1v
                       run t1v_${fold}_gru_noobs_s${s} artifacts/streaming_t1v/${fold}_gru_noobs_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module gru_noobs --seed $s --out artifacts/streaming_t1v
                       run t1v_${fold}_ema_s${s} artifacts/streaming_t1v/${fold}_ema_seed${s}.json $PY code/streaming_update_train.py --fold $fold --module ema --seed $s --out artifacts/streaming_t1v; done; done ;;
 p2rt) for tr in native avgpool2; do for fold in "$@"; do run p2rt_${tr}_${fold} artifacts/resolution_rt/p2_$tr/holdout_${fold}_seed1.json $PY code/cache_decoder_train.py --cache olmo_base_p2_rt --grid-transform $tr --fold holdout_$fold --seed 1 --out artifacts/resolution_rt/p2_$tr; done; done ;;
esac; log "done $KIND $*"
