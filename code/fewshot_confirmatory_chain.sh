#!/usr/bin/env bash
# 8지역 확증: fixed-update(A0,A1,A4w) → fixed-exposure(A1,A4w). GPU1. 판정 규칙: config/fewshot_a1_vs_a4_prereg_v0.json confirmatory_protocol_registered_now
cd /home/work/data/olmoearth; LOG=logs/fewshot_conf_chain.log
run(){ echo "$(date -u +%FT%TZ) start $*" >> $LOG; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/fewshot_a1_a4.py --confirmatory "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
run --arms A0,A1,A4w --exposure fixed_update --out artifacts/fewshot_confirmatory/fu > logs/fewshot_conf_fu.log 2>&1
run --arms A1,A4w --exposure fixed_exposure --out artifacts/fewshot_confirmatory/fe > logs/fewshot_conf_fe.log 2>&1
echo "$(date -u +%FT%TZ) CONF_CHAIN_DONE" >> $LOG
