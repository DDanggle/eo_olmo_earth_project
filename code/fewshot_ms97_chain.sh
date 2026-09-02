#!/usr/bin/env bash
# MS-97: (1) A4w0 + A4h + A4p, stratified, fixed-update, 8지역 → (2) random support: A0,A1,A4w,A4w0,A4h. fe 완료 대기 후 시작(GPU 경합 회피).
cd /home/work/data/olmoearth; LOG=logs/fewshot_ms97_chain.log
for i in $(seq 1 480); do grep -q FE_DONE logs/fewshot_conf_chain.log 2>/dev/null && break; sleep 30; done
run(){ echo "$(date -u +%FT%TZ) start $*" >> $LOG; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv-master/bin/python code/fewshot_a1_a4.py --confirmatory "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
run --arms A4w0,A4h,A4p --exposure fixed_update --support stratified --out artifacts/fewshot_confirmatory/fu_rawctl > logs/fewshot_ms97_rawctl.log 2>&1
run --arms A0,A1,A4w,A4w0,A4h --exposure fixed_update --support random --out artifacts/fewshot_confirmatory/fu_random > logs/fewshot_ms97_random.log 2>&1
echo "$(date -u +%FT%TZ) MS97_CHAIN_DONE" >> $LOG
