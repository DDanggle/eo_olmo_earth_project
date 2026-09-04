#!/usr/bin/env bash
# Clay few-shot: CLAY_SOURCE_DONE → A0,A1 층화 fu → random → fe. 8 확증 지역, 기존 manifest. raw arm(A4w/A4h)은 Sen12 확증 report 와 대조.
cd /home/work/data/olmoearth; LOG=logs/clay_fewshot_chain.log
for i in $(seq 1 2000); do grep -q CLAY_SOURCE_DONE logs/clay_chain.log 2>/dev/null && break; sleep 60; done
run(){ echo "$(date -u +%FT%TZ) start $*" >> $LOG; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv-master/bin/python code/fewshot_a1_a4.py --clay "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
run --arms A0,A1 --exposure fixed_update --support stratified --out artifacts/clay_fewshot/fu > logs/clay_fs_fu.log 2>&1
run --arms A0,A1 --exposure fixed_update --support random --out artifacts/clay_fewshot/fu_random > logs/clay_fs_random.log 2>&1
run --arms A1 --exposure fixed_exposure --support stratified --out artifacts/clay_fewshot/fe > logs/clay_fs_fe.log 2>&1
echo "$(date -u +%FT%TZ) CLAY_FEWSHOT_DONE" >> $LOG
