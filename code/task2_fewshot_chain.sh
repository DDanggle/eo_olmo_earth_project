#!/usr/bin/env bash
# Task-2 헤드라인: source ALL_DONE → manifests → few-shot fu(A0,A1,A4w,A4h) 층화 → random → fe(A1,A4w). GPU1.
cd /home/work/data/olmoearth; LOG=logs/task2_fewshot_chain.log
for i in $(seq 1 2000); do grep -q ALL_DONE logs/task2_source.log 2>/dev/null && break; sleep 60; done
env -u PYTHONPATH ./.venv-master/bin/python code/task2_fewshot_manifests.py > logs/task2_manifests.log 2>&1; echo "$(date -u +%FT%TZ) manifests rc=$?" >> $LOG
run(){ echo "$(date -u +%FT%TZ) start $*" >> $LOG; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv-master/bin/python code/fewshot_a1_a4.py --task2 "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
run --arms A0,A1,A4w,A4h --exposure fixed_update --support stratified --out artifacts/task2_fewshot/fu > logs/task2_fs_fu.log 2>&1
run --arms A0,A1,A4w,A4h --exposure fixed_update --support random --out artifacts/task2_fewshot/fu_random > logs/task2_fs_random.log 2>&1
run --arms A1,A4w --exposure fixed_exposure --support stratified --out artifacts/task2_fewshot/fe > logs/task2_fs_fe.log 2>&1
echo "$(date -u +%FT%TZ) TASK2_FEWSHOT_DONE" >> $LOG
