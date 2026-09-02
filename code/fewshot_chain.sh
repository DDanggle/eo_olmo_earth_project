#!/usr/bin/env bash
# 순서: (지금) A1 fixed-exposure → P2 source ALL_DONE 대기 → A4w fixed-update → A1,A4w fixed-exposure. GPU1.
cd /home/work/data/olmoearth; LOG=logs/fewshot_chain.log
run(){ echo "$(date -u +%FT%TZ) start $*" >> $LOG; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/fewshot_a1_a4.py "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
while pgrep -f "fewshot_a1_a4.py --arms A0,A1,A4s" >/dev/null; do sleep 30; done
run --arms A1 --exposure fixed_exposure --out artifacts/fewshot_a1_a4/fe_a1 > logs/fewshot_fe_a1.log 2>&1
for i in $(seq 1 480); do grep -q ALL_DONE logs/cachetune_source_p2.log 2>/dev/null && break; sleep 30; done
run --arms A0,A4w --exposure fixed_update --out artifacts/fewshot_a1_a4/fu_a4w > logs/fewshot_fu_a4w.log 2>&1
run --arms A4w --exposure fixed_exposure --out artifacts/fewshot_a1_a4/fe_a4w > logs/fewshot_fe_a4w.log 2>&1
echo "$(date -u +%FT%TZ) CHAIN_DONE" >> $LOG
