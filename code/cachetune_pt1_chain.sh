#!/usr/bin/env bash
# source P4(6 run) 완료(ALL_DONE) 후 PT-1 자동 기동. GPU1.
LOG=/home/work/data/olmoearth/logs/cachetune_chain.log
cd /home/work/data/olmoearth
for i in $(seq 1 360); do grep -q ALL_DONE logs/cachetune_source_p4.log 2>/dev/null && break; sleep 30; done
echo "$(date -u +%FT%TZ) source done; launching PT-1" >> $LOG
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/cachetune_pt1.py > logs/cachetune_pt1.log 2>&1
echo "$(date -u +%FT%TZ) PT-1 exited rc=$?" >> $LOG
