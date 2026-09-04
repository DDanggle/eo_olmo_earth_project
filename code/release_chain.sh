#!/usr/bin/env bash
# A chain: R6 heads (v1.2-native, folds 0,1) -> bridge screen. GPU1.
set -euo pipefail
cd /home/work/data/olmoearth; LOG=logs/release_chain.log
if env -u PYTHONPATH bash code/run_task2_source_v12.sh > logs/task2_source_v12.log 2>&1; then echo "$(date -u +%FT%TZ) R6 rc=0" >> $LOG; else rc=$?; echo "$(date -u +%FT%TZ) R6 rc=$rc" >> $LOG; exit $rc; fi
if env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/release_bridge_screen.py > logs/release_screen.log 2>&1; then echo "$(date -u +%FT%TZ) SCREEN rc=0" >> $LOG; else rc=$?; echo "$(date -u +%FT%TZ) SCREEN rc=$rc" >> $LOG; exit $rc; fi
grep -q "SCREEN DONE" logs/release_screen.log && echo "$(date -u +%FT%TZ) RELEASE_SCREEN_DONE" >> $LOG
