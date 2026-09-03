#!/usr/bin/env bash
# Task-2 체인: FETCH_DONE → 전량 감사 → 폴드 봉인 → 캐시 추출(GPU1, 여유 ≥12 GB 확인).
cd /home/work/data/olmoearth; LOG=logs/task2_chain.log
for i in $(seq 1 720); do grep -q FETCH_DONE logs/task2_fetch.log && break; sleep 30; done
echo "$(date -u +%FT%TZ) extraction done" >> $LOG
env -u PYTHONPATH ./.venv-master/bin/python code/audit_task2_contract.py > logs/task2_audit_full.log 2>&1; echo "$(date -u +%FT%TZ) audit rc=$?" >> $LOG
env -u PYTHONPATH ./.venv-master/bin/python code/build_task2_geo_folds.py > logs/task2_folds.log 2>&1; echo "$(date -u +%FT%TZ) folds rc=$?" >> $LOG
for i in $(seq 1 144); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1); [ "$free" -ge 12000 ] && break; sleep 600; done
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/extract_task2_cache.py > logs/task2_cache.log 2>&1; echo "$(date -u +%FT%TZ) cache rc=$?" >> $LOG
echo "$(date -u +%FT%TZ) TASK2_CHAIN_DONE" >> $LOG
