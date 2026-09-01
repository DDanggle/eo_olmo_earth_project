#!/usr/bin/env bash
# GPU1이 완전히 빌 때만 C1b 러너를 기동함 (CLAUDE.md 4b: 다른 프로세스가 있으면 멈춘다 → 빈 시점에만 시작).
LOG=/home/work/data/olmoearth/logs/c1b_watcher.log
for i in $(seq 1 288); do  # 10분 간격 최대 48h
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  echo "$(date -u +%FT%TZ) gpu1_used=${used}MiB" >> $LOG
  if [ "$used" -lt 100 ]; then
    echo "$(date -u +%FT%TZ) GPU1 empty → launching C1b" >> $LOG
    cd /home/work/data/olmoearth && env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 bash code/run_c1b_presto_native.sh >> /home/work/data/olmoearth/logs/c1b_native.log 2>&1
    echo "$(date -u +%FT%TZ) C1b runner exited rc=$?" >> $LOG
    exit 0
  fi
  sleep 600
done
echo "$(date -u +%FT%TZ) gave up after 48h" >> $LOG
