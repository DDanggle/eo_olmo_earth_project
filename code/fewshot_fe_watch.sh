#!/usr/bin/env bash
# GPU1 여유 메모리 >= 12 GB 일 때만 fixed-exposure 확증 재실행 (OOM 재발 방지). 10분 폴링, 24h 한도.
cd /home/work/data/olmoearth; LOG=logs/fewshot_conf_chain.log
for i in $(seq 1 144); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  echo "$(date -u +%FT%TZ) gpu1_free=${free}MiB" >> logs/fewshot_fe_watch.log
  if [ "$free" -ge 12000 ]; then
    echo "$(date -u +%FT%TZ) start fe retry (free=${free})" >> $LOG
    rm -rf artifacts/fewshot_confirmatory/fe
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv-master/bin/python code/fewshot_a1_a4.py --confirmatory --arms A1,A4w --exposure fixed_exposure --out artifacts/fewshot_confirmatory/fe > logs/fewshot_conf_fe.log 2>&1
    rc=$?; echo "$(date -u +%FT%TZ) fe retry rc=$rc" >> $LOG
    grep -q DONE logs/fewshot_conf_fe.log && { echo "$(date -u +%FT%TZ) FE_DONE" >> $LOG; exit 0; }
  fi
  sleep 600
done
