#!/usr/bin/env bash
# Wait until GPU1 has no compute process (CLAUDE.md 4b), then run E2 on GPU1 in one process
# (probe, 3 reader + 3 blind trainings, evaluations). Prereg: config/e2_multi_reader_prereg_v0.json.
set -u
cd /home/work/data/olmoearth
LOG=logs/e2_multi_reader_v0.log
U=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U"; do
  echo "$(date -Is) GPU1 busy, waiting" >> "$LOG"
  sleep 120
done
echo "$(date -Is) GPU1 free, starting E2" >> "$LOG"
CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH .venv-master/bin/python -B code/e2_multi_reader_v0.py --out e2_multi_reader_v0 >> "$LOG" 2>&1
echo "$(date -Is) E2 process exited with $?" >> "$LOG"
