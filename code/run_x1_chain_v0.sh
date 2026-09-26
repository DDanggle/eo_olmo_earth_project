#!/usr/bin/env bash
# X1 chain (prereg config/x1_italy_external_prereg_v0.json): wait for a producer process to exit, check its
# output is complete, then run the saved-reader evaluation on the given GPU. No settings are changed on failure.
#   bash code/run_x1_chain_v0.sh <wait_pid> <gpu> <models e2|e7>
set -u
cd /home/work/data/olmoearth
PID=$1; GPU=$2; MODELS=$3; LOG=logs/x1_chain_${MODELS}.log
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
N=$(ls x1_italy_stream/emb_time_fp16 | wc -l)
echo "$(date -Is) producer $PID exited; italy embeddings $N" >> "$LOG"
if [ "$N" -lt 963 ]; then echo "$(date -Is) incomplete extraction ($N/963), not launching" >> "$LOG"; exit 1; fi
if [ "$MODELS" = "e7" ] && [ ! -f e7_multi_reader_v0/final.json ]; then echo "$(date -Is) E7 not finished, not launching" >> "$LOG"; exit 1; fi
CUDA_VISIBLE_DEVICES=$GPU env -u PYTHONPATH .venv-master/bin/python -B code/x1_eval_saved_readers_v0.py --models "$MODELS" --out "x1_italy_eval_${MODELS}" >> "$LOG" 2>&1
echo "$(date -Is) eval exited $?" >> "$LOG"
