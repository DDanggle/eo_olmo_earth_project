#!/usr/bin/env bash
# Run a command on a GPU after the given PIDs exit and the GPU has no other user's process.
#   bash code/run_after_pids_v0.sh "<pid1 pid2 ...>" <gpu> <log> <command...>
set -u
PIDS=$1; GPU=$2; LOG=$3; shift 3
cd /home/work/data/olmoearth
for p in $PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
U=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
while nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$U" | awk -F, '{print $1}' | xargs -r -n1 ps -o user= -p | grep -vq '^work$'; do
  echo "$(date -Is) GPU$GPU has another user's process; waiting" >> "$LOG"; sleep 120; done
echo "$(date -Is) start: $*" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPU env -u PYTHONPATH "$@" >> "$LOG" 2>&1
echo "$(date -Is) exit $?" >> "$LOG"
