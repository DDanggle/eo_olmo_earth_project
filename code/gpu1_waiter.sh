#!/usr/bin/env bash
# GPU1 이 타 프로세스 없이 비면 지정한 체인을 1회 기동한다. 규약 4b(GPU1 만, 남의 작업 있으면 중단)를
# 자동화한 것. 체인 자체의 gpu_guard 가 단계마다 다시 확인하므로 중간에 누가 잡아도 안전하게 멈춘다.
# 사용: bash code/gpu1_waiter.sh <chain.sh> [poll_sec=120] [max_hours=24]
set -uo pipefail
cd /home/work/data/olmoearth
CHAIN="${1:?chain script}"; POLL="${2:-120}"; MAXH="${3:-24}"
LOG=logs/gpu1_waiter.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1 | tr -d ' ')
foreign_on_gpu1() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}' | \
    while read p; do ps -o cmd= -p "$p" 2>/dev/null | grep -q "venv-master/bin/python code/" || echo "$p"; done
}
note "START waiter chain=$CHAIN poll=${POLL}s max=${MAXH}h"
deadline=$(( $(date +%s) + MAXH*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  f=$(foreign_on_gpu1)
  if [ -z "$f" ]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    note "GPU1 free (used=${used} MiB) → 기동 $CHAIN"
    rm -f logs/solar_second_fm_FAILED.json
    setsid nohup bash "$CHAIN" > /dev/null 2>&1 < /dev/null &
    note "launched pid=$!"
    printf '{"launched":"%s","utc":"%s"}\n' "$CHAIN" "$(date -u +%FT%TZ)" > logs/gpu1_waiter_LAUNCHED.json
    exit 0
  fi
  note "GPU1 busy (foreign pids: $(echo $f | tr '\n' ' ')) — ${POLL}s 후 재확인"
  sleep "$POLL"
done
note "TIMEOUT ${MAXH}h — 기동하지 않음"
printf '{"status":"timeout","hours":%s}\n' "$MAXH" > logs/gpu1_waiter_TIMEOUT.json
exit 2
