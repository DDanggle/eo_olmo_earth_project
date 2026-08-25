#!/usr/bin/env bash
# 캐시 완료를 기다렸다가 P1/P2/P4 pilot을 자동 실행한다. GPU1 전용, 무인 실행용.
set -uo pipefail
V=/home/work/data/olmoearth/.venv-master/bin/python
LOG=/home/work/data/olmoearth/logs
CACHE=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
EXPECT=6834
cd /home/work/data
mkdir -p "$LOG"

echo "[chain] $(date +%F' '%T) 캐시 대기 시작 (목표 $EXPECT)"
for i in $(seq 1 240); do          # 최대 2시간
    N=$(ls "$CACHE/mask_u8" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${N:-0}" -ge "$EXPECT" ]; then
        echo "[chain] $(date +%F' '%T) 캐시 완료 $N/$EXPECT"
        break
    fi
    if ! pgrep -f extract_sen12_fold_cache.py > /dev/null; then
        echo "[chain] 추출 프로세스가 사라졌다. 현재 $N/$EXPECT — 중단"
        exit 1
    fi
    sleep 30
done

N=$(ls "$CACHE/mask_u8" 2>/dev/null | wc -l | tr -d ' ')
if [ "${N:-0}" -lt "$EXPECT" ]; then
    echo "[chain] 시간 초과. $N/$EXPECT — pilot을 돌리지 않는다"
    exit 1
fi

echo "[chain] $(date +%F' '%T) pilot 시작"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$V" code/pilot_sen12_gp_heads.py \
    > "$LOG/gp_pilot.log" 2>&1
echo "[chain] $(date +%F' '%T) pilot 종료 (exit $?)"
tail -40 "$LOG/gp_pilot.log"
