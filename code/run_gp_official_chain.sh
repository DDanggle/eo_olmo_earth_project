#!/usr/bin/env bash
# 월 추출 완료 대기 → P1/P2공식/P3공식/P4 4-arm pilot. GPU1 전용, 무인 실행.
set -uo pipefail
V=/home/work/data/olmoearth/.venv-master/bin/python
LOG=/home/work/data/olmoearth/logs
C=/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani
EXPECT=6834
cd /home/work/data
mkdir -p "$LOG"

echo "[chain2] $(date +%F' '%T) months 대기 (목표 $EXPECT)"
for i in $(seq 1 120); do
    N=$(wc -l < "$C/months.jsonl" 2>/dev/null | tr -d ' ')
    [ "${N:-0}" -ge "$EXPECT" ] && { echo "[chain2] months 완료 $N"; break; }
    pgrep -f extract_sen12_months.py > /dev/null || { echo "[chain2] 추출 프로세스 없음. $N/$EXPECT — 중단"; exit 1; }
    sleep 20
done
N=$(wc -l < "$C/months.jsonl" 2>/dev/null | tr -d ' ')
[ "${N:-0}" -ge "$EXPECT" ] || { echo "[chain2] months 미완 $N/$EXPECT — 중단"; exit 1; }

echo "[chain2] $(date +%F' '%T) 4-arm smoke (2 epoch) 시작"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$V" code/pilot_sen12_gp_heads.py \
    --arms P1,P2,P3,P4 --epochs 2 \
    --out /home/work/data/olmoearth/sen12_gp_official_smoke > "$LOG/gp_official_smoke.log" 2>&1
RC=$?
echo "[chain2] smoke exit $RC"
if [ "$RC" != "0" ]; then tail -25 "$LOG/gp_official_smoke.log"; exit 1; fi

echo "[chain2] $(date +%F' '%T) 4-arm full (40 epoch) 시작"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$V" code/pilot_sen12_gp_heads.py \
    --arms P1,P2,P3,P4 --epochs 40 \
    --out /home/work/data/olmoearth/sen12_gp_official > "$LOG/gp_official_full.log" 2>&1
echo "[chain2] $(date +%F' '%T) full exit $?"
tail -30 "$LOG/gp_official_full.log"
