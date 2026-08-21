#!/usr/bin/env bash
# OlmoEarth H200 모니터 — 로컬(맥)에서 실행하면 30초마다 상태를 다시 그린다.
# 사용법: h100-setup 루트에서  ./projects/olmoearth/watch.sh  [간격초]
# 종료: Ctrl+C   (bash 3.2 호환)
set -u
cd "$(dirname "$0")/../.."
INTERVAL=${1:-30}

while true; do
    OUT_GPU=$(./nexus run "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null)
    OUT_JOBS=$(./nexus job list 2>/dev/null)
    OUT_DISK=$(./nexus run "df -h /home/work/data | tail -1; du -sh /home/work/data/olmoearth 2>/dev/null" 2>/dev/null)
    clear
    echo "── OlmoEarth @ H200 ── $(date '+%H:%M:%S') (매 ${INTERVAL}s 갱신, Ctrl+C 종료)"
    echo
    echo "[GPU]  (idx, util, mem used/total, temp)"
    echo "${OUT_GPU:-  세션 응답 없음 — ./nexus doctor 확인}"
    echo
    echo "[Jobs]"
    echo "${OUT_JOBS:-  (없음)}"
    echo
    echo "[영구 저장소]"
    echo "${OUT_DISK:-  ?}"
    sleep "$INTERVAL"
done
