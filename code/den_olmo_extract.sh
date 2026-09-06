#!/usr/bin/env bash
# DEN chips → OlmoEarth base 캐시. GPU0 (사용자 이번 건 승인, 규약 4b 일회 해제).
# GPU0에 우리 것 아닌 프로세스 있으면 중단.
set -uo pipefail
cd /home/work/data/olmoearth
LOG=logs/den_olmo_extract.log; : > $LOG
note(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
# 변환 완료 대기 (최대 30분)
for i in $(seq 1 180); do grep -q TILES_DONE logs/tiles_den_test.log 2>/dev/null && break; sleep 10; done
note "tiles ready: $(ls geobench_tiles/dynamic_earthnet/raw_u16 | wc -l) chips"
# GPU0 가드
foreign=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i 0 | wc -l)
if [ "$foreign" -gt 0 ]; then note "GPU0에 프로세스 있음 → 중단"; echo FAIL > logs/den_olmo_FAILED.json; exit 3; fi
note "BEGIN olmo extract on GPU0"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 ./.venv-master/bin/python code/extract_olmo_variants.py --size base --src geobench_tiles/dynamic_earthnet --out olmo_den > logs/x_olmo_den.log 2>&1
rc=$?; note "END olmo extract rc=$rc"
[ $rc -eq 0 ] && printf "{\"status\":\"ok\"}\n" > logs/den_olmo_DONE.json || printf "{\"rc\":%s}\n" "$rc" > logs/den_olmo_FAILED.json
