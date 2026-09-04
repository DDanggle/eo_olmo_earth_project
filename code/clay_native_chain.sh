#!/usr/bin/env bash
set -euo pipefail; cd /home/work/data/olmoearth; LOG=logs/clay_native_chain.log
env -u PYTHONPATH ./.venv-master/bin/python code/build_olmo_pool16.py >> $LOG 2>&1
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/extract_clay_cache.py --grid native16 --temporal mean --out clay_cache_native16 > logs/clay_cache_native16.log 2>&1 && echo "$(date -u +%FT%TZ) native16 mean done" >> $LOG
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 ./.venv-master/bin/python code/extract_clay_cache.py --grid native16 --temporal last --out clay_cache_native16_last > logs/clay_cache_native16_last.log 2>&1 && echo "$(date -u +%FT%TZ) native16 last done" >> $LOG
echo "$(date -u +%FT%TZ) CLAY_NATIVE_CACHES_DONE" >> $LOG
