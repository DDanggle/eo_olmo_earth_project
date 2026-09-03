#!/usr/bin/env bash
# Task-2 (등록: config/task2_extension_prereg_v0.json) — Ai2 파트너 과업 Solar Farm tar 다운로드·SHA 봉인. CPU 전용, 라벨 미열람.
set -euo pipefail
D=/home/work/data/task2_solar_farm; mkdir -p $D; cd $D
URL=https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/solar_farm.tar
curl -sI "$URL" | grep -iE "content-length|HTTP/" | tee head.txt
[ -f solar_farm.tar ] || curl -sL -o solar_farm.tar "$URL"
sha256sum solar_farm.tar | tee SHA256SUMS
mkdir -p extracted && tar -xf solar_farm.tar -C extracted
find extracted -maxdepth 3 | head -30 > tree_head.txt; du -sh extracted >> tree_head.txt; cat tree_head.txt
echo FETCH_DONE
