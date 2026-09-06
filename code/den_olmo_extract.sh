#!/usr/bin/env bash
# DEN chips → OlmoEarth base cache. Default GPU1 per repository rule 4b.
# An explicit DEN_GPU_INDEX override is allowed only under a separately recorded exception.
set -euo pipefail
cd /home/work/data/olmoearth
mkdir -p logs
exec 9>logs/den_olmo_extract.lock
if ! flock -n 9; then
  echo "another DEN OlmoEarth extractor holds logs/den_olmo_extract.lock" >&2
  exit 75
fi
LOG=logs/den_olmo_extract.log
GPU_INDEX=${DEN_GPU_INDEX:-1}
note(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
SNAPSHOT_DIR="logs/den_olmo_snapshots/$RUN_ID"
mkdir -p "$SNAPSHOT_DIR"
cp code/extract_olmo_variants.py code/den_olmo_extract.sh "$SNAPSHOT_DIR"/
sha256sum "$SNAPSHOT_DIR"/extract_olmo_variants.py "$SNAPSHOT_DIR"/den_olmo_extract.sh > "$SNAPSHOT_DIR/SHA256SUMS"
printf '{"status":"running","started_at":"%s","pid":%s,"snapshot":"%s"}\n' "$(date -u +%FT%TZ)" "$$" "$SNAPSHOT_DIR" > logs/den_olmo_RUNNING.json
rm -f logs/den_olmo_DONE.json logs/den_olmo_FAILED.json
on_exit(){
  rc=$?
  rm -f logs/den_olmo_RUNNING.json
  if [ "$rc" -eq 0 ]; then
    printf '{"status":"ok","finished_at":"%s"}\n' "$(date -u +%FT%TZ)" > logs/den_olmo_DONE.json
  else
    printf '{"status":"failed","finished_at":"%s","rc":%s}\n' "$(date -u +%FT%TZ)" "$rc" > logs/den_olmo_FAILED.json
  fi
}
trap on_exit EXIT
# 변환 완료 대기 (최대 30분)
for i in $(seq 1 180); do grep -q TILES_DONE logs/tiles_den_test.log 2>/dev/null && break; sleep 10; done
note "tiles ready: $(ls geobench_tiles/dynamic_earthnet/raw_u16 | wc -l) chips"
# GPU guard
foreign=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i "$GPU_INDEX" | wc -l)
if [ "$foreign" -gt 0 ]; then note "GPU${GPU_INDEX} has a process -> stop"; exit 3; fi
note "BEGIN olmo extract on GPU${GPU_INDEX}"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU_INDEX" ./.venv-master/bin/python "$SNAPSHOT_DIR/extract_olmo_variants.py" --size base --src geobench_tiles/dynamic_earthnet --out olmo_den > logs/x_olmo_den.log 2>&1
rc=$?; note "END olmo extract rc=$rc"
exit "$rc"
