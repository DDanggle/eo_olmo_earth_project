#!/usr/bin/env bash
# GEO-Bench-2 데이터 확보 체인. GPU 사용 안 함.
# 순서: numpy 정합 -> fotw(4GB, 경로 검증) -> 적재 검증 -> pastis(53GB) -> 적재 검증
#       dynamic_earthnet(44GB)은 위 둘이 통과한 뒤 별도로 받는다.
# rc 는 명령 직후에 잡는다 (M107). 실패하면 즉시 멈추고 FAILED.json 을 남긴다.
set -uo pipefail
cd /home/work/data/olmoearth
ROOTDIR=/home/work/data/olmoearth/geobench2
LOG=logs/geobench_fetch.log
PY=./.venv-geobench/bin/python
: > "$LOG"
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
step() {
  local name="$1"; shift
  note "BEGIN $name"
  env -u PYTHONPATH PIP_CONSTRAINT= "$@" >> "$LOG" 2>&1
  local rc=$?
  note "END   $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    note "FAILED $name rc=$rc"
    printf '{"failed_step":"%s","rc":%s,"utc":"%s"}\n' "$name" "$rc" "$(date -u +%FT%TZ)" \
      > logs/geobench_fetch_FAILED.json
    exit "$rc"
  fi
}

note "START"
mkdir -p "$ROOTDIR"

# 디스크 가드: 최소 200GB 남아 있어야 진행 (총 102GB + 여유)
avail_gb=$(df -BG --output=avail "$ROOTDIR" | tail -1 | tr -dc '0-9')
note "disk avail=${avail_gb}GB"
if [ "${avail_gb:-0}" -lt 200 ]; then
  note "FAILED disk_guard avail=${avail_gb}GB < 200GB"
  printf '{"failed_step":"disk_guard","avail_gb":%s}\n' "${avail_gb:-0}" > logs/geobench_fetch_FAILED.json
  exit 1
fi

# 컨테이너 torch 는 NumPy 1.x 로 컴파일됐다. venv 의 numpy 2.x 가 이를 깨뜨린다.
step numpy_pin ./.venv-geobench/bin/pip install -q "numpy<2"
step numpy_check $PY -c "
import numpy, torch
print('numpy', numpy.__version__, 'torch', torch.__version__)
t = torch.arange(6).reshape(2,3)
a = t.numpy()                      # NumPy 1.x/2.x 불일치면 여기서 터진다
assert a.shape == (2,3), a.shape
print('tensor->numpy OK', a.tolist())
"

# 병렬 다운로더(실측 4.3 MB/s vs 단일 1.0 MB/s). sha256 을 대조하고 재개 가능.
step test_download    $PY code/test_geobench_download.py
step fetch_fotw       $PY code/geobench_parallel_download.py fotw --root "$ROOTDIR" --workers 8
step verify_fotw      $PY code/geobench_verify_one.py fotw      --root "$ROOTDIR"
step fetch_pastis     $PY code/geobench_parallel_download.py pastis --root "$ROOTDIR" --workers 12
step verify_pastis    $PY code/geobench_verify_one.py pastis    --root "$ROOTDIR"
step fetch_den        $PY code/geobench_parallel_download.py dynamic_earthnet --root "$ROOTDIR" --workers 12
step verify_den       $PY code/geobench_verify_one.py dynamic_earthnet --root "$ROOTDIR"

note "ALL_DONE"
printf '{"status":"ok","utc":"%s"}\n' "$(date -u +%FT%TZ)" > logs/geobench_fetch_DONE.json
