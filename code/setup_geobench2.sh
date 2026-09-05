#!/usr/bin/env bash
# GEO-Bench-2 격리 설치. .venv-master 는 건드리지 않는다 (확증 실행 경로 보호).
#
# 실패 원인(2026-09-06): NVIDIA 컨테이너가 pip constraint 로 torch 를 고정
#   geobenchv2 0.9 depends on torch>=2.0
#   The user requested (constraint) torch==2.7.0a0+...nv25.3
# 고정된 nv 빌드는 PyPI 에 없어 해석 불가. 해법 두 가지를 함께 쓴다:
#   (1) venv 를 --system-site-packages 로 만들어 컨테이너 torch 를 그대로 본다
#   (2) PIP_CONSTRAINT 를 비워 constraint 해석을 끈다
# rc 는 명령 직후에 잡는다 (M107).
set -uo pipefail
cd /home/work/data/olmoearth
LOG=logs/geobench_setup.log
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
    printf '{"failed_step":"%s","rc":%s}\n' "$name" "$rc" > logs/geobench_setup_FAILED.json
    exit "$rc"
  fi
}

note "START"
note "container torch: $(./.venv-master/bin/python -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"

rm -rf .venv-geobench
step venv python3 -m venv --system-site-packages .venv-geobench
step pip_upgrade ./.venv-geobench/bin/pip install -q --upgrade pip
step install ./.venv-geobench/bin/pip install -q GeoBenchV2
step import ./.venv-geobench/bin/python -c "import geobench_v2, torch; print('geobench_v2', geobench_v2.__file__); print('torch', torch.__version__)"

note "cli:"
ls .venv-geobench/bin | grep -i geobench >> "$LOG" 2>&1
note "SETUP_DONE"
printf '{"status":"ok"}\n' > logs/geobench_setup_DONE.json
