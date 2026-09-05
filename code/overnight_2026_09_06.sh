#!/usr/bin/env bash
# 2026-09-06 밤 체인 — CPU/네트워크만. GPU 사용 안 함 (GPU1에 타 사용자 작업 있음, 규약 4b).
#
# rc 처리: `echo "$(date) name rc=$?"` 는 $(date) 가 $? 를 덮어쓴다(M107).
# 여기서는 실행 직후 rc 를 변수에 담고, 실패하면 즉시 멈춘다(fail-closed).
set -uo pipefail
cd /home/work/data/olmoearth
PY=./.venv-master/bin/python
LOG=logs/overnight_20260906.log
: > "$LOG"

stamp() { date -u +%FT%TZ; }
note()  { echo "$(stamp) $*" >> "$LOG"; }

fail() {                      # fail <step> <rc>
  note "FAILED step=$1 rc=$2"
  printf '{"failed_step":"%s","rc":%s,"utc":"%s"}\n' "$1" "$2" "$(stamp)" > logs/overnight_20260906_FAILED.json
  exit "$2"
}

run() {                       # run <name> <cmd...>   — rc 를 올바르게 잡는다
  local name="$1"; shift
  note "BEGIN $name"
  env -u PYTHONPATH "$@" >> "$LOG" 2>&1
  local rc=$?                 # <-- 반드시 명령 직후. $(date) 보다 먼저.
  note "END   $name rc=$rc"
  [ "$rc" -ne 0 ] && fail "$name" "$rc"
  return 0
}

note "START overnight_20260906"

# --- 0) rc 포착 자체를 자가 검증한다 (M107 재발 방지) ---
( exit 7 ); selfrc=$?
if [ "$selfrc" -ne 7 ]; then fail "self_check_rc_capture" 1; fi
note "self_check rc capture OK (got $selfrc)"

# --- 0b) GPU 가드: 우리 것이 아닌 작업이 있으면 GPU 단계는 아예 만들지 않는다 ---
gpu1_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
note "GPU1 used=${gpu1_mb:-unknown} MiB (이 체인은 GPU를 쓰지 않는다)"

# --- 1) 단위 테스트 (실패하면 여기서 끝) ---
run unit_tests $PY code/test_cache_probes.py

# --- 2) 라벨 없는 프로브 14 캐시 ---
CACHES="olmo_cache_pool16 olmo_nano olmo_tiny olmo_base_half \
        clay_cache_native16 clay_cache_native16_last clay_cache_in256 clay_in256_half \
        galileo_cache galileo_cache_groupcat galileo_nano galileo_tiny galileo_base_half \
        prithvi_cache"
for c in $CACHES; do
  run "probe_$c" $PY code/cache_probes.py "$c"
done

# --- 3) 상관 분석 (사전 등록 게이트) ---
run probe_correlation $PY code/probe_correlation.py

# --- 4) GEO-Bench-2 격리 설치 ---
if [ ! -d .venv-geobench ]; then
  run geobench_venv python3 -m venv .venv-geobench
fi
run geobench_pip ./.venv-geobench/bin/pip install -q --upgrade pip
run geobench_install ./.venv-geobench/bin/pip install -q GeoBenchV2
run geobench_import ./.venv-geobench/bin/python -c "import geobench_v2; print(geobench_v2.__file__)"

note "ALL_DONE"
printf '{"status":"ok","utc":"%s"}\n' "$(stamp)" > logs/overnight_20260906_DONE.json
