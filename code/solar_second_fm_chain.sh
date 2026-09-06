#!/usr/bin/env bash
# Solar(task2) 에서 두 번째 FM 캐시 비교 — MS-102 설계를 두 번째 downstream 과업에 복제한다.
# 왜: "어느 표현이 캐시 가치가 있나" 결론이 지금 Sen12 한 과업에만 얹혀 있다(최대 구멍).
# GPU1 만 쓴다(규약 4b). 남의 프로세스가 GPU1 에 있으면 시작하지 않는다.
# rc 는 명령 직후 포착(M107). 실패하면 fail-closed. 다운로드(네트워크)와는 독립이라 함께 돌아도 된다.
set -uo pipefail
cd /home/work/data/olmoearth
PY=./.venv-master/bin/python
LOG=logs/solar_second_fm.log
: > "$LOG"
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
fail() { note "FAILED step=$1 rc=$2"; printf '{"failed_step":"%s","rc":%s,"utc":"%s"}\n' "$1" "$2" "$(date -u +%FT%TZ)" > logs/solar_second_fm_FAILED.json; exit "$2"; }
run()  { local name="$1"; shift; note "BEGIN $name"; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$@" >> "$LOG" 2>&1; local rc=$?; note "END   $name rc=$rc"; [ "$rc" -ne 0 ] && fail "$name" "$rc"; return 0; }

gpu_guard() {
  # GPU1 에 우리 것이 아닌 compute 프로세스가 있으면 멈춘다.
  local uuid; uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1 | tr -d ' ')
  local foreign; foreign=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}' | while read p; do ps -o cmd= -p "$p" 2>/dev/null | grep -q "venv-master/bin/python code/" || echo "$p"; done)
  if [ -n "$foreign" ]; then note "GPU1 에 타 프로세스 있음: $foreign — 규약 4b 대로 중단"; fail "gpu_guard" 3; fi
  note "GPU1 free (used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ') MiB)"
}

note "START solar_second_fm"
gpu_guard

# 0) 보호 4파일 해시 기록 (이 체인은 그 파일을 건드리지 않는다)
for f in pilot_sen12_gp_heads.py sen12_official_baselines.py extract_sen12_fold_cache.py audit_sen12_fold_cache.py; do note "protected $f $(sha256sum code/$f | cut -c1-16) $(stat -c %Y code/$f)"; done

# 1) Galileo base 를 Solar 타일(task2_cache, T=4)에 추출
run x_galileo_task2 $PY code/extract_galileo_cache.py --size base --src task2_cache --out galileo_task2

# 2) 동일 디코더로 8폴드: OlmoEarth(task2_cache) 와 Galileo(galileo_task2) — 매칭 비교
FOLDS="holdout_task2_fold0 holdout_task2_fold1 holdout_task2_fold2 holdout_task2_fold3 holdout_task2_fold4 holdout_task2_fold5 holdout_task2_fold6 holdout_task2_fold7"
for cache in task2_cache galileo_task2; do
  for fold in $FOLDS; do
    out="bv1_runs_task2/$cache"
    [[ -f "$out/${fold}_seed1.json" ]] && { note "skip $cache $fold (exists)"; continue; }
    gpu_guard
    run "dec_${cache}_${fold}" $PY code/cache_decoder_train.py --cache "$cache" --fold "$fold" --seed 1 \
        --folds task2_contract/loco_folds.json --contract task2_contract/sample_contract.jsonl --out "$out"
  done
done

note "SOLAR_SECOND_FM_DONE"
printf '{"status":"ok","utc":"%s"}\n' "$(date -u +%FT%TZ)" > logs/solar_second_fm_DONE.json
