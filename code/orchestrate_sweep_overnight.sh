#!/usr/bin/env bash
# 야간 오케스트레이터 — 서버 상주, SSH 터널과 무관하게 sweep을 완주함.
# 각 지역: 이전 지역 DONE 대기 → post gate(서버) → 좌표 추출 → 판독(JSON만 생성,
# 사람이 읽는 것은 아침에) → 다음 지역 병렬 runner.
# pre manifest는 로컬에서 생성·커밋·푸시된 것을 runner가 검증함.
set -uo pipefail
cd /home/work/data
PY="env -u PYTHONPATH /home/work/data/olmoearth/.venv-master/bin/python"
MAN=/home/work/data/olmoearth/confirmatory_manifests
LOGDIR=/home/work/data/logs

wait_done () {  # $1=fold
  until grep -q "CONFIRMATORY $1 DONE" "$LOGDIR/confirm_${1#holdout_}.log" 2>/dev/null; do sleep 180; done
}

close_region () {  # $1=fold  $2=allow_flag("--allow-no-snapshot" or "")
  local F=$1
  $PY code/verify_confirmatory_release.py --fold "$F" --mode post \
      --recipe /home/work/data/olmoearth/recipe_frozen_v2.json \
      --results-root /home/work/data/olmoearth/confirmatory \
      --manifest-root "$MAN" \
      --folds-json /home/work/data/olmoearth/sen12_gp_contract/loco_folds.json \
      --live-code /home/work/data/code $2 || { echo "POST GATE FAIL: $F"; return 1; }
  $PY code/export_fold_coords.py "$F" \
      "/home/work/data/olmoearth/confirmatory/$F/P4_seed1/per_sample/$F/P4_test.jsonl" || true
  $PY code/read_confirmatory_region.py --fold "$F" \
      --root /home/work/data/olmoearth/confirmatory \
      --gate "$MAN/${F}_post.json" \
      --coords "/home/work/data/olmoearth/gp_official_bundle/tile_coords_${F}.json" \
      --out "/home/work/data/olmoearth/confirmatory/$F/read_summary.json" \
      > /dev/null 2>&1 || echo "read 경고: $F"
  echo "CLOSED: $F"
}

run_region () {  # $1=fold
  local F=$1
  bash code/run_confirmatory_region_parallel.sh "$F" \
      > "$LOGDIR/confirm_${F#holdout_}.log" 2>&1
}

echo "[orchestrator] indonesia 대기"
wait_done holdout_indonesia
close_region holdout_indonesia ""

for F in holdout_itogon holdout_kyrgyzstan1 holdout_kyrgyzstan2 holdout_newzealand; do
  echo "[orchestrator] $F 시작 $(date -Iseconds)"
  run_region "$F" || { echo "RUNNER FAIL: $F"; exit 1; }
  close_region "$F" ""
done
echo "SWEEP ALL DONE $(date -Iseconds)"
