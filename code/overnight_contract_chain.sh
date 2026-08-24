#!/usr/bin/env bash
# 야간 무인 체인 — 계약 dose-response를 v1에서 마무리하고 v1.2로 복제한다.
#
# 단계
#   1. 실행 중인 v1 dose run 완료 대기
#   2. v1 분석 -> contract_dose_v1_analysis.json
#   3. v1.2 dose run (같은 8 site-years, 같은 밴드 순서 축)
#   4. v1.2 분석 -> contract_dose_v12_analysis.json
#   5. OVERNIGHT_COMPLETE.json
#
# 묻는 것: 밴드 순서 계약 불일치의 용량-반응과 `진단 눈멂`이 두 번째 릴리스에서도 재현되는가.
# 한 릴리스에서만 나오면 일반 주장을 할 수 없다.
#
# 안전장치: 어느 단계든 실패하면 즉시 멈추고 FAILED marker를 남긴다.
# 선택한 GPU에 다른 프로세스가 있으면 dose 스크립트가 스스로 거부한다.
set -euo pipefail

ROOT=/home/work/data/olmoearth
GPU="${GPU:-1}"
DS="$ROOT/release_audit_p0/smoke_dataset"
PY="$ROOT/.venv-data/bin/python"
RSLEARN="$ROOT/.venv-master/bin/rslearn"
STATE="$ROOT/artifacts/overnight_chain"
mkdir -p "$STATE" "$ROOT/artifacts/results"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() { log "FAILED at: $*"; echo "{\"failed_at\":\"$*\"}" > "$STATE/FAILED.json"; exit 1; }

cd "$ROOT"

ckpt() {
  env -u PYTHONPATH "$PY" -c "
import json,sys
d=json.load(open('release_audit_p0/checkpoints.json'))
print([m['snapshot_path'] for m in d['models'] if m['repo_id']==sys.argv[1]][0])
" "$1"
}

# ---- 1. v1 run 완료 대기 (최대 60분) ----
log "stage 1: waiting for v1 dose run"
for i in $(seq 1 120); do
  [ -f "$ROOT/artifacts/contract_dose_v1/RUNS_COMPLETE.json" ] && break
  grep -q "RuntimeError\|REFUSED" "$ROOT/.jobs/dose_v1.log" 2>/dev/null && fail "v1 dose run errored"
  sleep 30
done
[ -f "$ROOT/artifacts/contract_dose_v1/RUNS_COMPLETE.json" ] || fail "v1 dose run did not complete in 60min"
log "stage 1 done"

# ---- 2. v1 분석 ----
log "stage 2: analysing v1"
env -u PYTHONPATH "$ROOT/.venv-master/bin/python" code/analyze_contract_dose_response.py \
  --dataset-root "$DS" \
  --work-dir "$ROOT/artifacts/contract_dose_v1" \
  --layer-prefix embeddings_dose_ \
  --out "$ROOT/artifacts/results/contract_dose_v1_analysis.json" \
  > "$STATE/analyze_v1.log" 2>&1 || fail "v1 analysis"
tail -12 "$STATE/analyze_v1.log"
log "stage 2 done"

# ---- 3. v1.2 dose run ----
log "stage 3: v1.2 dose run on GPU$GPU"
MP12=$(ckpt allenai/OlmoEarth-v1_2-Base) || fail "v1.2 checkpoint lookup"
env -u PYTHONPATH "$PY" code/contract_dose_response.py \
  --base-config config/olmo_release_v1_2_legacy.yaml \
  --dataset-root "$DS" \
  --rslearn "$RSLEARN" \
  --model-path "$MP12" \
  --model-env OLMO_V1_2_MODEL_PATH \
  --work-dir "$ROOT/artifacts/contract_dose_v12" \
  --layer-prefix embeddings_dose_v12_ \
  --gpu "$GPU" --execute \
  > "$STATE/run_v12.log" 2>&1 || fail "v1.2 dose run"
log "stage 3 done"

# ---- 4. v1.2 분석 ----
log "stage 4: analysing v1.2"
env -u PYTHONPATH "$ROOT/.venv-master/bin/python" code/analyze_contract_dose_response.py \
  --dataset-root "$DS" \
  --work-dir "$ROOT/artifacts/contract_dose_v12" \
  --layer-prefix embeddings_dose_v12_ \
  --out "$ROOT/artifacts/results/contract_dose_v12_analysis.json" \
  > "$STATE/analyze_v12.log" 2>&1 || fail "v1.2 analysis"
tail -12 "$STATE/analyze_v12.log"
log "stage 4 done"

# ---- 5. 완료 marker ----
env -u PYTHONPATH "$PY" - <<'EOF' > "$STATE/OVERNIGHT_COMPLETE.json"
import json, datetime
from pathlib import Path
root = Path("/home/work/data/olmoearth/artifacts/results")
out = {"finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "releases": {}}
for tag, name in (("v1", "contract_dose_v1_analysis.json"),
                  ("v1_2", "contract_dose_v12_analysis.json")):
    p = root / name
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    out["releases"][tag] = {
        "summary_by_dose": d["summary_by_dose"],
        "diagnostic_blindness": d["diagnostic_blindness"],
    }
blind = {t: [k for k, v in r["diagnostic_blindness"].items()
             if v["cka_stays_high_while_recall_collapses"]]
         for t, r in out["releases"].items()}
out["blind_doses_by_release"] = blind
out["replicates_across_releases"] = bool(
    len(out["releases"]) == 2 and all(blind.values()))
out["reading"] = (
    "replicates_across_releases=true 이면 밴드 순서 계약 불일치에 대한 진단 눈멂이 "
    "두 릴리스에서 모두 나타난 것이다. false 이면 한 릴리스 특성일 수 있으므로 "
    "일반 주장을 하지 않는다.")
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF
log "ALL STAGES COMPLETE -> $STATE/OVERNIGHT_COMPLETE.json"
