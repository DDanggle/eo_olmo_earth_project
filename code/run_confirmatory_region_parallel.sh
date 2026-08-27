#!/usr/bin/env bash
# 확증 runner v2 — seed당 3 arm **병렬** (사용자 승인 2026-08-27: GPU1 풀 사용).
# 절차는 순차판과 동일함: pre gate manifest 요구 → snapshot 봉인 → 봉인본 실행.
# 병렬이 결과를 바꾸지 않는 근거: 결정성 계약은 프로세스 단위이며 M38 보강에서
# 경합 하에서도 산출물이 비트 단위 동일함을 실증했음. 벽시계는 비용 지표에서 이미 제외(M38).
# 메모리: P4 ~10GB + P2 ~16GB + P3 ~10GB = ~36GB < 143GB.
set -euo pipefail
FOLD="${1:?fold 이름 필요}"
cd /home/work/data
PYBIN="/home/work/data/olmoearth/.venv-master/bin/python"
RUN="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PYBIN"
CACHE="/home/work/data/olmoearth/sen12_pilot/$FOLD"
OUTROOT="/home/work/data/olmoearth/confirmatory/$FOLD"
SNAP="$OUTROOT/code_snapshot"

if [ -e "$OUTROOT" ]; then echo "중단: $OUTROOT 존재" >&2; exit 2; fi

PRE="/home/work/data/olmoearth/confirmatory_manifests/${FOLD}_pre.json"
echo "=== [0/4] pre gate manifest 확인 ==="
if [ ! -f "$PRE" ]; then echo "중단: $PRE 없음" >&2; exit 3; fi
$PYBIN - "$PRE" /home/work/data/olmoearth/recipe_frozen_v2.json "$FOLD" <<'PYCHK'
import hashlib, json, sys
man = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2])); fold = sys.argv[3]
assert man.get("verdict") == "PASS" and man.get("fold") == fold
body = dict(rec); body.pop("self_sha256", None)
h = hashlib.sha256(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True).encode()).hexdigest()
assert rec.get("self_sha256") == h and man["recipe"]["declared_self_sha256"] == h
print(f"pre gate PASS 확인 · recipe {h[:16]}")
PYCHK

echo "=== [1/4] 코드 스냅샷 봉인 ==="
mkdir -p "$SNAP"
for f in pilot_sen12_gp_heads.py sen12_official_baselines.py \
         extract_sen12_fold_cache.py audit_sen12_fold_cache.py; do
  cp -p "code/$f" "$SNAP/$f"
done
( cd "$SNAP" && sha256sum ./*.py > SHA256SUMS.txt )
date -Iseconds > "$SNAP/started_at.txt"
chmod -w "$SNAP"/*.py "$SNAP/SHA256SUMS.txt" || true
cat "$SNAP/SHA256SUMS.txt"

if [ ! -d "$CACHE/emb_fp16" ]; then
  echo "=== [2/4] 캐시 추출 (snapshot) ==="
  $RUN "$SNAP/extract_sen12_fold_cache.py" --fold "$FOLD" --out /home/work/data/olmoearth/sen12_pilot
fi
echo "=== [3/4] 캐시 감사 (snapshot) ==="
$RUN "$SNAP/audit_sen12_fold_cache.py" --cache "$CACHE" --fold "$FOLD" | tail -6

echo "=== [4/4] seed당 3 arm 병렬 ==="
for S in 1 2 3; do
  echo "--- seed $S : P4 & P2 & P3 병렬 시작 ---"
  pids=()
  for ARM in P4 P2 P3; do
    $RUN "$SNAP/pilot_sen12_gp_heads.py" --arms "$ARM" --epochs 40 --seed "$S" \
         --save-probs --fold "$FOLD" --cache "$CACHE" \
         --out "$OUTROOT/${ARM}_seed${S}" \
         > "$OUTROOT/${ARM}_seed${S}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [ "$fail" -ne 0 ]; then echo "seed $S 에서 실패 — 중단" >&2; exit 4; fi
  echo "--- seed $S 완료 ---"
done
echo "CONFIRMATORY $FOLD DONE"
