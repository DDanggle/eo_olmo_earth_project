#!/usr/bin/env bash
# 확증 실행 — 동결된 recipe로 미열람 지역 하나를 연다.
# 사용법: bash code/run_confirmatory_region.sh <fold_name>
#
# M57 이후 재설계 (감사 [P1] 3건 반영):
#   (1) pre gate를 **첫 단계로 강제**한다. PASS가 아니면 아무것도 만들지 않고 종료한다.
#   (2) 소스를 snapshot으로 봉인하고 **그 snapshot을 실제로 실행한다.**
#       복사만 하고 live 경로를 실행하면 보호장치가 아니라 착각이다(이전 버전의 결함).
#   (3) OUTROOT가 이미 있으면 덮어쓰지 않고 즉시 실패한다.
#
# 동결 내용 (변경 금지): arms P4·P2·P3 / seeds 1 2 3 / epochs 40 / lr 1e-3 /
# clip 없음 / threshold 0.5 / 주지표 양성 macro IoU
set -euo pipefail
FOLD="${1:?fold 이름이 필요하다 (예: holdout_hiroshima)}"
cd /home/work/data
PYBIN="/home/work/data/olmoearth/.venv-master/bin/python"
RUN="env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PYBIN"
CACHE="/home/work/data/olmoearth/sen12_pilot/$FOLD"
OUTROOT="/home/work/data/olmoearth/confirmatory/$FOLD"
SNAP="$OUTROOT/code_snapshot"

# ── (3) 기존 결과 보호 ──
if [ -e "$OUTROOT" ]; then
  echo "중단: $OUTROOT 가 이미 존재한다. 재실행은 사후 선택의 여지를 만든다." >&2
  exit 2
fi

# ── (1) pre gate 강제 ──
# 서버에는 git 저장소가 없으므로 `clean_worktree` 검사가 서버에서는 공허하게 통과한다.
# 그래서 pre gate는 **로컬에서** 돌리고(진짜 git 상태를 봄) 그 manifest를 여기로
# 푸시해야 한다. runner는 그 manifest의 존재·verdict·recipe 해시 일치를 요구한다.
PRE="/home/work/data/olmoearth/confirmatory_manifests/${FOLD}_pre.json"
echo "=== [0/4] pre gate manifest 확인 ==="
if [ ! -f "$PRE" ]; then
  echo "중단: $PRE 가 없다. 로컬에서 pre gate를 통과시키고 manifest를 푸시하라." >&2
  exit 3
fi
$PYBIN - "$PRE" /home/work/data/olmoearth/recipe_frozen_v2.json "$FOLD" <<'PYCHK'
import hashlib, json, sys
man = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2])); fold = sys.argv[3]
assert man.get("verdict") == "PASS", f"pre gate verdict={man.get('verdict')}"
assert man.get("fold") == fold, f"manifest fold 불일치: {man.get('fold')}"
body = dict(rec); body.pop("self_sha256", None)
h = hashlib.sha256(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True).encode()).hexdigest()
assert rec.get("self_sha256") == h, "recipe self_sha 불일치"
assert man["recipe"]["declared_self_sha256"] == h, "manifest가 다른 recipe로 만들어졌다"
print(f"pre gate PASS 확인 · recipe {h[:16]} · HEAD {man.get('git',{}).get('head','?')[:8]}")
PYCHK

# ── (2) 소스 실물 봉인. 이후 모든 실행은 이 사본에서만 한다 ──
echo "=== [1/4] 코드 스냅샷 봉인 ==="
mkdir -p "$SNAP"
for f in pilot_sen12_gp_heads.py sen12_official_baselines.py \
         extract_sen12_fold_cache.py audit_sen12_fold_cache.py; do
  cp -p "code/$f" "$SNAP/$f"
done
( cd "$SNAP" && sha256sum ./*.py > SHA256SUMS.txt )
date -Iseconds > "$SNAP/started_at.txt"
chmod -w "$SNAP"/*.py "$SNAP/SHA256SUMS.txt" || true   # 실수로 덮어쓰는 것을 막는다
echo "봉인 완료: $SNAP"
cat "$SNAP/SHA256SUMS.txt"

if [ ! -d "$CACHE/emb_fp16" ]; then
  echo "=== [2/4] 캐시 추출 (snapshot 실행) ==="
  $RUN "$SNAP/extract_sen12_fold_cache.py" --fold "$FOLD" \
       --out /home/work/data/olmoearth/sen12_pilot
fi

echo "=== [3/4] 캐시 감사 (snapshot 실행) ==="
$RUN "$SNAP/audit_sen12_fold_cache.py" --cache "$CACHE" --fold "$FOLD" | tail -20

echo "=== [4/4] 3 arm x 3 seed (snapshot 실행) ==="
for S in 1 2 3; do
  for ARM in P4 P2 P3; do
    echo "--- $FOLD / $ARM / seed $S ---"
    $RUN "$SNAP/pilot_sen12_gp_heads.py" --arms "$ARM" --epochs 40 --seed "$S" \
         --save-probs --fold "$FOLD" --cache "$CACHE" \
         --out "$OUTROOT/${ARM}_seed${S}"
  done
done
echo "CONFIRMATORY $FOLD DONE"
