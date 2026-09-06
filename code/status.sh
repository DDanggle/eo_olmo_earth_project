#!/usr/bin/env bash
# 한 명령으로 서버 전체 상태를 본다. 어느 세션에서든 이것만 돌리면 "지금 뭐가 돌고 뭐가 죽었나"가 나온다.
# 판정 규칙(2026-09-06): 체인 생사는 nx status 가 아니라 pgrep + GPU + 로그 mtime + DONE/FAILED 마커로 본다.
cd /home/work/data/olmoearth 2>/dev/null || { echo "작업 디렉토리 없음"; exit 1; }
now=$(date -u +%s)
age() { local f=$1; [ -e "$f" ] || { echo "-"; return; }; echo "$(( (now - $(stat -c %Y "$f")) / 60 ))분전"; }

echo "=== $(date -u +%FT%TZ)  ($(date +%H:%M) KST) ==="
echo
echo "## GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
  awk -F', ' '{printf "  GPU%s  %s  %s/%s\n",$1,$2,$3,$4}'
ours=$(pgrep -f "venv-master/bin/python code/|venv-geobench/bin/python code/" | wc -l)
echo "  우리 파이썬 프로세스: ${ours}개"
echo
echo "## 실행 중 체인/작업"
pgrep -af "^bash .*code/.*\.sh|python code/" 2>/dev/null | grep -v "bash -lc\|status.sh" | \
  sed 's/^\([0-9]*\) .*code\//  \1  /' | cut -c1-110 || echo "  (없음)"
echo
echo "## 마커 (DONE / FAILED)"
for f in logs/*_DONE.json logs/*_FAILED.json logs/*FAILED.json; do
  [ -e "$f" ] && printf "  %-46s %s\n" "$(basename $f)" "$(age $f)"
done 2>/dev/null | sort -u
echo
echo "## GEO-Bench-2 데이터"
for d in fotw pastis dynamic_earthnet; do
  p=geobench2/$d
  if [ -d "$p" ]; then
    sz=$(du -sb "$p" 2>/dev/null | cut -f1)
    parts=$(ls -d $p/*.parts 2>/dev/null | wc -l)
    v=artifacts/geobench_verify_$d.json
    vs="미검증"; [ -e "$v" ] && vs=$(python3 -c "import json;print('검증OK' if json.load(open('$v'))['ok'] else '검증FAIL')" 2>/dev/null)
    # 우선순위: 명시적 실패 > 검증 성공 > DONE > partial/incomplete. .parts는 전송 구현
    # 세부사항일 뿐이며 검증 완료 상태를 다시 "받는중"으로 내리지 않는다.
    st="미완료"
    [ $parts -gt 0 ] && st="받는중"
    [ -e "logs/${d}_DONE.json" ] && st="완료  "
    [ "$vs" = "검증OK" ] && st="검증완료"
    [ -e "logs/${d}_FAILED.json" ] && st="실패!!"
    printf "  %-18s %7.2f GB  %s  %s\n" "$d" "$(echo $sz | awk '{print $1/1e9}')" "$st" "$vs"
  else
    printf "  %-18s 없음\n" "$d"
  fi
done
echo
echo "## 최근 갱신 로그 (10분 이내)"
find logs -name "*.log" -mmin -10 2>/dev/null | while read f; do printf "  %-40s %s\n" "$(basename $f)" "$(age $f)"; done
echo
echo "## 보호 4파일 (규약 4c)"
for f in pilot_sen12_gp_heads.py sen12_official_baselines.py extract_sen12_fold_cache.py audit_sen12_fold_cache.py; do
  printf "  %-32s %s\n" "$f" "$(sha256sum code/$f 2>/dev/null | cut -c1-16)"
done
