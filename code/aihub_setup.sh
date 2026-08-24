#!/usr/bin/env bash
# AI-Hub aihubshell 설치 · 인증확인 · 목록조회 · 다운로드. 서버(H200)에서 직접 실행한다.
#
# 실제 스크립트(aihubshell v0.6, 2026-08-25 확인)에서 확정한 사실:
#   설치      curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
#   인증      -aihubapikey '<키>'  또는  환경변수 AIHUB_APIKEY  (line 94의 fallback)
#   모드      l 조회 / d 다운로드 / pl 패키지조회 / pd 패키지다운로드
#   filekey   -filekey 로 부분 선택. 기본값은 "all" 이므로 생략하면 전체를 받는다
#   동작      CWD에 download.tar를 받아 풀고 .part 파일을 자동 병합한다
#
# 키는 명령행 인자로 받지 않는다. AIHUB_APIKEY를 export해서만 전달한다
# (셸 히스토리·프로세스 목록 노출 방지).
#
# 사용:
#   set -a; . /home/work/data/olmoearth/.env.aihub; set +a
#   bash code/aihub_setup.sh check
#   bash code/aihub_setup.sh list 71363               # filekey 목록 (무엇을 받을지 먼저 본다)
#   bash code/aihub_setup.sh get 71363 <filekey,...>  # 부분 다운로드 + manifest 기록
set -euo pipefail

ROOT="/home/work/data/olmoearth/aihub"
BIN="$ROOT/bin/aihubshell"

if [ -z "${AIHUB_APIKEY:-}" ]; then
    echo "AIHUB_APIKEY가 비어 있다. /home/work/data/olmoearth/.env.aihub를 source한다." >&2
    exit 1
fi
export AIHUB_APIKEY

mkdir -p "$(dirname "$BIN")" "$ROOT/logs"
if [ ! -x "$BIN" ]; then
    echo "▸ aihubshell 다운로드"
    curl -fsSL -o "$BIN" https://api.aihub.or.kr/api/aihubshell.do
    chmod +x "$BIN"
fi
echo "▸ aihubshell $(grep -m1 -oE "version [0-9.]+ v[0-9.]+" "$BIN" || echo "(버전 미확인)")"

case "${1:-check}" in
check)
    # 인증만 확인한다. 키는 출력하지 않는다.
    if "$BIN" -mode l -aihubapikey "$AIHUB_APIKEY" >"$ROOT/logs/auth_check.log" 2>&1; then
        echo "▸ 인증 OK. 목록 $(wc -l <"$ROOT/logs/auth_check.log") 행 수신 → $ROOT/logs/auth_check.log"
    else
        echo "▸ 인증 실패. $ROOT/logs/auth_check.log 확인" >&2
        exit 1
    fi
    ;;
list)
    KEY="${2:?datasetkey가 필요하다}"
    "$BIN" -mode l -datasetkey "$KEY" -aihubapikey "$AIHUB_APIKEY" \
        | tee "$ROOT/logs/dataset_${KEY}_filetree.txt"
    echo "▸ filekey 목록 → $ROOT/logs/dataset_${KEY}_filetree.txt"
    echo "▸ 다음: 10m Sentinel-2에 해당하는 filekey만 골라 get에 넘긴다. 전체를 받지 않는다."
    ;;
get)
    KEY="${2:?datasetkey가 필요하다}"
    FILEKEYS="${3:?filekey가 필요하다. 전체 다운로드를 막기 위해 필수로 둔다}"
    DEST="$ROOT/raw/$KEY"
    mkdir -p "$DEST"
    echo "▸ 다운로드 datasetkey=$KEY filekey=$FILEKEYS → $DEST"
    ( cd "$DEST" && "$BIN" -mode d -datasetkey "$KEY" -filekey "$FILEKEYS" \
        -aihubapikey "$AIHUB_APIKEY" ) 2>&1 | tee "$ROOT/logs/download_${KEY}.log"
    # 논문 재현성용 manifest. 원본을 공개하지 않고 이것만 공개한다 (AIHUB_INQUIRY.md 질문 2).
    ( cd "$DEST" && find . -type f ! -name "download*.tar" -print0 \
        | sort -z | xargs -0 sha256sum ) > "$ROOT/manifest_${KEY}.sha256"
    echo "▸ manifest: $ROOT/manifest_${KEY}.sha256 ($(wc -l <"$ROOT/manifest_${KEY}.sha256") 파일)"
    du -sh "$DEST"
    ;;
*)
    echo "사용법: aihub_setup.sh {check | list <datasetkey> | get <datasetkey> <filekey,...>}" >&2
    exit 2
    ;;
esac
