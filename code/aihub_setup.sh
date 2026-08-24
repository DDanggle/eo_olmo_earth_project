#!/usr/bin/env bash
# AI-Hub aihubshell 설치 + 인증 확인. 서버에서 실행한다.
#
# 검증 출처: https://aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100 (2026-08-25 확인)
#   -mode l   데이터셋 정보 조회
#   -mode d   데이터셋 다운로드
#   -mode pl  데이터패키지 정보 조회
#   -mode pd  데이터패키지 다운로드
#   인증: -aihubapikey '<키>'   (특수문자 때문에 홑따옴표 필수)
#
# 키는 인자로 받지 않는다. 셸 히스토리와 프로세스 목록에 남기지 않기 위해
# 반드시 환경변수 AIHUB_API_KEY로만 전달한다.
#
# 사용:
#   set -a; . /home/work/data/olmoearth/.env.aihub; set +a
#   bash code/aihub_setup.sh check
#   bash code/aihub_setup.sh list                  # 전체 데이터셋 목록
#   bash code/aihub_setup.sh list 71641            # 특정 datasetkey의 filekey 목록
set -euo pipefail

BIN_DIR="/home/work/data/olmoearth/bin"
SHELL_BIN="$BIN_DIR/aihubshell"
LOG_DIR="/home/work/data/olmoearth/aihub"

if [ -z "${AIHUB_API_KEY:-}" ]; then
    echo "AIHUB_API_KEY가 비어 있다. .env.aihub를 source한 뒤 다시 실행한다." >&2
    exit 1
fi

mkdir -p "$BIN_DIR" "$LOG_DIR"
if [ ! -x "$SHELL_BIN" ]; then
    echo "▸ aihubshell 다운로드"
    curl -fsSL -o "$SHELL_BIN" https://api.aihub.or.kr/api/aihubshell.do
    chmod +x "$SHELL_BIN"
fi
echo "▸ aihubshell: $SHELL_BIN ($(wc -c <"$SHELL_BIN") bytes, sha256 $(sha256sum "$SHELL_BIN" | cut -c1-16)…)"

case "${1:-check}" in
    check)
        # 인증만 확인한다. 목록 앞부분만 보고 키 노출 없이 성공/실패를 판정한다.
        if "$SHELL_BIN" -mode l -aihubapikey "$AIHUB_API_KEY" 2>&1 | head -20 | tee "$LOG_DIR/auth_check.log" | grep -qi "인증\|success\|데이터셋"; then
            echo "▸ 인증 확인됨. 로그: $LOG_DIR/auth_check.log"
        else
            echo "▸ 인증 실패로 보인다. $LOG_DIR/auth_check.log 확인" >&2
            exit 1
        fi
        ;;
    list)
        if [ -n "${2:-}" ]; then
            "$SHELL_BIN" -mode l -datasetkey "$2" -aihubapikey "$AIHUB_API_KEY" \
                | tee "$LOG_DIR/dataset_$2_files.txt"
            echo "▸ filekey 목록 저장: $LOG_DIR/dataset_$2_files.txt"
        else
            "$SHELL_BIN" -mode l -aihubapikey "$AIHUB_API_KEY" | tee "$LOG_DIR/dataset_list.txt"
            echo "▸ 전체 목록 저장: $LOG_DIR/dataset_list.txt"
        fi
        ;;
    *)
        echo "사용법: aihub_setup.sh {check|list [datasetkey]}" >&2
        exit 2
        ;;
esac
