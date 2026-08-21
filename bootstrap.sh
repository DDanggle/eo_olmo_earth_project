#!/usr/bin/env bash
# OlmoEarth 세션 부트스트랩 — H200 컨테이너 안에서 실행.
#
# 사용법 (로컬에서):
#   ./nexus push projects/olmoearth/bootstrap.sh olmoearth/
#   ./nexus run "bash /home/work/data/olmoearth/bootstrap.sh"
#
# 멱등: 세션이 회수돼도 이 스크립트 한 번이면 작업 환경이 복원된다.
# 원칙: 영구 저장소(/home/work/data)에만 상태를 남긴다 (h100-setup/CLAUDE.md 참고).
#
# 주의: olmoearth-runner 가 Python <3.12 를 요구하는데 NGC 이미지는 py3.12 이므로
# 시스템 파이썬을 쓰지 않고 uv 로 Python 3.11 venv 를 영구 저장소에 만든다.
# (torch 는 venv 안에 별도 설치됨 — uv 캐시가 영구라 최초 1회만 느리다)
set -euo pipefail

DATA=/home/work/data
PROJ=$DATA/olmoearth
REPO=$PROJ/olmoearth_projects
VENV=$PROJ/.venv

# 캐시·툴 전부 영구 저장소로
export UV_INSTALL_DIR=$DATA/.local/bin
export UV_CACHE_DIR=$DATA/.cache/uv
export UV_PYTHON_INSTALL_DIR=$DATA/.local/uv-python
export PIP_CACHE_DIR=$DATA/.cache/pip
export HF_HOME=$DATA/.cache/huggingface
export TORCH_HOME=$DATA/.cache/torch
export PATH=$UV_INSTALL_DIR:$PATH

mkdir -p "$PROJ"/{checkpoints,logs,scratch} "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

echo "==> 1/5 uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

echo "==> 2/5 레포"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only || echo "    (pull 실패 — 로컬 변경이 있으면 정상, 계속 진행)"
else
    git clone https://github.com/allenai/olmoearth_projects "$REPO"
fi

echo "==> 3/5 Python 3.11 venv + 패키지 (uv.lock 고정 버전 — fresh resolve 금지)"
# 교훈(2026-08-13): fresh resolve로 설치하면 rslearn/runner가 lock보다 최신으로 풀려
# model.yaml과 버전 스큐가 남 (예: enable_confusion_matrix 미지원 → rslearn exit 2).
export UV_PROJECT_ENVIRONMENT=$VENV
(cd "$REPO" && uv sync --frozen --extra dev)

echo "==> 4/5 환경변수를 .bashrc 에 심기 (다음 SSH 접속부터 자동 적용)"
MARKER="# >>> olmoearth bootstrap >>>"
if ! grep -qF "$MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF
$MARKER
export PATH=$UV_INSTALL_DIR:\$PATH
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR
source $VENV/bin/activate
# <<< olmoearth bootstrap <<<
EOF
fi

echo "==> 5/5 스모크 테스트"
"$VENV/bin/python" - <<'EOF'
import torch
import rslearn
import olmoearth_run
import olmoearth_projects
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")
print("rslearn / olmoearth_run / olmoearth_projects import OK")
EOF

echo
echo "부트스트랩 완료. 프로젝트 루트: $PROJ / venv: $VENV"
echo "다음 단계: sample 프로젝트 엔드투엔드 추론 (GOAL.md 루프 1 참고)"
