"""제주 v8 추적기의 파일시스템 계약.

기존 146개 스크립트는 `/home/work/data/olmoearth/...` 를 하드코딩해 로컬에서 돌지 않는다.
그 파일들은 이력·차단 상태이므로 건드리지 않고, **v8 이후 새 코드만 이 모듈을 쓴다.**

세 뿌리를 구분한다.

- ``ARTIFACT_ROOT`` — `_work/artifacts`. **git 에 추적된다**(`.gitignore` 의 `!artifacts/**`).
  보고서·매니페스트·작은 JSON 처럼 **눈으로 확인한 근거**만 넣는다.
- ``CACHE_ROOT`` — `_work/.cache`. **추적하지 않는다.** npz 큐브·임베딩처럼 재생성 가능한
  대용량은 전부 여기. 네팔의 `research-private/artifacts` 와 같은 자리다.
- ``WEB_DATA_ROOT`` — 공개 추적 지도가 읽는 파생 자산.

환경변수로 덮어쓸 수 있어, 원격에서 돌릴 때 소스를 고치지 않아도 된다.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(env_var: str, default: Path) -> Path:
    configured = os.environ.get(env_var)
    return Path(configured).expanduser().resolve() if configured else default


#: 추적되는 근거. 큰 배열을 넣지 말 것 — `.gitignore` 가 artifacts 를 강제 포함한다.
ARTIFACT_ROOT = _root("JEJU_ARTIFACT_ROOT", REPO_ROOT / "artifacts")

#: 추적하지 않는 재생성 가능 대용량 (npz 큐브, 임베딩).
CACHE_ROOT = _root("JEJU_CACHE_ROOT", REPO_ROOT / ".cache")

#: 공개 지도가 읽는 파생 자산.
WEB_DATA_ROOT = _root("JEJU_WEB_DATA_ROOT", REPO_ROOT / "apps/oreum-web/public/data")

#: 사전 등록된 계약. 첫 임베딩 실행 전에 커밋되어야 한다.
CONTRACT_ROOT = ARTIFACT_ROOT / "contracts"


def display_path(path: Path) -> str:
    """가능하면 저장소 상대 경로로 출처를 적는다. 보고서에 절대경로가 새지 않게."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def ensure(path: Path) -> Path:
    """디렉터리를 만들고 그대로 돌려준다."""
    path.mkdir(parents=True, exist_ok=True)
    return path
