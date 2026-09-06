"""긴 작업 전에 강제로 도는 5초짜리 예비검사.

왜 생겼나 (2026-09-06): GEO-Bench fotw 검증 스크립트가 존재하지 않는 모달리티 이름
`rgb` 를 넘겨 실패했다. 그런데 그 실패가 4GB 다운로드가 끝난 **뒤에** 드러났고,
체인이 fail-closed 로 멈추면서 뒤따르는 97GB 다운로드가 시작조차 못 했다.
클래스 메타데이터만 읽었으면 5초 만에 잡혔을 버그다.

규칙: **비싼 단계(다운로드·추출·학습) 앞에는 반드시 이 예비검사를 둔다.**
예비검사는 네트워크·GPU를 쓰지 않고 계약(이름·모양·인자)만 확인한다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS: list[str] = []


def check(name: str, cond: bool, detail="") -> bool:
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return bool(cond)


def preflight_geobench(datasets: list[str]) -> None:
    """다운로드 전에: 클래스 해석 / url·paths·sha 존재 / band_order 형태 / 검증기 인자 정합."""
    import inspect
    from geobench_fetch_one import resolve, DATASETS
    from geobench_verify_one import OUR_S2

    for name in datasets:
        print(f"[{name}]")
        if not check(f"{name}: DATASETS 에 등록", name in DATASETS):
            continue
        try:
            cls = resolve(name)
        except Exception as e:
            check(f"{name}: 클래스 해석", False, f"{type(e).__name__}: {e}")
            continue
        check(f"{name}: 클래스 = {cls.__name__}", True)
        url = getattr(cls, "url", "")
        paths = list(getattr(cls, "paths", []) or [])
        shas = list(getattr(cls, "sha256str", []) or [])
        check(f"{name}: url 템플릿 존재", bool(url) and "{}" in url, url[:60])
        check(f"{name}: paths 비어있지 않음", bool(paths), paths)
        check(f"{name}: sha256 개수 == paths 개수", len(shas) == len(paths),
              f"{len(shas)} vs {len(paths)}")

        # band_order 는 dict(모달리티별) 또는 tuple(단일). 검증기가 둘 다 다뤄야 한다.
        bdo = getattr(cls, "band_default_order", None)
        check(f"{name}: band_default_order 존재", bdo is not None, type(bdo).__name__)
        if isinstance(bdo, dict):
            mods = list(bdo)
            check(f"{name}: 모달리티 {mods}", True)
            if "s2" in bdo:
                have = [b for b in OUR_S2 if b in list(bdo["s2"])]
                check(f"{name}: 우리 10밴드 중 {len(have)}개 보유", len(have) >= 4, have)
        elif bdo is not None:
            check(f"{name}: 단일 모달리티 밴드 {list(bdo)}", True)

        # __init__ 이 요구하는 필수 인자를 우리가 전부 넘기는지
        try:
            sig = inspect.signature(cls.__init__)
            required = [p.name for p in sig.parameters.values()
                        if p.name != "self" and p.default is inspect.Parameter.empty
                        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            supplied = {"root", "split", "band_order", "download"}
            missing = [r for r in required if r not in supplied]
            check(f"{name}: 필수 인자 모두 공급 (필수={required})", not missing, missing)
        except Exception as e:
            check(f"{name}: 서명 검사", False, str(e)[:80])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["geobench"])
    ap.add_argument("--datasets", nargs="+", default=["fotw", "pastis", "dynamic_earthnet"])
    a = ap.parse_args()
    print(f"=== PREFLIGHT [{a.kind}] — 네트워크·GPU 없이 계약만 검사 ===")
    if a.kind == "geobench":
        preflight_geobench(a.datasets)
    print()
    if FAILS:
        print(f"PREFLIGHT FAILED ({len(FAILS)}): {FAILS}")
        print("비싼 단계를 시작하지 않는다.")
        return 1
    print("PREFLIGHT PASSED — 비싼 단계 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
