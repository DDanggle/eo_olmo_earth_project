"""저장소 가드: 체인 스크립트의 `rc=$?` 오용을 탐지한다.

버그(M107): `echo "$(date -u +%FT%TZ) name rc=$?"` 에서 `$(date)` 명령치환이 먼저 실행되며
`$?` 를 덮어쓴다. 결과적으로 로그의 rc 는 항상 date 의 종료코드(0)이며, 단계 실패를 숨긴다.

    $ bash -c 'false; echo "$(date -u +%FT%TZ) rc=$?"'
    2026-09-05T11:11:40Z rc=0        <-- false 인데 0

올바른 형태:  cmd; rc=$?; echo "$(date -u +%FT%TZ) name rc=$rc"

이 테스트는 code/*.sh 전체를 훑어 위반을 찾는다. 위반이 있으면 exit 1.
새 체인을 쓸 때 이 테스트를 먼저 돌린다.
"""
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

# 실제 버그 형태는 **같은 큰따옴표 문자열 안에** `$(` 와 `$?` 가 함께 있는 것이다:
#     echo "$(date -u +%FT%TZ) name rc=$?"
# 따옴표 밖에서 `rc=$?` 를 먼저 잡는 코드는 올바르므로 오탐이면 안 된다:
#     else rc=$?; echo "$(date ...) rc=$rc"
DQ = re.compile(r'"(?:[^"\\]|\\.)*"')       # 큰따옴표 문자열

# 이 스크립트들은 이미 봉인된 결과를 **생산한** 코드다. 계보 보존을 위해 수정하지 않는다.
# 결과는 rc 로그가 아니라 실물 산출물로 검증됐다(audit_summary_vs_raw.py, verify_arch_axes.py).
# 다시 실행할 일이 생기면 그때 사본을 만들어 고친다.
HISTORICAL = {
    "bv1_chain.sh", "bv1_chain2.sh", "bv1_chain3.sh", "bv1_chain4.sh",
    "clay_native_chain.sh", "fewshot_chain.sh", "fewshot_confirmatory_chain.sh",
    "fewshot_ms97_chain.sh", "task2_chain.sh", "task2_fewshot_chain.sh",
    "task2_source_chain.sh", "cachetune_pt1_chain.sh", "c1b_gpu1_watcher.sh",
    "run_gp_official_chain.sh", "run_gp_pilot_chain.sh",
}
ALLOWLIST: set[str] = set(HISTORICAL)


def scan(path: Path) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for q in DQ.findall(line):
            if "$(" in q and "$?" in q:
                hits.append((i, s))
                break
    return hits


def main() -> int:
    files = sorted(CODE.glob("*.sh"))
    if not files:
        print("FAIL: code/*.sh 를 하나도 못 찾음 — 경로 오류일 수 있다 (공허한 통과 방지)")
        return 1
    violations = {}
    for f in files:
        if f.name in ALLOWLIST:
            continue
        h = scan(f)
        if h:
            violations[f.name] = h
    print(f"검사한 스크립트: {len(files)}개")
    if not violations:
        print("PASS: `$(...)` 뒤에서 `$?` 를 읽는 줄 없음")
        return 0
    n = sum(len(v) for v in violations.values())
    print(f"FAIL: {len(violations)}개 파일, {n}개 위반\n")
    for name, hits in sorted(violations.items()):
        print(f"  {name}")
        for ln, txt in hits:
            print(f"    :{ln}  {txt[:120]}")
    print("\n고치는 법: `cmd; rc=$?; echo \"$(date ...) name rc=$rc\"`")
    return 1


if __name__ == "__main__":
    sys.exit(main())
