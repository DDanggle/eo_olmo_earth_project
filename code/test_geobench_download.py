"""geobench_parallel_download.plan_ranges 검증 — 빈틈·겹침·누락이 없어야 한다."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geobench_parallel_download import plan_ranges

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

for total, workers in [(1, 1), (10, 3), (100, 7), (3_990_000_000, 8), (53_400_000_000, 12),
                       (5, 8), (2, 2), (999983, 6)]:
    r = plan_ranges(total, workers)
    covered = sum(e - s + 1 for s, e in r)
    gaps = [(r[i][1], r[i + 1][0]) for i in range(len(r) - 1) if r[i + 1][0] != r[i][1] + 1]
    check(f"total={total} w={workers}: 전체 덮음", covered == total, f"{covered} vs {total}")
    check(f"total={total} w={workers}: 빈틈/겹침 없음", not gaps, gaps[:3])
    check(f"total={total} w={workers}: 시작 0, 끝 total-1",
          r[0][0] == 0 and r[-1][1] == total - 1, f"{r[0][0]}..{r[-1][1]}")
    check(f"total={total} w={workers}: 청크 수 <= workers", len(r) <= max(workers, 1), len(r))

for bad in [(0, 4), (-5, 4), (10, 0)]:
    try:
        plan_ranges(*bad); check(f"잘못된 입력 {bad} 거부", False, "예외 없음")
    except ValueError:
        check(f"잘못된 입력 {bad} 거부", True)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS[:5]}"); sys.exit(1)
print("ALL TESTS PASSED")
