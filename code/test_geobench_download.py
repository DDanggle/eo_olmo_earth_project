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
print("RANGE TESTS DONE")

# --- 재시도 로직: sha 불일치가 N-1 번 나고 마지막에 맞으면 성공, N 번 다 틀리면 실패 ---
import geobench_parallel_download as G
import tempfile, os

class _Fake:
    def __init__(self, fail_times): self.calls = 0; self.fail_times = fail_times
    def __call__(self, url, out, want, workers):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise G.ChecksumMismatch(f"fake mismatch #{self.calls}")
        return {"file": out.name, "status": "downloaded", "sha256_ok": True}

_orig_once, _orig_sleep = G._download_once, G.time.sleep
G.time.sleep = lambda *_: None
try:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.bin"
        f = _Fake(fail_times=2); G._download_once = f
        r = G.download_file("u", out, "deadbeef", 4, max_attempts=3)
        check("불일치 2회 후 3번째 성공 → attempts=3", r.get("attempts") == 3 and f.calls == 3, r)
        f = _Fake(fail_times=5); G._download_once = f
        try:
            G.download_file("u", out, "deadbeef", 4, max_attempts=3)
            check("3회 모두 불일치면 SystemExit", False, "예외 없음")
        except SystemExit:
            check("3회 모두 불일치면 SystemExit", f.calls == 3, f"calls={f.calls}")
        f = _Fake(fail_times=0); G._download_once = f
        r = G.download_file("u", out, "deadbeef", 4, max_attempts=3)
        check("첫 시도 성공 → attempts=1", r.get("attempts") == 1 and f.calls == 1, r)
finally:
    G._download_once, G.time.sleep = _orig_once, _orig_sleep

if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS[:5]}"); sys.exit(1)
print("RETRY TESTS PASSED")
