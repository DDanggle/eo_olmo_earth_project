"""GEO-Bench-2 파일을 병렬 range 요청으로 받고 sha256 으로 검증한다.

패키지 기본 다운로더는 단일 연결이라 서버에서 ~1.0 MB/s 다. 실측상 6 병렬이면 4.3 MB/s
(pastis 53GB 기준 24시간 -> 3.5시간). 청크를 병렬로 받아 이어붙이고 sha256 을 대조한다.

이미 있고 해시가 맞는 파일은 건너뛴다(재개 가능). 해시가 틀리면 실패로 종료한다 —
받다 만 파일을 성공으로 세지 않는다.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geobench_fetch_one import resolve, DATASETS

UA = {"User-Agent": "curl/8"}


def sha256_of(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def plan_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    """[start, end] 폐구간 목록. 빈틈·겹침이 없어야 한다."""
    if total <= 0 or workers < 1:
        raise ValueError(f"total={total} workers={workers}")
    workers = min(workers, total)
    step = -(-total // workers)                       # 올림 나눗셈
    out = []
    s = 0
    while s < total:
        e = min(s + step, total) - 1
        out.append((s, e))
        s = e + 1
    return out


def fetch_range(url: str, start: int, end: int, dest: Path, retries: int = 5) -> int:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                n = 0
                while chunk := r.read(4 << 20):
                    f.write(chunk); n += len(chunk)
            want = end - start + 1
            if n != want:
                raise IOError(f"short read {n} != {want}")
            return n
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return 0


MAX_ATTEMPTS = 3   # sha256 불일치 시 파일 전체 재시도 횟수 (2026-09-06: 청크 1개 손상으로 20GB 폐기된 사고 이후)


def _download_once(url: str, out: Path, want_sha: str | None, workers: int) -> dict:
    if out.exists() and want_sha:
        got = sha256_of(out)
        if got == want_sha:
            return {"file": out.name, "status": "skip_verified", "bytes": out.stat().st_size}
        out.unlink()
    total = remote_size(url)
    parts = plan_ranges(total, workers)
    tmpdir = out.parent / (out.name + ".parts")
    tmpdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_range, url, s, e, tmpdir / f"{i:05d}")
                for i, (s, e) in enumerate(parts)]
        for f in futs:
            f.result()                                # 예외는 여기서 터진다
    with open(out, "wb") as w:
        for i in range(len(parts)):
            with open(tmpdir / f"{i:05d}", "rb") as r:
                while chunk := r.read(8 << 20):
                    w.write(chunk)
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    dt = time.time() - t0
    got = sha256_of(out)
    ok = (want_sha is None) or (got == want_sha)
    if not ok:
        out.unlink()
        raise ChecksumMismatch(f"sha256 불일치 {out.name}: got {got[:16]} want {want_sha[:16]}")
    return {"file": out.name, "status": "downloaded", "bytes": total,
            "seconds": round(dt, 1), "MB_per_s": round(total / 1e6 / max(dt, 1e-9), 2),
            "sha256_ok": ok}


class ChecksumMismatch(Exception):
    pass


def download_file(url: str, out: Path, want_sha: str | None, workers: int,
                  max_attempts: int = MAX_ATTEMPTS) -> dict:
    """sha256 불일치면 파일 전체를 다시 받는다(최대 max_attempts). 그래도 틀리면 실패로 종료.

    이미 있고 해시가 맞는 파일은 _download_once 가 skip_verified 로 즉시 반환한다.
    """
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = _download_once(url, out, want_sha, workers)
            r["attempts"] = attempt
            return r
        except ChecksumMismatch as e:
            last = e
            print(f"  [attempt {attempt}/{max_attempts}] {e} — 재시도", flush=True)
            time.sleep(5 * attempt)
    raise SystemExit(f"{out.name}: {max_attempts}회 모두 sha256 불일치 — {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cls = resolve(a.dataset)
    root = Path(a.root) / a.dataset
    root.mkdir(parents=True, exist_ok=True)
    paths = list(getattr(cls, "paths", []))
    shas = list(getattr(cls, "sha256str", []) or [])
    if not paths:
        raise SystemExit(f"{a.dataset}: paths 가 비어 있다")
    tmpl = cls.url
    print(f"{a.dataset}: {len(paths)} files, workers={a.workers}", flush=True)
    for i, p in enumerate(paths):
        url = tmpl.format(p)
        want = shas[i] if i < len(shas) else None
        r = download_file(url, root / p, want, a.workers)
        print(f"  {r}", flush=True)
    print(f"{a.dataset}: ALL FILES OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
