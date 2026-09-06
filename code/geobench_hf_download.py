"""GEO-Bench-2 파일을 huggingface_hub 공식 다운로더로 받고 sha256 검증한다.

우리 병렬 range 다운로더가 pastis 0001(20GB)에서 결정적으로 잘못된 해시를 냈다
(재시도마다 동일 got). HF LFS/xet 서빙에 대한 range 병합 문제로 보여, 공식 도구로 우회한다.
공식 다운로더는 청크·재개·무결성을 자체 처리한다. 받은 뒤 클래스의 sha256str 로 재확인한다.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geobench_fetch_one import resolve, DATASETS
from huggingface_hub import hf_hub_download


def sha256_of(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    cls = resolve(a.dataset)
    root = Path(a.root) / a.dataset
    root.mkdir(parents=True, exist_ok=True)
    paths = list(getattr(cls, "paths", []))
    shas = list(getattr(cls, "sha256str", []) or [])
    # url = https://hf.co/datasets/<repo>/resolve/main/{}  → repo_id 추출
    url = cls.url
    repo_id = url.split("/datasets/")[1].split("/resolve/")[0]
    print(f"{a.dataset}: repo={repo_id}, {len(paths)} files", flush=True)
    for i, p in enumerate(paths):
        dest = root / p
        want = shas[i] if i < len(shas) else None
        if dest.exists() and want and sha256_of(dest) == want:
            print(f"  skip_verified {p}", flush=True); continue
        print(f"  downloading {p} ...", flush=True)
        got_path = hf_hub_download(repo_id=repo_id, filename=p, repo_type="dataset",
                                   local_dir=str(root), local_dir_use_symlinks=False)
        got = sha256_of(Path(got_path))
        ok = (want is None) or (got == want)
        print(f"  {p}: sha256 {'OK' if ok else 'MISMATCH got '+got[:16]+' want '+(want or '')[:16]}", flush=True)
        if not ok:
            raise SystemExit(f"{p}: sha256 불일치")
    print(f"{a.dataset}: ALL FILES OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
