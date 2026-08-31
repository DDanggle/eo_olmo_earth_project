#!/usr/bin/env python3
"""C1 캐시 봉인 — decoder 학습 전 필수 (계약 remaining_before_full_run 4항).

emb_fp16/*.npy 전수: 파일별 sha256 → manifest, 집계 sha, 개수·크기 검증,
무작위 40타일 finite/shape/dtype 검사. 이후 캐시는 읽기 전용으로 잠근다.
"""
import hashlib, json, random
from pathlib import Path
import numpy as np

ROOT = Path("/home/work/data/olmoearth/presto_c1/holdout_chimanimani")
EMB = ROOT / "emb_fp16"
rng = random.Random(20260901)

files = sorted(EMB.glob("*.npy"))
assert len(files) == 6834, f"타일 수 {len(files)} != 6834"
manifest, agg = {}, hashlib.sha256()
for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest[p.stem] = h
    agg.update(f"{p.stem}:{h}\n".encode())

bad = []
for p in rng.sample(files, 40):
    a = np.load(p)
    if a.shape != (128, 128, 128) or a.dtype != np.float16 or not np.isfinite(a).all():
        bad.append(p.stem)

seal = {
    "schema": "presto-c1-cache-seal-v1",
    "n_files": len(files),
    "bytes_total": sum(p.stat().st_size for p in files),
    "manifest_sha256": agg.hexdigest(),
    "spot_check_40": {"failed": bad, "pass": not bad},
    "expected_shape": [128, 128, 128], "dtype": "float16",
}
(ROOT / "seal_manifest.json").write_text(
    json.dumps({"aggregate": seal, "files": manifest}, indent=2) + "\n")
print(json.dumps(seal, indent=2))
if not bad:
    import subprocess
    subprocess.run(["chmod", "-R", "a-w", str(EMB)], check=True)
    print("cache locked read-only")
