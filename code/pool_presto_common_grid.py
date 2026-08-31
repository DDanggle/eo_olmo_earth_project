#!/usr/bin/env python3
"""C1a common-grid 캐시 — 봉인된 Presto 128x128x128 를 사전등록 4x4 mean pooling 으로
128x32x32 로 축소한다 (계약 registered_readouts.primary_common_grid). CPU 전용.

성능을 보고 pooling을 고르는 것은 금지 조항이므로, 이 규칙(비중첩 4x4 산술평균)은
계약에 이미 고정되어 있고 여기서 그대로 구현만 한다.
"""
import hashlib, json
from pathlib import Path
import numpy as np

SRC = Path("/home/work/data/olmoearth/presto_c1/holdout_chimanimani/emb_fp16")
DST = Path("/home/work/data/olmoearth/presto_c1_common32/emb_fp16")
SEAL_IN = Path("/home/work/data/olmoearth/presto_c1/holdout_chimanimani/seal_manifest.json")
DST.mkdir(parents=True, exist_ok=True)

seal_in = json.loads(SEAL_IN.read_text())
assert seal_in["aggregate"]["spot_check_40"]["pass"] and seal_in["aggregate"]["n_files"] == 6834

files = sorted(SRC.glob("*.npy"))
assert len(files) == 6834
agg = hashlib.sha256()
for i, p in enumerate(files, 1):
    a = np.load(p).astype("float32")                       # 128,128,128
    pooled = a.reshape(128, 32, 4, 32, 4).mean(axis=(2, 4)).astype("float16")
    np.save(DST / p.name, pooled, allow_pickle=False)
    agg.update(pooled.tobytes())
    if i % 1000 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}]", flush=True)

out = {
    "schema": "presto-c1a-common32-seal-v1",
    "rule": "non-overlapping channel-wise 4x4 arithmetic mean, preregistered in presto_c1_contract.json",
    "source_manifest_sha256": seal_in["aggregate"]["manifest_sha256"],
    "n_files": len(files),
    "content_sha256": agg.hexdigest(),
    "shape": [128, 32, 32], "dtype": "float16",
}
(DST.parent / "seal_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
import subprocess
subprocess.run(["chmod", "-R", "a-w", str(DST)], check=True)
print("DONE locked")
