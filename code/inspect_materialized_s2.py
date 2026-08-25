#!/usr/bin/env python3
"""물질화된 12밴드 큐브를 육안·수치로 검증한다 (L5). 빈 큐브가 성공으로 기록되는 것을 막는다."""
import json, sys, pathlib
import numpy as np

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                 else "/home/work/data/olmoearth/aihub/s2_probe")
rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l]
print(f"manifest {len(rows)}건")
for r in rows:
    a = np.load(d / "arrays" / (r["key"] + ".npy"))
    nz = [round(float((a[i] > 0).mean()), 4) for i in range(a.shape[0])]
    print(f'{r["key"]} cc={r["cloud_cover"]} item={r["item_id"][:36]} '
          f'mgrs={r["mgrs"]} miss={r["missing_bands"]} {a.shape} {a.dtype} '
          f'cand={r["n_candidates"]} plat={r["platform_stac"]}/{r["platform_meta"]}')
    print(f'   nonzero비율/밴드 = {nz}')
    for name, i in (("B04", 2), ("B08", 3), ("B11", 8)):
        b = a[i]
        print(f'   {name}: min {b.min()} med {int(np.median(b))} '
              f'p99 {int(np.percentile(b, 99))} max {b.max()}')
x = d / "excluded.jsonl"
print("excluded:", x.read_text()[:600] if x.exists() else "없음")
