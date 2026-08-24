#!/usr/bin/env python3
"""PhilEO n-shot의 S2 배열 형태와 task/shot 인벤토리 — P0 성립 조건의 마지막 확인.

라벨은 [N,128,128,1]로 확인됐다. 남은 것은 입력 S2의 채널 수와 시간 차원 유무다.
OlmoEarth는 시계열 입력을 기대하므로 단일 시점이면 그 자체가 계약 차이다.
"""
from __future__ import annotations

import collections
import glob
import json
import re
import zipfile
from pathlib import Path

import numpy as np

OUT = Path("/home/work/data/olmoearth/phileo_probe")
zpath = glob.glob(str(OUT / "**/downstream_datasets_nshot.zip"), recursive=True)[0]
zf = zipfile.ZipFile(zpath)

s2 = []
for name in sorted(n for n in zf.namelist() if n.endswith("_s2.npy"))[:4]:
    with zf.open(name) as fh:
        arr = np.load(fh)
    s2.append({
        "member": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    })

inventory = collections.defaultdict(set)
for name in zf.namelist():
    m = re.match(r"downstream_datasets_nshot/(\d+)_shot_(\w+)/", name)
    if m:
        inventory[m.group(2)].add(int(m.group(1)))

labels = {}
for name in sorted(n for n in zf.namelist() if "_label_" in n)[:6]:
    with zf.open(name) as fh:
        arr = np.load(fh)
    labels[name.split("/")[-1]] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                   "unique_head": np.unique(arr)[:12].tolist()}

result = {
    "schema": "phileo-s2-shape-v1",
    "s2_samples": s2,
    "task_inventory": {k: sorted(v) for k, v in inventory.items()},
    "label_samples": labels,
    "verdict": [
        f"S2 채널 {s2[0]['shape'][-1]}개 (OlmoEarth S2 12밴드와 차이 {12 - s2[0]['shape'][-1]:+d})" if s2 else "S2 없음",
        f"S2 차원 {len(s2[0]['shape'])} → {'시간 축 없음(단일 시점)' if len(s2[0]['shape']) == 4 else '확인 필요'}" if s2 else "",
        f"task: {sorted(inventory)} — building density {'있음' if 'building' in inventory else '없음(대용량 아카이브 필요)'}",
    ],
}
(OUT / "phileo_s2_shape.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
