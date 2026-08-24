#!/usr/bin/env python3
"""PhilEO n-shot 서브셋(3GB)만 받아 P0 성립 조건 4개를 확인한다.

전체 PhilEO-downstream은 949GB(분할 zip 5세트)라 부분 접근이 불가능하다.
`downstream_datasets_nshot.zip`은 3GB 단일 파일이므로 이것으로 판정한다.

확인:
  1) 시점 구조 — 단일 시점인가 (OlmoEarth 시계열 입력과의 정합)
  2) 밴드 구성 — 카드상 11밴드(SCL 포함). OlmoEarth S2 12밴드와의 차이
  3) tile 크기·개수 — 통계 단위(독립 위치 수)
  4) 3-task 라벨이 같은 tile을 공유하는가 — shared cache 실험의 전제
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = "PhilEO-community/PhilEO-downstream"
MEMBER = "data/downstream_datasets_nshot.zip"
OUT = Path("/home/work/data/olmoearth/phileo_probe")


def main() -> None:
    import numpy as np
    from huggingface_hub import hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    zpath = Path(hf_hub_download(REPO, MEMBER, repo_type="dataset", cache_dir=str(OUT)))
    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()

    ext = Counter(Path(n).suffix.lower() for n in names)
    # 파일명에서 task / split / n-shot 구조 추론
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        low = n.lower()
        key = (
            "roads" if "road" in low else
            "building" if "building" in low else
            "landcover" if "lc" in low or "landcover" in low else
            "image" if "s2" in low or "image" in low else
            "other"
        )
        groups[key].append(n)

    result = {
        "schema": "phileo-nshot-probe-v1",
        "zip_bytes": zpath.stat().st_size,
        "member_count": len(names),
        "extensions": dict(ext.most_common(8)),
        "groups": {k: {"count": len(v), "examples": sorted(v)[:5]} for k, v in groups.items()},
        "shot_tokens": dict(Counter(m.group(0) for n in names
                                   for m in [re.search(r"\d+_?shot|_\d+shot", n.lower())] if m).most_common(8)),
        "top_dirs": dict(Counter(n.split("/")[0] for n in names).most_common(8)),
    }

    # 배열 표본 검사: 가장 작은 .npy 몇 개
    npys = sorted((n for n in names if n.endswith(".npy")), key=lambda n: zf.getinfo(n).file_size)
    samples = []
    for n in npys[:6]:
        with zf.open(n) as fh:
            arr = np.load(fh, allow_pickle=False)
        entry = {"member": n, "shape": list(arr.shape), "dtype": str(arr.dtype),
                 "min": float(np.min(arr)), "max": float(np.max(arr))}
        samples.append(entry)
    result["npy_samples"] = samples

    # 판정
    v = []
    imgs = [s for s in samples if s["shape"][-1] in (10, 11, 12, 13) or (len(s["shape"]) > 2 and s["shape"][1] in (10, 11, 12))]
    if imgs:
        ch = imgs[0]["shape"][-1]
        v.append(f"이미지 채널 {ch}개 → OlmoEarth S2 12밴드와 차이 {12-ch:+d}")
    v.append("shot 토큰: " + (str(result["shot_tokens"]) if result["shot_tokens"] else "없음"))
    v.append(f"3-task 라벨 동시 존재: roads={len(groups['roads'])} building={len(groups['building'])} landcover={len(groups['landcover'])}")
    result["verdict"] = v

    (OUT / "phileo_nshot_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)[:3500])
    print("DONE")


if __name__ == "__main__":
    main()
