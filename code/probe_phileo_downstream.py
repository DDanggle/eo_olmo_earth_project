#!/usr/bin/env python3
"""PhilEO-downstream 착수 전 확인 — 다운로드 없이 파일 목록부터, 그다음 최소 표본만.

확인할 4개 (P0 설계가 성립하는지 결정한다):
  1) 단일 시점인가 시계열인가        → OlmoEarth 시계열 입력과의 정합
  2) S2 product level (L1C/L2A)      → 우리 정규화 계약
  3) 밴드 구성                        → 카드상 11밴드(SCL 포함), B01·B09 부재로 보임.
                                        zero-fill은 계약 불일치를 주입하므로 M3 dose와 직결
  4) tile 크기·개수                   → 통계 단위(독립 위치 수)

출력: artifacts/results/phileo_probe.json
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = "PhilEO-community/PhilEO-downstream"
OUT = Path("/home/work/data/olmoearth/phileo_probe")
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024  # 표본만. 전체 받지 않는다.


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)

    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    ext = Counter(Path(n).suffix.lower() for n, _ in files)
    total = sum(sz for _, sz in files)

    # 파일명 규칙에서 task/지역/시점 구조 추론
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, size in files:
        low = name.lower()
        key = (
            "label_roads" if "road" in low else
            "label_building" if "building" in low else
            "label_landcover" if ("landcover" in low or "lc" in low) else
            "image_s2" if ("s2" in low or "sentinel" in low or "image" in low) else
            "other"
        )
        buckets[key].append((name, size))

    result = {
        "schema": "phileo-probe-v1",
        "repo": REPO,
        "license_on_card": "MIT (card 기재, 파일 확인 별도)",
        "file_count": len(files),
        "total_bytes": total,
        "extensions": dict(ext.most_common()),
        "buckets": {k: {"count": len(v), "bytes": sum(s for _, s in v),
                        "examples": [n for n, _ in sorted(v, key=lambda x: x[1])[:4]]}
                    for k, v in buckets.items()},
    }

    # 시점 구조: 파일명에 날짜/회차 패턴이 있는지
    dates = Counter()
    for name, _ in files:
        for pat in (r"\d{4}[-_]?\d{2}[-_]?\d{2}", r"_t\d+_", r"_(\d)of3_"):
            m = re.search(pat, name)
            if m:
                dates[pat] += 1
    result["filename_temporal_patterns"] = dict(dates)

    # 최소 표본 다운로드 후 실제 배열 검사
    samples, budget = {}, MAX_DOWNLOAD_BYTES
    for key in ("image_s2", "label_landcover", "label_building", "label_roads"):
        for name, size in sorted(buckets.get(key, []), key=lambda x: x[1]):
            if size == 0 or size > budget:
                continue
            try:
                path = Path(hf_hub_download(REPO, name, repo_type="dataset", cache_dir=str(OUT)))
            except Exception as exc:  # noqa: BLE001
                samples[key] = {"file": name, "error": repr(exc)[:200]}
                break
            budget -= size
            entry = {"file": name, "bytes": size, "suffix": path.suffix}
            if path.suffix == ".npy":
                import numpy as np
                arr = np.load(path, mmap_mode="r")
                entry |= {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                if arr.ndim >= 3:
                    entry["interpretation"] = (
                        "H,W,C" if arr.shape[-1] <= 16 else "C,H,W" if arr.shape[0] <= 16 else "unclear"
                    )
                sub = np.asarray(arr[..., 0] if arr.ndim == 3 else arr).astype("float64")
                entry["value_range_first_channel"] = [float(sub.min()), float(sub.max())]
            elif path.suffix in (".tif", ".tiff"):
                import rasterio
                with rasterio.open(path) as src:
                    entry |= {"count": src.count, "width": src.width, "height": src.height,
                              "dtype": str(src.dtypes[0]), "crs": str(src.crs),
                              "res": list(src.res), "descriptions": list(src.descriptions)}
            samples[key] = entry
            break
    result["samples"] = samples

    verdict = []
    img = samples.get("image_s2", {})
    ch = None
    if "shape" in img:
        s = img["shape"]
        ch = s[-1] if img.get("interpretation") == "H,W,C" else s[0] if img.get("interpretation") == "C,H,W" else None
    elif "count" in img:
        ch = img["count"]
    if ch:
        verdict.append(f"채널 {ch}개 — OlmoEarth S2 12밴드와 차이: {12 - ch:+d}")
    verdict.append("시점 구조: 파일명 패턴 " + (str(dates) if dates else "없음 → 단일 시점 가능성"))
    result["verdict"] = verdict

    path = Path("/home/work/data/olmoearth/phileo_probe/phileo_probe.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)[:4000])
    print("DONE")


if __name__ == "__main__":
    main()
