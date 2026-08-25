#!/usr/bin/env python3
"""AI-Hub 71363 타일에 대응하는 Sentinel-2 **12밴드**를 STAC에서 물질화한다. GPU 불필요.

왜 필요한가 (M28): AI-Hub 원천 Sentinel-2는 **3밴드 uint8 RGB**(1024×1024, EPSG:32652, 10 m)다.
OlmoEarth `sentinel2_l2a` 계약은 10~12밴드 반사도를 요구하므로 그대로 쓸 수 없다.
3밴드를 12밴드 슬롯에 억지로 채우면 M3(밴드순서 dose-response)와 M8(조용한 계약 위반)이
동시에 걸린다. 그래서 우리가 직접 받는다.

사전 등록 규칙 (2026-08-26. 실행 중 바꾸지 않는다)
  후보 선택   같은 날짜(UTC 00:00~23:59)·같은 bbox의 `sentinel-2-l2a` item을 **id 정렬 후 첫 항목**.
              수작업 cherry-pick 금지. 같은 입력이면 같은 item이 나온다(C2-C S3에서 확인).
  구름 상한   `eo:cloud_cover <= 60`. 초과는 **버리지 않고 `excluded` 로 기록**한다.
              20표본 게이트에서 cc가 0.19~100.0까지 퍼졌고 cc=100 장면이 실재했다.
  격자        AI-Hub 타일의 EPSG:32652 격자를 **그대로** 쓴다. 좌상단 좌표는 M9에서 확정했다
              (중위 4.2e-05 m 정확일치). 1024×1024 @10 m.
  리샘플링    **nearest**. 20 m·60 m 밴드를 10 m로 올릴 때 값을 새로 만들지 않는다.
              bilinear은 없던 값을 만들므로 기본으로 쓰지 않는다. 이 선택을 산출물에 기록하고
              나중에 민감도로 확인한다.
  밴드 순서   v1 band-set 순서로 고정한다 (M27과 동일):
              B02 B03 B04 B08 / B05 B06 B07 B8A B11 B12 / B01 B09
  dtype       uint16. L2A 반사도가 정수이므로 무손실이다.

산출물
  s2_12band/<tile>_<date>.npy   (12, 1024, 1024) uint16
  manifest.jsonl                item id · platform · cloud_cover · sha256 · 결측 밴드
  excluded.jsonl                구름 초과·item 없음 등 제외 사유
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

BANDS = ["B02", "B03", "B04", "B08",
         "B05", "B06", "B07", "B8A", "B11", "B12",
         "B01", "B09"]
CLOUD_MAX = 60.0
RESAMPLING = "nearest"
TILE_PX = 1024
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path,
                   default=Path("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/aihub/s2_12band"))
    p.add_argument("--limit", type=int, default=0, help="0이면 전체")
    p.add_argument("--start", type=int, default=0)
    return p.parse_args()


def main() -> None:
    import numpy as np
    import planetary_computer as pc
    import rasterio
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    npy_dir = args.out / "arrays"
    npy_dir.mkdir(exist_ok=True)
    man_p, exc_p = args.out / "manifest.jsonl", args.out / "excluded.jsonl"

    done = set()
    for path in (man_p, exc_p):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    done.add(json.loads(line)["key"])

    rows = [json.loads(l) for l in
            args.inventory.read_text(encoding="utf-8").splitlines() if l]
    rows.sort(key=lambda r: r["key"])
    todo = [r for r in rows[args.start:] if r["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"대상 {len(todo)} / 전체 {len(rows)} (이미 처리 {len(done)})", flush=True)

    client = Client.open(STAC_URL, modifier=pc.sign_inplace)
    resamp = getattr(Resampling, RESAMPLING)

    ok = skipped = failed = 0
    t0 = time.perf_counter()
    with man_p.open("a", encoding="utf-8") as mf, exc_p.open("a", encoding="utf-8") as xf:
        for i, r in enumerate(todo, 1):
            key, d = r["key"], r["date"]
            iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            x0, y0, x1, y1 = r["utm52n_bbox"]
            try:
                search = client.search(
                    collections=["sentinel-2-l2a"], bbox=r["wgs84_bbox"],
                    datetime=f"{iso}T00:00:00Z/{iso}T23:59:59Z", limit=20)
                items = sorted(search.item_collection(), key=lambda it: it.id)
                if not items:
                    xf.write(json.dumps({"key": key, "reason": "no_stac_item",
                                         "date": iso}, ensure_ascii=False) + "\n")
                    xf.flush(); skipped += 1; continue
                it = items[0]
                cc = it.properties.get("eo:cloud_cover")
                if cc is not None and float(cc) > CLOUD_MAX:
                    xf.write(json.dumps({"key": key, "reason": "cloud_over_max",
                                         "cloud_cover": cc, "cloud_max": CLOUD_MAX,
                                         "item_id": it.id}, ensure_ascii=False) + "\n")
                    xf.flush(); skipped += 1; continue

                cube = np.zeros((len(BANDS), TILE_PX, TILE_PX), dtype="uint16")
                missing = []
                for bi, b in enumerate(BANDS):
                    asset = it.assets.get(b)
                    if asset is None:
                        missing.append(b); continue
                    with rasterio.open(asset.href) as src:
                        # AI-Hub 격자를 그대로 쓴다. S2 over Korea도 EPSG:32652이므로
                        # 재투영 없이 bounds 창으로 읽고 1024x1024로 리샘플한다.
                        win = from_bounds(x0, y0, x1, y1, transform=src.transform)
                        arr = src.read(1, window=win, out_shape=(TILE_PX, TILE_PX),
                                       resampling=resamp, boundless=True, fill_value=0)
                    cube[bi] = np.clip(arr, 0, 65535).astype("uint16")

                target = npy_dir / f"{key}.npy"
                np.save(target, cube, allow_pickle=False)
                mf.write(json.dumps({
                    "key": key, "date": iso, "item_id": it.id,
                    "platform_stac": it.properties.get("platform"),
                    "platform_meta": r["platform"],
                    "cloud_cover": cc,
                    "mgrs": it.properties.get("s2:mgrs_tile"),
                    "utm52n_bbox": r["utm52n_bbox"],
                    "bands": BANDS, "missing_bands": missing,
                    "resampling": RESAMPLING, "dtype": "uint16",
                    "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(cube.tobytes()).hexdigest(),
                    "n_candidates": len(items),
                    "candidate_ids": [x.id for x in items[:5]],
                }, ensure_ascii=False) + "\n")
                mf.flush(); ok += 1
            except Exception as exc:  # noqa: BLE001
                xf.write(json.dumps({"key": key, "reason": "error",
                                     "error": repr(exc)[:300]}, ensure_ascii=False) + "\n")
                xf.flush(); failed += 1
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                rate = i / max(el, 1e-9)
                print(f"  [{i}/{len(todo)}] ok={ok} skip={skipped} fail={failed} "
                      f"{rate:.2f}/s 남은 {(len(todo)-i)/max(rate,1e-9)/60:.1f}분", flush=True)

    total_bytes = sum(p.stat().st_size for p in npy_dir.glob("*.npy"))
    summary = {
        "schema": "aihub-s2-12band-materialize-v1",
        "preregistered": {
            "candidate_pick": "same date+bbox sentinel-2-l2a, sorted by id, first",
            "cloud_max": CLOUD_MAX, "resampling": RESAMPLING,
            "bands": BANDS, "dtype": "uint16",
            "grid": "AI-Hub EPSG:32652 upper-left origin, 1024x1024 @10m (M9)",
        },
        "processed": ok, "excluded": skipped, "failed": failed,
        "bytes_total": total_bytes,
        "reason": "AI-Hub 원천은 3밴드 uint8 RGB이므로 직접 물질화한다 (M28)",
    }
    (args.out / "materialize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
