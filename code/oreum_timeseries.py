#!/usr/bin/env python3
"""상위 오름의 월별 시계열 프레임 — 지도 슬라이더용. **보는 용도**이고 채점에는 쓰지 않는다.

월마다 후보 장면(같은 타일, 궤도 무관)을 오름 창의 SCL 로 재서 **사이트 판독 가능 비율이 가장 높은 장면**을 고른다.
장면 구름 메타데이터로 고르지 않는다(S0 실측: 장면 1% 구름인데 오름 0/12 판독). 판독 비율이 MIN_CLEAR 미만인 월은
프레임을 만들지 않고 `null` 로 남긴다 — 슬라이더에서 빈 칸이 "볼 수 없었던 달" 이다.

RGB 는 오름별 **시계열 공통 스트레치**(전 프레임 2~98% 분위)로 만든다. 달마다 따로 늘리면 계절 변화가 사라진다.

산출물: <WEB_DATA_ROOT>/series/<oreum_id>/<YYYY-MM>.jpg, <WEB_DATA_ROOT>/series/<oreum_id>.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, CONTRACT_ROOT, WEB_DATA_ROOT, display_path, ensure
from oreum_prepare import BANDS, GRID_SNAP_M, SIZE, load_oreum

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SCL_CLEAR = (4, 5, 6, 7)
MIN_CLEAR = 0.35              # 이보다 흐린 달은 프레임 없음(빈 칸)
MAX_CANDIDATES = 6            # 월별 후보 상한 (SCL 창 읽기 비용)
START, END = date(2023, 1, 1), date(2026, 9, 30)


def window_bounds(ref_href: str, lon: float, lat: float):
    with rasterio.open(ref_href) as ref:
        xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
    half = SIZE * 10 / 2
    x0 = np.floor((xs[0] - half) / GRID_SNAP_M) * GRID_SNAP_M
    y0 = np.floor((ys[0] - half) / GRID_SNAP_M) * GRID_SNAP_M
    return (x0, y0, x0 + SIZE * 10, y0 + SIZE * 10)


def read(href: str, bounds, resampling, dtype, retries: int = 4):
    """blob 이 가끔 TIFF 가 아닌 응답을 돌려준다(스로틀). 지수 대기로 재시도하고 그래도 실패하면 올린다."""
    for k in range(retries):
        try:
            with rasterio.open(href) as ds:
                win = from_bounds(*bounds, transform=ds.transform)
                return ds.read(1, window=win, out_shape=(SIZE, SIZE), boundless=True, fill_value=0, resampling=resampling).astype(dtype)
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))


def months():
    y, m = START.year, START.month
    while (y, m) <= (END.year, END.month):
        yield f"{y}-{m:02d}", f"{y}-{m:02d}-01", (f"{y+1}-01-01" if m == 12 else f"{y}-{m+1:02d}-01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def build(oid: str, o: dict, tile: str, cat, out_root) -> dict:
    out_dir = ensure(out_root / oid)
    coll = "sentinel-2-l2a"
    frames, rgb_stack = {}, []
    bounds = None
    for key, d0, d1 in months():
        items = list(cat.search(collections=[coll], datetime=f"{d0}/{d1}",
                                query={"s2:mgrs_tile": {"eq": tile}, "eo:cloud_cover": {"lt": 80}}).items())
        # 같은 날짜 재처리 중복 제거 → 장면 구름 낮은 순으로 후보 상한
        by_day = {}
        for it in items:
            by_day.setdefault(it.datetime.date(), it)
        cands = sorted(by_day.values(), key=lambda i: i.properties["eo:cloud_cover"])[:MAX_CANDIDATES]
        best, best_clear = None, -1.0
        for it in cands:
            s = pc.sign(it)
            if bounds is None:
                bounds = window_bounds(s.assets["B04"].href, o["lon"], o["lat"])
            try:
                scl = read(s.assets["SCL"].href, bounds, Resampling.nearest, "uint8")
            except Exception:
                continue
            clear = float(np.isin(scl, SCL_CLEAR).mean())
            if clear > best_clear:
                best, best_clear, best_signed = it, clear, s
        if best is None or best_clear < MIN_CLEAR:
            frames[key] = None if best is None else {"date": str(best.datetime.date()), "site_clear": round(best_clear, 3),
                                                       "scene_cloud": round(best.properties["eo:cloud_cover"], 1), "frame": None}
            continue
        try:
            cube = np.stack([read(best_signed.assets[b].href, bounds, Resampling.bilinear, "float32") for b in BANDS])
            scl_best = read(best_signed.assets["SCL"].href, bounds, Resampling.nearest, "uint8")
        except Exception as e:      # 이 달만 건너뛴다. 오름 전체를 잃지 않는다.
            frames[key] = {"date": str(best.datetime.date()), "site_clear": round(best_clear, 3),
                           "scene_cloud": round(best.properties["eo:cloud_cover"], 1), "frame": None, "error": type(e).__name__}
            continue
        cube[cube <= 0] = np.nan
        np.savez_compressed(ensure(CACHE_ROOT / "jeju_v8/series" / oid) / f"{key}.npz", cube=cube, scl=scl_best,
                            date=str(best.datetime.date()), item_id=best.id, bands=np.array(BANDS))
        rgb = cube[[BANDS.index(b) for b in ("B04", "B03", "B02")]]
        rgb_stack.append((key, rgb))
        frames[key] = {"date": str(best.datetime.date()), "site_clear": round(best_clear, 3),
                       "scene_cloud": round(best.properties["eo:cloud_cover"], 1), "orbit": best.properties.get("sat:relative_orbit"),
                       "item_id": best.id, "frame": f"{key}.jpg"}
    # 공통 스트레치
    if rgb_stack:
        from PIL import Image
        allv = np.concatenate([r[np.isfinite(r)] for _, r in rgb_stack])
        lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
        for key, rgb in rgb_stack:
            a = np.clip((np.nan_to_num(rgb, nan=lo) - lo) / max(hi - lo, 1.0), 0, 1) ** 0.8
            Image.fromarray((np.transpose(a, (1, 2, 0)) * 255).astype("uint8")).save(out_dir / f"{key}.jpg", quality=82)
    meta = {"oreum_id": oid, "name": o["name"], "tile": tile, "min_site_clear": MIN_CLEAR,
            "rule": "월별 후보 중 오름 창 SCL 판독 비율 최고 장면. 궤도 무관(보는 용도). 채점에 쓰지 않는다.",
            "n_months": len(frames), "n_frames": len(rgb_stack), "frames": frames}
    (out_root / f"{oid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    scan = json.loads((ARTIFACT_ROOT / f"results/{a.contract}_scan_p4.json").read_text())
    assign = contract["optical"]["frame_a_assignment"]
    oreum = {o["oreum_id"]: o for o in load_oreum(None)}
    ids = scan["ranking"][: a.top]
    out_root = ensure(WEB_DATA_ROOT / "series")
    cat = pystac_client.Client.open(STAC)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build, oid, oreum[oid], assign[oid]["tile"], cat, out_root): oid for oid in ids}
        for f, oid in futs.items():
            try:
                m = f.result(); print(f"  {oid} {m['name']}: 프레임 {m['n_frames']}/{m['n_months']}개월  ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  {oid} 실패 {type(e).__name__}: {e}", flush=True)
    (out_root / "index.json").write_text(json.dumps({"oreum_ids": ids, "months": [k for k, _, _ in months()]}, ensure_ascii=False))
    print(f"→ {display_path(out_root)}")


if __name__ == "__main__":
    main()
