#!/usr/bin/env python3
"""v8 준비 단계 — 동결된 계약대로 오름 창 큐브를 로컬에 만든다.

기존 제주 파이프라인은 원격 클러스터(`/home/work/data/olmoearth`)에 의존해 로컬에서 전혀
돌지 않았다. 여기서는 네팔과 같이 **Planetary Computer COG 를 직접 창 단위로 읽어** 캐시한다.

밴드 계약 (조용히 틀리기 쉬운 곳)
---------------------------------
OlmoEarth 에는 sentinel2 modality 가 **둘** 있다.
  * ``sentinel2``      — 13밴드, B10 포함. **L1C 용.**
  * ``sentinel2_l2a``  — 12밴드, B10 없음. **L2A 용이고 우리가 쓰는 것.**
L2A 산출물에는 B10(권운)이 아예 없으므로 13밴드 spec 에 맞추려고 B10 을 0 으로 채우면
없는 관측을 지어내는 것이 된다. 여기서는 12밴드 spec 을 그대로 쓴다.
(이 구분은 기존 `model_s2.yaml` 이 이미 옳게 하고 있었다.)

SCL 재표본 (합성기의 교훈)
--------------------------
`scl_compositor.py` 의 `Sentinel2SCLBestClearNearest` 는 rslearn 데이터셋 경로용 어댑터라
여기서 직접 import 하지 않는다. 그 **원칙**은 그대로 지킨다 —
반사율은 bilinear, **SCL 은 categorical 이므로 nearest**. 보간하면 `scl == 9` 같은 등식이 깨진다.

"최적 장면 선택"은 계약 단계에서 이미 끝났다(타일·연도별 1장 동결). 여기서는 읽기만 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from shapely.geometry import shape

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, CONTRACT_ROOT, display_path, ensure

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
SIZE = 256                      # 2.56 km / 10 m
YEARS = ["2023", "2024", "2025", "2026"]
# SCL 에서 '쓸 만한 지표' 로 보는 클래스. 4 식생 5 나지 6 물 7 미분류 11 눈.
# 구름(8,9,10)·그림자(3)·결측(0)·포화(1)·어두운영역(2)은 제외한다.
SCL_CLEAR = {4, 5, 6, 7, 11}


def load_oreum(limit: int | None) -> list[dict]:
    import csv
    path = ARTIFACT_ROOT / "external_data/kearth_oreum_v1/oreum_registry_368.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    geo = [r for r in rows
           if r.get("location_status") == "resolved_offline_osm_peak" and r.get("lat") and r.get("lon")]
    out = [{"oreum_id": r["oreum_id"], "name": r["name"],
            "lat": float(r["lat"]), "lon": float(r["lon"])} for r in geo]
    return out[:limit] if limit else out


def assign_tiles(oreum: list[dict], contract: dict) -> dict[str, str]:
    """타일 배정은 **계약에서 읽는다.** 여기서 다시 고르지 않는다.

    예전에는 이 함수가 구름 평균으로 직접 골랐고, 그 바람에 계약과 갈라져 오름 34곳이
    한 해만 덮는 타일에 배정돼 조용히 빈 큐브가 됐다. 무엇을 보느냐는 계약이 정하고
    파이프라인은 따르기만 한다.
    """
    assign = contract["optical"]["frame_a_assignment"]
    out, missing = {}, []
    for o in oreum:
        rec = assign.get(o["oreum_id"])
        if not rec or not rec.get("tile"):
            missing.append(o["oreum_id"]); continue
        out[o["oreum_id"]] = rec["tile"]
    if missing:
        print(f"  계약에 타일이 없는 오름 {len(missing)}곳 → {missing[:5]}")
    n_obs = sum(1 for o in oreum if assign.get(o["oreum_id"], {}).get("observable_all_years"))
    print(f"  네 해 모두 판독 가능 {n_obs} / {len(oreum)} (나머지는 unobservable 로 보고, 분모에서 빼지 않는다)")
    return out


def read_window(item, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray]:
    """한 오름 창의 12밴드(bilinear)와 SCL(nearest)을 읽는다."""
    signed = pc.sign(item)
    with rasterio.open(signed.assets[BANDS[0]].href) as ref:
        xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
        cx, cy = xs[0], ys[0]
    half = SIZE * 10 / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)

    cube = np.full((len(BANDS), SIZE, SIZE), np.nan, dtype="float32")
    for bi, band in enumerate(BANDS):
        with rasterio.open(signed.assets[band].href) as ds:
            win = from_bounds(*bounds, transform=ds.transform)
            arr = ds.read(1, window=win, out_shape=(SIZE, SIZE), boundless=True,
                          fill_value=0, resampling=Resampling.bilinear).astype("float32")
        arr[arr <= 0] = np.nan                       # 0 은 장면 밖/결측
        cube[bi] = arr
    with rasterio.open(signed.assets["SCL"].href) as ds:   # ← categorical: nearest 만
        win = from_bounds(*bounds, transform=ds.transform)
        scl = ds.read(1, window=win, out_shape=(SIZE, SIZE), boundless=True,
                      fill_value=0, resampling=Resampling.nearest).astype("uint8")
    return cube, scl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--limit", type=int, help="오름 개수 제한 (스모크용)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    scenes = contract["optical"]["scenes"]
    cat = pystac_client.Client.open(STAC)

    print("동결된 item 을 STAC 에서 해석하는 중…")
    coll = cat.get_collection("sentinel-2-l2a")
    items = {(t, y): coll.get_item(scenes[t][y]["item_id"]) for t in scenes for y in YEARS}
    print(f"  {len(items)} 건")

    oreum = load_oreum(a.limit)
    tiles = assign_tiles(oreum, contract)
    print(f"오름 {len(oreum)}곳, 타일 배정 {len(tiles)}곳")

    out_dir = ensure(CACHE_ROOT / f"{a.contract}/prepare")
    todo = [o for o in oreum if o["oreum_id"] in tiles
            and not (out_dir / f"{o['oreum_id']}.npz").exists()]
    print(f"새로 받을 곳 {len(todo)} / 캐시됨 {len(oreum) - len(todo)}")

    t0, done, failed = time.time(), 0, []

    def work(o: dict) -> None:
        nonlocal done
        tile = tiles[o["oreum_id"]]
        cube = np.full((len(BANDS), len(YEARS), SIZE, SIZE), np.nan, dtype="float32")
        sclc = np.zeros((len(YEARS), SIZE, SIZE), dtype="uint8")
        for yi, year in enumerate(YEARS):
            for attempt in range(4):
                try:
                    c, s = read_window(items[(tile, year)], o["lon"], o["lat"])
                    cube[:, yi], sclc[yi] = c, s
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(4 * (attempt + 1))
        clear = np.isin(sclc, list(SCL_CLEAR))
        np.savez_compressed(
            out_dir / f"{o['oreum_id']}.npz",
            cube=cube, scl=sclc, clear=clear, years=np.array(YEARS),
            lonlat=np.array([o["lon"], o["lat"]]), tile=tile,
            item_ids=np.array([scenes[tile][y]["item_id"] for y in YEARS]),
            bands=np.array(BANDS))
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for o, fut in [(o, ex.submit(work, o)) for o in todo]:
            try:
                fut.result()
            except Exception as e:
                failed.append((o["oreum_id"], f"{type(e).__name__}: {e}"))

    manifest = {
        "schema": "jeju-v8-prepare-v1", "contract": a.contract,
        "contract_sha256": contract.get("_self_sha256"),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "modality": "sentinel2_l2a (12밴드, B10 없음 — L2A 에 존재하지 않는다)",
        "bands": BANDS, "window_px": SIZE, "scl_clear_classes": sorted(SCL_CLEAR),
        "resampling": {"reflectance": "bilinear", "scl": "nearest (categorical)"},
        "oreum_total": len(oreum), "cached": len(oreum) - len(todo) + done,
        "failed": failed, "elapsed_s": round(time.time() - t0, 1),
        "tile_assignment": tiles,
    }
    mpath = ensure(ARTIFACT_ROOT / "results") / f"{a.contract}_prepare_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\n완료 {done}, 실패 {len(failed)}  → {display_path(mpath)}")
    print(f"큐브 → {display_path(out_dir)}")
    for oid, err in failed[:5]:
        print(f"  실패 {oid}: {err[:90]}")


if __name__ == "__main__":
    main()
