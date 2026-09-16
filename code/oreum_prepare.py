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
GRID_SNAP_M = 40                # 창 원점을 맞추는 절대 격자(UTM). 40 m 토큰과 20 m 토큰의 공배수.
YEARS = ["2023", "2024", "2025", "2026"]
# SCL 에서 '쓸 만한 지표' 로 보는 클래스. 4 식생 5 나지 6 물 7 미분류 11 눈.
# 구름(8,9,10)·그림자(3)·결측(0)·포화(1)·어두운영역(2)은 제외한다.
SCL_CLEAR = {4, 5, 6, 7}   # 동결기·게이트와 같은 정의. 눈(11) 은 8~9월 제주에 없고 밝은 표면의 오분류다.


def load_frame_b(limit: int | None) -> list[dict]:
    """프레임 B — 제주 본섬 2.56 km 격자점. 공간 귀무 풀·외부 검정 전용. 오름 결과로 제시 금지."""
    import csv
    rows = list(csv.DictReader((ARTIFACT_ROOT / "external_data/kearth_oreum_v1/frame_b_grid.csv").open()))
    out = [{"oreum_id": r["point_id"], "name": r["point_id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in rows]
    return out[:limit] if limit else out


def assign_tiles_by_containment(points: list[dict], items: dict, scenes: dict) -> dict[str, str]:
    """프레임 B 는 계약에 배정이 없다. 네 해 모두 footprint 가 점을 담는 타일 중 최소 면적이 큰 것."""
    from shapely.geometry import Point
    geoms = {k: shape(v.geometry) for k, v in items.items()}
    out = {}
    for o in points:
        pt = Point(o["lon"], o["lat"])
        cands = [t for t in scenes if all(geoms[(t, y)].contains(pt) for y in YEARS)]
        if cands:
            out[o["oreum_id"]] = max(cands, key=lambda t: min(geoms[(t, y)].area for y in YEARS))
    print(f"  프레임 B 배정 {len(out)} / {len(points)}")
    return out


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


def read_window(item, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray, tuple]:
    """한 오름 창의 12밴드(bilinear)와 SCL(nearest)을 읽는다."""
    signed = pc.sign(item)
    with rasterio.open(signed.assets[BANDS[0]].href) as ref:
        xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
        cx, cy = xs[0], ys[0]
    half = SIZE * 10 / 2
    # 창 원점을 UTM 상의 GRID_SNAP_M 배수에 맞춘다. 40 은 20 의 배수이므로 patch_size 4 와 2
    # 의 토큰 격자가 둘 다 같은 절대 격자 위에 놓인다. 그러면 (1) 토큰 footprint 가 좌표로
    # 재현되고 (2) PNU 필지 polygon 과의 조인이 "어느 창의 몇 번째 토큰" 이 아니라 절대 위치로
    # 떨어진다. 스냅으로 창 중심이 오름 좌표에서 최대 20 m 비껴갈 수 있는데, 2.56 km 창에서
    # 무시할 수 있는 양이다. 원점은 반환값으로 기록한다.
    x0 = np.floor((cx - half) / GRID_SNAP_M) * GRID_SNAP_M
    y0 = np.floor((cy - half) / GRID_SNAP_M) * GRID_SNAP_M
    bounds = (x0, y0, x0 + SIZE * 10, y0 + SIZE * 10)

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
    return cube, scl, (float(x0), float(y0), str(ref.crs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--limit", type=int, help="오름 개수 제한 (스모크용)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frame", default="A", choices=("A", "B"), help="A=오름 243 (계약 배정) · B=제주 격자 (공간 귀무)")
    a = ap.parse_args()

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    scenes = contract["optical"]["scenes"]
    cat = pystac_client.Client.open(STAC)

    print("동결된 item 을 STAC 에서 해석하는 중…")
    coll = cat.get_collection("sentinel-2-l2a")
    items = {(t, y): coll.get_item(scenes[t][y]["item_id"]) for t in scenes for y in YEARS}
    print(f"  {len(items)} 건")

    if a.frame == "B":
        oreum = load_frame_b(a.limit)
        tiles = assign_tiles_by_containment(oreum, items, scenes)
    else:
        oreum = load_oreum(a.limit)
        tiles = assign_tiles(oreum, contract)
    print(f"오름 {len(oreum)}곳, 타일 배정 {len(tiles)}곳")

    out_dir = ensure(CACHE_ROOT / (f"{a.contract}/prepare_B" if a.frame == "B" else f"{a.contract}/prepare"))
    todo = [o for o in oreum if o["oreum_id"] in tiles
            and not (out_dir / f"{o['oreum_id']}.npz").exists()]
    print(f"새로 받을 곳 {len(todo)} / 캐시됨 {len(oreum) - len(todo)}")

    t0, done, failed = time.time(), 0, []

    def work(o: dict) -> None:
        nonlocal done
        tile = tiles[o["oreum_id"]]
        cube = np.full((len(BANDS), len(YEARS), SIZE, SIZE), np.nan, dtype="float32")
        sclc = np.zeros((len(YEARS), SIZE, SIZE), dtype="uint8")
        origins: list[tuple] = []
        for yi, year in enumerate(YEARS):
            for attempt in range(4):
                try:
                    c, s, origin = read_window(items[(tile, year)], o["lon"], o["lat"])
                    cube[:, yi], sclc[yi] = c, s
                    origins.append(origin)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(4 * (attempt + 1))
        # 네 해가 같은 타일·같은 CRS 이므로 스냅된 원점은 네 해 모두 같아야 한다. 다르면
        # Δz 가 서로 다른 땅을 비교하는 것이므로 여기서 멈춘다.
        assert len(set(origins)) == 1, f"{o['oreum_id']}: 연도별 창 원점이 다르다 {origins}"
        clear = np.isin(sclc, list(SCL_CLEAR))
        np.savez_compressed(
            out_dir / f"{o['oreum_id']}.npz",
            cube=cube, scl=sclc, clear=clear, years=np.array(YEARS),
            lonlat=np.array([o["lon"], o["lat"]]), tile=tile,
            item_ids=np.array([scenes[tile][y]["item_id"] for y in YEARS]),
            bands=np.array(BANDS),
            origin_xy=np.array(origins[0][:2]), crs=origins[0][2], grid_snap_m=GRID_SNAP_M)
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
        "schema": "jeju-v8-prepare-v1", "contract": a.contract, "frame": a.frame,
        "contract_sha256": contract.get("_self_sha256"),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "modality": "sentinel2_l2a (12밴드, B10 없음 — L2A 에 존재하지 않는다)",
        "bands": BANDS, "window_px": SIZE, "scl_clear_classes": sorted(SCL_CLEAR),
        "resampling": {"reflectance": "bilinear", "scl": "nearest (categorical)"},
        "grid_snap_m": GRID_SNAP_M,
        "grid_snap_note": "창 원점을 UTM 40 m 배수에 맞춤. 40 m·20 m 토큰 격자가 같은 절대 격자에 놓이고 필지 조인이 절대 좌표로 떨어진다.",
        "oreum_total": len(oreum), "cached": len(oreum) - len(todo) + done,
        "failed": failed, "elapsed_s": round(time.time() - t0, 1),
        "tile_assignment": tiles,
    }
    mpath = ensure(ARTIFACT_ROOT / "results") / (f"{a.contract}_prepare_manifest.json" if a.frame == "A" else f"{a.contract}_prepare_B_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\n완료 {done}, 실패 {len(failed)}  → {display_path(mpath)}")
    print(f"큐브 → {display_path(out_dir)}")
    for oid, err in failed[:5]:
        print(f"  실패 {oid}: {err[:90]}")


if __name__ == "__main__":
    main()
