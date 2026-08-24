#!/usr/bin/env python3
"""P0 1단계 — AI-Hub 71363 (10m S2) 타일 인벤토리 + 계약 감사. 네트워크 미사용.

STAC 질의를 하기 전에 반드시 확인해야 하는 것들이다. 순서를 바꾸면 헛수고가 된다.

  A1 ID 조인 감사 — 메타데이터 ID(`SA…`)와 라벨 ID(`SB…`)가 실제로 이어지는가.
     Major TOM에서 `unique_id` 교집합이 0이었던 함정과 같은 종류다. 안 이어지면
     "라벨이 있는 타일의 좌표를 모른다"는 뜻이므로 물질화 자체가 불가능하다.
  A2 기하 해석 — 메타데이터 `coordinates`가 타일 중심인가 좌상단인가.
     라벨 GeoJSON 폴리곤의 실제 범위와 대조해 판정한다. 추측하지 않는다.
  A3 인벤토리 — (타일ID, 촬영일, 위성, EPSG:32652 bbox, WGS84 bbox)를 확정한다.

산출물은 STAC 질의에 그대로 넣을 수 있는 형태여야 한다.
인코딩: 라벨 GeoJSON은 UTF-8, 메타데이터는 cp949 (실측, M-기록 참조).
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("/home/work/data/olmoearth/aihub/raw/71363")
OUT = Path("/home/work/data/olmoearth/aihub/inventory")
TILE_PX = 1024
RES_M = 10.0
EXTENT_M = TILE_PX * RES_M  # 10,240 m

# A2 판정 허용오차: 중심/좌상단 가설 중 어느 쪽이 폴리곤 범위와 맞는지 가릴 때 쓴다.
GEOM_TOL_M = 600.0


def find_zip(pattern: str) -> Path:
    hits = sorted(RAW.rglob(pattern))
    if not hits:
        raise SystemExit(f"zip 없음: {pattern}")
    return hits[0]


def read_json(data: bytes) -> object:
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return json.loads(data.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("디코딩 실패")


def utm52n_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """EPSG:32652 -> WGS84. pyproj가 없는 환경도 있으므로 표준 역투영을 직접 구현한다."""
    import math
    a, f = 6378137.0, 1 / 298.257223563
    e2 = 2 * f - f * f
    k0, e0, n0, lon0 = 0.9996, 500000.0, 0.0, math.radians(129.0)  # UTM zone 52
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    m = (y - n0) / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu))
    c1 = e2 * math.cos(phi1) ** 2 / (1 - e2)
    t1 = math.tan(phi1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = (x - e0) / (n1 * k0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e2 / (1 - e2)) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e2 / (1 - e2) - 3 * c1**2) * d**6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e2 / (1 - e2) + 24 * t1**2)
                  * d**5 / 120) / math.cos(phi1)
    return math.degrees(lon), math.degrees(lat)


def poly_bounds(geom: dict) -> tuple[float, float, float, float] | None:
    """MultiPolygon/Polygon의 좌표 범위."""
    xs, ys = [], []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
                xs.append(float(node[0])); ys.append(float(node[1]))
            else:
                for c in node:
                    walk(c)

    walk((geom or {}).get("coordinates"))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"schema": "aihub-71363-inventory-v1",
                    "constants": {"tile_px": TILE_PX, "res_m": RES_M,
                                  "extent_m": EXTENT_M, "geom_tol_m": GEOM_TOL_M}}

    # ---- 메타데이터 읽기 ----
    meta: dict[str, dict] = {}
    with zipfile.ZipFile(find_zip("01.메타데이터_03*.zip")) as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".json")]:
            for rec in (read_json(zf.read(name)) or []):
                if not isinstance(rec, dict):
                    continue
                img_id = str(rec.get("img_id", ""))
                try:
                    x, y = [float(v) for v in str(rec.get("coordinates", "")).split(",")]
                except (ValueError, TypeError):
                    continue
                meta[img_id] = {"x": x, "y": y, "date": str(rec.get("img_time", "")),
                                "platform": str(rec.get("img_type", "")),
                                "w": rec.get("img_width"), "h": rec.get("img_height")}
    result["metadata_records"] = len(meta)

    # ---- 라벨 읽기 (폴리곤 범위만; 전체 전개 없이) ----
    label_bounds: dict[str, tuple] = {}
    label_split: dict[str, str] = {}
    for split, pattern in (("train", "TL_02.JSON_03*.zip"), ("valid", "VL_02.JSON_03*.zip")):
        with zipfile.ZipFile(find_zip(pattern)) as zf:
            for name in [n for n in zf.namelist() if n.lower().endswith(".json")]:
                obj = read_json(zf.read(name))
                if not isinstance(obj, dict):
                    continue
                key = str(obj.get("name") or Path(name).stem)
                bb = None
                for feat in obj.get("features") or []:
                    b = poly_bounds((feat or {}).get("geometry") or {})
                    if b is None:
                        continue
                    bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]),
                                               max(bb[2], b[2]), max(bb[3], b[3]))
                if bb:
                    label_bounds[key] = bb
                    label_split[key] = split
    result["label_tiles"] = len(label_bounds)

    # ---- A1 ID 조인 감사 ----
    mk, lk = set(meta), set(label_bounds)
    result["A1_id_join"] = {
        "metadata_ids": len(mk), "label_ids": len(lk),
        "intersection": len(mk & lk),
        "metadata_only": len(mk - lk), "label_only": len(lk - mk),
        "metadata_prefixes": dict(Counter(k[:2] for k in mk).most_common()),
        "label_prefixes": dict(Counter(k[:2] for k in lk).most_common()),
        "sample_metadata_ids": sorted(mk)[:3], "sample_label_ids": sorted(lk)[:3],
    }
    joinable = mk & lk

    # ---- A2 기하 해석: coordinates가 중심인가 좌상단인가 ----
    votes = Counter()
    diffs = {"center": [], "upper_left": [], "lower_left": []}
    for key in sorted(joinable)[:400]:
        mx, my = meta[key]["x"], meta[key]["y"]
        x0, y0, x1, y1 = label_bounds[key]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cand = {
            "center": ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5,
            "upper_left": ((mx - x0) ** 2 + (my - y1) ** 2) ** 0.5,
            "lower_left": ((mx - x0) ** 2 + (my - y0) ** 2) ** 0.5,
        }
        best = min(cand, key=cand.get)
        votes[best] += 1
        for k, v in cand.items():
            diffs[k].append(v)
    result["A2_geometry"] = {
        "votes": dict(votes),
        "median_distance_m": {k: (sorted(v)[len(v) // 2] if v else None)
                              for k, v in diffs.items()},
        "note": "라벨 폴리곤은 타일 전체를 덮지 않을 수 있어 거리가 0은 아니다. "
                "상대 비교로 해석을 고른다.",
    }
    interp = votes.most_common(1)[0][0] if votes else None
    result["A2_interpretation"] = interp

    # ---- A3 인벤토리 ----
    inv = []
    for key in sorted(joinable):
        m = meta[key]
        mx, my = m["x"], m["y"]
        if interp == "center":
            x0, y0 = mx - EXTENT_M / 2, my - EXTENT_M / 2
        elif interp == "upper_left":
            x0, y0 = mx, my - EXTENT_M
        else:
            x0, y0 = mx, my
        x1, y1 = x0 + EXTENT_M, y0 + EXTENT_M
        lon0, lat0 = utm52n_to_wgs84(x0, y0)
        lon1, lat1 = utm52n_to_wgs84(x1, y1)
        tile_id, _, date = key.rpartition("_")
        inv.append({
            "key": key, "tile_id": tile_id, "date": date,
            "platform": m["platform"], "split": label_split.get(key),
            "utm52n_bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "wgs84_bbox": [round(min(lon0, lon1), 6), round(min(lat0, lat1), 6),
                           round(max(lon0, lon1), 6), round(max(lat0, lat1), 6)],
        })
    (OUT / "inventory.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inv), encoding="utf-8")

    by_date = Counter(r["date"] for r in inv)
    by_tile = Counter(r["tile_id"] for r in inv)
    result["A3_inventory"] = {
        "rows": len(inv),
        "unique_tiles": len(by_tile),
        "unique_dates": len(by_date),
        "split_counts": dict(Counter(r["split"] for r in inv)),
        "platform_counts": dict(Counter(r["platform"] for r in inv)),
        "dates_per_tile_min_max": [min(by_tile.values()), max(by_tile.values())] if by_tile else None,
        "wgs84_extent": [
            round(min(r["wgs84_bbox"][0] for r in inv), 4),
            round(min(r["wgs84_bbox"][1] for r in inv), 4),
            round(max(r["wgs84_bbox"][2] for r in inv), 4),
            round(max(r["wgs84_bbox"][3] for r in inv), 4),
        ] if inv else None,
        "file": str(OUT / "inventory.jsonl"),
    }
    result["gates"] = {
        "A1_join_nonzero": len(joinable) > 0,
        "A1_join_covers_labels": len(joinable) >= 0.9 * len(lk) if lk else False,
        "A2_interpretation_decided": interp is not None,
        "A3_bbox_in_korea": bool(inv) and 124.0 < result["A3_inventory"]["wgs84_extent"][0] < 132.0,
    }
    (OUT / "inventory_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
