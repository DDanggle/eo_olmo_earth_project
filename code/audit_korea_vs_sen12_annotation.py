#!/usr/bin/env python3
"""M13 — 한국(AI-Hub 71363) 산사태 폴리곤을 Sen12Landslides 16지역과 같은 자로 잼.

목적: `Italy -> Korea` transfer arm이 성립하는지 판정함.
G-A 3단계 통제표에 한국을 얹으려면 한국의 annotation descriptor가 필요함.

측정 대상은 AI-Hub 71363 라벨 GeoJSON의 `ANN_CD=80` (산사태·토석류 피해지) 폴리곤임.
좌표는 EPSG:32652 미터이므로 shoelace로 면적을 바로 구함 (재투영 불필요).
비교 대상은 `sen12landslides/audit/annotation_audit.json`의 지역별 면적 통계임.

부가 측정: `ANN_CD=70`(벌목지)도 같이 냄. 한국 auxiliary head의 descriptor로 씀.
"""
from __future__ import annotations

import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("/home/work/data/olmoearth/aihub/raw/71363")
S12 = Path("/home/work/data/olmoearth/sen12landslides/audit/annotation_audit.json")
ASSIGN = Path("/home/work/data/olmoearth/aihub/splits/tile_assignment.jsonl")
OUT = Path("/home/work/data/olmoearth/aihub/audit")
TARGETS = {80: "산사태및토석류피해지", 70: "벌목지"}


def read_json(data: bytes):
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return json.loads(data.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("디코딩 실패")


def ring_area(ring) -> float:
    """shoelace. 좌표가 미터(EPSG:32652)이므로 결과는 m^2임."""
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def poly_area(geom: dict) -> float:
    """Polygon / MultiPolygon 면적. 내부 링(구멍)은 뺌."""
    t = (geom or {}).get("type")
    c = (geom or {}).get("coordinates") or []
    total = 0.0
    if t == "Polygon":
        polys = [c]
    elif t == "MultiPolygon":
        polys = c
    else:
        return 0.0
    for poly in polys:
        for i, ring in enumerate(poly):
            a = ring_area(ring)
            total += a if i == 0 else -a
    return max(0.0, total)


def pctl(v: list[float], q: float) -> float:
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split_of = {}
    if ASSIGN.exists():
        for line in ASSIGN.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                split_of[r["tile_id"]] = r["split"]

    areas = defaultdict(list)            # code -> [m2]
    areas_by_split = defaultdict(list)   # (code, split) -> [m2]
    tiles_with = defaultdict(set)
    for pattern in ("TL_02.JSON_03*.zip", "VL_02.JSON_03*.zip"):
        hits = sorted(RAW.rglob(pattern))
        if not hits:
            continue
        with zipfile.ZipFile(hits[0]) as zf:
            for name in [n for n in zf.namelist() if n.lower().endswith(".json")]:
                obj = read_json(zf.read(name))
                if not isinstance(obj, dict):
                    continue
                key = str(obj.get("name") or Path(name).stem)
                tile_id = key.rpartition("_")[0]
                sp = split_of.get(tile_id, "(unassigned)")
                for feat in obj.get("features") or []:
                    props = (feat or {}).get("properties") or {}
                    cd = props.get("ANN_CD")
                    if cd not in TARGETS:
                        continue
                    a = poly_area((feat or {}).get("geometry") or {})
                    if a <= 0:
                        continue
                    areas[cd].append(a)
                    areas_by_split[(cd, sp)].append(a)
                    tiles_with[cd].add(tile_id)

    korea = {}
    for cd, nm in TARGETS.items():
        v = areas[cd]
        korea[f"{cd}:{nm}"] = {
            "polygons": len(v), "tiles": len(tiles_with[cd]),
            "min": pctl(v, 0.0), "p1_mmu": pctl(v, 0.01), "median": pctl(v, 0.50),
            "p99": pctl(v, 0.99), "max": pctl(v, 1.0),
            "log10_hist": dict(sorted(Counter(
                int(math.floor(math.log10(a))) for a in v if a > 0).items())),
            "by_split": {sp: {"polygons": len(areas_by_split[(cd, sp)]),
                              "median": pctl(areas_by_split[(cd, sp)], 0.50)}
                         for sp in ("train", "val", "test", "excluded", "(unassigned)")
                         if areas_by_split[(cd, sp)]},
        }

    result = {"schema": "korea-vs-sen12-annotation-v1", "korea": korea}

    # ---- Sen12Landslides와 같은 표에 얹음 ----
    if S12.exists():
        s12 = json.loads(S12.read_text(encoding="utf-8"))
        rows = []
        for loc, v in s12["per_region"].items():
            a = v["A2_area_m2"]
            top = max(v["A3_authors"], key=v["A3_authors"].get) if v["A3_authors"] else ""
            rows.append({"region": loc, "polygons": v["A1_polygons"],
                         "p1_mmu": a["p1_mmu"], "median": a["median"], "p99": a["p99"],
                         "author": top.replace("HÃ¶hn", "Hohn")})
        k = korea["80:산사태및토석류피해지"]
        rows.append({"region": "Korea_AIHub", "polygons": k["polygons"],
                     "p1_mmu": k["p1_mmu"], "median": k["median"], "p99": k["p99"],
                     "author": "AI-Hub/NIA (2023)"})
        rows.sort(key=lambda r: r["median"])
        result["combined"] = rows

        # Italy 및 Hohn 지역과의 MMU 비
        italy = next((r for r in rows if r["region"] == "Italy"), None)
        hohn = [r for r in rows if r["author"].startswith("Hohn") and r["polygons"] >= 100]
        k_mmu = k["p1_mmu"]
        result["verdict"] = {
            "korea_mmu_m2": k_mmu,
            "korea_median_m2": k["median"],
            "italy_mmu_m2": italy["p1_mmu"] if italy else None,
            "italy_vs_korea_mmu_ratio": (round(max(k_mmu, italy["p1_mmu"])
                                               / min(k_mmu, italy["p1_mmu"]), 2)
                                         if italy and min(k_mmu, italy["p1_mmu"]) > 0 else None),
            "italy_vs_korea_median_ratio": (round(max(k["median"], italy["median"])
                                                  / min(k["median"], italy["median"]), 2)
                                            if italy else None),
            "hohn_mmu_range": [min(r["p1_mmu"] for r in hohn),
                               max(r["p1_mmu"] for r in hohn)] if hohn else None,
            "korea_within_hohn_mmu_range": (
                bool(hohn) and min(r["p1_mmu"] for r in hohn) <= k_mmu
                <= max(r["p1_mmu"] for r in hohn)),
        }

    (OUT / "korea_annotation_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== 한국 AI-Hub 71363 ===")
    for k, v in korea.items():
        print("  %-24s 폴리곤%6d 타일%5d  MMU=%9.1f median=%10.1f p99=%11.1f"
              % (k, v["polygons"], v["tiles"], v["p1_mmu"], v["median"], v["p99"]))
        print("     split별:", {s: d["polygons"] for s, d in v["by_split"].items()})
    if "combined" in result:
        print("\n=== median 면적 오름차순 (한국 포함) ===")
        print("%-16s %8s %10s %11s %12s  %s"
              % ("region", "polys", "MMU(p1)", "median", "p99", "author"))
        for r in result["combined"]:
            mark = " <<<" if r["region"] == "Korea_AIHub" else ""
            print("%-16s %8d %10.1f %11.1f %12.1f  %-28s%s"
                  % (r["region"], r["polygons"], r["p1_mmu"], r["median"], r["p99"],
                     r["author"][:28], mark))
        print("\n판정:", json.dumps(result["verdict"], ensure_ascii=False, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
