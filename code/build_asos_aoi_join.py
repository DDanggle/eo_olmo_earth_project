#!/usr/bin/env python3
"""ASOS 지점 → AI-Hub 71363 AOI 군집 최근접 결합. `era5_10` 슬롯 재료를 만듦.

배경
  M14  OlmoEarth에 기상 전용 **비공간** modality `era5_10`이 있음
       (2m-temperature / dewpoint / surface-pressure / 10m u·v wind / total-precipitation)
  M17  ASOS가 그 6변수를 **과거 날짜로** 준다. 단 응답에 지점 좌표가 없어 막혔음
  M19  좌표는 `stn_inf.php`에 있음. apihub 활용신청으로 열렸음
  M20  GK2A 격자는 `x0` 해석 불가로 봉인 보류. **이 경로는 격자와 무관하므로 막히지 않음**

이 스크립트
  1. `stn_inf.php`에서 96지점의 (STN, LON, LAT, HT)를 파싱함
  2. M10에서 동결한 13개 AOI 군집 중심을 WGS84로 변환함
  3. 군집별 최근접 지점을 정하고 **거리를 함께 기록함**
  4. 사전 등록 판정 — 거리가 멀면 그 군집의 기상 residual은 신뢰할 수 없음

사전 등록 판정 (L4: 실험 전에 기준을 정함)
  J1 모든 군집에 최근접 지점이 존재함
  J2 최근접 거리 중위값 ≤ 30 km
  J3 최근접 거리가 60 km를 넘는 군집을 **명시적으로 표시**하고, 그 군집의 기상 residual은
     "약함"으로 보고함. 조용히 쓰지 않음
  J4 한 지점이 4개 이상 군집을 담당하면 표시함 (같은 값이 여러 군집에 복제되어
     지역 간 변동을 없애므로 negative control에서 이득이 남을 위험이 있음)

보간하지 않음. **최근접 지점 값**부터 시작함 — 보간은 그 자체로 계약 변경임(M8 계열 위험).
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from collections import Counter
from pathlib import Path

APIHUB = "https://apihub.kma.go.kr/api/typ01/url"
ART = Path(__file__).resolve().parent.parent / "artifacts"
OUT = Path(os.environ.get("ASOS_OUT",
           str(Path.home() / "dong/ai_projects/data/asos")))

MEDIAN_KM_MAX = 30.0
FAR_KM = 60.0
MAX_CLUSTERS_PER_STN = 4


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=90) as r:
        raw = r.read()
    for enc in ("cp949", "utf-8", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def parse_stations(text: str) -> list[dict]:
    """`stn_inf.php` 파싱. 주석(#)은 정의 줄이므로 버림.

    실측 컬럼 순서:
      STN LON LAT STN_SP HT HT_PA HT_TA HT_WD HT_RN STN_AD STN_KO STN_EN
      FCT_ID LAW_ID BASIN LAW_ADDR...
    한글 지점명에 공백이 없다는 보장이 없으므로 **앞쪽 수치 필드만** 위치로 읽음.
    """
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) < 12:
            continue
        try:
            stn = int(f[0])
            lon, lat = float(f[1]), float(f[2])
            ht = float(f[4])
        except (ValueError, IndexError):
            continue
        out.append({"stn": stn, "lon": lon, "lat": lat, "ht": ht,
                    "name_ko": f[10] if len(f) > 10 else "",
                    "fct_id": f[12] if len(f) > 12 else "",
                    "law_id": f[13] if len(f) > 13 else ""})
    return out


def utm52n_to_wgs84(x: float, y: float) -> tuple[float, float]:
    a, f = 6378137.0, 1 / 298.257223563
    e2 = 2 * f - f * f
    k0, e0, n0, lon0 = 0.9996, 500000.0, 0.0, math.radians(129.0)
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


def haversine_km(lo1: float, la1: float, lo2: float, la2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def main() -> None:
    key = os.environ.get("KMA_API_HUB", "")
    if not key:
        raise SystemExit("KMA_API_HUB 없음")
    OUT.mkdir(parents=True, exist_ok=True)

    text = fetch_text(f"{APIHUB}/stn_inf.php?inf=SFC&stn=&tm=202204170900&help=0&authKey={key}")
    (OUT / "stn_inf_sfc.txt").write_text(text, encoding="utf-8")
    stations = parse_stations(text)

    holdout = json.loads((ART / "aihub71363_spatial_holdout.json").read_text(encoding="utf-8"))
    assign = {}
    for line in (ART / "aihub71363_tile_assignment.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            assign.setdefault(r["cluster"], r["split"])

    clusters = []
    for cid, st in sorted(holdout["cluster_stats"].items()):
        x0, y0, x1, y1 = st["utm_extent"]
        lon, lat = utm52n_to_wgs84((x0 + x1) / 2, (y0 + y1) / 2)
        best = min(stations, key=lambda s: haversine_km(lon, lat, s["lon"], s["lat"]))
        dist = haversine_km(lon, lat, best["lon"], best["lat"])
        # 상위 3개도 남김 — 최근접이 멀 때 대안을 보려면 필요함
        top3 = sorted(stations, key=lambda s: haversine_km(lon, lat, s["lon"], s["lat"]))[:3]
        clusters.append({
            "cluster": cid, "split": assign.get(cid), "tiles": st["tiles"],
            "pairs": st["pairs"],
            "landslide_tiles": st["tiles_per_class"].get("80:산사태및토석류피해지", 0),
            "logging_tiles": st["tiles_per_class"].get("70:벌목지", 0),
            "centroid": [round(lon, 5), round(lat, 5)],
            "nearest_stn": best["stn"], "nearest_name": best["name_ko"],
            "nearest_km": round(dist, 2),
            "alternatives": [{"stn": s["stn"], "name": s["name_ko"],
                              "km": round(haversine_km(lon, lat, s["lon"], s["lat"]), 2)}
                             for s in top3],
        })

    dists = sorted(c["nearest_km"] for c in clusters)
    median = dists[len(dists) // 2]
    far = [c["cluster"] for c in clusters if c["nearest_km"] > FAR_KM]
    load = Counter(c["nearest_stn"] for c in clusters)
    overloaded = {s: n for s, n in load.items() if n >= MAX_CLUSTERS_PER_STN}

    result = {
        "schema": "asos-aoi-join-v1",
        "station_count": len(stations),
        "cluster_count": len(clusters),
        "thresholds": {"median_km_max": MEDIAN_KM_MAX, "far_km": FAR_KM,
                       "max_clusters_per_stn": MAX_CLUSTERS_PER_STN},
        "distance_km": {"min": dists[0], "median": median, "max": dists[-1]},
        "gates": {
            "J1_all_clusters_matched": len(clusters) == len(holdout["cluster_stats"]),
            "J2_median_within_limit": median <= MEDIAN_KM_MAX,
            "J3_far_clusters_flagged": True,   # 표시하는 것이 통과 조건임
            "J4_no_station_overloaded": not overloaded,
        },
        "far_clusters": far,
        "overloaded_stations": overloaded,
        "clusters": clusters,
    }
    result["verdict"] = ("결합 사용 가능" if all(result["gates"].values())
                         else "결합 사용 가능하나 표시된 군집은 '약함'으로 보고함")
    (OUT / "asos_aoi_join.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("ASOS 지점 %d개, 군집 %d개" % (len(stations), len(clusters)))
    print("최근접 거리 km: 최소 %.1f · 중위 %.1f · 최대 %.1f"
          % (dists[0], median, dists[-1]))
    print("\n%-5s %-6s %5s %5s %5s  %-14s %7s" %
          ("군집", "split", "타일", "산사태", "벌목", "최근접 지점", "km"))
    for c in sorted(clusters, key=lambda d: -d["nearest_km"]):
        flag = "  <<< 멀다" if c["nearest_km"] > FAR_KM else ""
        print("%-5s %-6s %5d %5d %5d  %-14s %7.1f%s" %
              (c["cluster"], c["split"], c["tiles"], c["landslide_tiles"],
               c["logging_tiles"], f'{c["nearest_stn"]} {c["nearest_name"]}',
               c["nearest_km"], flag))
    print("\n게이트:", json.dumps(result["gates"], ensure_ascii=False))
    if overloaded:
        print("과부하 지점:", overloaded)
    print("판정:", result["verdict"])
    print("DONE")


if __name__ == "__main__":
    main()
