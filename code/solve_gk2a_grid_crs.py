#!/usr/bin/env python3
"""GK2A 2km 격자의 좌표계를 실측으로 확정함 (추측 금지, 반증 가능한 가설 검정).

문제: 한반도(All) 응답에 CRS가 없음 — `gridKm=2.0 xdim=320 ydim=397 x0=63.0 y0=333.0` 뿐임.
좌표를 못 붙이면 격자를 AOI에 쓸 수 없음.

가설 H1: 기상청 동네예보와 **같은 Lambert Conformal Conic**을 격자간격만 2 km로 쓴 것이다.
  동네예보 표준 파라미터: RE=6371.00877 km, SLAT1=30, SLAT2=60, OLON=126, OLAT=38,
  5 km 격자의 원점 오프셋 XO=43, YO=136.
  2 km로 환산하면 XO2 = 43×2.5 = 107.5, YO2 = 136×2.5 = 340.
  응답의 x0/y0는 전체 2 km 격자 안에서 우리가 받은 부분집합의 시작 인덱스로 본다.

검정 방법: `Area` 계열이 행정동코드에 대해 (lon, lat, value)를 줌.
  같은 시각의 `All` 격자에서 그 lon/lat이 떨어지는 칸의 값을 뽑아 **일치율**을 본다.
  가설이 맞으면 일치율이 높아야 하고, 틀리면 우연 수준(구름탐지 3클래스 → 약 33%)이어야 함.

사전 등록 판정:
  일치율 ≥ 0.90  → H1 채택
  0.50 ~ 0.90    → 오프셋이 다름. XO2/YO2를 격자 탐색으로 보정한 뒤 재검정
  < 0.50         → H1 기각. 문서 확보 전까지 격자를 쓰지 않음

앵커는 검증된 행정동코드 4개임 (경도 126.97~129.34, 위도 35.10~37.58).
Area 응답도 2일 보존이므로 앵커 수집을 미루면 이 검정 자체가 불가능해짐.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

GK2A_ROOT = Path(os.environ.get("GK2A_ROOT",
                 str(Path.home() / "dong/ai_projects/data/gk2a")))
OUT = GK2A_ROOT / "_crs"
BASE = "https://apis.data.go.kr/1360000/CloudSatlitInfoService"

# 검증된 행정동코드 (실측: resultCode 00)
ANCHORS = ["1111051500", "1114052000", "2611051000", "3111051000"]

# 기상청 동네예보 LCC 표준 파라미터
RE, SLAT1, SLAT2, OLON, OLAT = 6371.00877, 30.0, 60.0, 126.0, 38.0
XO_5KM, YO_5KM, GRID_5KM = 43.0, 136.0, 5.0


def lonlat_to_grid(lon: float, lat: float, grid_km: float,
                   xo: float, yo: float) -> tuple[float, float]:
    """기상청 LCC 정변환. 반환은 격자 좌표(실수).

    동네예보 `dfs_xy_conv`와 같은 식이며 격자간격과 원점 오프셋만 매개변수화했음.
    """
    DEGRAD = math.pi / 180.0
    re = RE / grid_km
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD
    sn = (math.tan(math.pi * 0.25 + slat2 * 0.5)
          / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    return ra * math.sin(theta) + xo, ro - ra * math.cos(theta) + yo


def load_grid(day_dir: Path, op: str, rt: str, hhmm: str):
    f = day_dir / f"{op}_{rt}_{hhmm}.json.gz"
    if not f.exists():
        return None
    it = json.loads(gzip.open(f).read())["response"]["body"]["items"]["item"][0]
    return {"xdim": int(float(it["xdim"])), "ydim": int(float(it["ydim"])),
            "x0": float(it["x0"]), "y0": float(it["y0"]),
            "gridKm": float(it["gridKm"]),
            "values": it["value"].split(",")}


def fetch_area(key: str, op: str, rt: str, dt: str, dong: str):
    u = (f"{BASE}/{op}?ServiceKey={key}&pageNo=1&numOfRows=1&dataType=JSON"
         f"&dateTime={dt}&resultType={rt}&dongCode={dong}")
    try:
        d = json.loads(urllib.request.urlopen(u, timeout=30).read().decode())
    except Exception as exc:  # noqa: BLE001
        return None
    if d["response"]["header"]["resultCode"] != "00":
        return None
    it = ((d["response"].get("body") or {}).get("items") or {}).get("item") or [{}]
    it = it[0]
    if "lon" not in it:
        return None
    return float(it["lon"]), float(it["lat"]), it.get("value")


def main() -> None:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    if not key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY 없음")
    OUT.mkdir(parents=True, exist_ok=True)

    # 우리가 이미 받아둔 격자 파일들에서 (날짜, 시각)을 모음
    days = sorted(p for p in GK2A_ROOT.glob("*/*/*") if p.is_dir())
    if not days:
        raise SystemExit(f"수집된 격자가 없음: {GK2A_ROOT}")

    # 부분집합 원점을 전체 2km 격자 인덱스로 환산한 가설값
    xo2, yo2 = XO_5KM * (GRID_5KM / 2.0), YO_5KM * (GRID_5KM / 2.0)

    # Area 응답을 캐시함. 2일 보존이라 재호출이 항상 가능하지 않고, 변형 검정을
    # 여러 번 돌려야 하므로 한 번 받은 것은 디스크에 남김.
    cache_p = OUT / "area_anchors.jsonl"
    cache = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                cache[(r["dateTime"], r["dong"])] = (r["lon"], r["lat"], r["value"])

    trials, hits = [], Counter()
    cache_f = cache_p.open("a", encoding="utf-8")
    for day in days:
        stem = f"{day.parent.parent.name}{day.parent.name}{day.name}"
        for hhmm in [f"{h:02d}00" for h in range(0, 24, 2)]:
            g = load_grid(day, "getGk2acldAll", "CLD", hhmm)
            if g is None:
                continue
            dt = f"{stem}{hhmm}"
            for dong in ANCHORS:
                ck = (dt, dong)
                if ck in cache:
                    a = cache[ck]
                else:
                    a = fetch_area(key, "getGk2acldArea", "CLD", dt, dong)
                    time.sleep(3)
                    if a is not None:
                        cache[ck] = a
                        cache_f.write(json.dumps(
                            {"dateTime": dt, "dong": dong, "lon": a[0], "lat": a[1],
                             "value": a[2]}, ensure_ascii=False) + "\n")
                        cache_f.flush()
                if a is None:
                    continue
                lon, lat, area_val = a
                gx, gy = lonlat_to_grid(lon, lat, g["gridKm"], xo2, yo2)
                # 네 가지 인덱싱 변형을 모두 시험함.
                #   y축은 래스터 관행상 뒤집혀 있을 수 있음(행 0 = 북쪽).
                #   저장 순서도 행우선/열우선 둘 다 가능함.
                i = int(round(gx)) - int(g["x0"])
                variants = {
                    "row_major_ydown": (i, int(round(gy)) - int(g["y0"])),
                    "row_major_yup":   (i, int(g["y0"]) - int(round(gy))),
                    "col_major_ydown": (i, int(round(gy)) - int(g["y0"])),
                    "col_major_yup":   (i, int(g["y0"]) - int(round(gy))),
                }
                got = {}
                for name, (ii, jj) in variants.items():
                    if not (0 <= ii < g["xdim"] and 0 <= jj < g["ydim"]):
                        continue
                    idx = (jj * g["xdim"] + ii) if name.startswith("row") \
                          else (ii * g["ydim"] + jj)
                    if 0 <= idx < len(g["values"]):
                        got[name] = g["values"][idx]
                        hits[name + "_n"] += 1
                        if g["values"][idx] == area_val:
                            hits[name] += 1
                trials.append({"dateTime": dt, "dong": dong, "lon": lon, "lat": lat,
                               "area_value": area_val,
                               "grid_xy": [round(gx, 2), round(gy, 2)],
                               "grid_values": got})
                if got:
                    hits["comparable"] += 1
                else:
                    hits["outside"] += 1
    cache_f.close()

    n = hits["comparable"]
    result = {
        "schema": "gk2a-grid-crs-v1",
        "hypothesis": "기상청 동네예보 LCC, 격자 2km, XO2=%.1f YO2=%.1f" % (xo2, yo2),
        "lcc_params": {"RE": RE, "SLAT1": SLAT1, "SLAT2": SLAT2,
                       "OLON": OLON, "OLAT": OLAT, "XO2": xo2, "YO2": yo2},
        "trials": len(trials), "comparable": n, "outside_subset": hits["outside"],
        "match_rate": {o: (round(hits[o] / hits[o + "_n"], 4) if hits[o + "_n"] else None)
                       for o in ("row_major_ydown", "row_major_yup",
                                 "col_major_ydown", "col_major_yup")},
        "compared_per_variant": {o: hits[o + "_n"] for o in
                                 ("row_major_ydown", "row_major_yup",
                                  "col_major_ydown", "col_major_yup")},
        "samples": trials[:20],
    }
    cands = ("row_major_ydown", "row_major_yup", "col_major_ydown", "col_major_yup")
    best = max(cands, key=lambda o: (hits[o] / hits[o + "_n"]) if hits[o + "_n"] else 0)
    rate = (hits[best] / hits[best + "_n"]) if hits[best + "_n"] else 0.0
    result["verdict"] = (
        f"H1 채택 ({best}, 일치율 {rate:.3f})" if rate >= 0.90 else
        f"오프셋 보정 필요 ({best}, 일치율 {rate:.3f})" if rate >= 0.50 else
        f"H1 기각 (최고 일치율 {rate:.3f}) — 문서 확보 전까지 격자를 쓰지 않음")
    (OUT / "grid_crs_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    if trials:
        print("\n표본 3개:")
        for t in trials[:3]:
            print("  ", json.dumps(t, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
