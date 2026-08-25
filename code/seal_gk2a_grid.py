#!/usr/bin/env python3
"""GK2A KO/2km 격자를 공식 KMA 파일로 봉인함. 역공학을 대체함.

배경: M15의 역공학(LCC 오프셋 적합)은 **철회**됐음. 고유 앵커 4곳에서 오프셋을 적합하고
같은 4곳에서 평가한 in-sample fit이었으며, 캐시 키에 `resultType`이 없어 FOG가 CLD를
덮어쓸 수 있는 결함도 있었음. 기상청이 공식 KO/2km lat/lon을 **격자 저장 순서대로**
제공하므로 역공학 자체가 불필요함.

공식 경로 (apihub.kma.go.kr, 2026-08-25 확인):
  ASCII   /api/typ01/cgi-bin/url/nph-gk2a_latlon_api?area=KO&grid=2&latlon=lon|lat&disp=A&authKey=
  NetCDF  /api/typ01/url/gk2a_latlon_file_down.php?area=KO&grid=2&authKey=
  키 없이 호출하면 `401 유효한 인증키가 아닙니다.` — data.go.kr 키와 **별개**임.

이 스크립트가 하는 일
  1. lon·lat 배열을 공식 경로로 받음
  2. **사전 등록 검증 4개**를 통과해야 봉인함
  3. 통과 시 원본 응답의 SHA-256과 파생 인덱스를 봉인 파일에 씀

사전 등록 검증 (하나라도 실패하면 봉인하지 않음)
  V1 개수    lon·lat 원소 수가 GK2A 응답의 `xdim × ydim` = 320 × 397 = 127,040과 같음
  V2 순서    row-major (397, 320)로 재구성했을 때 한 행 안에서 경도가 단조증가하고
             한 열 안에서 위도가 단조(증가 또는 감소)함. 아니면 저장 순서가 다른 것임
  V3 범위    경도 120~135, 위도 30~45 안에 들어옴 (한반도 영역)
  V4 앵커    **자유 매개변수 0개**로 검증함. Area API가 준 앵커의 lon/lat에 대해 공식
             격자에서 최근접 칸을 찾고, 같은 시각 격자값이 Area 값과 일치하는지 봄.
             일치율 ≥ 0.90 이면 통과.

V4가 M15와 결정적으로 다른 점: **아무것도 적합하지 않음.** 대응 관계가 공식 파일에서
직접 오므로, 높은 일치율은 증거가 되고 낮은 일치율은 반증이 됨. 앵커가 4곳이라는 한계는
남지만 in-sample fit 문제는 사라짐.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

APIHUB = "https://apihub.kma.go.kr"
GK2A_ROOT = Path(os.environ.get("GK2A_ROOT",
                 str(Path.home() / "dong/ai_projects/data/gk2a")))
OUT = GK2A_ROOT / "_grid"

EXPECT_X, EXPECT_Y = 320, 397          # GK2A 응답의 xdim, ydim (실측)
LON_RANGE, LAT_RANGE = (120.0, 135.0), (30.0, 45.0)
ANCHOR_MATCH_MIN = 0.90


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        return exc.code, exc.read()
    except Exception as exc:  # noqa: BLE001
        return -1, f"EXC {type(exc).__name__}: {exc}".encode()


def parse_ascii(body: bytes) -> list[float]:
    """ASCII 응답을 실수 배열로 만듦. 주석·헤더 줄은 버림."""
    vals: list[float] = []
    for tok in body.decode("utf-8", "replace").replace(",", " ").split():
        if tok.startswith("#"):
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals


def load_grid_values(day_dir: Path, hhmm: str) -> list[str] | None:
    f = day_dir / f"getGk2acldAll_CLD_{hhmm}.json.gz"
    if not f.exists():
        return None
    it = json.loads(gzip.open(f).read())["response"]["body"]["items"]["item"][0]
    return it["value"].split(",")


def main() -> None:
    key = (os.environ.get("KMA_API_HUB")
           or os.environ.get("KMA_APIHUB_KEY") or "")
    if not key:
        print("KMA_API_HUB 없음.\n"
              "  apihub.kma.go.kr 에서 인증키를 발급받아 .env 에 넣어야 함.\n"
              "  data.go.kr 키와 별개이며, 키 없이 호출하면 401임.", file=sys.stderr)
        raise SystemExit(2)

    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"schema": "gk2a-ko2km-grid-seal-v1",
                    "source": f"{APIHUB}/api/typ01/cgi-bin/url/nph-gk2a_latlon_api",
                    "params": {"area": "KO", "grid": 2, "disp": "A"},
                    "expect": {"xdim": EXPECT_X, "ydim": EXPECT_Y,
                               "cells": EXPECT_X * EXPECT_Y}}

    arrays: dict[str, list[float]] = {}
    for which in ("lon", "lat"):
        url = (f"{APIHUB}/api/typ01/cgi-bin/url/nph-gk2a_latlon_api"
               f"?area=KO&grid=2&latlon={which}&disp=A&authKey={key}")
        status, body = fetch(url)
        raw_path = OUT / f"ko2km_{which}.txt"
        result[which] = {"http": status, "bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest()}
        if status != 200:
            result[which]["head"] = body[:200].decode("utf-8", "replace")
            result["verdict"] = f"{which} 요청 실패 (http {status}) — 봉인하지 않음"
            (OUT / "grid_seal.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        raw_path.write_bytes(body)
        arrays[which] = parse_ascii(body)
        result[which]["count"] = len(arrays[which])
        result[which]["raw_file"] = str(raw_path)

    lon, lat = arrays["lon"], arrays["lat"]
    n = EXPECT_X * EXPECT_Y

    # ---- V1 개수 ----
    v1 = len(lon) == n and len(lat) == n

    # ---- V2 순서 (row-major (ydim, xdim) 가정) ----
    v2 = False
    row_mono = col_mono = None
    if v1:
        def row(j):
            return lon[j * EXPECT_X:(j + 1) * EXPECT_X]

        def col(i):
            return [lat[j * EXPECT_X + i] for j in range(EXPECT_Y)]

        rows_ok = sum(1 for j in range(0, EXPECT_Y, 20)
                      if all(b > a for a, b in zip(row(j), row(j)[1:])))
        rows_tested = len(range(0, EXPECT_Y, 20))
        cols_inc = sum(1 for i in range(0, EXPECT_X, 20)
                       if all(b > a for a, b in zip(col(i), col(i)[1:])))
        cols_dec = sum(1 for i in range(0, EXPECT_X, 20)
                       if all(b < a for a, b in zip(col(i), col(i)[1:])))
        cols_tested = len(range(0, EXPECT_X, 20))
        row_mono = f"{rows_ok}/{rows_tested}"
        col_mono = f"inc {cols_inc}/{cols_tested}, dec {cols_dec}/{cols_tested}"
        v2 = rows_ok == rows_tested and max(cols_inc, cols_dec) == cols_tested
        result["v2_detail"] = {"lon_row_monotonic": row_mono, "lat_col_monotonic": col_mono,
                              "lat_direction": ("north_to_south" if cols_dec == cols_tested
                                                else "south_to_north" if cols_inc == cols_tested
                                                else "불규칙")}

    # ---- V3 범위 ----
    v3 = (v1 and LON_RANGE[0] <= min(lon) and max(lon) <= LON_RANGE[1]
          and LAT_RANGE[0] <= min(lat) and max(lat) <= LAT_RANGE[1])
    if v1:
        result["extent"] = {"lon": [round(min(lon), 5), round(max(lon), 5)],
                            "lat": [round(min(lat), 5), round(max(lat), 5)]}

    # ---- V4 앵커 (자유 매개변수 0개) ----
    anchors_p = GK2A_ROOT / "_crs" / "area_anchors.jsonl"
    hits = Counter()
    samples = []
    if v1 and anchors_p.exists():
        # resultType 을 키에 포함함 — M15의 결함(FOG가 CLD를 덮어씀)을 반복하지 않음
        recs = []
        for line in anchors_p.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            if r.get("resultType", "CLD") != "CLD":
                continue
            recs.append(r)
        for r in recs:
            dt = r["dateTime"]
            day = GK2A_ROOT / dt[0:4] / dt[4:6] / dt[6:8]
            vals = load_grid_values(day, dt[8:12])
            if vals is None or len(vals) != n:
                continue
            # 최근접 칸 (공식 lon/lat 배열에서 직접 찾음)
            best_idx, best_d = None, float("inf")
            for k in range(n):
                d = (lon[k] - r["lon"]) ** 2 + (lat[k] - r["lat"]) ** 2
                if d < best_d:
                    best_d, best_idx = d, k
            hits["compared"] += 1
            match = vals[best_idx] == r["value"]
            if match:
                hits["match"] += 1
            if len(samples) < 8:
                samples.append({"dateTime": dt, "dong": r["dong"],
                                "anchor_lonlat": [r["lon"], r["lat"]],
                                "grid_index": best_idx,
                                "grid_lonlat": [round(lon[best_idx], 5),
                                                round(lat[best_idx], 5)],
                                "dist_deg": round(math.sqrt(best_d), 5),
                                "area_value": r["value"], "grid_value": vals[best_idx],
                                "match": match})
    rate = (hits["match"] / hits["compared"]) if hits["compared"] else None
    v4 = rate is not None and rate >= ANCHOR_MATCH_MIN
    result["v4_anchor"] = {"compared": hits["compared"], "match": hits["match"],
                           "match_rate": (round(rate, 4) if rate is not None else None),
                           "threshold": ANCHOR_MATCH_MIN,
                           "unique_anchor_points": len({s["dong"] for s in samples}),
                           "samples": samples,
                           "note": "자유 매개변수 0개. 대응은 공식 격자에서 직접 옴"}

    result["gates"] = {"V1_cell_count": v1, "V2_storage_order": v2,
                       "V3_extent_in_korea": v3, "V4_anchor_match": v4}
    sealed = all(result["gates"].values())
    if sealed:
        payload = json.dumps({k: v for k, v in result.items() if k != "seal_sha256"},
                             ensure_ascii=False, sort_keys=True)
        result["seal_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        result["verdict"] = "봉인 완료 — 이 격자 대응을 실험에 사용할 수 있음"
    else:
        failed = [k for k, v in result["gates"].items() if not v]
        result["verdict"] = f"봉인 보류 — 실패 게이트 {failed}. 격자를 실험에 쓰지 않음"

    (OUT / "grid_seal.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    slim = {k: v for k, v in result.items() if k != "v4_anchor"}
    slim["v4_anchor"] = {k: v for k, v in result["v4_anchor"].items() if k != "samples"}
    print(json.dumps(slim, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
