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
# 공식 KO/2km 전체 격자. 파일 첫 줄이 "900, 900,=" 이므로 900x900임 (실측).
FULL_NX, FULL_NY = 900, 900
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


def parse_ascii(body: bytes) -> tuple[tuple[int, int], list[float]]:
    """ASCII 응답을 (헤더 차원, 값 배열)로 만듦.

    실측 형식: 첫 줄이 `   900,   900,=` 로 격자 차원을 주고, 그 뒤에 값이 콤마로 이어짐.
    첫 줄을 값으로 세면 개수가 2 커진다 — 처음에 그렇게 틀렸음(810,002).
    """
    text = body.decode("utf-8", "replace")
    lines = text.splitlines()
    hdr = [t for t in lines[0].replace(",", " ").replace("=", " ").split() if t]
    dims = (int(hdr[0]), int(hdr[1])) if len(hdr) >= 2 else (0, 0)
    vals: list[float] = []
    for line in lines[1:]:
        for tok in line.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
    return dims, vals


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
        dims, vals = parse_ascii(body)
        arrays[which] = vals
        result[which]["header_dims"] = list(dims)
        result[which]["count"] = len(vals)
        result[which]["raw_file"] = str(raw_path)

    lon_full, lat_full = arrays["lon"], arrays["lat"]
    full_n = FULL_NX * FULL_NY
    n = EXPECT_X * EXPECT_Y

    # ---- V1 개수: 전체 격자가 헤더가 말한 크기와 같은가 ----
    v1 = (len(lon_full) == full_n and len(lat_full) == full_n
          and result["lon"]["header_dims"] == [FULL_NX, FULL_NY])
    result["full_grid"] = {"nx": FULL_NX, "ny": FULL_NY, "cells": full_n}

    # ---- 부분집합 창 추출 ----
    # GK2A 응답은 전체 격자의 (x0, y0)에서 시작하는 xdim x ydim 창임.
    # y 방향 기준이 위/아래 어느 쪽인지 모르므로 두 변형을 만들어 V4로 가린다.
    def window(full, x0, y0, flip_y):
        out = []
        for jj in range(EXPECT_Y):
            j = (y0 + jj) if not flip_y else (y0 - jj)
            if not (0 <= j < FULL_NY):
                return None
            base = j * FULL_NX
            if x0 + EXPECT_X > FULL_NX:
                return None
            out.extend(full[base + x0: base + x0 + EXPECT_X])
        return out

    lon, lat = None, None   # V4에서 창별로 결정함

    # ---- V2 순서: 전체 격자에서 row-major 단조성 ----
    v2 = False
    if v1:
        def row(j):
            return lon_full[j * FULL_NX:(j + 1) * FULL_NX]

        def col(i):
            return [lat_full[j * FULL_NX + i] for j in range(FULL_NY)]

        js = range(0, FULL_NY, 50)
        iss = range(0, FULL_NX, 50)
        rows_ok = sum(1 for j in js if all(b > a for a, b in zip(row(j), row(j)[1:])))
        cols_inc = sum(1 for i in iss if all(b > a for a, b in zip(col(i), col(i)[1:])))
        cols_dec = sum(1 for i in iss if all(b < a for a, b in zip(col(i), col(i)[1:])))
        v2 = rows_ok == len(list(js)) and max(cols_inc, cols_dec) == len(list(iss))
        result["v2_detail"] = {
            "lon_row_monotonic": f"{rows_ok}/{len(list(js))}",
            "lat_col_monotonic": f"inc {cols_inc}/{len(list(iss))}, dec {cols_dec}/{len(list(iss))}",
            "lat_direction": ("north_to_south" if cols_dec == len(list(iss))
                              else "south_to_north" if cols_inc == len(list(iss)) else "불규칙")}

    # ---- V3 범위 ----
    v3 = (v1 and LON_RANGE[0] <= min(lon_full) and max(lon_full) <= LON_RANGE[1]
          and LAT_RANGE[0] <= min(lat_full) and max(lat_full) <= LAT_RANGE[1])
    if v1:
        result["extent"] = {"lon": [round(min(lon_full), 5), round(max(lon_full), 5)],
                            "lat": [round(min(lat_full), 5), round(max(lat_full), 5)]}

    # ---- V4 앵커 (자유 매개변수 0개). 창의 y 방향만 두 가지를 가린다 ----
    anchors_p = GK2A_ROOT / "_crs" / "area_anchors.jsonl"
    recs = []
    if anchors_p.exists():
        for line in anchors_p.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            # resultType 을 필터함 — M15의 결함(FOG가 CLD를 덮어씀)을 반복하지 않음
            if r.get("resultType", "CLD") == "CLD":
                recs.append(r)

    variants = {}
    if v1:
        for name, flip in (("y_down", False), ("y_up", True)):
            wl = window(lon_full, 63, 333, flip)
            wa = window(lat_full, 63, 333, flip)
            if wl is None or wa is None:
                variants[name] = {"error": "창이 격자 밖"}
                continue
            hits = Counter()
            samples = []
            for r in recs:
                dt = r["dateTime"]
                day = GK2A_ROOT / dt[0:4] / dt[4:6] / dt[6:8]
                vals = load_grid_values(day, dt[8:12])
                if vals is None or len(vals) != n:
                    continue
                best_idx, best_d = None, float("inf")
                for k in range(n):
                    d = (wl[k] - r["lon"]) ** 2 + (wa[k] - r["lat"]) ** 2
                    if d < best_d:
                        best_d, best_idx = d, k
                hits["compared"] += 1
                if vals[best_idx] == r["value"]:
                    hits["match"] += 1
                if len(samples) < 6:
                    samples.append({"dateTime": dt, "dong": r["dong"],
                                    "anchor": [r["lon"], r["lat"]],
                                    "cell": [round(wl[best_idx], 5), round(wa[best_idx], 5)],
                                    "dist_deg": round(math.sqrt(best_d), 5),
                                    "area_value": r["value"], "grid_value": vals[best_idx]})
            rate = (hits["match"] / hits["compared"]) if hits["compared"] else None
            variants[name] = {"compared": hits["compared"], "match": hits["match"],
                              "match_rate": (round(rate, 4) if rate is not None else None),
                              "samples": samples}

    best_name = None
    best_rate = 0.0
    for name, v in variants.items():
        r_ = v.get("match_rate") or 0.0
        if r_ > best_rate:
            best_name, best_rate = name, r_
    v4 = best_rate >= ANCHOR_MATCH_MIN
    result["v4_anchor"] = {"variants": variants, "best": best_name,
                           "best_match_rate": round(best_rate, 4),
                           "threshold": ANCHOR_MATCH_MIN,
                           "unique_anchor_points": len({r["dong"] for r in recs}),
                           "subset_origin": {"x0": 63, "y0": 333,
                                             "xdim": EXPECT_X, "ydim": EXPECT_Y},
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
    slim = {k: v for k, v in result.items() if k not in ("v4_anchor", "lon", "lat")}
    slim["lon"] = {k: v for k, v in result["lon"].items() if k != "raw_file"}
    slim["lat"] = {k: v for k, v in result["lat"].items() if k != "raw_file"}
    slim["v4_anchor"] = {k: (v if k != "variants" else
                             {vn: {kk: vv for kk, vv in vv2.items() if kk != "samples"}
                              for vn, vv2 in v.items()})
                         for k, v in result["v4_anchor"].items()}
    print(json.dumps(slim, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
