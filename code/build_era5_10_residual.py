#!/usr/bin/env python3
"""ASOS → `era5_10` 형태 residual 생성. 71363 촬영일 60개 전부.

`era5_10` 슬롯 (M14, 실측):
  2m-temperature · 2m-dewpoint-temperature · surface-pressure
  10m-u-component-of-wind · 10m-v-component-of-wind · total-precipitation

ASOS 시간자료(`kma_sfctm2.php`)는 `stn=0`으로 **96지점을 한 번에** 주므로,
촬영일 60개 × 1시각 = 60회 호출로 끝남. 지점별 반복이 필요 없음.

사전 등록 (L4)
  시각      Sentinel-2 통과가 현지 10:30 전후이므로 **11:00 KST**를 주 시각으로 씀.
            민감도 확인용으로 09·10·12·13시도 같이 받음. 주 시각을 사후에 바꾸지 않음.
  결측      ASOS 결측은 `-9`, `-9.0` 등으로 오므로 **버리지 않고 None으로 보존**함.
            보간하지 않음 — 보간은 그 자체로 계약 변경임(M8 계열).
  바람      `WD`(36방위)와 `WS`(m/s)를 같은 시각에서 받아 u/v로 분해함.
            36방위이므로 방위각 = WD × 10도. 기상 관례상 WD는 **바람이 불어오는 방향**이라
            u = -WS·sin(θ), v = -WS·cos(θ) 로 둠. 이 규약을 결과에 명시함.
  단위      변환하지 않고 **원단위와 규약을 함께 기록**함. ERA5는 K·Pa, ASOS는 °C·hPa다.
            변환은 학습 직전 단계에서 명시적으로 하고, 여기서는 원값을 보존함.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections import Counter
from pathlib import Path

APIHUB = "https://apihub.kma.go.kr/api/typ01/url"
ART = Path(__file__).resolve().parent.parent / "artifacts"
OUT = Path(os.environ.get("ASOS_OUT",
           str(Path.home() / "dong/ai_projects/data/asos")))
PRIMARY_HOUR = 11
SENSITIVITY_HOURS = (9, 10, 12, 13)
SLEEP_S = 3
MISSING = {"-9", "-9.0", "-99", "-99.0", "-999", "-999.0", ""}

# `kma_sfctm2.php` 컬럼 순서 (help=1 헤더 실측)
COLS = ["TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
        "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
        "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN"]


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=90) as r:
        raw = r.read()
    for enc in ("cp949", "utf-8", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def num(tok: str):
    if tok in MISSING:
        return None
    try:
        v = float(tok)
    except ValueError:
        return None
    return None if v <= -9 and abs(v) in (9.0, 99.0, 999.0) else v


def parse_sfctm2(text: str) -> dict[int, dict]:
    """지점번호 -> 컬럼 dict. 주석(#)은 정의 줄이므로 버림."""
    out: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) < 20:
            continue
        try:
            stn = int(f[1])
        except ValueError:
            continue
        rec = {}
        for k, tok in zip(COLS, f):
            rec[k] = tok
        out[stn] = rec
    return out


def wind_uv(wd_tok: str, ws_tok: str):
    """36방위 풍향 + 풍속 -> (u, v). WD는 바람이 불어오는 방향(기상 관례)."""
    wd, ws = num(wd_tok), num(ws_tok)
    if wd is None or ws is None:
        return None, None
    theta = math.radians(wd * 10.0)
    return round(-ws * math.sin(theta), 4), round(-ws * math.cos(theta), 4)


def main() -> None:
    key = os.environ.get("KMA_API_HUB", "")
    if not key:
        raise SystemExit("KMA_API_HUB 없음")
    OUT.mkdir(parents=True, exist_ok=True)

    join = json.loads((OUT / "asos_aoi_join.json").read_text(encoding="utf-8"))
    cl2stn = {c["cluster"]: c["nearest_stn"] for c in join["clusters"]}
    stn_km = {c["cluster"]: c["nearest_km"] for c in join["clusters"]}

    # 촬영일: 봉인된 inventory에서 가져옴 (곱집합이 아니라 실제 관측 조합)
    inv = ART / "aihub71363_tile_assignment.jsonl"
    dates = set()
    seal = json.loads((ART / "aihub71363_spatial_holdout.json").read_text(encoding="utf-8"))
    # inventory.jsonl 은 서버에만 있으므로 군집 통계의 dates 수만 참조하고,
    # 실제 날짜 목록은 메타데이터 감사에서 확정된 60개를 쓴다.
    dates_file = OUT / "acquisition_dates.txt"
    if dates_file.exists():
        dates = {d.strip() for d in dates_file.read_text(encoding="utf-8").splitlines()
                 if d.strip()}
    if not dates:
        print(f"촬영일 목록이 없음. {dates_file} 에 YYYYMMDD 를 한 줄씩 넣어야 함.\n"
              "  (서버 inventory.jsonl 의 date 고유값 60개)")
        raise SystemExit(2)

    cache_p = OUT / "asos_hourly_cache.jsonl"
    cache: dict[str, dict] = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                cache[r["tm"]] = r["stations"]

    hours = (PRIMARY_HOUR,) + SENSITIVITY_HOURS
    rows, missing_counter = [], Counter()
    with cache_p.open("a", encoding="utf-8") as cf:
        for date in sorted(dates):
            for hh in hours:
                tm = f"{date}{hh:02d}00"
                if tm in cache:
                    stations = cache[tm]
                else:
                    text = fetch(f"{APIHUB}/kma_sfctm2.php?tm={tm}&stn=0&help=0"
                                 f"&authKey={key}")
                    time.sleep(SLEEP_S)
                    stations = {str(k): v for k, v in parse_sfctm2(text).items()}
                    cache[tm] = stations
                    cf.write(json.dumps({"tm": tm, "stations": stations},
                                        ensure_ascii=False) + "\n")
                    cf.flush()
                for cid, stn in cl2stn.items():
                    rec = stations.get(str(stn))
                    if rec is None:
                        missing_counter[f"{cid}:no_station_row"] += 1
                        continue
                    u, v = wind_uv(rec.get("WD", ""), rec.get("WS", ""))
                    row = {
                        "cluster": cid, "date": date, "hour_kst": hh,
                        "is_primary": hh == PRIMARY_HOUR,
                        "stn": stn, "stn_km": stn_km[cid],
                        # era5_10 대응. 원단위 보존.
                        "temperature_C": num(rec.get("TA", "")),
                        "dewpoint_C": num(rec.get("TD", "")),
                        "surface_pressure_hPa": num(rec.get("PA", "")),
                        "wind_u_ms": u, "wind_v_ms": v,
                        # 강수는 공백이 "무강수"를 뜻함 (M22 실측: 4~10월 98.2%,
                        # 11~3월 98.8%가 공백 — 계절 규칙이 아니라 0을 생략하는 것).
                        # 원값과 가정을 분리해 둘 다 보존함. 조용히 0으로 채우지 않음.
                        "precipitation_mm": num(rec.get("RN", "")),
                        "precipitation_mm_zerofilled": (
                            num(rec.get("RN", "")) if num(rec.get("RN", "")) is not None
                            else 0.0),
                        "precip_was_blank": num(rec.get("RN", "")) is None,
                        "precipitation_day_mm": num(rec.get("RN_DAY", "")),
                        "precip_intensity_mmh": num(rec.get("RN_INT", "")),
                        # 보조 (관측조건·눈사태용)
                        "cloud_total_tenths": num(rec.get("CA_TOT", "")),
                        "snow_depth_cm": num(rec.get("SD_TOT", "")),
                    }
                    for k in ("temperature_C", "dewpoint_C", "surface_pressure_hPa",
                              "wind_u_ms"):
                        if row[k] is None:
                            missing_counter[k] += 1
                    rows.append(row)

    (OUT / "era5_10_residual.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    primary = [r for r in rows if r["is_primary"]]
    n = len(primary)
    def cov(k):
        return round(sum(1 for r in primary if r[k] is not None) / n, 4) if n else None

    summary = {
        "schema": "era5-10-residual-v1",
        "convention": {
            "primary_hour_kst": PRIMARY_HOUR,
            "sensitivity_hours_kst": list(SENSITIVITY_HOURS),
            "wind": "WD(36방위)×10도를 바람이 불어오는 방향으로 보고 u=-WS·sin(θ), v=-WS·cos(θ)",
            "units": "원단위 보존 (°C, hPa, m/s, mm). ERA5의 K·Pa 변환은 학습 직전 단계에서 명시적으로 함",
            "interpolation": "없음. 최근접 지점 값만 사용",
            "precipitation": ("ASOS는 무강수를 생략함(M22). 원값 precipitation_mm(None 가능)과 "
                              "가정 적용값 precipitation_mm_zerofilled(0.0 채움)을 분리 보존하고 "
                              "precip_was_blank 로 표시함. 산사태 forcing에는 시각값이 아니라 "
                              "선행강우 누적이 필요하므로 별도 설계가 남았음"),
        },
        "dates": len(dates), "clusters": len(cl2stn),
        "rows_total": len(rows), "rows_primary": n,
        "coverage_primary": {k: cov(k) for k in
                             ("temperature_C", "dewpoint_C", "surface_pressure_hPa",
                              "wind_u_ms", "precipitation_mm",
                              "precipitation_mm_zerofilled", "precipitation_day_mm",
                              "cloud_total_tenths", "snow_depth_cm")},
        "precip_blank_rate_primary": (
            round(sum(1 for r in primary if r["precip_was_blank"]) / n, 4) if n else None),
        "missing_counts": dict(missing_counter.most_common(12)),
        "file": str(OUT / "era5_10_residual.jsonl"),
    }
    (OUT / "era5_10_residual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
