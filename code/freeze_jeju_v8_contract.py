#!/usr/bin/env python3
"""제주 v8 계약을 굽는다 — 첫 임베딩 실행 **전에** 돌리고 커밋한다.

왜 이 파일이 있는가
-------------------
v7.10 시간계약 감사가 치명적 결함을 찾았다: 2025 창과 rolling-2026 창이 **184일 겹치고**,
"4기간"이 12개 모자이크의 역순 앞 4개라 **계절이 어긋났다**(달력연도는 9~12월, rolling 은 3~6월).
임베딩 모델은 계절 교란을 안정적으로 재현하므로, 그 축 위에서 나온 숫자는 전부 무효다.

그래서 v8 은 달력·rolling 창을 버리고 **좁은 계절 층(8/10~9/25) 안의 실제 획득 장면**을 고른다.
날짜가 아니라 **STAC item ID 를 동결한다** — Planetary Computer 는 재처리하므로 같은 날짜가
다른 자산을 가리킬 수 있다.

선택 규칙 (결정적, 사람 손 안 탐)
---------------------------------
타일마다 네 해에서 한 장면씩, **(1) 연도 간 DOY 폭 최소화 → (2) 최대 구름 최소화** 순으로 고른다.
구름만 보고 고르면 타일별로 날짜가 4주까지 벌어져 계절 정렬이 다시 깨진다.

레이더에 대한 정직한 한계
-------------------------
S1 궤도 134 하강은 2025·2026 에만 있고 **2023·2024 Aug–Sep 획득이 없다.**
따라서 레이더는 사건 쌍만 만들 수 있고 **시간 귀무가 불가능하다.** 계약에 그렇게 적고,
레이더에는 보정된 순위나 p 값을 붙이지 않는다.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime

import pystac_client
from shapely.geometry import shape

from jeju_paths import CONTRACT_ROOT, display_path, ensure

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [126.14, 33.18, 126.98, 33.57]          # 위치 확인된 오름 243곳 + 여유
TILES = ["51SYS", "51SYT", "52SBB", "52SBC", "52SCB", "52SCC"]
YEARS = [2023, 2024, 2025, 2026]
STRATUM = ("08-10", "09-25")                    # 좁은 계절 층
CLOUD_MAX = 15.0
S1_ORBIT, S1_STATE = 134, "descending"
WINDOW_M, PATCH, TOKEN_M = 2560, 4, 40
MIN_VALID_TOKEN_FRAC = 0.2                      # 네팔과 같은 관측가능성 게이트
GAP_TOLERANCE_DAYS = 30                         # 연간 쌍 간 간격 차이 허용치

doy = lambda item: item.datetime.timetuple().tm_yday


def pick_optical(cat) -> dict:
    """타일별로 **하나의 상대궤도** 안에서 DOY 폭 최소 → 최대 구름 최소 순으로 네 해를 고른다.

    궤도를 묶는 이유는 두 가지이고 둘 다 실측으로 확인했다.

    1. **간격 규율.** 궤도가 다르면 관측 기하가 달라지고, 그 차이가 연간 Δz 에 그대로 실린다.
       네팔이 궤도 121 과 19 를 절대 한 쌍에 섞지 않은 것과 같은 이유다.
    2. **커버리지.** 같은 MGRS 타일이라도 궤도마다 granule 이 조각날 수 있다. 궤도 제약 없이
       구름만 최소화했더니 52SBB 의 2023~2025 가 면적 0.11 짜리 R103 조각으로 잡히고 2026 만
       면적 1.16 의 R003 전체 장면이 잡혀, **오름 34곳이 네 해를 다 갖지 못했다.**
    """
    chosen = {}
    for tile in TILES:
        by_orbit: dict[int, dict[int, list]] = {}
        for year in YEARS:
            items = [
                i for i in cat.search(
                    collections=["sentinel-2-l2a"], bbox=BBOX,
                    datetime=f"{year}-{STRATUM[0]}/{year}-{STRATUM[1]}").items()
                if i.properties.get("s2:mgrs_tile") == tile
                and (i.properties.get("eo:cloud_cover") or 100) <= CLOUD_MAX
            ]
            if not items:
                raise SystemExit(f"REFUSED: {tile} {year} 에 구름 {CLOUD_MAX}% 이하 장면이 없다. "
                                 "그 해를 계약에서 빼거나 층을 넓히되, 넓힌 사실을 기록하라.")
            for i in items:
                by_orbit.setdefault(i.properties["sat:relative_orbit"], {}).setdefault(year, []).append(i)

        # 네 해를 모두 가진 궤도만 후보다.
        cands = {o: y for o, y in by_orbit.items() if len(y) == len(YEARS)}
        if not cands:
            raise SystemExit(f"REFUSED: {tile} 에 네 해를 모두 덮는 상대궤도가 없다. "
                             "궤도를 섞는 대신 이 타일을 계약에서 빼라.")

        # 궤도 안에서는 DOY 폭 → 최대 구름. 궤도끼리는 **granule 완전성이 먼저**다.
        # 같은 타일의 두 궤도가 다 네 해를 가져도 한쪽이 조각 granule 일 수 있고, 그 경우
        # 조각을 고르면 해당 지역 오름이 통째로 분석에서 빠진다(실측: 52SBB R103 은 면적
        # 0.11, R003 은 1.16 이고 그 차이가 오름 34곳이었다). 덜 촘촘한 계절 정렬은
        # 각주로 적을 수 있지만, 덮이지 않은 땅은 각주로 되살릴 수 없다.
        per_orbit = {}
        for orbit, per_year in cands.items():
            combo = min(itertools.product(*(sorted(per_year[y], key=lambda i: i.datetime) for y in YEARS)),
                        key=lambda c: (max(map(doy, c)) - min(map(doy, c)),
                                       max(i.properties["eo:cloud_cover"] for i in c)))
            per_orbit[orbit] = (combo, min(shape(i.geometry).area for i in combo))
        best_orbit = min(per_orbit,
                         key=lambda o: (-round(per_orbit[o][1], 2),
                                        max(map(doy, per_orbit[o][0])) - min(map(doy, per_orbit[o][0])),
                                        max(i.properties["eo:cloud_cover"] for i in per_orbit[o][0])))
        best = per_orbit[best_orbit][0]
        best_key = (max(map(doy, best)) - min(map(doy, best)),
                    max(i.properties["eo:cloud_cover"] for i in best))

        chosen[tile] = {
            str(y): {"item_id": i.id, "datetime": str(i.datetime), "doy": doy(i),
                     "cloud_cover": round(i.properties["eo:cloud_cover"], 2),
                     "relative_orbit": i.properties["sat:relative_orbit"]}
            for y, i in zip(YEARS, best)
        }
        chosen[tile]["_relative_orbit"] = best_orbit
        chosen[tile]["_min_footprint_area_deg2"] = round(per_orbit[best_orbit][1], 4)
        chosen[tile]["_doy_spread_days"] = best_key[0]
        chosen[tile]["_max_cloud"] = round(best_key[1], 2)
    return chosen


def pick_radar(cat) -> dict:
    out = {}
    for year in YEARS:
        items = [i for i in cat.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                                       datetime=f"{year}-{STRATUM[0]}/{year}-{STRATUM[1]}").items()
                 if i.properties.get("sat:relative_orbit") == S1_ORBIT]
        out[str(year)] = [{"item_id": i.id, "datetime": str(i.datetime), "doy": doy(i)}
                          for i in sorted(items, key=lambda i: i.datetime)]
    return out


def annual_gaps(optical: dict, a: int, b: int) -> list[int]:
    """타일별 실제 획득 간격(일). 실제 장면이므로 정확히 365 는 아니다."""
    gaps = []
    for tile in TILES:
        t0 = datetime.fromisoformat(optical[tile][str(a)]["datetime"]).replace(tzinfo=None)
        t1 = datetime.fromisoformat(optical[tile][str(b)]["datetime"]).replace(tzinfo=None)
        gaps.append((t1 - t0).days)
    return gaps


def main() -> None:
    cat = pystac_client.Client.open(STAC)
    optical = pick_optical(cat)
    radar = pick_radar(cat)

    event = annual_gaps(optical, 2025, 2026)
    null_a = annual_gaps(optical, 2024, 2025)
    null_b = annual_gaps(optical, 2023, 2024)

    # 네팔의 `assert gaps(event) == gaps(placebo)` 를 실제 획득에 맞게 완화하되 강제한다.
    # 정확한 동수는 실제 장면으로 불가능하므로, 평균 간격 차이를 허용치 안으로 묶는다.
    mean = lambda g: sum(g) / len(g)
    for name, g in (("null_2024_2025", null_a), ("null_2023_2024", null_b)):
        diff = abs(mean(event) - mean(g))
        if diff > GAP_TOLERANCE_DAYS:
            raise SystemExit(f"REFUSED: {name} 평균 간격이 사건 쌍과 {diff:.0f}일 차이난다 "
                             f"(허용 {GAP_TOLERANCE_DAYS}일). 간격이 안 맞는 귀무는 문턱을 왜곡한다.")

    contract = {
        "schema": "jeju-oreum-contract-v8",
        "frozen_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "supersedes": "v7.10 (2025 창과 rolling-2026 창 184일 중첩, 계절 정렬 false)",
        "why": ("달력·rolling 창을 버리고 좁은 계절 층 안의 실제 획득 장면을 동결한다. "
                "날짜가 아니라 STAC item ID 를 고정한다 — PC 는 재처리한다."),

        "frames": {
            "A": {"name": "위치 확인 오름 243곳", "role": "분석 단위",
                  "rule": "top-k 없이 전수 채점. 미해결 125곳은 unresolved 로 보고하되 분모에서 빼지 않는다.",
                  "source": "artifacts/external_data/kearth_oreum_v1/oreum_registry_368.csv"},
            "B": {"name": "제주 전역 격자", "role": "공간 귀무 풀·외부 검정 전용",
                  "rule": "오름 결과로 제시하지 않는다."},
            "C": {"name": "기존 14후보 pool", "role": "동결된 이력",
                  "rule": "A 순위 봉인 후 회고 한 줄만. 재순위·병합·문턱 선택 금지."},
        },

        "geometry": {"bbox": BBOX, "window_m": WINDOW_M, "patch_size": PATCH,
                     "token_ground_m": TOKEN_M, "crs": "EPSG:4326 for selection, UTM for windows"},

        "optical": {
            "collection": "sentinel-2-l2a", "tiles": TILES, "years": YEARS,
            "seasonal_stratum": f"{STRATUM[0]} ~ {STRATUM[1]}",
            "cloud_max_scene": CLOUD_MAX,
            "selection_rule": "타일별로 (1) 연도 간 DOY 폭 최소화 (2) 최대 구름 최소화",
            "compositor": "code/scl_compositor.py :: Sentinel2SCLBestClearNearest "
                          "(SCL class ID 는 nearest 로만 재표본; 보간하면 scl==9 같은 등식이 깨진다)",
            "scenes": optical,
        },

        "radar": {
            "collection": "sentinel-1-rtc", "relative_orbit": S1_ORBIT, "orbit_state": S1_STATE,
            "input_unit": "gamma0 dB — RTC 자산은 linear power 이므로 10*log10 변환 필수",
            "unit_note": "이 변환 누락이 네팔 M75 철회의 원인이었다. model.yaml 은 S1 을 선언하고도 "
                         "Sentinel1ToDecibels 가 없다 — model_s1db.yaml 을 따를 것.",
            "temporal_null_available": False,
            "limitation": "궤도 134 하강에 2023·2024 Aug–Sep 획득이 없다. 사건 쌍만 가능하고 "
                          "시간 귀무를 만들 수 없으므로 레이더에 보정된 순위·p 값을 붙이지 않는다.",
            "scenes": radar,
        },

        "pairs": {
            "event": {"from": 2025, "to": 2026, "gap_days_per_tile": event,
                      "gap_days_mean": round(mean(event), 1)},
            "null_temporal_primary": {"from": 2024, "to": 2025, "gap_days_per_tile": null_a,
                                      "gap_days_mean": round(mean(null_a), 1)},
            "null_temporal_secondary": {"from": 2023, "to": 2024, "gap_days_per_tile": null_b,
                                        "gap_days_mean": round(mean(null_b), 1)},
            "gap_tolerance_days": GAP_TOLERANCE_DAYS,
            "null_bias": "귀무 해에도 실제 변화가 있어 문턱이 부풀고, 따라서 검정은 보수적(과소 탐지)이다. "
                         "안전한 방향이며 보고서에 적는다.",
        },

        "scoring": {
            "change_score": "Δz = 1 − cos(z_before, z_after) per 40 m token",
            "model": "OlmoEarth v1 Base, frozen, patch_size=4",
            "threshold_temporal": "귀무 쌍 유효 토큰의 p99",
            "threshold_spatial": "같은 사건 쌍을 프레임 B 전역에 풀링한 p99",
            "site_score": "그 문턱을 넘은 유효 토큰의 비율",
            "null_flag_rate_expected": 0.01,
            "observability_gate": {
                "token_invalid_if": "픽셀 50% 초과가 구름·결측",
                "site_unobservable_if": f"유효 토큰 비율 < {MIN_VALID_TOKEN_FRAC}",
                "gk2a": "천리안 2A 구름탐지로 독립 판정. SCL 과 불일치하면 그 자체를 기록한다.",
                "abstain": "관측 불가 기간에는 점수를 만들지 않고 abstain 으로 기록한다.",
            },
        },

        "claims_this_contract_can_earn": [
            "이 오름의 유효 토큰 중 X%가 자기 지역의 계절 일치 연간 귀무 p99 를 넘었다.",
            "두 해 모두 관측 가능한 오름은 243 중 N 곳이다.",
        ],
        "claims_forbidden": [
            "오름 훼손을 탐지했다",
            "허가·시설이 이 변화의 원인이다 (causal_or_facility_type_claims_allowed: false 유지)",
            "레이더 순위의 p 값",
            "95.64% 를 탐지 정확도로 (한 window 의 입력 품질 지표다)",
            "368 을 분모로 사용",
            "프레임 A 와 C 를 같은 성공률에 합산",
        ],
    }

    payload = json.dumps(contract, indent=1, ensure_ascii=False)
    contract["_self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out = ensure(CONTRACT_ROOT) / "jeju_v8_contract.json"
    out.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n")

    print(f"계약 동결 → {display_path(out)}")
    print(f"  sha256 {contract['_self_sha256'][:16]}…")
    print(f"  광학 타일 {len(TILES)} × 연도 {len(YEARS)} = {len(TILES)*len(YEARS)} 장면")
    for t in TILES:
        print(f"    {t}  DOY폭 {optical[t]['_doy_spread_days']:2}d  최대구름 {optical[t]['_max_cloud']:5.1f}%")
    print(f"  간격(일) 사건 {mean(event):.0f} · 귀무A {mean(null_a):.0f} · 귀무B {mean(null_b):.0f}")
    print(f"  레이더 획득: " + ", ".join(f"{y}:{len(v)}" for y, v in radar.items()) + "  (시간 귀무 불가)")


if __name__ == "__main__":
    main()
