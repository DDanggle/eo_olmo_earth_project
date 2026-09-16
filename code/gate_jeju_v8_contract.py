#!/usr/bin/env python3
"""v8 계약 사전 게이트 — 임베딩을 돌리기 전에, 그리고 이후 매번 통과해야 한다.

`audit_jeju_candidate_time_contract.py` 는 이미 나온 후보를 **사후** 감사했다.
그때는 늦었다 — 184일 중첩이 발견됐을 때 숫자는 이미 발표돼 있었다.
이 파일은 같은 검사를 **사전**에 건다. 실패하면 파이프라인이 시작되지 않는다.

검사 항목
---------
1. 계절 정렬: 타일별 연도 간 DOY 폭이 한계 이내인가 (v7.10 은 몇 달 어긋났다)
2. 간격 일치: 사건 쌍과 귀무 쌍의 평균 간격 차이가 허용치 이내인가
3. 창 중첩: 쌍의 두 관측이 서로 다른 해인가 (v7.10 은 184일 겹쳤다)
4. item ID 유효: 동결된 STAC ID 가 지금도 조회되는가 (PC 재처리 감지)
5. 주장 경계: 금지 주장 목록이 계약에 남아 있는가
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

from jeju_paths import CONTRACT_ROOT, display_path

MAX_DOY_SPREAD = 14        # 2주. v7.10 은 계절 자체가 달랐다.
REQUIRED_FORBIDDEN = ["오름 훼손을 탐지했다", "레이더 순위의 p 값", "368 을 분모로 사용"]


YEARS_STR = ("2023", "2024", "2025", "2026")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--check-stac", action="store_true",
                    help="동결된 item ID 를 STAC 에 재조회한다 (네트워크 필요)")
    a = ap.parse_args()

    path = CONTRACT_ROOT / f"{a.contract}_contract.json"
    if not path.exists():
        print(f"FAIL  계약 없음: {display_path(path)}  — freeze_jeju_v8_contract.py 를 먼저 돌려라")
        return 1
    c = json.loads(path.read_text())
    fails: list[str] = []
    # 연도·쌍·DOY 규칙은 계약에서 읽는다 (v8x 처럼 6개년으로 늘어나도 게이트가 같은 규칙을 적용하도록)
    years_str = tuple(str(y) for y in c["optical"]["years"])
    pair_keys = [k for k, v in c["pairs"].items() if isinstance(v, dict) and "from" in v]
    doy_rule = c["pairs"].get("doy_rule", "all")
    print(f"계약 {display_path(path)}  frozen {c['frozen_at_utc']}")

    # 1. 계절 정렬
    print("\n[1] 계절 정렬 (타일별 연도 간 DOY 폭)")
    for tile, rec in c["optical"]["scenes"].items():
        if doy_rule == "adjacent":     # 인접 연도 쌍마다 폭을 잰다 — 비교는 항상 쌍이다
            doys = [rec[y]["doy"] for y in years_str]
            spread = max(abs(doys[k + 1] - doys[k]) for k in range(len(doys) - 1))
        else:
            spread = rec["_doy_spread_days"]
        ok = spread <= MAX_DOY_SPREAD
        print(f"    {tile}  {spread:3}d  최대구름 {rec['_max_cloud']:5.1f}%  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{tile} DOY 폭 {spread}d > {MAX_DOY_SPREAD}d")

    # 2. 간격 일치
    print("\n[2] 간격 일치 (사건 쌍 대비)")
    pairs, tol = c["pairs"], c["pairs"]["gap_tolerance_days"]
    ev = pairs["event"]["gap_days_mean"]
    for key in [k for k in pair_keys if k != "event"]:
        g = pairs[key]["gap_days_mean"]
        diff = abs(ev - g)
        ok = diff <= tol
        print(f"    {key:26} {g:6.1f}d  차이 {diff:4.1f}d  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{key} 간격 차이 {diff:.0f}d > {tol}d")

    # 3. 창 중첩 — v7.10 의 치명적 결함
    print("\n[3] 창 중첩 (v7.10 은 184일 겹쳤다)")
    for key in pair_keys:
        y0, y1 = pairs[key]["from"], pairs[key]["to"]
        overlap = 0 if y1 > y0 else -1
        print(f"    {key:26} {y0}→{y1}  중첩 {overlap}일  {'ok' if overlap == 0 else 'FAIL'}")
        if overlap != 0:
            fails.append(f"{key} 창 순서 이상")

    # 4. item ID 유효성
    print("\n[4] 동결된 STAC item ID")
    if a.check_stac:
        import pystac_client
        cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        checked = missing = 0
        for tile, rec in c["optical"]["scenes"].items():
            for year in years_str:
                iid = rec[year]["item_id"]
                checked += 1
                try:
                    cat.get_collection("sentinel-2-l2a").get_item(iid)
                except Exception:
                    missing += 1
                    fails.append(f"item 사라짐: {iid}")
        print(f"    {checked}건 확인, 실패 {missing}건  {'ok' if missing == 0 else 'FAIL'}")
    else:
        n = sum(len([k for k in r if k.isdigit()]) for r in c["optical"]["scenes"].values())
        print(f"    {n}건 동결됨 (재조회 생략 — --check-stac 로 확인)")

    # 4b. 궤도 일관성 + 프레임 A 커버리지
    #  둘 다 실측 결함에서 나왔다. 궤도 제약이 없을 때 52SBB 는 2023~2025 를 면적 0.11 짜리
    #  조각 granule(R103)로, 2026 만 전체 장면(R003)으로 잡았고, 그 결과 오름 34곳이 네 해를
    #  다 갖지 못한 채 조용히 분석에서 빠졌다. 조용히 빠지는 것이 이 게이트가 막는 종류다.
    print("\n[4b] 궤도 일관성 · 프레임 A 커버리지")
    for tile, rec in c["optical"]["scenes"].items():
        orbits = {rec[y]["relative_orbit"] for y in years_str}
        ok = len(orbits) == 1
        print(f"    {tile:8s} 궤도 {sorted(orbits)}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{tile} 연도 간 궤도 불일치 {sorted(orbits)}")
    if a.check_stac:
        import sys as _sys
        _sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from shapely.geometry import shape, Point
        from oreum_prepare import load_oreum
        coll = cat.get_collection("sentinel-2-l2a")
        g = {(t, y): shape(coll.get_item(r[y]["item_id"]).geometry)
             for t, r in c["optical"]["scenes"].items() for y in years_str}
        oreum = load_oreum(None)
        bad = [o["oreum_id"] for o in oreum
               if not any(all(g[(t, y)].contains(Point(o["lon"], o["lat"])) for y in years_str)
                          for t in c["optical"]["scenes"])]
        print(f"    오름 {len(oreum)}곳 중 네 해 모두 덮임 {len(oreum) - len(bad)}  "
              f"{'ok' if not bad else 'FAIL ' + str(bad[:5])}")
        if bad:
            fails.append(f"네 해를 다 덮지 못하는 오름 {len(bad)}곳")
    else:
        print("    커버리지 재조회 생략 — --check-stac 로 확인")

    # 5. 주장 경계
    print("\n[5] 주장 경계")
    forb = c.get("claims_forbidden", [])
    for needed in REQUIRED_FORBIDDEN:
        ok = any(needed in f for f in forb)
        print(f"    금지 유지: {needed[:28]:30} {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"금지 주장 누락: {needed}")
    rn = c["radar"]["temporal_null_available"]
    print(f"    레이더 시간귀무 = {rn}  {'ok (없음을 명시)' if rn is False else 'FAIL'}")
    if rn is not False:
        fails.append("레이더 시간귀무가 없는데 있다고 표기됨")

    print("\n" + ("=" * 58))
    if fails:
        print(f"GATE FAIL — {len(fails)}건")
        for f in fails:
            print(f"  · {f}")
        return 1
    print("GATE PASS — 임베딩 실행을 허용한다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
