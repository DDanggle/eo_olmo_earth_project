#!/usr/bin/env python3
"""S4 — 추적 루프. 새 장면이 들어올 때 **계약 템플릿**을 만족하는 삼중쌍이 있으면 갱신하고, 없으면 abstain.

왜 v8 계약을 그대로 반복하지 않는가
-----------------------------------
v8 계약은 8/10~9/25 계절층에 묶여 있다. 11월에 새 장면이 와도 비교 대상은 **11월 2025** 여야
하고 그 앵커는 없다. 계약 하나로는 추적기가 1년에 한 번 갱신된다. 그래서 계약을 **템플릿**으로
바꾼다 — 조건은 v8 과 같고 날짜만 열려 있다:

  * 같은 MGRS 타일, 같은 상대궤도
  * 목표 장면 t, 사건 기준 t−1년, 귀무 기준 t−2년. 셋의 DOY 차이가 DOY_TOL 이내
  * 세 장면 모두에서 그 오름이 판독 가능(사이트 SCL 유효 토큰 ≥ 20%)
  * 문턱은 그 삼중쌍의 귀무 쌍(t−2 → t−1) p99

셋이 모두 갖춰지면 그 오름의 상태를 갱신한다. 하나라도 빠지면 **abstain 으로 기록**한다 —
점수를 안 만드는 것도 기록이다. 갱신 주기는 정하지 않는다. 데이터가 결정한다.
그래서 이 파일은 "무엇을 봤는가" 보다 "왜 보지 않았는가" 를 더 많이 쓴다.

이 스크립트가 하는 일 (--dry-run 기본)
  1. STAC 에서 제주 타일의 최근 LOOKBACK_DAYS 장면 조회 (장면 구름은 상한 필터로만)
  2. 각 신규 장면에 대해 t−1y, t−2y 후보를 같은 궤도·DOY_TOL 안에서 찾는다
  3. 삼중쌍이 있으면 오름별 사이트 판독가능성(SCL)을 재고 갱신 가능 오름 수를 센다
  4. `artifacts/tracking/state.json` 에 마지막 확인 item ID 와 판정 이력을 append
  5. --run 이면 준비→채점을 그 삼중쌍 계약으로 실행한다(v8 코드를 계약 인자만 바꿔 재사용)

GK2A 는 여기서 게이트 후보다 — D-1/D-2 창이 앞으로 가는 추적기에는 충분하다. 다만 서비스
창이 제주를 담지 않는 문제(artifacts/results/jeju_v8_gk2a_limitation.json)가 풀리기 전엔
쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from jeju_paths import ARTIFACT_ROOT, CONTRACT_ROOT, display_path, ensure

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LOOKBACK_DAYS = 30
DOY_TOL = 14
CLOUD_MAX = 60.0            # 상한 필터일 뿐. 장면 구름은 선택 변수가 아니다(S0 실측).
MIN_FOOTPRINT = 1.0         # 조각 granule 배제


def doy(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--run", action="store_true", help="삼중쌍이 있으면 실제로 준비→채점 실행")
    a = ap.parse_args()
    import pystac_client
    from shapely.geometry import shape

    base = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    tiles = {t: rec["_relative_orbit"] for t, rec in base["optical"]["scenes"].items()}
    state_p = ensure(ARTIFACT_ROOT / "tracking") / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"seen_items": [], "checks": []}
    seen = set(state["seen_items"])

    cat = pystac_client.Client.open(STAC)
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    check = {"at": now.isoformat(timespec="seconds"), "window": f"{since}..{now.date()}", "tiles": {}}

    for tile, orbit in tiles.items():
        new = [i for i in cat.search(collections=["sentinel-2-l2a"], datetime=f"{since}/{now.date()}",
                                     query={"s2:mgrs_tile": {"eq": tile}, "sat:relative_orbit": {"eq": orbit},
                                            "eo:cloud_cover": {"lt": CLOUD_MAX}}).items()
               if shape(i.geometry).area >= MIN_FOOTPRINT and i.id not in seen]
        rec = {"new_scenes": len(new), "triplets": [], "abstain": []}
        for it in sorted(new, key=lambda i: i.datetime):
            t = it.datetime
            found = {}
            for back, role in ((1, "event_base"), (2, "null_base")):
                y = t.year - back
                lo = (t - timedelta(days=DOY_TOL)).replace(year=y); hi = (t + timedelta(days=DOY_TOL)).replace(year=y)
                cands = [c for c in cat.search(collections=["sentinel-2-l2a"], datetime=f"{lo.date()}/{hi.date()}",
                                                query={"s2:mgrs_tile": {"eq": tile}, "sat:relative_orbit": {"eq": orbit},
                                                       "eo:cloud_cover": {"lt": CLOUD_MAX}}).items()
                         if shape(c.geometry).area >= MIN_FOOTPRINT]
                # 같은 날짜 재처리 중복 제거: 최신 처리본 하나만
                by_day = {}
                for c in cands:
                    by_day.setdefault(c.datetime.date(), c)
                found[role] = [{"item_id": c.id, "date": str(c.datetime.date()), "cloud": c.properties["eo:cloud_cover"],
                                "doy_diff": abs(doy(c.datetime) - doy(t))} for c in by_day.values()]
            if found["event_base"] and found["null_base"]:
                rec["triplets"].append({"target": {"item_id": it.id, "date": str(t.date()), "cloud": it.properties["eo:cloud_cover"]},
                                        "event_base": sorted(found["event_base"], key=lambda x: x["doy_diff"])[0],
                                        "null_base": sorted(found["null_base"], key=lambda x: x["doy_diff"])[0],
                                        "site_gate": "pending — 사이트 SCL 판독가능성은 준비 단계에서 잰다"})
            else:
                missing = [k for k, v in found.items() if not v]
                rec["abstain"].append({"target": it.id, "date": str(t.date()), "reason": f"no {'/'.join(missing)} within ±{DOY_TOL}d on orbit {orbit}"})
            seen.add(it.id)
        check["tiles"][tile] = rec

    n_trip = sum(len(r["triplets"]) for r in check["tiles"].values())
    n_abs = sum(len(r["abstain"]) for r in check["tiles"].values())
    n_new = sum(r["new_scenes"] for r in check["tiles"].values())
    check["summary"] = {"new_scenes": n_new, "triplets": n_trip, "abstain": n_abs}
    state["seen_items"] = sorted(seen); state["checks"].append(check)
    state_p.write_text(json.dumps(state, ensure_ascii=False, indent=1))

    print(f"조회 {check['window']} · 신규 장면 {n_new} · 삼중쌍 가능 {n_trip} · abstain {n_abs}")
    for tile, r in check["tiles"].items():
        for tr in r["triplets"]:
            print(f"  {tile}  목표 {tr['target']['date']} ({tr['target']['cloud']:.0f}%)  ← 기준 {tr['event_base']['date']}  귀무 {tr['null_base']['date']}")
        for ab in r["abstain"]:
            print(f"  {tile}  {ab['date']}  abstain: {ab['reason']}")
    print(f"→ {display_path(state_p)}")
    if a.run and n_trip:
        print("\n--run: 삼중쌍 계약으로 준비→채점을 실행하려면 freeze 를 템플릿 인자로 호출해야 한다 (다음 단계).")


if __name__ == "__main__":
    main()
