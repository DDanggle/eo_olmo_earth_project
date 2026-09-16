#!/usr/bin/env python3
"""S2 — 관측가능성 게이트. **점수를 내기 전에 무엇을 볼 수 있는지 먼저 정한다.**

왜 이 단계가 따로 있는가
------------------------
이전 라운드에서 오름 후보 8곳이 8곳 모두 구름·해무로 기각됐다. 원인은 두 가지였고 둘 다
여기서 다룬다.

1. **장면 선택 실패.** 장면 구름으로 고르면 맑은 바다와 구름 쓴 산을 고른다. 이건 S0 계약
   단계에서 이미 고쳤다(오름 자리의 SCL 로 고른다).
2. **사이트별 게이트 부재.** 장면이 맑다는 것과 *이 오름이* 판독된다는 것은 다른 명제다.
   이 파일이 그 둘째를 맡는다.

토큰 규칙은 네팔에서 그대로 가져온다 — patch_size=4 이므로 40 m 토큰이고, 창 256 px 는
64×64 토큰이 된다. 토큰 안 16 픽셀 중 절반 넘게 불량이면 그 토큰은 무효, 유효 토큰이
20% 미만이면 그 해 그 오름은 **unobservable** 이다.

가장 중요한 규칙
----------------
**관측 불가는 변화 없음이 아니다.** 여기서 unobservable 로 판정된 곳은 분모에서 빠지지
않는다. 점수를 만들지 않고 `abstain` 으로 남긴다. 지도도 그렇게 그린다.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from jeju_paths import CACHE_ROOT, CONTRACT_ROOT, ensure, display_path, ARTIFACT_ROOT

YEARS = ["2023", "2024", "2025", "2026"]          # 기본(v8). 계약이 있으면 contract_years() 가 덮어쓴다.

# 단일 정의. 준비기와 동결기가 각자 상수를 들고 있다가 갈라진 적이 있어(하나는 11 눈을
# 포함했다) 여기서 raw SCL 로부터 다시 계산한다. 저장된 `clear` 배열은 쓰지 않는다.
# 눈(11)은 제외한다 — 8~9월 제주에 눈은 없고, 그 라벨은 밝은 표면의 오분류다. 포함하면
# 관측가능성이 부풀고 그건 안전하지 않은 방향이다.
SCL_CLEAR = (4, 5, 6, 7)

PATCH = 4                       # 40 m 토큰 (10 m × 4)
TOKEN_BAD_PIXEL_FRAC = 0.5      # 토큰 안 불량 픽셀이 이보다 많으면 토큰 무효
MIN_VALID_TOKEN_FRAC = 0.2      # 유효 토큰이 이보다 적으면 그 해는 관측 불가

# 쌍 정의는 계약을 따른다. 사건은 2025→2026, 시간 귀무는 2024→2025 와 2023→2024.
PAIRS = {"event": ("2025", "2026"),
         "null_temporal_primary": ("2024", "2025"),
         "null_temporal_secondary": ("2023", "2024")}


def contract_years_pairs(contract: dict) -> tuple[list[str], dict]:
    """계약에서 연도와 쌍을 읽는다. v8x 처럼 연도가 늘어나면 추가 귀무(null_YYYY_YYYY)가 함께 온다."""
    years = [str(y) for y in contract["optical"]["years"]]
    pairs = {k: (str(v["from"]), str(v["to"])) for k, v in contract["pairs"].items() if isinstance(v, dict) and "from" in v}
    return years, pairs


def token_valid_fraction(scl: np.ndarray) -> float:
    """이 해 이 창에서 유효한 40 m 토큰의 비율."""
    h, w = scl.shape
    ht, wt = h // PATCH, w // PATCH
    good = np.isin(scl[: ht * PATCH, : wt * PATCH], SCL_CLEAR)
    per_token = good.reshape(ht, PATCH, wt, PATCH).mean(axis=(1, 3))
    return float((per_token >= 1.0 - TOKEN_BAD_PIXEL_FRAC).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    a = ap.parse_args()

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    global YEARS, PAIRS
    YEARS, PAIRS = contract_years_pairs(contract)
    assign = contract["optical"]["frame_a_assignment"]
    cache = CACHE_ROOT / f"{a.contract}/prepare"

    sites, missing = {}, []
    for oid in sorted(assign):
        f = cache / f"{oid}.npz"
        if not f.exists():
            missing.append(oid); continue
        d = np.load(f, allow_pickle=True)
        scl = d["scl"]
        frac = {y: token_valid_fraction(scl[i]) for i, y in enumerate(YEARS)}
        obs = {y: bool(v >= MIN_VALID_TOKEN_FRAC) for y, v in frac.items()}
        pairs = {k: bool(obs[b] and obs[t]) for k, (b, t) in PAIRS.items()}
        sites[oid] = {
            "tile": str(d["tile"]),
            "valid_token_fraction": {y: round(v, 4) for y, v in frac.items()},
            "observable_by_year": obs,
            "observable_by_pair": pairs,
            "verdict": "scorable" if pairs["event"] else "abstain",
            "abstain_reason": None if pairs["event"] else
                              "event pair unobservable: " + ", ".join(
                                  y for y in PAIRS["event"] if not obs[y]),
        }

    n = len(sites)
    summary = {
        "frame_a_total": len(assign),
        "cubes_present": n,
        "cubes_missing": missing,
        "observable_by_year": {y: sum(s["observable_by_year"][y] for s in sites.values())
                               for y in YEARS},
        "observable_by_pair": {k: sum(s["observable_by_pair"][k] for s in sites.values())
                               for k in PAIRS},
        "scorable": sum(s["verdict"] == "scorable" for s in sites.values()),
        "abstain": sum(s["verdict"] == "abstain" for s in sites.values()),
        "event_and_primary_null": sum(
            s["observable_by_pair"]["event"] and s["observable_by_pair"]["null_temporal_primary"]
            for s in sites.values()),
    }

    out = {
        "schema": "jeju-v8-observability-v1",
        "contract": a.contract,
        "contract_sha256": contract.get("_self_sha256"),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "rule": {
            "scl_clear_classes": list(SCL_CLEAR),
            "scl_clear_note": "눈(11) 제외 — 8~9월 제주에 눈은 없고 그 라벨은 밝은 표면의 오분류다",
            "token_ground_m": 40, "patch_size": PATCH,
            "token_bad_pixel_frac": TOKEN_BAD_PIXEL_FRAC,
            "min_valid_token_frac": MIN_VALID_TOKEN_FRAC,
        },
        "claims": {
            "denominator": "프레임 A 243곳. abstain 은 분모에서 빼지 않는다.",
            "forbidden": ["관측 불가를 '변화 없음' 으로 표시", "abstain 한 곳에 점수 부여"],
        },
        "summary": summary,
        "sites": sites,
    }
    path = ensure(ARTIFACT_ROOT / "results") / f"{a.contract}_observability.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"프레임 A {summary['frame_a_total']}곳 · 큐브 {n}곳"
          + (f" · 누락 {len(missing)}곳" if missing else ""))
    print("\n연도별 관측 가능 (유효 토큰 20% 이상)")
    for y in YEARS:
        print(f"   {y}   {summary['observable_by_year'][y]:3d} / {n}")
    print("\n쌍 관측 가능")
    for k in PAIRS:
        print(f"   {k:24s} {summary['observable_by_pair'][k]:3d} / {n}")
    print(f"\n채점 가능(scorable) {summary['scorable']}  ·  보류(abstain) {summary['abstain']}")
    print(f"사건+주 귀무 둘 다 관측 가능 {summary['event_and_primary_null']}")
    print(f"\n→ {display_path(path)}")


if __name__ == "__main__":
    main()
