#!/usr/bin/env python3
"""M59 독립 재검증 — read_confirmatory_region.py와 **다른 코드 경로**로 다시 계산한다.

같은 스크립트를 두 번 돌리는 것은 검증이 아니다. 여기서는:
  - 지표를 다른 방식으로 구현(누적합 대신 리스트 평균, 조건 순서 변경)
  - 봉인 계약과 sample ID 대조
  - **thrissur fold의 val이 chimanimani**(다회 노출 개발 지역)라는 구조적 사실 확인
  - 지역 특성 비교(양성 비율 등)로 "쉬운 지역이라 이긴 것" 가설 점검
"""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np

E = pathlib.Path("evidence/confirmatory/holdout_thrissur")
OUT = pathlib.Path("evidence/confirmatory/holdout_thrissur/independent_verification.json")
ARMS = {"reuse": "P4", "raw_strong": "P2", "raw_efficient": "P3"}


def rows(arm, seed, split="test"):
    p = E / f"{arm}_seed{seed}_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def macro_alt(rs):
    """read_ 스크립트와 다른 구현: 리스트를 먼저 만들고 조건을 나중에 건다."""
    out = []
    for r in rs:
        den = r["tp"] + r["fp"] + r["fn"]
        if den == 0:
            continue
        if r["mask_positive_pixels"] <= 0:
            continue
        out.append(r["tp"] / den)
    return float(sum(out) / len(out)) if out else float("nan")


def main():
    res = {"schema": "thrissur-independent-verification-v1"}

    # ── 1. 주지표 재계산 (다른 구현) ──
    tab = {}
    for role, arm in ARMS.items():
        vals = [macro_alt(rows(arm, s)) for s in (1, 2, 3)]
        tab[role] = {"per_seed": [round(v, 6) for v in vals],
                     "mean": round(float(np.mean(vals)), 6)}
    res["recomputed_primary"] = tab
    res["matches_read_summary"] = {
        "reuse_mean": tab["reuse"]["mean"], "expected": 0.358835,
        "raw_strong_mean": tab["raw_strong"]["mean"], "expected_rs": 0.231513,
        "agree": abs(tab["reuse"]["mean"] - 0.358835) < 1e-6
                 and abs(tab["raw_strong"]["mean"] - 0.231513) < 1e-6}

    # ── 2. 봉인 계약 대조 ──
    ids = sorted(r["sample_id"] for r in rows("P4", 1))
    res["sample_id_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    res["n_test_tiles"] = len(ids)
    res["all_nine_same_ids"] = len({
        hashlib.sha256("\n".join(sorted(r["sample_id"] for r in rows(a, s))).encode()).hexdigest()
        for a in ARMS.values() for s in (1, 2, 3)}) == 1

    # ── 3. 구조적 사실: val이 chimanimani ──
    res["fold_structure"] = {
        "test_region": "thrissur",
        "val_region": "chimanimani",
        "concern": ("thrissur의 epoch 선택에 **다회 노출된 개발 지역(chimanimani)**이 "
                    "val로 쓰였다. test set 자체는 미열람이지만, 선택 신호가 "
                    "우리가 깊이 들여다본 지역에서 온다"),
        "mitigation": "두 arm이 같은 val을 쓰므로 대칭적이다. 다만 recipe(arm 구성·"
                      "decoder 크기)를 chimanimani test 성능을 보고 골랐다는 점에서 "
                      "간접 오염이 남는다",
    }
    val = {}
    for role, arm in ARMS.items():
        try:
            v = [macro_alt(rows(arm, s, "val")) for s in (1, 2, 3)]
            val[role] = [round(x, 6) for x in v]
        except FileNotFoundError:
            val[role] = None
    res["val_chimanimani_primary"] = val

    # ── 4. 지역 난이도 비교 ──
    r0 = rows("P4", 1)
    npos = sum(1 for r in r0 if r["mask_positive_pixels"] > 0)
    pos_px = [r["mask_positive_pixels"] for r in r0 if r["mask_positive_pixels"] > 0]
    res["region_difficulty"] = {
        "thrissur": {"n_tiles": len(r0), "n_positive_tiles": npos,
                     "positive_tile_frac": round(npos / len(r0), 4),
                     "median_positive_px": int(np.median(pos_px)),
                     "mean_positive_px": round(float(np.mean(pos_px)), 1)},
        "chimanimani_reference": {"n_tiles": 1133, "n_positive_tiles": 423,
                                  "positive_tile_frac": 0.3734,
                                  "median_positive_px": 207},
        "note": ("양성 타일 비율이 크게 다르면 주지표(양성 macro)의 절대값은 "
                 "비교 불가다. arm 간 격차는 같은 지역 안에서만 비교한다"),
    }

    # ── 5. 격차가 특정 타일 소수에 몰려 있는가 ──
    a1 = {r["sample_id"]: r for r in rows("P4", 1)}
    b1 = {r["sample_id"]: r for r in rows("P2", 1)}
    diffs = []
    for s in ids:
        da = a1[s]["tp"] + a1[s]["fp"] + a1[s]["fn"]
        db = b1[s]["tp"] + b1[s]["fp"] + b1[s]["fn"]
        if a1[s]["mask_positive_pixels"] > 0 and da > 0 and db > 0:
            diffs.append(a1[s]["tp"] / da - b1[s]["tp"] / db)
    d = np.array(diffs)
    res["gap_distribution_seed1"] = {
        "n_positive_tiles": len(d),
        "reuse_wins_tiles": int((d > 0).sum()),
        "reuse_wins_frac": round(float((d > 0).mean()), 4),
        "median_gap": round(float(np.median(d)), 6),
        "top10pct_share_of_total_gap": round(
            float(np.sort(d)[::-1][:max(1, len(d) // 10)].sum() / d.sum()), 4)
        if d.sum() != 0 else None,
        "note": "소수 타일이 격차를 만들면 top10pct 비중이 1에 가깝다",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
