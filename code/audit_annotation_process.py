#!/usr/bin/env python3
"""G-A — annotation-process 감사. 지역 차이와 라벨 생성 차이를 분리하기 위한 측정.

문제 (R2): leave-one-region-out으로 성능이 떨어질 때, 그것이
  (a) 지형·기후가 달라서(domain shift)   인지
  (b) 라벨을 다르게 그려서(annotation shift) 인지
분리되지 않으면 transfer 주장을 할 수 없다.

Sen12Landslides inventory는 이 교락을 눈으로 볼 수 있게 해준다. `location` 16개와
`author` 5개가 거의 1:1로 붙어 있다 (Italy 47,522 = Ferrario 47,522).
따라서 leave-one-region-out은 부분적으로 leave-one-annotator-out이다.

여기서 측정하는 annotation descriptor (전부 inventory 속성에서 직접 나온다):
  A1 폴리곤 개수
  A2 면적 분포 — min / p1(MMU 대리) / median / p99 / max, log10 면적 히스토그램
  A3 저자(라벨 생성 주체) 구성
  A4 event_type(유발요인) 구성
  A5 날짜 3종(event/pre/post)의 존재율과 event_conf 분포
  A6 type(현상) 구성

판정 (사전 등록):
  - MMU 비 = max(p1) / min(p1) 를 지역쌍에 대해 계산. **10배 이상**이면 그 쌍은
    직접 비교 불가로 표시한다.
  - 한 지역의 폴리곤 90% 이상이 단일 저자에서 나오면 `author-confounded`로 표시한다.
  - 위 둘 중 하나라도 걸리면 **원자료(raw) 결과와 조화(harmonized) 결과를 모두 보고**한다.
    제외가 아니라 조화가 기본 조치다 (§harmonization).

harmonization: 모든 지역에 공통 면적 하한(전 지역 p1의 최댓값)을 적용해 작은 폴리곤을
버린 뒤 다시 평가한다. 이렇게 하면 "작게 그리는 저자" 효과가 제거된다.
"""
from __future__ import annotations

import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

DBF = Path("/home/work/data/olmoearth/sen12landslides/inventories/inventories.dbf")
OUT = Path("/home/work/data/olmoearth/sen12landslides/audit")
MMU_RATIO_LIMIT = 10.0        # 지역쌍 MMU 비가 이 값 이상이면 직접 비교 불가
AUTHOR_DOMINANCE = 0.90       # 한 저자가 이 비율 이상이면 author-confounded


def read_dbf(path: Path) -> list[dict]:
    raw = path.read_bytes()
    n_rec, hdr_len, rec_len = struct.unpack("<IHH", raw[4:12])
    fields, off = [], 32
    while raw[off] != 0x0D:
        name = raw[off:off + 11].split(b"\x00")[0].decode("latin1")
        fields.append((name, chr(raw[off + 11]), raw[off + 16]))
        off += 32
    rows = []
    for i in range(n_rec):
        rec = raw[hdr_len + i * rec_len: hdr_len + (i + 1) * rec_len]
        if len(rec) < rec_len:
            break
        pos, d = 1, {}
        for name, _t, flen in fields:
            d[name] = rec[pos:pos + flen].decode("latin1").strip()
            pos += flen
        rows.append(d)
    return rows


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_dbf(DBF)

    by_loc = defaultdict(list)
    for r in rows:
        by_loc[r.get("location", "")].append(r)

    per_region = {}
    for loc, rs in sorted(by_loc.items(), key=lambda kv: -len(kv[1])):
        areas = sorted(float(r["area"]) for r in rs
                       if r.get("area") and r["area"].replace(".", "", 1).isdigit())
        authors = Counter(r.get("author", "") for r in rs)
        top_author, top_n = (authors.most_common(1) or [("", 0)])[0]
        confs = sorted(float(r["event_conf"]) for r in rs
                       if r.get("event_conf") and r["event_conf"].replace(".", "", 1).isdigit())
        per_region[loc] = {
            "A1_polygons": len(rs),
            "A2_area_m2": {
                "min": areas[0] if areas else None,
                "p1_mmu": pct(areas, 0.01),
                "median": pct(areas, 0.50),
                "p99": pct(areas, 0.99),
                "max": areas[-1] if areas else None,
                "log10_hist": dict(Counter(
                    int(math.floor(math.log10(a))) for a in areas if a > 0).most_common()),
            },
            "A3_authors": dict(authors.most_common()),
            "A3_top_author_share": round(top_n / len(rs), 4) if rs else None,
            "A4_event_type": dict(Counter(r.get("event_type", "") for r in rs).most_common()),
            "A5_dates_present": {
                k: round(sum(1 for r in rs if r.get(k)) / len(rs), 4)
                for k in ("event_dt2", "pre_dt2", "post_dt2")},
            "A5_event_conf": {"median": pct(confs, 0.50), "p10": pct(confs, 0.10),
                              "frac_eq_1": round(sum(1 for c in confs if c >= 1.0)
                                                 / len(confs), 4) if confs else None},
            "A6_type": dict(Counter(r.get("type", "") for r in rs).most_common()),
        }

    # ---- 판정 ----
    mmus = {loc: v["A2_area_m2"]["p1_mmu"] for loc, v in per_region.items()
            if v["A2_area_m2"]["p1_mmu"] == v["A2_area_m2"]["p1_mmu"]}
    incomparable_pairs = []
    locs = sorted(mmus)
    for i, a in enumerate(locs):
        for b in locs[i + 1:]:
            hi, lo = max(mmus[a], mmus[b]), min(mmus[a], mmus[b])
            if lo > 0 and hi / lo >= MMU_RATIO_LIMIT:
                incomparable_pairs.append({"a": a, "b": b, "mmu_a": mmus[a],
                                           "mmu_b": mmus[b], "ratio": round(hi / lo, 2)})
    author_confounded = [loc for loc, v in per_region.items()
                         if (v["A3_top_author_share"] or 0) >= AUTHOR_DOMINANCE]

    # ---- harmonization: 공통 면적 하한 ----
    floor = max(mmus.values()) if mmus else None
    harmonized = {}
    for loc, rs in by_loc.items():
        kept = [r for r in rs
                if r.get("area") and r["area"].replace(".", "", 1).isdigit()
                and float(r["area"]) >= (floor or 0)]
        harmonized[loc] = {"kept": len(kept), "dropped": len(rs) - len(kept),
                           "keep_rate": round(len(kept) / len(rs), 4) if rs else None}

    result = {
        "schema": "sen12landslides-annotation-audit-v1",
        "source": "inventories.shp (Sen12Landslides, CC BY 4.0)",
        "total_polygons": len(rows),
        "thresholds": {"mmu_ratio_limit": MMU_RATIO_LIMIT,
                       "author_dominance": AUTHOR_DOMINANCE},
        "per_region": per_region,
        "verdict": {
            "region_count": len(per_region),
            "author_confounded_regions": author_confounded,
            "author_confounded_count": len(author_confounded),
            "incomparable_pair_count": len(incomparable_pairs),
            "incomparable_pairs_top": sorted(
                incomparable_pairs, key=lambda d: -d["ratio"])[:15],
            "harmonization_area_floor_m2": floor,
            "harmonized_keep_rate": {k: v["keep_rate"] for k, v in
                                     sorted(harmonized.items(),
                                            key=lambda kv: -kv[1]["kept"])},
        },
    }
    (OUT / "annotation_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    # 콘솔 요약
    print("총 폴리곤 %d, 지역 %d, 저자 %d"
          % (len(rows), len(per_region), len({a for v in per_region.values()
                                              for a in v["A3_authors"]})))
    print("\n%-16s %7s %10s %10s %10s  %-30s %5s" %
          ("region", "polys", "MMU(p1)", "median", "p99", "top author", "share"))
    for loc, v in sorted(per_region.items(), key=lambda kv: -kv[1]["A1_polygons"]):
        a = v["A2_area_m2"]
        top = max(v["A3_authors"], key=v["A3_authors"].get) if v["A3_authors"] else ""
        print("%-16s %7d %10.1f %10.1f %10.1f  %-30s %5.2f" %
              (loc, v["A1_polygons"], a["p1_mmu"] or 0, a["median"] or 0,
               a["p99"] or 0, top[:30], v["A3_top_author_share"] or 0))
    print("\nauthor-confounded 지역 %d/%d" % (len(author_confounded), len(per_region)))
    print("MMU %gx 이상 차이나는 지역쌍 %d" % (MMU_RATIO_LIMIT, len(incomparable_pairs)))
    for p in sorted(incomparable_pairs, key=lambda d: -d["ratio"])[:8]:
        print("   %-16s vs %-16s ratio=%.1f" % (p["a"], p["b"], p["ratio"]))
    print("\nharmonization 면적 하한 = %.1f m2" % (floor or 0))
    print("DONE")


if __name__ == "__main__":
    main()
