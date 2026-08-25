#!/usr/bin/env python3
"""GK2A 응답 320x397 창이 공식 900x900 격자의 어디를 가리키는지 규약을 가림.

연속 매개변수를 적합하지 않음. **유한한 규약 집합**에서 고를 뿐임:
  창 행 원점   북쪽 기준(j = y0+jj)  vs  남쪽 기준(j = 899-(y0+jj))
  창 열 원점   서쪽 기준(i = x0+ii)  vs  동쪽 기준(i = 899-(x0+ii))
  응답 배열    그대로  vs  행 뒤집기  vs  열 뒤집기  vs  둘 다
따라서 24개 조합을 전부 재고 **최고와 차선의 격차**를 함께 보고함.
격차가 크면 규약이 결정된 것이고, 비슷하면 4개 앵커로는 못 가린다는 뜻임.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("GK2A_ROOT", str(Path.home() / "dong/ai_projects/data/gk2a")))
GRID = ROOT / "_grid"
FULL = 900
XD, YD = 320, 397
X0, Y0 = 63, 333


def read_grid(p: Path) -> list[float]:
    lines = p.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[1:]:
        for tok in line.replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def main() -> None:
    lon = read_grid(GRID / "ko2km_lon.txt")
    lat = read_grid(GRID / "ko2km_lat.txt")
    assert len(lon) == FULL * FULL and len(lat) == FULL * FULL, (len(lon), len(lat))

    recs = []
    for line in (ROOT / "_crs/area_anchors.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            if r.get("resultType", "CLD") == "CLD":
                recs.append(r)
    grids = {}
    for r in recs:
        dt = r["dateTime"]
        f = ROOT / dt[0:4] / dt[4:6] / dt[6:8] / f"getGk2acldAll_CLD_{dt[8:12]}.json.gz"
        if f.exists() and dt not in grids:
            it = json.loads(gzip.open(f).read())["response"]["body"]["items"]["item"][0]
            grids[dt] = it["value"].split(",")
    print("앵커 관측 %d건, 고유 지점 %d, 격자 시각 %d"
          % (len(recs), len({r["dong"] for r in recs}), len(grids)))

    results = []
    for row_from in ("north", "south"):
        for col_from in ("west", "east"):
            # 창의 (jj, ii) -> 전체 격자 인덱스
            def full_idx(jj, ii, rf=row_from, cf=col_from):
                j = (Y0 + jj) if rf == "north" else (FULL - 1 - (Y0 + jj))
                i = (X0 + ii) if cf == "west" else (FULL - 1 - (X0 + ii))
                if not (0 <= j < FULL and 0 <= i < FULL):
                    return None
                return j * FULL + i

            ok = all(full_idx(jj, ii) is not None
                     for jj in (0, YD - 1) for ii in (0, XD - 1))
            if not ok:
                continue
            wl = [lon[full_idx(jj, ii)] for jj in range(YD) for ii in range(XD)]
            wa = [lat[full_idx(jj, ii)] for jj in range(YD) for ii in range(XD)]
            for vflip in ("none", "rows", "cols", "both"):
                hits = Counter()
                for r in recs:
                    vals = grids.get(r["dateTime"])
                    if vals is None or len(vals) != XD * YD:
                        continue
                    # 응답 배열을 규약대로 재배치
                    def at(jj, ii, vf=vflip):
                        j = jj if vf in ("none", "cols") else YD - 1 - jj
                        i = ii if vf in ("none", "rows") else XD - 1 - ii
                        return vals[j * XD + i]
                    best_k, best_d = None, float("inf")
                    for k in range(XD * YD):
                        d = (wl[k] - r["lon"]) ** 2 + (wa[k] - r["lat"]) ** 2
                        if d < best_d:
                            best_d, best_k = d, k
                    jj, ii = divmod(best_k, XD)
                    hits["n"] += 1
                    if at(jj, ii) == r["value"]:
                        hits["m"] += 1
                if hits["n"]:
                    results.append({
                        "row_from": row_from, "col_from": col_from, "value_flip": vflip,
                        "n": hits["n"], "match": hits["m"],
                        "rate": round(hits["m"] / hits["n"], 4),
                        "window_lon": [round(min(wl), 4), round(max(wl), 4)],
                        "window_lat": [round(min(wa), 4), round(max(wa), 4)]})

    results.sort(key=lambda d: -d["rate"])
    print("\n%-7s %-6s %-6s %5s %5s  %-22s %-22s" %
          ("row", "col", "flip", "rate", "n", "lon 범위", "lat 범위"))
    for d in results:
        print("%-7s %-6s %-6s %5.3f %5d  %-22s %-22s" %
              (d["row_from"], d["col_from"], d["value_flip"], d["rate"], d["n"],
               str(d["window_lon"]), str(d["window_lat"])))
    if len(results) >= 2:
        print("\n최고 %.4f · 차선 %.4f · 격차 %.4f"
              % (results[0]["rate"], results[1]["rate"],
                 results[0]["rate"] - results[1]["rate"]))
    (GRID / "window_convention.json").write_text(
        json.dumps({"results": results, "anchors": len(recs),
                    "unique_points": len({r["dong"] for r in recs})},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
