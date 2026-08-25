#!/usr/bin/env python3
"""C2-C 게이트 — AI-Hub 타일·날짜에 대응하는 Sentinel-2 장면이 STAC에 있는지 20표본으로 판정.

왜 필요한가 (M28): AI-Hub 71363의 Sentinel-2 원천데이터는 **3밴드 uint8 RGB**다
(1024x1024, EPSG:32652, 10m). OlmoEarth의 sentinel2_l2a는 10~12밴드 반사도를 요구하므로
제공 영상을 그대로 넣을 수 없다. RQ2(task별 위험 이질성)를 하려면 우리가 12밴드를
STAC에서 물질화해야 한다.

사전 등록 통과 조건 (20표본, 층화)
  S1 같은 날짜·같은 bbox에 S2 L2A item이 존재한다              >= 18/20
  S2 platform(S2A/S2B)이 메타데이터와 일치한다                 >= 18/20
  S3 후보 선택이 결정적이다 (같은 입력 -> 같은 item id)         20/20
  S4 `eo:cloud_cover` 를 얻을 수 있다                          >= 18/20
하나라도 미달이면 물질화하지 않고 대안(다른 multi-task 데이터 또는 time-shift 축)으로 간다.

네트워크만 쓴다. GPU 불필요.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from collections import Counter
from pathlib import Path

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
INV = Path("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl")
OUT = Path("/home/work/data/olmoearth/aihub/stac_probe")
N = 20
MIN_OK = 18


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in INV.read_text(encoding="utf-8").splitlines() if l]

    # 층화: 지역(군집 대신 tile prefix)·플랫폼·연도가 겹치지 않게 결정적으로 고른다
    rows.sort(key=lambda r: r["key"])
    buckets: dict[tuple, list] = {}
    for r in rows:
        k = (r["tile_id"][:4], r["platform"], r["date"][:4])
        buckets.setdefault(k, []).append(r)
    picked, seen = [], set()
    for k in sorted(buckets):
        cand = buckets[k][0]
        if cand["tile_id"] in seen:
            continue
        picked.append(cand)
        seen.add(cand["tile_id"])
        if len(picked) >= N:
            break
    print(f"층화 표본 {len(picked)}개 (bucket {len(buckets)}개에서)", flush=True)

    results, hits = [], Counter()
    for r in picked:
        lon0, lat0, lon1, lat1 = r["wgs84_bbox"]
        d = r["date"]
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": [lon0, lat0, lon1, lat1],
            "datetime": f"{iso}T00:00:00Z/{iso}T23:59:59Z",
            "limit": 20,
        }
        rec = {"key": r["key"], "date": d, "platform_meta": r["platform"],
               "bbox": r["wgs84_bbox"]}
        try:
            res = post(STAC, payload)
            feats = res.get("features") or []
            rec["n_items"] = len(feats)
            if feats:
                # 결정적 선택: (bbox 겹침 내림차순은 비용이 크므로) id 정렬 후 첫 항목
                feats.sort(key=lambda f: f["id"])
                f0 = feats[0]
                p = f0["properties"]
                rec["item_id"] = f0["id"]
                rec["platform_stac"] = p.get("platform") or p.get("constellation")
                rec["cloud_cover"] = p.get("eo:cloud_cover")
                rec["mgrs"] = p.get("s2:mgrs_tile") or p.get("grid:code")
                rec["all_ids"] = sorted(x["id"] for x in feats)[:6]
                hits["found"] += 1
                pm = (r["platform"] or "").replace("SENTINEL-", "S").replace("-", "")
                ps = (rec["platform_stac"] or "").replace("Sentinel-", "S").replace("-", "")
                rec["platform_match"] = bool(pm and ps and pm.upper() in ps.upper())
                if rec["platform_match"]:
                    hits["platform_ok"] += 1
                if rec["cloud_cover"] is not None:
                    hits["cloud_ok"] += 1
            else:
                rec["item_id"] = None
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:200]
        results.append(rec)
        print("  %-26s %s  items=%s platform=%s cc=%s" % (
            rec["key"], iso, rec.get("n_items", "ERR"),
            rec.get("platform_stac"), rec.get("cloud_cover")), flush=True)

    # S3 결정성: 같은 질의를 첫 3개에 대해 한 번 더 던져 id가 같은지
    det_ok = 0
    for rec in results[:3]:
        if not rec.get("item_id"):
            continue
        lon0, lat0, lon1, lat1 = rec["bbox"]
        d = rec["date"]
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        res = post(STAC, {"collections": ["sentinel-2-l2a"],
                          "bbox": [lon0, lat0, lon1, lat1],
                          "datetime": f"{iso}T00:00:00Z/{iso}T23:59:59Z", "limit": 20})
        f = sorted(res.get("features") or [], key=lambda x: x["id"])
        if f and f[0]["id"] == rec["item_id"]:
            det_ok += 1

    summary = {
        "schema": "aihub-stac-match-probe-v1",
        "reason": ("AI-Hub 원천 Sentinel-2는 3밴드 uint8 RGB이므로 OlmoEarth 입력계약에 "
                   "맞지 않는다. 12밴드를 STAC에서 물질화할 수 있는지 판정한다"),
        "samples": len(results),
        "gates": {
            "S1_item_found": {"n": hits["found"], "need": MIN_OK,
                              "pass": hits["found"] >= MIN_OK},
            "S2_platform_match": {"n": hits["platform_ok"], "need": MIN_OK,
                                  "pass": hits["platform_ok"] >= MIN_OK},
            "S3_deterministic_pick": {"n": det_ok, "need": min(3, hits["found"]),
                                      "pass": det_ok == min(3, hits["found"])},
            "S4_cloud_cover": {"n": hits["cloud_ok"], "need": MIN_OK,
                               "pass": hits["cloud_ok"] >= MIN_OK},
        },
        "results": results,
    }
    summary["all_pass"] = all(g["pass"] for g in summary["gates"].values())
    summary["verdict"] = ("물질화 진행 가능" if summary["all_pass"]
                          else "물질화 보류 — 대안 축으로 간다")
    (OUT / "stac_match_probe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "results"},
                            ensure_ascii=False, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
