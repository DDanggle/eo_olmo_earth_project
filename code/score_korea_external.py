#!/usr/bin/env python3
"""S6 — 한국 근거 결합. **창이 아니라 필지 단위**로, permutation 으로 채점한다.

질문은 하나다: 2026년에 개발행위허가가 난 필지가 덮는 토큰들은, 같은 오름 창 안의
다른 유효 토큰보다 사건 쌍(2025→2026)에서 문턱을 더 자주 넘는가?

왜 이렇게 하는가
----------------
* 허가 필지 중앙값이 1,216 m² 로 40 m 토큰(1,600 m²)보다 **작다.** 그래서 40 m 에서는
  필지가 최소 MIN_TOKENS 개 토큰을 덮는 경우만 주검정에 넣고, 20 m(p2) 결과를 부검정으로
  같이 낸다. 라벨이 해상도보다 작은 문제는 네팔(라벨이 너무 컸다)의 반대쪽이다.
* 대조는 **같은 창 안의** 유효 토큰이다. 창을 넘어 비교하면 타일·구름·지형이 섞인다.
* 허가는 **행정적 상관**이다. 허가가 났다고 공사가 그 해에 있었다는 뜻이 아니고, 깃발이
  섰다고 허가 때문이라는 뜻도 아니다. 원인·시설 종류 주장은 계약이 금지한다.
* 검정력: 쓸 수 있는 필지가 MIN_PARCELS 미만이면 p 값을 내지 않고 "검정력 부족" 으로 적는다.

층화 태그(자연환경보전지역·문화재보호구역·국립공원)는 검정이 아니라 **분모 나누기**다 —
보전지역 안/밖의 깃발율을 나란히 보여줄 뿐 우열을 말하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, CONTRACT_ROOT, display_path, ensure

MIN_TOKENS = 4          # 필지가 덮는 토큰 수 바닥 (주검정)
MIN_PARCELS = 20        # 이 미만이면 p 값 없이 보고
N_PERM = 200_000
RNG = np.random.default_rng(20260916)


def parcel_token_mask(geom, origin_xy, crs, patch, n_tok):
    """필지 polygon(EPSG:4326) 이 덮는 토큰(중심점 포함) 마스크."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as sh_transform
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    g = sh_transform(lambda x, y, z=None: tf.transform(x, y), shape(geom))
    x0, y0 = origin_xy
    step = patch * 10.0
    minx, miny, maxx, maxy = g.bounds
    mask = np.zeros((n_tok, n_tok), dtype=bool)
    # 창 원점은 좌하단(y0 = min y). 래스터 행 0 은 위(북)쪽이므로 행 = n_tok-1-j.
    j0 = max(int((miny - y0) // step), 0); j1 = min(int((maxy - y0) // step) + 1, n_tok)
    i0 = max(int((minx - x0) // step), 0); i1 = min(int((maxx - x0) // step) + 1, n_tok)
    if j0 >= j1 or i0 >= i1:
        return mask
    from shapely.geometry import Point
    for j in range(j0, j1):
        for i in range(i0, i1):
            cx, cy = x0 + (i + 0.5) * step, y0 + (j + 0.5) * step
            if g.contains(Point(cx, cy)):
                mask[n_tok - 1 - j, i] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--patch", type=int, default=4, choices=(2, 4))
    a = ap.parse_args()
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    scan = json.loads((ARTIFACT_ROOT / f"results/{a.contract}_scan_p{a.patch}.json").read_text())
    thr = scan["threshold"]["value"]
    korea = json.loads((ARTIFACT_ROOT / "external_data/korea_public_v1/jeju_v8_korea_evidence_polygons.json").read_text())
    n_tok = 256 // a.patch
    prep = CACHE_ROOT / f"{a.contract}/prepare"; sc = CACHE_ROOT / f"{a.contract}/scan_p{a.patch}"

    # 창 footprint (EPSG:4326 근사: 원점+2560 m 를 역변환)
    from pyproj import Transformer
    windows = {}
    for oid, rec in scan["sites"].items():
        if rec["verdict"] != "scored":
            continue
        d = np.load(prep / f"{oid}.npz", allow_pickle=True)
        x0, y0 = map(float, d["origin_xy"]); crs = str(d["crs"])
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = inv.transform([x0, x0 + 2560, x0 + 2560, x0], [y0, y0, y0 + 2560, y0 + 2560])
        from shapely.geometry import Polygon
        windows[oid] = {"poly": Polygon(zip(xs, ys)), "origin": (x0, y0), "crs": crs}

    # ---- 허가 필지 → 창 → 토큰 -------------------------------------------------------
    parcels = [p for p in korea["permits_2026"] if p.get("geometry")]
    tree_ids = list(windows); tree = STRtree([windows[k]["poly"] for k in tree_ids])
    rows, too_small, outside = [], 0, 0
    for p in parcels:
        g = shape(p["geometry"]); hits = [tree_ids[i] for i in tree.query(g) if windows[tree_ids[i]]["poly"].intersects(g)]
        if not hits:
            outside += 1; continue
        oid = hits[0]
        w = windows[oid]
        m = parcel_token_mask(p["geometry"], w["origin"], w["crs"], a.patch, n_tok)
        dd = np.load(sc / f"{oid}_delta.npz")
        v = dd["v_event"]; de = dd["d_event"]
        m &= v
        if m.sum() < MIN_TOKENS:
            too_small += 1; continue
        ctrl = v & ~m
        if ctrl.sum() < 10:
            continue
        rows.append({"pnu": p["pnu"], "oreum_id": oid, "area_m2": p["area_m2"], "act": p["act"], "purpose": p["purpose"],
                     "n_tokens": int(m.sum()), "parcel_flag_frac": float((de[m] > thr).mean()),
                     "window_ctrl_flag_frac": float((de[ctrl] > thr).mean()),
                     "parcel_median_delta": float(np.median(de[m])), "ctrl_median_delta": float(np.median(de[ctrl])),
                     "_m": m, "_ctrl": ctrl, "_de": de})

    # ---- permutation: 필지 토큰 집합의 깃발율 vs 같은 창에서 같은 수의 토큰을 무작위로 -----
    result = {"schema": "jeju-v8-korea-external-v1", "patch_size": a.patch, "token_ground_m": a.patch * 10,
              "threshold_p99": thr, "min_tokens": MIN_TOKENS, "min_parcels": MIN_PARCELS,
              "permits_2026_total": len(korea["permits_2026"]), "with_geometry": len(parcels),
              "outside_any_scored_window": outside, "too_small_for_min_tokens": too_small, "tested": len(rows),
              "claims": {"allowed": ["허가 필지 토큰의 깃발율이 같은 창 대조보다 높은가 (행정적 상관)"],
                         "forbidden": ["허가가 변화의 원인", "시설 종류", "보전지역 안/밖의 우열"]}}
    if len(rows) < MIN_PARCELS:
        result["verdict"] = f"검정력 부족 — 필지 {len(rows)}곳 < {MIN_PARCELS}. p 값을 내지 않는다."
        result["parcels"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    else:
        obs = np.mean([r["parcel_flag_frac"] for r in rows]) - np.mean([r["window_ctrl_flag_frac"] for r in rows])
        # 같은 창의 유효 토큰 N 개 중 깃발 K 개. 필지가 n 개를 덮을 때 무작위 n 개 안의 깃발 수는
        # 초기하분포다. 필지별로 N_PERM 번 뽑아 (필지 깃발율 − 나머지 깃발율) 을 평균한다.
        # 순수 루프(필지 × 20만 × 토큰 셔플)는 수십 분이 걸려 벡터화했다. 분포는 동일하다.
        null = np.zeros(N_PERM)
        for r in rows:
            flags = r["_de"][r["_m"] | r["_ctrl"]] > thr
            N, K, n = int(flags.size), int(flags.sum()), int(r["n_tokens"])
            x = RNG.hypergeometric(K, N - K, n, size=N_PERM)
            null += x / n - (K - x) / (N - n)
        null /= len(rows)
        p = float((null >= obs).mean())
        result.update({"verdict": "tested", "observed_mean_diff": float(obs), "p_one_sided": p, "n_perm": N_PERM,
                       "parcel_mean_flag": float(np.mean([r["parcel_flag_frac"] for r in rows])),
                       "control_mean_flag": float(np.mean([r["window_ctrl_flag_frac"] for r in rows]))})
        result["parcels"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    # ---- 층화: 보전지역 안/밖 깃발율 (검정 아님) ---------------------------------------
    strata = {}
    for lyr, rec in korea["layers"].items():
        geoms = [shape(f["geometry"]) for f in rec["features"] if f.get("geometry")]
        t = STRtree(geoms)
        inside, outside_ = [], []
        for oid, s in scan["sites"].items():
            if s["verdict"] != "scored":
                continue
            c = windows[oid]["poly"].centroid
            (inside if any(geoms[i].contains(c) for i in t.query(c)) else outside_).append(s["event"]["flag_frac"])
        strata[rec["name"]] = {"n_inside": len(inside), "n_outside": len(outside_),
                               "mean_flag_inside": float(np.mean(inside)) if inside else None,
                               "mean_flag_outside": float(np.mean(outside_)) if outside_ else None,
                               "note": "분모 나누기다. 우열을 말하지 않는다."}
    result["strata"] = strata
    result["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()

    path = ensure(ARTIFACT_ROOT / "results") / f"{a.contract}_korea_external_p{a.patch}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"허가 2026 {result['permits_2026_total']} · 폴리곤 {result['with_geometry']} · 채점 창 밖 {outside} · "
          f"토큰 {MIN_TOKENS}개 미만 {too_small} · 검정 {len(rows)}")
    print(result["verdict"] if isinstance(result["verdict"], str) and result["verdict"] != "tested"
          else f"필지 깃발율 {result['parcel_mean_flag']*100:.2f}% vs 창 대조 {result['control_mean_flag']*100:.2f}%  "
               f"차이 {result['observed_mean_diff']*100:+.2f}pp  p(one-sided) = {result['p_one_sided']:.4f}")
    for k, v in strata.items():
        print(f"  {k}: 안 {v['n_inside']}곳 {v['mean_flag_inside'] and round(v['mean_flag_inside']*100,2)}%  "
              f"밖 {v['n_outside']}곳 {v['mean_flag_outside'] and round(v['mean_flag_outside']*100,2)}%")
    print(f"→ {display_path(path)}")


if __name__ == "__main__":
    main()
