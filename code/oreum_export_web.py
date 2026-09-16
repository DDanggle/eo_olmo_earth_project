#!/usr/bin/env python3
"""S5 — 지도가 읽는 자산을 봉인된 결과에서만 만든다.

산출물 (apps/oreum-web/public/data/)
  oreum.geojson            243 점. 점수·판정·순위·연도별 관측가능성·한국 레이어 태그.
  summary.json             분모·채점·보류·문턱·깃발율·계약 해시 — 지도의 모든 숫자는 여기서만 온다.
  frames/<id>_<year>.jpg   연도별 RGB (B04,B03,B02). 네 해 **공통 스트레치** — 프레임마다 따로 늘리면
                           전후 비교가 다른 척도를 비교하게 된다(네팔에서 고친 오류).
  frames/<id>_delta.png    사건 Δz 토큰 맵 + 문턱 초과 표시. 유효하지 않은 토큰은 투명.

규칙
  * abstain 오름은 점수 필드가 null 이다. 0 이 아니다.
  * 20 m(p2) 결과는 `secondary` 아래에만 넣고 색계급에 쓰지 않는다.
  * 색계급 경계는 채점된 오름의 event flag_frac 분위(20/40/60/80%)에서 잡아 summary 에 적는다.
    지도와 범례가 같은 배열을 읽어야 어긋나지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, CONTRACT_ROOT, WEB_DATA_ROOT, display_path, ensure
from oreum_prepare import BANDS, load_oreum

YEARS = ["2023", "2024", "2025", "2026"]


def rgb_common_stretch(cube: np.ndarray) -> list[np.ndarray]:
    """네 해 RGB 를 한 스트레치로. cube: (12, 4, H, W) raw DN."""
    idx = [BANDS.index(b) for b in ("B04", "B03", "B02")]
    rgb = cube[idx]                                   # (3, 4, H, W)
    finite = rgb[np.isfinite(rgb)]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    out = []
    for yi in range(rgb.shape[1]):
        a = np.clip((rgb[:, yi] - lo) / max(hi - lo, 1.0), 0, 1)
        a = np.nan_to_num(a, nan=0.0)
        out.append((np.transpose(a, (1, 2, 0)) ** 0.8 * 255).astype("uint8"))
    return out


def delta_png(d: np.ndarray, v: np.ndarray, thr: float, vmax: float) -> np.ndarray:
    """Δz 토큰 맵 RGBA. 청록→주황→자홍, 문턱 초과는 자홍 테두리 없이 그대로(색계급이 말한다)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("oreum", ["#0fcb8c", "#f09158", "#b8306f"])
    x = np.clip(d / max(vmax, 1e-6), 0, 1)
    rgba = (cmap(x) * 255).astype("uint8")
    rgba[..., 3] = np.where(v, 235, 0)
    return rgba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    from PIL import Image

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    res = ARTIFACT_ROOT / "results"
    p4 = json.loads((res / f"{a.contract}_scan_p4.json").read_text())
    p2_path = res / f"{a.contract}_scan_p2.json"
    p2 = json.loads(p2_path.read_text()) if p2_path.exists() else None
    obs = json.loads((res / f"{a.contract}_observability.json").read_text())
    korea_path = ARTIFACT_ROOT / "external_data/korea_public_v1/jeju_v8_korea_evidence_polygons.json"
    korea = json.loads(korea_path.read_text()) if korea_path.exists() else None

    out = ensure(WEB_DATA_ROOT); frames = ensure(out / "frames")
    oreum = {o["oreum_id"]: o for o in load_oreum(None)}
    thr = p4["threshold"]["value"]

    # 색계급: 채점된 오름의 event flag_frac 분위
    scored_vals = sorted(s["event"]["flag_frac"] for s in p4["sites"].values() if s["verdict"] == "scored")
    breaks = [float(np.percentile(scored_vals, q)) for q in (20, 40, 60, 80)] if scored_vals else [0, 0, 0, 0]

    # 한국 레이어 태그 (점이 폴리곤 안에 있는가)
    tags = {}
    if korea:
        from shapely.geometry import Point, shape
        from shapely.strtree import STRtree
        layers = {}
        for lyr, rec in korea["layers"].items():
            geoms = [shape(f["geometry"]) for f in rec["features"] if f.get("geometry")]
            names = [f["properties"].get("uname") or f["properties"].get("park_name") or rec["name"]
                     for f in rec["features"] if f.get("geometry")]
            layers[lyr] = (STRtree(geoms), geoms, names, rec["name"])
        for oid, o in oreum.items():
            p = Point(o["lon"], o["lat"]); t = {}
            for lyr, (tree, geoms, names, lname) in layers.items():
                hit = [names[i] for i in tree.query(p) if geoms[i].contains(p)]
                if hit:
                    t[lname] = sorted(set(hit))
            tags[oid] = t

    features = []
    vmax = float(np.percentile([s["event"]["median_delta"] for s in p4["sites"].values()
                                if s["event"]["median_delta"] is not None], 99)) * 3 if scored_vals else 1.0
    for oid, o in sorted(oreum.items()):
        s4 = p4["sites"].get(oid); ob = obs["sites"].get(oid, {})
        s2 = p2["sites"].get(oid) if p2 else None
        scored = bool(s4 and s4["verdict"] == "scored")
        props = {
            "oreum_id": oid, "name": o["name"],
            "verdict": "scored" if scored else "abstain",
            "abstain_reason": None if scored else (ob.get("abstain_reason") or "no cube"),
            "rank": s4.get("rank") if s4 else None,
            "event_flag_frac": s4["event"]["flag_frac"] if scored else None,
            "event_n_flag": s4["event"]["n_flag"] if scored else None,
            "null_flag_frac": s4["null_temporal_primary"]["flag_frac"] if s4 else None,
            "persistent_tokens": s4["persistent_tokens"] if s4 else None,
            "event_only_tokens": s4["event_only_tokens"] if s4 else None,
            "valid_token_fraction": ob.get("valid_token_fraction"),
            "observable_by_year": ob.get("observable_by_year"),
            "tile": s4["tile"] if s4 else None,
            "secondary_20m": ({"verdict": s2["verdict"], "event_flag_frac": s2["event"]["flag_frac"],
                               "rank": s2.get("rank")} if s2 else None),
            "korea_tags": tags.get(oid, {}),
        }
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
                         "properties": props})

        if a.no_frames:
            continue
        cube_path = CACHE_ROOT / f"{a.contract}/prepare/{oid}.npz"
        if not cube_path.exists():
            continue
        d = np.load(cube_path, allow_pickle=True)
        for yi, img in enumerate(rgb_common_stretch(d["cube"])):
            Image.fromarray(img).save(frames / f"{oid}_{YEARS[yi]}.jpg", quality=82)
        dl = CACHE_ROOT / f"{a.contract}/scan_p4/{oid}_delta.npz"
        if dl.exists():
            dd = np.load(dl)
            Image.fromarray(delta_png(dd["d_event"], dd["v_event"], thr, vmax), "RGBA").resize((256, 256), Image.NEAREST).save(frames / f"{oid}_delta.png")

    (out / "oreum.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False))

    summary = {
        "contract": a.contract, "contract_sha256": contract.get("_self_sha256"),
        "scan_p4_sha256": hashlib.sha256((res / f"{a.contract}_scan_p4.json").read_bytes()).hexdigest(),
        "frame_a_total": len(oreum),
        "scored": p4["summary"]["scored"], "abstain": len(oreum) - p4["summary"]["scored"],
        "sites_with_any_event_flag": p4["summary"]["sites_with_any_event_flag"],
        "threshold_p99": thr, "flag_rate_pooled": p4["flag_rate_pooled"],
        "pairs": p4["pairs"], "token_ground_m": p4["token_ground_m"],
        "class_breaks": breaks, "class_colors": ["#0fcb8c", "#7fd2b0", "#f5c9a6", "#f09158", "#b8306f"],
        "anchor_dates": {t: {y: contract["optical"]["scenes"][t][y]["datetime"][:10] for y in YEARS}
                         for t in contract["optical"]["scenes"]},
        "secondary_20m": ({"scored": p2["summary"]["scored"], "flag_rate_pooled": p2["flag_rate_pooled"],
                           "threshold_p99": p2["threshold"]["value"]} if p2 else None),
        "korea_layers": ({k: {"name": v["name"], "n": v["n"]} for k, v in korea["layers"].items()} if korea else None),
        "claims": p4["claims"],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"oreum.geojson {len(features)}점 · summary.json · frames {len(list(frames.glob('*')))}개 → {display_path(out)}")
    print(f"채점 {summary['scored']} · 보류 {summary['abstain']} · 색계급 경계 {[round(b*100,2) for b in breaks]}%")


if __name__ == "__main__":
    main()
