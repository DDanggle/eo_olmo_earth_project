#!/usr/bin/env python3
"""Run model 용 소형 Area 폴리곤 생성: 제주 오름 폴리곤 중 중앙값에 가까운 N개의 bbox + 여유. 비용(compute units)을
작게 유지하려는 목적 — 전체 제주(≈1,850 km²)는 예측 비용을 예측할 수 없다."""
import json, statistics, sys
from pathlib import Path
src = Path("artifacts/studio_audit/jeju_import/jeju_oreum_polys_studio.geojson"); n = int(sys.argv[1]) if len(sys.argv) > 1 else 15; pad = 0.004  # ≈400 m
feats = json.load(open(src))["features"]
def centroid(f):
    ring = f["geometry"]["coordinates"][0]; xs = [p[0] for p in ring]; ys = [p[1] for p in ring]; return (sum(xs)/len(xs), sum(ys)/len(ys))
cs = [(centroid(f), f["properties"].get("oreum_id")) for f in feats]
mx = statistics.median(c[0][0] for c in cs); my = statistics.median(c[0][1] for c in cs)
near = sorted(cs, key=lambda c: (c[0][0]-mx)**2 + (c[0][1]-my)**2)[:n]
xs = [c[0][0] for c in near]; ys = [c[0][1] for c in near]
x0, x1, y0, y1 = min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad
km = (x1-x0)*111*0.87 * (y1-y0)*111  # cos(33.4°)≈0.83~0.87
gj = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":f"jeju_oreum_core_{n}"},"geometry":{"type":"Polygon","coordinates":[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}}]}
out = Path("artifacts/studio_audit/jeju_import")/f"jeju_area_core_{n}.geojson"; out.write_text(json.dumps(gj)); 
print(f"{out}  bbox lon {x0:.4f}–{x1:.4f} lat {y0:.4f}–{y1:.4f}  ≈{km:.1f} km²  포함 오름 {n}개 (ids: {[c[1] for c in near][:8]}…)")
