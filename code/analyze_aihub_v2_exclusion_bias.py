"""AI-Hub 큐브 v2 제외 163건의 선택 편향 진단 (gate 6 보고용).

읽기 전용. 질문: 제외가 (a) 날짜에 뭉쳤나 (b) 지리에 뭉쳤나 (c) split을 깨나.

사전 등록 판정 기준 (L4 — 결과 보기 전에 고정):
  G1 지리 붕괴  : 전체 date를 잃은 tile이 1개라도 있으면 "지역 소실" — 보고 필수.
  G2 지리 편향  : tile별 제외율의 상위 10% tile이 전체 제외의 50% 이상을 차지하면 "공간 집중".
  G3 시간 집중  : 단일 date가 전체 제외의 30% 이상이면 "날짜 집중" — 지리 편향이 아니라
                  그 날짜 STAC 커버리지 문제로 해석한다.
  G4 split 편향 : train/val/test 제외율 최대-최소 차가 3%p 초과면 "split 불균형".
어느 것도 걸리지 않으면 제외는 무작위에 가깝다고 보고한다.
"""
import json, collections
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth/aihub")
OUT = ROOT / "s2_12band_v2"

inv = {}
for line in open(ROOT / "inventory/inventory.jsonl"):
    if line.strip():
        r = json.loads(line); inv[r["key"]] = r

exc = [json.loads(l) for l in open(OUT / "excluded.jsonl") if l.strip()]
man = [json.loads(l) for l in open(OUT / "manifest.jsonl") if l.strip()]
exc_keys = {e["key"] for e in exc}
man_keys = {m["key"] for m in man}
sel_keys = exc_keys | man_keys

res = {"n_selected": len(sel_keys), "n_manifest": len(man_keys), "n_excluded": len(exc_keys)}

# --- tile 단위 ---
tile_sel = collections.Counter(); tile_exc = collections.Counter()
for k in sel_keys:
    t = inv.get(k, {}).get("tile_id", k.split("_")[0]); tile_sel[t] += 1
for k in exc_keys:
    t = inv.get(k, {}).get("tile_id", k.split("_")[0]); tile_exc[t] += 1

fully_lost = sorted(t for t in tile_sel if tile_exc[t] == tile_sel[t])
res["n_tiles_selected"] = len(tile_sel)
res["n_tiles_touched_by_exclusion"] = len(tile_exc)
res["G1_tiles_fully_lost"] = fully_lost
res["G1_triggered"] = len(fully_lost) > 0

ranked = sorted(tile_exc.items(), key=lambda x: -x[1])
top10pct = max(1, len(tile_sel) // 10)
share = sum(c for _, c in ranked[:top10pct]) / len(exc_keys)
res["G2_top10pct_tile_share_of_exclusions"] = round(share, 4)
res["G2_triggered"] = share >= 0.50
res["top_excluded_tiles"] = ranked[:10]

# --- 날짜 단위 ---
date_exc = collections.Counter(inv.get(k, {}).get("date", k.split("_")[-1]) for k in exc_keys)
date_sel = collections.Counter(inv.get(k, {}).get("date", k.split("_")[-1]) for k in sel_keys)
dr = sorted(date_exc.items(), key=lambda x: -x[1])
res["G3_top_date"] = dr[0] if dr else None
res["G3_top_date_share"] = round(dr[0][1] / len(exc_keys), 4) if dr else None
res["G3_triggered"] = bool(dr and dr[0][1] / len(exc_keys) >= 0.30)
res["top_excluded_dates"] = [(d, n, date_sel[d], round(n / date_sel[d], 3)) for d, n in dr[:10]]

# --- split 단위 ---
sp_sel = collections.Counter(inv.get(k, {}).get("split", "?") for k in sel_keys)
sp_exc = collections.Counter(inv.get(k, {}).get("split", "?") for k in exc_keys)
rates = {s: round(sp_exc[s] / sp_sel[s], 4) for s in sp_sel}
res["split_selected"] = dict(sp_sel); res["split_excluded"] = dict(sp_exc)
res["split_exclusion_rate"] = rates
res["G4_spread_pp"] = round((max(rates.values()) - min(rates.values())) * 100, 2) if rates else 0
res["G4_triggered"] = res["G4_spread_pp"] > 3.0

# --- 사유별 x 날짜 (no_stac_item이 어디서 오나) ---
by_reason_date = collections.defaultdict(collections.Counter)
for e in exc:
    d = inv.get(e["key"], {}).get("date", e["key"].split("_")[-1])
    by_reason_date[e["reason"]][d] += 1
res["reason_x_top_dates"] = {r: c.most_common(5) for r, c in by_reason_date.items()}

# --- 위경도 분포 (제외 vs 유지) ---
def centroid(k):
    b = inv.get(k, {}).get("wgs84_bbox")
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else None
ec = [c for c in map(centroid, exc_keys) if c]
mc = [c for c in map(centroid, man_keys) if c]
def stats(cs):
    if not cs: return None
    lon = sorted(c[0] for c in cs); lat = sorted(c[1] for c in cs)
    q = lambda v, p: v[int(p * (len(v) - 1))]
    return {"n": len(cs), "lon_p05": round(q(lon,.05),4), "lon_med": round(q(lon,.5),4),
            "lon_p95": round(q(lon,.95),4), "lat_p05": round(q(lat,.05),4),
            "lat_med": round(q(lat,.5),4), "lat_p95": round(q(lat,.95),4)}
res["centroid_excluded"] = stats(ec); res["centroid_kept"] = stats(mc)

triggered = [g for g in ("G1","G2","G3","G4") if res[f"{g}_triggered"]]
res["verdict"] = ("무작위에 가까움 (사전 등록 기준 전부 미발동)" if not triggered
                  else "발동: " + ",".join(triggered))
(OUT / "exclusion_bias.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
