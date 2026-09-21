#!/usr/bin/env python3
"""라벨 대상 명단 생성 (WP1). 상위 30 + 대조 30 + 재판독 5 → apps/oreum-web/public/data/label_targets.json
- 상위: v8 순위 1..30 (채점된 오름만).
- 대조: 채점됐지만 사건 깃발율이 채점 중앙값 미만인 오름에서 무작위 30 (seed 고정 = 재현 가능).
- 재판독: 상위∪대조 60곳 중 무작위 5 (자기 일치율용). 명단은 계약 sha 와 함께 봉인된다.
대조군이 없으면 "상위 = 변화" 가 순환논리가 되므로, 대조는 옵션이 아니다."""
import argparse, hashlib, json, random, statistics, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT/"apps/oreum-web/public/data"
ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=30); ap.add_argument("--control", type=int, default=30); ap.add_argument("--recheck", type=int, default=5); ap.add_argument("--seed", type=int, default=20260922); a = ap.parse_args()
g = json.load(open(DATA/"oreum.geojson")); s = json.load(open(DATA/"summary.json"))
scored = sorted([f["properties"] for f in g["features"] if f["properties"]["verdict"] == "scored"], key=lambda p: p["rank"])
top = [p["oreum_id"] for p in scored[:a.top]]
med = statistics.median(p["event_flag_frac"] for p in scored)
pool = [p["oreum_id"] for p in scored if p["event_flag_frac"] < med and p["oreum_id"] not in top]
rng = random.Random(a.seed); control = sorted(rng.sample(pool, a.control))
recheck = sorted(rng.sample(top + control, a.recheck))
TAGS = [
    {"tag": "trail_erosion", "short": "탐방로 침식·나지", "long": "탐방로 주변 나지 확대·침식(등급제 3~5등급 지표: 침식 깊이, 뿌리·암석 노출)"},
    {"tag": "veg_loss", "short": "식생 손실", "long": "병해·기후·산불 등으로 식생이 사라짐(IUCN 기후 위협)"},
    {"tag": "facility", "short": "시설·도로·개간", "long": "새 시설물·도로·개간·건축(개발행위허가와 대조 가능)"},
    {"tag": "grazing", "short": "방목·경작", "long": "방목·경작·초지 관리로 인한 변화"},
    {"tag": "coastal", "short": "해안 침식", "long": "해안형 오름의 침식·붕괴(성산일출봉·송악산 등)"},
    {"tag": "other", "short": "기타", "long": "위 어디에도 맞지 않음 — 메모에 무엇을 봤는지 반드시 적는다"},
]
out = {"schema": "oreum-label-targets/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "contract_sha256": s.get("contract_sha256"),
       "seed": a.seed, "control_rule": f"scored & event_flag_frac < median({med:.4f})", "top": top, "control": control, "recheck": recheck, "tags": TAGS}
body = json.dumps({k: v for k, v in out.items() if k != "generated"}, ensure_ascii=False, sort_keys=True).encode(); out["targets_sha256"] = hashlib.sha256(body).hexdigest()
(DATA/"label_targets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
name = {p["oreum_id"]: p["name"] for p in scored}
print(f"top {len(top)} · control {len(control)} (pool {len(pool)}, median flag {med:.3%}) · recheck {recheck} · sha {out['targets_sha256'][:12]}")
print("control:", ", ".join(f"{i} {name[i]}" for i in control))
