"""요약 수치가 원시 산출물에서 재계산되는지 검증한다 (CVPR 감사 항목 1).

MS-102 표의 macro 값을 bv1_runs/<cache>/holdout_*.json 원본에서 다시 계산해
artifacts/bv1_diagnostics_summary.json 및 bv1_summary.py 출력과 대조한다.
불일치가 있으면 표가 아니라 원본이 정답이다.
"""
import json, statistics
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
FOLDS = ["hiroshima", "hokkaido", "indonesia", "itogon",
         "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur"]
CLAIMED = {  # MEASURED_FINDINGS.md MS-102 표에 적힌 값
    "olmo_cache_pool16": 0.219, "clay_cache_native16": 0.155,
    "clay_cache_native16_last": 0.110, "clay_cache_in256": 0.195,
    "galileo_cache": 0.153, "prithvi_cache": 0.025,
    "galileo_cache_groupcat": None,  # 오늘 신규 — 장부 미등재
}
# bv1_summary.py와 동일 경로를 쓴다: d["test"]["positive_patch_macro_iou"]
METRIC_PATH = ("test", "positive_patch_macro_iou")

out = {"checked": {}, "mismatch": [], "missing": []}
for cache, claim in CLAIMED.items():
    vals, used_key = [], None
    for f in FOLDS:
        p = ROOT / "bv1_runs" / cache / f"holdout_{f}_seed1.json"
        if not p.exists():
            out["missing"].append(f"{cache}/{f}"); continue
        d = json.loads(p.read_text())
        node = d
        for seg in METRIC_PATH:
            if not isinstance(node, dict) or seg not in node:
                node = None; break
            node = node[seg]
        if node is None:
            out["missing"].append(f"{cache}/{f}:no {'.'.join(METRIC_PATH)}"); continue
        used_key = ".".join(METRIC_PATH); vals.append(float(node))
    if not vals:
        continue
    recomputed = round(statistics.fmean(vals), 4)
    rec = {"n_folds": len(vals), "metric_key": used_key,
           "recomputed_macro": recomputed, "claimed_macro": claim,
           "per_fold": [round(v, 4) for v in vals]}
    if claim is not None:
        rec["abs_diff"] = round(abs(recomputed - claim), 4)
        rec["agrees_to_3dp"] = rec["abs_diff"] <= 0.0005
        if not rec["agrees_to_3dp"]:
            out["mismatch"].append({"cache": cache, **rec})
    out["checked"][cache] = rec

# 공허한 통과 금지: 실제로 대조한 캐시가 0이면 통과라고 말하지 않는다.
n_compared = sum(1 for r in out["checked"].values() if r.get("claimed_macro") is not None)
out["n_compared"] = n_compared
if n_compared == 0:
    out["verdict"] = "검증 불능 — 대조한 요약값이 0건 (통과 아님)"
elif out["mismatch"]:
    out["verdict"] = f"불일치 {len(out['mismatch'])}건 — 원본 우선"
else:
    out["verdict"] = f"{n_compared}개 캐시의 요약값이 전부 원본에서 재계산됨"
(ROOT / "artifacts/summary_vs_raw_audit.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False))
for c, r in out["checked"].items():
    print(f"{c:26s} n={r['n_folds']} 재계산={r['recomputed_macro']:.4f} "
          f"장부={r['claimed_macro']} 일치={r.get('agrees_to_3dp')}")
print("판정:", out["verdict"])
if out["missing"]:
    print(f"누락 {len(out['missing'])}건:", out["missing"][:6])
