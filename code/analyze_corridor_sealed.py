#!/usr/bin/env python3
"""회랑 27창 봉인 계약(S1+S2) Δz — baseline↔s1_live 토큰 Δ, 5앵커 매칭 placebo p99 임계(차용) 초과 비율로 순위.
사전 등록: 임계는 같은 모델·같은 계약의 5앵커 매칭 9쌍 토큰 풀 p99를 **차용**함(회랑 자체 placebo 확보 전).
라벨은 "candidate change (sealed, borrowed threshold)"까지만. 광학 전용 순위(M69 v2)와 Spearman·상위10 교집합 비교.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nepal_delta import find_embedding, load_cube, cosine_delta  # noqa
REPO = Path(__file__).resolve().parents[1]
CROOT = REPO / "artifacts/external_data/nepal_olmo_live_v1/materialized_corridor"
def emb(mode, wid):
    base = CROOT / mode / "dataset/windows/nepal" / wid / "layers/embeddings"
    tifs = sorted(base.rglob("*.tif")) if base.exists() else []
    return load_cube(tifs[0]) if len(tifs) == 1 else None
def main():
    matched = sorted((REPO / "artifacts/external_data/nepal_olmo_live_v1/delta_matched").glob("*/nepal_delta_matched_report.json"))
    thr = None
    if matched:
        mj = json.loads(matched[-1].read_text()); ths = [v["token"]["threshold_p99"] for v in mj["anchors"].values() if v.get("token")]
        thr = float(np.median(ths)) if ths else None
    v2 = json.loads((REPO / "artifacts/corridor_s2_candidates/embed_v2/report.json").read_text())
    s2rank = {w["id"]: w.get("rank") for w in v2["windows"]}
    rows = []
    for wid in sorted(s2rank):
        zb, zl = emb("baseline", wid), emb("s1_live", wid)
        if zb is None or zl is None: rows.append({"id": wid, "status": "missing"}); continue
        d = cosine_delta(zb, zl)
        rows.append({"id": wid, "status": "ok", "mean": float(d.mean()), "p95": float(np.quantile(d, 0.95)),
                     "frac_above_borrowed_p99": float((d > thr).mean()) if thr else None, "s2_only_rank": s2rank[wid]})
    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: -(r["frac_above_borrowed_p99"] if r["frac_above_borrowed_p99"] is not None else r["mean"]))
    for i, r in enumerate(ok): r["sealed_rank"] = i + 1
    from scipy.stats import spearmanr
    pairs = [(r["sealed_rank"], r["s2_only_rank"]) for r in ok if r["s2_only_rank"]]
    rho = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation) if len(pairs) > 3 else None
    top_s = {r["id"] for r in ok[:10]}; top_o = {r["id"] for r in sorted([r for r in ok if r["s2_only_rank"]], key=lambda r: r["s2_only_rank"])[:10]}
    out = {"schema": "corridor-sealed-delta-v1", "borrowed_threshold_p99": thr, "n_windows": len(ok), "spearman_vs_s2_only": rho,
           "top10_overlap_with_s2_only": len(top_s & top_o), "windows": ok, "claim": "candidate change (sealed S1+S2, borrowed 5-anchor matched threshold); not damage"}
    od = REPO / "artifacts/external_data/nepal_olmo_live_v1/corridor_sealed"; od.mkdir(parents=True, exist_ok=True)
    (od / "report.json").write_text(json.dumps(out, indent=1))
    print("thr", thr, "n", len(ok), "spearman", rho, "top10 overlap", len(top_s & top_o))
    for r in ok[:10]: print(r["sealed_rank"], r["id"], "frac %.3f mean %.4f s2rank %s" % (r["frac_above_borrowed_p99"] or 0, r["mean"], r["s2_only_rank"]))
    print("DONE")
if __name__ == "__main__": main()
