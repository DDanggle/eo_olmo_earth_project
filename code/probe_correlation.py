"""라벨 없는 프로브가 downstream 캐시 가치를 예측하는가.

자산: 캐시 14개 x 8폴드 downstream macro IoU (이미 측정됨).
질문: 라벨 0으로 계산되는 어떤 프로브가 그 macro 를 예측하는가.

=== 사전 등록 (결과 보기 전 고정) ===
C1 전체상관   : |Spearman(probe, macro)| >= 0.70  -> 후보
C2 차원 교란  : emb_dim 단독 |rho| 가 프로브보다 높으면 그 프로브는 기각
                (프로브가 단지 채널수 대리물이면 안 된다)
C3 family 교란: family 내부 상관도 같은 부호로 유지돼야 한다.
                3 family(OlmoEarth/Galileo/Clay) 중 >=2 에서 부호 일치 필요.
C4 외삽       : leave-one-family-out 으로 나머지 두 family 에 적합한 순위가
                빠진 family 의 순위와 Spearman >= 0.50
판정: C1 통과 AND C2 통과 AND C3 통과 -> PREDICTOR_CANDIDATE (C4 는 보고만)
      그 외 -> NOT_PREDICTIVE
n=14 는 작다. rho 는 부트스트랩 CI 와 함께 보고하고, 확증은 GEO-Bench 과업 확장 뒤로 미룬다.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
GATES = {"C1_rho_min": 0.70, "C3_family_sign_min": 2, "C4_loo_min": 0.50}
FAMILY = {"olmo": "OlmoEarth", "clay": "Clay", "galileo": "Galileo", "prithvi": "Prithvi"}
PROBES = ["effective_rank", "effective_rank_frac", "participation_ratio", "anisotropy",
          "dead_channel_frac", "spatial_gap", "phys_r2_pca_mean", "phys_r2_fulldim_mean",
          "emb_dim"]


def family_of(c):
    for k, v in FAMILY.items():
        if c.startswith(k):
            return v
    return "?"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if len(set(i.tolist())) < 3:
            continue
        r = spearman(a[i], b[i])
        if not np.isnan(r):
            vals.append(r)
    if not vals:
        return [float("nan")] * 2
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_truth():
    truth = {}
    for f in ["artifacts/bv1_diagnostics_summary.json", "artifacts/arch_axes_summary.json"]:
        p = ROOT / f
        if not p.exists():
            continue
        for c, v in json.load(open(p)).get("verdict", {}).items():
            if v.get("n") == 8:
                truth[c] = float(v["macro"])
    return truth


def main():
    truth = load_truth()
    probes = {x["cache"]: x for x in json.load(open(ROOT / "artifacts/cache_probes.json"))
              if "error" not in x}
    caches = sorted(set(truth) & set(probes))
    if len(caches) < 6:
        out = {"status": "INSUFFICIENT", "n": len(caches), "caches": caches,
               "note": "프로브 x 정답이 6개 미만이면 상관을 주장하지 않는다"}
        print(json.dumps(out, indent=1, ensure_ascii=False)); return out

    y = np.array([truth[c] for c in caches])
    fam = [family_of(c) for c in caches]
    res = {"n_caches": len(caches), "caches": caches, "gates": GATES,
           "macro": {c: truth[c] for c in caches},
           "family": dict(zip(caches, fam)), "probes": {}}

    dim_rho = abs(spearman([probes[c]["emb_dim"] for c in caches], y))
    res["emb_dim_abs_rho"] = dim_rho

    for pr in PROBES:
        x = [probes[c].get(pr) for c in caches]
        if any(v is None for v in x):
            continue
        rho = spearman(x, y)
        ci = boot_ci(x, y)
        # family 내부
        within = {}
        for f in sorted(set(fam)):
            idx = [i for i, ff in enumerate(fam) if ff == f]
            if len(idx) >= 3:
                within[f] = spearman([x[i] for i in idx], y[idx])
        signs = [np.sign(v) for v in within.values() if not np.isnan(v)]
        sign_ok = sum(1 for s in signs if s == np.sign(rho))
        # leave-one-family-out: 남은 두 family 로 선형 적합 -> 빠진 family 순위 예측
        loo = {}
        for f in sorted(set(fam)):
            tr = [i for i, ff in enumerate(fam) if ff != f]
            te = [i for i, ff in enumerate(fam) if ff == f]
            if len(tr) >= 4 and len(te) >= 3:
                A = np.vstack([np.array(x)[tr], np.ones(len(tr))]).T
                coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
                pred = np.array(x)[te] * coef[0] + coef[1]
                loo[f] = spearman(pred, y[te])
        c1 = abs(rho) >= GATES["C1_rho_min"]
        c2 = abs(rho) >= dim_rho or pr == "emb_dim"
        c3 = sign_ok >= GATES["C3_family_sign_min"]
        res["probes"][pr] = {
            "spearman": rho, "ci95": ci, "abs_rho": abs(rho),
            "within_family": within, "n_family_sign_agree": int(sign_ok),
            "loo_family_spearman": loo,
            "C1_pass": bool(c1), "C2_pass": bool(c2), "C3_pass": bool(c3),
            "verdict": "PREDICTOR_CANDIDATE" if (c1 and c2 and c3) else "NOT_PREDICTIVE",
            "values": dict(zip(caches, [float(v) for v in x])),
        }

    cands = [p for p, v in res["probes"].items() if v["verdict"] == "PREDICTOR_CANDIDATE"]
    res["candidates"] = cands
    res["overall_verdict"] = ("CANDIDATES_FOUND" if cands else "NO_PREDICTOR_AT_N14")
    res["caveat"] = ("n=14 캐시, downstream 과업 1개(Sen12). 확증은 GEO-Bench 과업 확장 뒤. "
                     "family 3개는 서로 독립 표본이 아니다(같은 폴드·시드 공유).")
    (ROOT / "artifacts/probe_correlation.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))

    print(f"n={res['n_caches']}  emb_dim |rho|={dim_rho:.3f}")
    print(f"{'probe':24s} {'rho':>7s} {'CI95':>18s} {'fam동의':>7s}  판정")
    for p, v in sorted(res["probes"].items(), key=lambda kv: -kv[1]["abs_rho"]):
        ci = f"[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
        print(f"{p:24s} {v['spearman']:+7.3f} {ci:>18s} {v['n_family_sign_agree']:>7d}  {v['verdict']}")
    print("\n판정:", res["overall_verdict"], "| 후보:", cands or "없음")
    return res


if __name__ == "__main__":
    main()
