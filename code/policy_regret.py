#!/usr/bin/env python3
"""Held-out action regret of deployment policies on a rectangular action matrix (CPU only).

Input: the g0 rectangular payload (rows: episode_id, task, action, seed, score, gpu_seconds, raw_bytes,
support_label_count, lower_anchor, upper_anchor). Policies:
  always:<ACTION>      static
  oracle               per-episode best mean score (upper bound, not a policy)
  rule:<json>          support-only rule evaluated leave-one-TASK-out, e.g. thresholds on Z0/K features
Outputs per policy: mean anchor-normalized regret, harm rate (chosen action worse than always:CACHED_HEAD by > eps),
mean gpu_s and raw bytes. Regret is averaged over episodes, seeds are repeated optimization (matrix must be rectangular).
"""
import argparse, json, statistics as S
from collections import defaultdict
from pathlib import Path

def norm(row):
    lo, hi = row["lower_anchor"], row["upper_anchor"]; return (row["score"] - lo) / (hi - lo) if hi > lo else float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out")
    ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--reference", default="CACHED_HEAD")
    ap.add_argument("--budget-gpu-s", type=float, default=float("inf")); ap.add_argument("--budget-raw-bytes", type=float, default=float("inf"))
    a = ap.parse_args(); P = json.loads(Path(a.input).read_text()); rows = P["rows"]
    ep = defaultdict(lambda: defaultdict(list))
    for r in rows: ep[r["episode_id"]][r["action"]].append(r)
    actions = sorted({r["action"] for r in rows})
    # rectangular check
    seeds = {(e, act): len(v) for e, d in ep.items() for act, v in d.items()}
    n_seed = set(seeds.values())
    if len(n_seed) != 1 or any(set(d) != set(actions) for d in ep.values()):
        print("NON-RECTANGULAR matrix; refusing to score policies", {e: sorted(d) for e, d in ep.items()}); return
    def mean_score(e, act): return S.mean(norm(r) for r in ep[e][act])
    def mean_cost(e, act, k): return S.mean(r[k] for r in ep[e][act])
    def eligible(e, act): return mean_cost(e, act, "gpu_seconds") <= a.budget_gpu_s and mean_cost(e, act, "raw_bytes") <= a.budget_raw_bytes
    policies = {f"always:{act}": (lambda e, act=act: act) for act in actions}
    policies["oracle"] = lambda e: max((x for x in actions if eligible(e, x)), key=lambda x: mean_score(e, x))
    res = {}
    for name, pol in policies.items():
        regs, harms, g, b = [], [], [], []
        for e in ep:
            act = pol(e)
            if not eligible(e, act): regs.append(float("nan")); continue
            best = max(mean_score(e, x) for x in actions if eligible(e, x))
            regs.append(best - mean_score(e, act)); harms.append(mean_score(e, act) < mean_score(e, a.reference) - a.eps)
            g.append(mean_cost(e, act, "gpu_seconds")); b.append(mean_cost(e, act, "raw_bytes"))
        res[name] = {"mean_regret": S.mean(regs), "max_regret": max(regs), "harm_rate": (sum(harms) / len(harms)) if harms else None,
                     "mean_gpu_s": S.mean(g) if g else None, "mean_raw_bytes": S.mean(b) if b else None, "n_episodes": len(ep)}
    out = {"actions": actions, "seeds_per_cell": n_seed.pop(), "eps": a.eps, "reference": a.reference, "policies": res,
           "note": "oracle is an upper bound, not a policy; static policies are the baselines any support-only rule must beat on regret AND harm"}
    print(json.dumps(out, indent=1))
    if a.out: Path(a.out).write_text(json.dumps(out, indent=1))
main()
