#!/usr/bin/env python3
"""N-task break-even for a materialized Earth-embedding cache (CPU only).

Compares, for N downstream tasks over the SAME tile set:
  cache path : encode once (GPU s) + store cache (bytes) + N x HEAD_ADAPT
  raw path   : store raw only        + N x RAW_FINETUNE (GPU s + raw bytes read)
Costs are combined with a declared price vector (gpu $/s, storage $/GB-month, egress $/GB)
so that the break-even N is reported per price scenario, never as one number.
All inputs come from config/cost_ledger_v0.json; provisional (null) fields abort.
"""
import argparse, json, sys
from pathlib import Path

def load(p):
    d = json.loads(Path(p).read_text())
    if d["encode"]["p4_gpu_seconds_total"] is None:
        print("p4 encode time not measured; using p2 as an UPPER bound for p4 (flagged in output)", file=sys.stderr)
        d["encode"]["p4_gpu_seconds_total"] = d["encode"]["p2_gpu_seconds_total"]; d["encode"]["p4_is_bound"] = True
    return d

def cost(gpu_s, bytes_store, bytes_read, price, months):
    return (gpu_s * price["gpu_usd_per_s"] + bytes_store / 1e9 * price["storage_usd_per_gb_month"] * months
            + bytes_read / 1e9 * price["read_usd_per_gb"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ledger", default="config/cost_ledger_v0.json")
    ap.add_argument("--out", default="artifacts/cost/breakeven_v0.json"); ap.add_argument("--months", type=float, default=12)
    ap.add_argument("--max-n", type=int, default=20); a = ap.parse_args()
    L = load(a.ledger); st, en, ad = L["storage"], L["encode"], L["per_task_adaptation_measured_range"]
    prices = {  # scenarios, not facts: declared so that reviewers can substitute their own
        "cloud_list":   {"gpu_usd_per_s": 3.0/3600, "storage_usd_per_gb_month": 0.023, "read_usd_per_gb": 0.09},
        "on_prem_gpu":  {"gpu_usd_per_s": 0.5/3600, "storage_usd_per_gb_month": 0.005, "read_usd_per_gb": 0.0},
        "gpu_only":     {"gpu_usd_per_s": 1.0,       "storage_usd_per_gb_month": 0.0,   "read_usd_per_gb": 0.0},
    }
    out = {"months": a.months, "p4_encode_is_upper_bound": en.get("p4_is_bound", False), "scenarios": {}}
    for name, pr in prices.items():
        res = {}
        for contract, cbytes, enc in (("p4", st["cache_p4_bytes"], en["p4_gpu_seconds_total"]), ("p2", st["cache_p2_bytes"], en["p2_gpu_seconds_total"])):
            for bound, i in (("best_for_cache", 0), ("worst_for_cache", 1)):
                ha_s, rf_s, rf_b = ad["HEAD_ADAPT"]["gpu_seconds"][i], ad["RAW_FINETUNE"]["gpu_seconds"][1-i], ad["RAW_FINETUNE"]["raw_bytes"][1-i]
                curve = []; be = None
                for n in range(0, a.max_n + 1):
                    c_cache = cost(enc + n*ha_s, st["raw_u16_bytes"] + cbytes, 0, pr, a.months)  # raw kept as archive in both paths
                    c_raw = cost(n*rf_s, st["raw_u16_bytes"], n*rf_b, pr, a.months)
                    curve.append({"n": n, "cache_usd": round(c_cache, 4), "raw_usd": round(c_raw, 4)})
                    if be is None and n > 0 and c_cache <= c_raw: be = n
                res[f"{contract}_{bound}"] = {"breakeven_n": be, "curve": curve}
        out["scenarios"][name] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    for name, res in out["scenarios"].items():
        print(name, {k: v["breakeven_n"] for k, v in res.items()})
main()
