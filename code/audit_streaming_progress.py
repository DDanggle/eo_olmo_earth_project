#!/usr/bin/env python3
"""Summarize copied reports only; never open cache labels, checkpoints, or GPU devices."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

FOLDS = ("holdout_hiroshima", "holdout_thrissur", "holdout_chimanimani", "holdout_newzealand")
ARMS = ("gru", "gru_noobs", "ema", "gru_dt", "gru_sp", "xattn")


def summarize(snapshot):
    errors, groups, refs = [], defaultdict(list), defaultdict(list)
    for folder in ("streaming_t1v", "streaming_t1arch"):
        for path in sorted((snapshot / "artifacts" / folder).glob("*.json")):
            raw = path.read_bytes()
            r = json.loads(raw)
            key = (r["fold"], r["module"])
            if key[0] not in FOLDS or key[1] not in ARMS or r["seed"] not in (1, 2, 3):
                errors.append(f"unexpected identity: {path.name}")
                continue
            ds = r["downstream_c12"]
            t, stale, s = [ds[k]["auprc_exact"] for k in ("teacher_full_reencode", "frozen_m4", "student")]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in (t, stale, s)):
                errors.append(f"invalid AP: {path.name}")
                continue
            if any(not math.isfinite(h[k]) for h in r["history"] for k in ("train", "val")):
                errors.append(f"nonfinite history: {path.name}")
                continue
            if r["best_val_epoch"] <= 0 or not math.isfinite(r["best_val_loss"]):
                errors.append(f"invalid selection: {path.name}")
                continue
            refs[key[0]].append((t, stale, r["n"]))
            groups[key].append({"seed": r["seed"], "student_ap": s, "teacher_ap": t,
                "stale_ap": stale, "recovery": (s-stale)/(t-stale) if t > stale+1e-9 else None,
                "params": r["params"], "sha256": hashlib.sha256(raw).hexdigest(), "file": str(path.relative_to(snapshot))})
    rows = []
    for fold in FOLDS:
        rr = refs[fold]
        if rr and any(abs(x[0]-rr[0][0]) > 1e-8 or abs(x[1]-rr[0][1]) > 1e-8 or x[2] != rr[0][2] for x in rr):
            errors.append(f"reference AP/count mismatch: {fold}")
        for arm in ARMS:
            gg = sorted(groups[(fold, arm)], key=lambda x: x["seed"])
            seeds = [x["seed"] for x in gg]
            if len(seeds) != len(set(seeds)):
                errors.append(f"duplicate seeds: {fold}/{arm}")
            mean = lambda k: statistics.mean(x[k] for x in gg) if gg and all(x[k] is not None for x in gg) else None
            rows.append({"fold": fold, "arm": arm, "seeds": seeds, "complete": seeds == [1, 2, 3],
                "ap": mean("student_ap"), "teacher_ap": mean("teacher_ap"), "stale_ap": mean("stale_ap"),
                "recovery": mean("recovery"), "params": sorted({x["params"] for x in gg}), "reports": gg})
    lookup = {(r["fold"], r["arm"]): r for r in rows}
    comparisons = []
    for arm in ("gru_dt", "gru_sp", "xattn"):
        evaluated, wins = [], 0
        for fold in FOLDS:
            a, b = lookup[(fold, arm)], lookup[(fold, "gru")]
            if a["complete"] and b["complete"] and a["recovery"] is not None and b["recovery"] is not None:
                delta = a["recovery"]-b["recovery"]
                evaluated.append({"fold": fold, "delta_recovery": delta, "delta_ap": a["ap"]-b["ap"]})
                wins += delta >= .05
        missing = 4-len(evaluated)
        comparisons.append({"arm": arm, "completed_folds": len(evaluated), "wins": wins,
            "max_possible_wins": wins+missing, "three_of_four_still_possible": wins+missing >= 3,
            "final_gate": None if missing or errors else wins >= 3, "comparisons": evaluated})
    meta_path = snapshot/"kurosiwo_s1_cache/meta.jsonl"
    meta = [json.loads(l) for l in meta_path.read_text().splitlines() if l]
    acts, samples = defaultdict(set), defaultdict(set)
    for r in meta:
        acts[r["split"]].add(r["actid"])
        samples[r["split"]].add(r["id"])
    pairwise = {f"{a}__{b}": {"sample_overlap": len(samples[a]&samples[b]), "activation_overlap": len(acts[a]&acts[b])}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    bad_log = (snapshot/"logs/ks_update_gru_s1.log").read_text()
    return {"scope": "retrospective copied reports, decoder seed1; no fresh test scoring or pre-run attestation",
        "errors": errors, "baseline_reports": sum(len(g) for (f,a),g in groups.items() if a in ("gru", "ema", "gru_noobs")),
        "architecture_reports": sum(len(g) for (f,a),g in groups.items() if a in ("gru_dt", "gru_sp", "xattn")),
        "rows": rows, "architecture_screen": comparisons,
        "kuro_manifest": {"rows": len(meta), "unique_ids": len({x["id"] for x in meta}),
            "splits": dict(Counter(x["split"] for x in meta)), "activations": {k:len(v) for k,v in acts.items()},
            "pairwise": pairwise, "spatial_geometry_overlap_audited": False,
            "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()},
        "kuro_gru_seed1": {"nan_epoch_count": sum("val nan" in l for l in bad_log.splitlines()),
            "says_done": "DONE update" in bad_log, "valid_scientific_negative": False}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.snapshot)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    compact = {k:v for k,v in result.items() if k != "rows"}
    compact["rows"] = [{k:v for k,v in r.items() if k != "reports"} for r in result["rows"]]
    print(json.dumps(compact, indent=2, allow_nan=False))
    raise SystemExit(bool(result["errors"]))
