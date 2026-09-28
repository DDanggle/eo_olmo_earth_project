#!/usr/bin/env python3
"""Summarize completed OE1 receipts and verify launch/source preservation locally."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(run):
    launch = read(run / "launch_manifest.json")
    status = read(run / "launch_status.json")
    assert status["status"] == "completed" and status["valid"] is True
    assert read(run / "protected_code_after.json") == launch["protected_code_before"]
    for name, expected in launch["source_snapshot_sha256"].items():
        assert hashlib.sha256((run / "code_snapshot" / name).read_bytes()).hexdigest() == expected
    summary = read(run / "training/summary.json")
    rows = [json.loads(line) for line in (run / "training/predictions.jsonl").read_text().splitlines()]
    items = {r["id"]: r for r in read(run / "training/items.json")}
    groups = {(a, c): {r["id"]: r for r in rows if r["arm"] == a and r["control"] == c}
              for a in ("frozen", "joint", "blind") for c in ("real", "zero", "observation_swap")}
    pairs = {}
    for other in ("frozen", "blind"):
        counts = Counter()
        by_mgrs = {}
        for ident, j in groups["joint", "real"].items():
            o = groups[other, "real"][ident]
            jc, oc = j["parsed"] == j["eval_gold"], o["parsed"] == o["eval_gold"]
            counts[f"joint_{jc}_other_{oc}"] += 1
            region = items[ident]["mgrs"]
            entry = by_mgrs.setdefault(region, {"n": 0, "joint_correct": 0, "other_correct": 0})
            entry["n"] += 1
            entry["joint_correct"] += int(jc)
            entry["other_correct"] += int(oc)
        pairs[other] = {"paired_counts": dict(counts), "by_mgrs": by_mgrs}
    swaps = {}
    for arm in ("frozen", "joint", "blind"):
        counts = Counter()
        for ident, donor in groups[arm, "observation_swap"].items():
            source = groups[arm, "real"][ident]
            counts["n"] += 1
            counts["changed_response"] += int(source["parsed"] != donor["parsed"])
            counts["both_source_and_donor_correct"] += int(source["parsed"] == source["eval_gold"] and donor["parsed"] == donor["eval_gold"])
        swaps[arm] = dict(counts)
    elapsed = (datetime.fromisoformat(status["finished_utc"]) - datetime.fromisoformat(status["started_utc"])).total_seconds()
    ties = {}
    for arm in ("frozen", "joint", "blind"):
        values = list(groups[arm, "real"].values())
        ties[arm] = {"yes_no_logit_ties": sum(r["yes_logit"] == r["no_logit"] for r in values),
                     "primary_constrained_disagreements": sum(r["parsed"] != r["constrained_prediction"] for r in values),
                     "non_tied_disagreements": sum(r["parsed"] != r["constrained_prediction"] and r["yes_logit"] != r["no_logit"] for r in values)}
    return {"schema": "oe1-launch-and-descriptive-summary-v0", "source_snapshot_hashes_valid": True,
            "protected_four_files_identical": True, "wall_seconds_including_preflight": elapsed,
            "trainer_seconds": summary["elapsed_seconds"],
            "peak_training_allocated_gib": max(x["peak_gpu_bytes"] for x in summary["training"].values()) / 2**30,
            "paired_real": pairs, "swap_descriptive": swaps, "real_input_tie_audit": ties,
            "raw_token_counts": {a + "/" + c: dict(Counter(r["raw_token"] for r in g.values())) for (a, c), g in groups.items()},
            "interpretation": "Post-result descriptive counts; no new decision threshold or IID significance claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
