#!/usr/bin/env python3
"""Read-only reproduction of MS-155 answer arithmetic; no models or data writes.

This checks the saved answer files, not image visibility or counterfactual labels.
Run locally from any directory with Python 3 (standard library only).
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "artifacts" / "spacenet7"


def records(filename):
    with (BASE / filename).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(row["id"], row["cond"]) for row in rows]
    assert len(keys) == len(set(keys)), "Duplicate item-condition records"
    return rows


def audit(reader):
    rows = records(f"diag_lite_v0_4_{reader}_answers.jsonl")
    with (BASE / f"diag_lite_v0_4_{reader}_scores.json").open() as handle:
        scores = json.load(handle)
    for cond, groups in scores["per_condition"].items():
        for question, result in groups.items():
            subset = [r for r in rows if r["cond"] == cond and r["q"] == question]
            if result is None:
                assert not subset
                continue
            correct = sum(r["pred"] == r["gold"] for r in subset)
            assert len(subset) == result["n"]
            assert abs(correct / len(subset) - result["acc"]) < 1e-12
    q2 = {r["id"]: r for r in rows if r["q"] == "Q2" and r["cond"] == "privileged_silver"}
    wrong = {r["id"]: r for r in rows if r["q"] == "Q2" and r["cond"] == "privileged_wrongcontent"}
    assert q2.keys() == wrong.keys()
    pairs = Counter()
    for key, row in q2.items():
        other = wrong[key]
        assert row["gold"] == other["gold"]
        pairs[f"priv_{int(row['pred'] == row['gold'])}_wrong_{int(other['pred'] == other['gold'])}"] += 1
    q1 = [r for r in rows if r["q"] == "Q1" and r["cond"] == "latest_k"]
    privileged_q1 = [r for r in rows if r["q"] == "Q1" and r["cond"] == "privileged_silver"]
    old_name = "diag_lite_v0_3_molmo_answers.jsonl" if reader == "molmo" else "diag_lite_v0_3_answers.jsonl"
    old_q2 = {r["id"] for r in records(old_name) if r["q"] == "Q2" and r["cond"] == "privileged_silver"}
    distinct = {(r["aoi"], r["id"].split("|")[-1], r["gold"]) for r in q2.values()}
    return {
        "all_reported_item_accuracies_match": True,
        "q2_items": len(q2),
        "q2_aois": len({r["aoi"] for r in q2.values()}),
        "q2_distinct_aoi_region_gold": len(distinct),
        "q2_same_item_ids_as_v03": len(q2.keys() & old_q2),
        "q2_correctness_pairs": dict(sorted(pairs.items())),
        "q2_same_prediction_after_swap": sum(q2[k]["pred"] == wrong[k]["pred"] for k in q2),
        "q1_latest_gold_counts": dict(Counter(r["gold"] for r in q1)),
        "q1_latest_confusion_gold_pred": dict(Counter(f"{r['gold']}_{r['pred']}" for r in q1)),
        "q1_privileged_gold_counts": dict(Counter(r["gold"] for r in privileged_q1)),
        "limitations": "No source frames/items/labels here; no visual correctness or alternate-quadrant gold audit. AOI bootstrap not rerun.",
    }


if __name__ == "__main__":
    print(json.dumps({reader: audit(reader) for reader in ("qwen", "molmo")}, indent=2))
