#!/usr/bin/env python3
"""Read-only offline audit of copied KR-4 reports, not a pixel-level reevaluation.

No torch, GPU, server access, training, label opening, or production imports.
Hashes identify retrieved inputs, not the code executed before the original runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

METRIC = {"land_cover": "cluster_macro_miou", "logged": "all_cluster_ap", "landslide": "all_cluster_ap"}


def zero_positive_probability(population, positive_chips, k):
    """Uniform sampling without replacement; pixel prevalence is not this input."""
    if not 0 <= positive_chips <= population or not 0 <= k <= population:
        raise ValueError("invalid chip population")
    if k > population - positive_chips:
        return 0.0
    return math.prod((population - positive_chips - i) / (population - i) for i in range(k))


def matched_row(task, rows, k, draw):
    paired = []
    for seed in (1, 2, 3):
        a = rows[f"CACHE_K_K{k}_{draw}_s{seed}"]
        b = rows[f"RAW_K_K{k}_{draw}_s{seed}"]
        assert a["support"]["support_ids"] == b["support"]["support_ids"]
        assert len(a["support"]["support_ids"]) == k
        assert len(set(a["support"]["support_ids"])) == k
        assert a["train"]["steps"] == b["train"]["steps"]
        av, bv = a["test"][METRIC[task]], b["test"][METRIC[task]]
        assert all(math.isfinite(v) for v in (av, bv))
        paired.append({"seed": seed, "cache": av, "raw": bv, "delta": av-bv,
                       "positive_support_chips": a["support"]["support_pos_chips"],
                       "steps": a["train"]["steps"],
                       "cache_train_s": a["train"]["gpu_s"], "raw_train_s": b["train"]["gpu_s"],
                       "cache_recorded_train_bytes": a["train"]["train_bytes_read"],
                       "raw_recorded_train_bytes": b["train"]["train_bytes_read"]})
    return {"K_total_labeled_chips": k, "draw": draw, "pairs": paired,
            "cache_mean": mean(x["cache"] for x in paired),
            "raw_mean": mean(x["raw"] for x in paired),
            "cache_sd": pstdev(x["cache"] for x in paired),
            "delta_mean": mean(x["delta"] for x in paired),
            "exact_seed_wins": sum(x["delta"] > 0 for x in paired),
            "rounded_4dp_seed_wins": sum(round(x["cache"], 4) > round(x["raw"], 4) for x in paired)}


def audit(root):
    paths = [root / "run_v1/report.json", root / "run_v1_fullraw/report.json",
             root / "korea_cache_audit.json", root / "label_inventory.json",
             root / "korea_chip_manifest_summary.json"]
    report, raw, cache, inventory, manifest = [json.loads(p.read_text()) for p in paths]
    assert report["n"] == raw["n"] == manifest["by_split"]
    tasks = {}
    for task, metric in METRIC.items():
        rows = report["tasks"][task]
        fc, fr = rows["FULL_CACHE"], raw["tasks"][task]["FULL_RAW"]
        assert len(fc["test"]["clusters"]) == len(fr["test"]["clusters"]) == 7
        assert fc["test"]["n_chips"] == fr["test"]["n_chips"] == report["n"]["test"]
        ref = {"cache": fc["test"][metric], "raw": fr["test"][metric],
               "delta": fc["test"][metric] - fr["test"][metric],
               "cache_steps": fc["train"]["steps"], "raw_steps": fr["train"]["steps"],
               "cache_train_s": fc["train"]["gpu_s"], "raw_train_s": fr["train"]["gpu_s"],
               "replicates_each": 1,
               "sample_exposures_from_report_steps_and_current_code_batch": {
                   "cache": 32 * fc["train"]["steps"], "raw": 16 * fr["train"]["steps"]}}
        if task != "land_cover":
            positive = fc["test"]["test_pos_px"]
            assert positive == fr["test"]["test_pos_px"]
            ref.update({"positive_pixels": positive,
                        "query_pixel_prevalence": positive / (report["n"]["test"] * 128 * 128),
                        "positive_clusters": {c: x["pos_px"] for c,x in fc["test"]["clusters"].items() if x["pos_px"]}})
        else:
            ref["cluster_wins"] = sum(fc["test"]["clusters"][c]["miou"] > fr["test"]["clusters"][c]["miou"] for c in fc["test"]["clusters"])
        tasks[task] = {"metric": metric, "full_reference": ref, "few_shot": {}}
        for k in (5,20):
            for draw in (["random"] if task == "land_cover" else ["random", "posaware"]):
                tasks[task]["few_shot"][f"K{k}_{draw}"] = matched_row(task, rows, k, draw)
    total_cache_train = sum(t["full_reference"]["cache_train_s"] for t in tasks.values())
    total_raw_train = sum(t["full_reference"]["raw_train_s"] for t in tasks.values())
    byte_shape = cache["n_chips"] * 768 * 32 * 32 * 2
    out = {"scope": "REPORT_ARITHMETIC_AND_CURRENT_CODE_AUDIT_NOT_PIXEL_RECOMPUTATION",
           "n": report["n"], "tasks": tasks,
           "random_primary_direction_only": {str(k): [t for t in tasks if tasks[t]["few_shot"][f"K{k}_random"]["delta_mean"] > 0] for k in (5,20)},
           "original_multi_task_primary_rule_not_met": not all(sum(tasks[t]["few_shot"][f"K{k}_random"]["delta_mean"] > 0 for t in tasks) >= 2 for k in (5,20)),
           "cost_partial_components_only": {"cache_extraction_reported_s": cache["elapsed_s"],
                 "three_cache_head_training_s": total_cache_train, "three_raw_training_s": total_raw_train,
                 "extraction_plus_three_heads_s": cache["elapsed_s"] + total_cache_train,
                 "ratio_to_three_raw_train_s": (cache["elapsed_s"] + total_cache_train) / total_raw_train,
                 "excludes": "evaluation latency, unreported I/O, upstream pretraining and operational amortization",
                 "array_payload_bytes_from_declared_shape": byte_shape, "payload_decimal_GB": byte_shape / 1e9,
                 "payload_GiB": byte_shape / 2**30, "physical_disk_size_verified": False},
           "retrieved_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    mp = root / "korea_chip_manifest.jsonl"
    if mp.exists():
        rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        assert hashlib.sha256(mp.read_bytes()).hexdigest() == manifest["manifest_sha256"]
        invalid = {x["key"] for x in inventory["unreadable_label_tifs"]}
        affected = []
        for c in rows:
            valid_keys = [k for k in c["keys"] if k not in invalid]
            if valid_keys and max(c["keys"]) > max(valid_keys):
                affected.append({"chip_id": c["chip_id"], "split": c["split"], "cluster": c["cluster"],
                                 "last_cache_key": max(c["keys"]), "last_label_key": max(valid_keys)})
        out["known_missing_label_cutoff_mismatch"] = {"n_chips": len(affected), "affected": affected,
            "scope": "Derived from manifest and the inventory known-missing keys; not an exhaustive scan of mask file presence"}
        out["retrieved_sha256"][mp.name] = hashlib.sha256(mp.read_bytes()).hexdigest()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": {t: x["full_reference"] for t,x in result["tasks"].items()},
                      "cost": result["cost_partial_components_only"],
                      "primary_not_met": result["original_multi_task_primary_rule_not_met"],
                      "cutoff_mismatch_chips": result.get("known_missing_label_cutoff_mismatch", {}).get("n_chips")}, indent=2))
