#!/usr/bin/env python3
"""Rescore OE1 development receipts; do not load model tensors or regenerate logits."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from audit_prepared_receipts import expected_donors, read_json, read_rows, require, score, sha256

ARMS = ("frozen", "joint", "blind")
CONTROLS = ("real", "zero", "observation_swap")


def check_prediction(record, items, donors):
    source = items[record["id"]]
    shown_id = donors[source["id"]] if record["control"] == "observation_swap" else source["id"]
    shown = items[shown_id]
    require(record["presented_id"] == shown_id, "Wrong presented donor ID")
    require(record["patch_id"] == source["patch_id"] and record["presented_patch_id"] == shown["patch_id"], "Wrong prediction image mapping")
    require(record["presented_date"] == shown["date"], "Wrong presented date")
    require(record["source_gold"] == source["output"] and record["eval_gold"] == shown["output"], "Wrong source/donor gold")
    if record["control"] == "observation_swap":
        require(source["input"] == shown["input"] and source["output"] != shown["output"], "Invalid donor interpretation")
    normalized = record["raw_token"].strip().lower()
    parsed = normalized if normalized in ("yes", "no") else None
    require(record["parsed"] == parsed, "Stored parse differs from raw token")
    require(all(math.isfinite(record[key]) for key in ("yes_logit", "no_logit", "constrained_yes_probability")), "Nonfinite prediction")
    difference = record["yes_logit"] - record["no_logit"]
    probability = 1 / (1 + math.exp(-difference)) if difference >= 0 else math.exp(difference) / (1 + math.exp(difference))
    require(math.isclose(record["constrained_yes_probability"], probability, abs_tol=2e-6), "Stored constrained probability differs from logits")
    constrained = "yes" if difference >= 0 else "no"
    require(record["constrained_prediction"] == constrained, "Stored constrained label differs from logits")
    require(record["image_tokens_used"] == (record["arm"] != "blind" and record["control"] != "zero"), "Image input flag differs")
    return dict(record, parsed=parsed, constrained_prediction=constrained,
                query_class=source.get("query_class") or "unparsed", queried_mgrs=source["mgrs"], presented_mgrs=shown["mgrs"])


def score_records(records):
    golds = [row["eval_gold"] for row in records]
    result = {"n": len(records), "support": dict(Counter(golds))}
    for field in ("parsed", "constrained_prediction"):
        calculated = score(golds, [row[field] for row in records])
        result[field] = {key: value for key, value in calculated.items() if key not in ("n", "support")}
    return result


def close_structure(actual, expected):
    if isinstance(actual, dict):
        return isinstance(expected, dict) and set(actual) == set(expected) and all(close_structure(value, expected[key]) for key, value in actual.items())
    if isinstance(actual, float):
        return isinstance(expected, (float, int)) and math.isclose(actual, expected, abs_tol=1e-10)
    return actual == expected


def audit(run_dir, data_dir=None):
    run_dir = Path(run_dir)
    status = read_json(run_dir / "status.json")
    manifest = read_json(run_dir / "manifest.json")
    require(status["status"] == "completed" and status["valid"] is True and manifest["valid"] is True, "Run is not completed and valid")
    require(sha256(run_dir / "manifest.json") == status["manifest_sha256"], "Final manifest SHA mismatch")
    verified, absent_checkpoints = {}, []
    for name, expected in manifest["files_sha256"].items():
        require(Path(name).name == name, "Nonlocal result path")
        target = run_dir / name
        if not target.is_file():
            require(name.endswith(".pt"), "Missing result receipt: " + name)
            absent_checkpoints.append(name)
        else:
            verified[name] = sha256(target)
            require(verified[name] == expected, "Result SHA mismatch: " + name)
    required = {"summary.json", "items.json", "predictions.jsonl", "run_config.json", "batches.json", "donor_plan.json", "training_log.jsonl", "training_summary.json", "model_sources.json"}
    require(required.issubset(verified), "Required receipts absent from manifest")
    config = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "summary.json")
    source_models = read_json(run_dir / "model_sources.json")
    items_list = read_json(run_dir / "items.json")
    items = {row["id"]: row for row in items_list}
    require(len(items) == len(items_list), "Repeated item ID")
    train_ids = {row["id"] for row in items_list if row["partition"] == "train"}
    dev_rows = [row for row in items_list if row["partition"] == "dev"]
    dev_ids = {row["id"] for row in dev_rows}
    require(len(train_ids) == config["n_train"] and len(dev_ids) == config["n_dev"] and train_ids | dev_ids == set(items), "Item split counts differ")
    require(config["arms"] == list(ARMS) and config["seed"] == 17 and config["batch_size"] == 8, "Unexpected pilot configuration")
    source_receipts_checked = []
    if data_dir is not None:
        data_dir = Path(data_dir)
        for name in ("manifest.json", "train.jsonl", "dev.jsonl"):
            original_path = str(Path(config["data_dir"]) / name)
            require(sha256(data_dir / name) == manifest["source_files_sha256"][original_path] == config["data_files_sha256"][original_path], "Original prepared receipt differs: " + name)
            source_receipts_checked.append(name)
        for split in ("train", "dev"):
            prepared = read_rows(data_dir / (split + ".jsonl"))
            require({row["id"] for row in prepared} == (train_ids if split == "train" else dev_ids), "Prepared IDs differ from run")
            require(all(all(items[row["id"]].get(key) == value for key, value in row.items()) for row in prepared), "Prepared item contents changed")
    donors = expected_donors(dev_rows)
    donor_receipt = read_json(run_dir / "donor_plan.json")
    require(donors == donor_receipt["mapping"] and len(donors) == donor_receipt["eligible"] and donor_receipt["total_dev"] == len(dev_ids), "Donor plan differs")
    groups = defaultdict(dict)
    raw_records = read_rows(run_dir / "predictions.jsonl")
    for record in raw_records:
        require(record["arm"] in ARMS and record["control"] in CONTROLS and record["id"] in dev_ids, "Unexpected prediction population")
        key = record["arm"] + "/" + record["control"]
        require(record["id"] not in groups[key], "Repeated prediction ID")
        groups[key][record["id"]] = check_prediction(record, items, donors)
    metrics, grouped = {}, {}
    for arm in ARMS:
        for control in CONTROLS:
            key = arm + "/" + control
            require(set(groups[key]) == (set(donors) if control == "observation_swap" else dev_ids), "Prediction coverage mismatch: " + key)
            records = list(groups[key].values())
            metrics[key] = score_records(records)
            require(close_structure(metrics[key], summary["arms"][key]), "Summary rescore mismatch: " + key)
            grouped[key] = {}
            for field in ("queried_mgrs", "presented_mgrs", "query_class"):
                buckets = defaultdict(list)
                for record in records:
                    buckets[record[field]].append(record)
                grouped[key][field] = {name: score_records(bucket) for name, bucket in sorted(buckets.items())}
    paired = {}
    for arm in ARMS:
        paired[arm] = {}
        for population, identifiers in (("all_dev", sorted(dev_ids)), ("donor_eligible", sorted(donors))):
            available = CONTROLS if population == "donor_eligible" else CONTROLS[:2]
            paired[arm][population] = {control: score_records([groups[arm + "/" + control][identifier] for identifier in identifiers]) for control in available}
            golds = [items[identifier]["output"] for identifier in identifiers]
            for control in available[1:]:
                original = [groups[arm + "/real"][identifier]["parsed"] for identifier in identifiers]
                altered = [groups[arm + "/" + control][identifier]["parsed"] for identifier in identifiers]
                valid_pairs = [(first, second) for first, second in zip(original, altered) if first in ("yes", "no") and second in ("yes", "no")]
                paired[arm][population][control + "_paired"] = {"source_label_agreement": score(golds, altered),
                    "valid_response_pairs": len(valid_pairs), "changed_valid_responses": sum(first != second for first, second in valid_pairs),
                    "source_label_agreement_mean_difference_from_real": (sum(value == gold for value, gold in zip(altered, golds)) - sum(value == gold for value, gold in zip(original, golds))) / len(golds) if golds else None}
    batches = read_json(run_dir / "batches.json")
    require(len(batches) == config["steps_per_arm"] and all(len(batch) == 8 and set(batch).issubset(train_ids) for batch in batches), "Training batch schedule differs")
    exposures = Counter(identifier for batch in batches for identifier in batch)
    require(sum(exposures.values()) == config["exposures_per_arm"], "Exposure count differs")
    logs = defaultdict(list)
    for row in read_rows(run_dir / "training_log.jsonl"):
        require(row["arm"] in ARMS, "Unknown training arm")
        logs[row["arm"]].append(row)
    training = read_json(run_dir / "training_summary.json")
    require(training == summary["training"] and set(training) == set(ARMS), "Training summary disagreement")
    before_hashes = set()
    for arm in ARMS:
        require([row["step"] for row in logs[arm]] == list(range(1, len(batches) + 1)), "Training log steps differ: " + arm)
        require(training[arm]["steps"] == len(batches) and training[arm]["exposures"] == sum(exposures.values()), "Arm exposure receipt differs")
        require(training[arm]["final_file_sha256"] == manifest["files_sha256"][arm + "_final.pt"], "Checkpoint SHA declarations disagree")
        for row in logs[arm]:
            require(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm_pre_clip"]) and row["encoder_gradient"]["finite"] is True, "Invalid training numerics")
            require(row["encoder_gradient"]["nonzero_grad_tensors"] > 0 if arm == "joint" else row["encoder_gradient"]["with_grad"] == 0, "Wrong encoder gradient receipt")
        update = training[arm]["encoder_update_audit"]
        before_hashes.add(update["before_sha256"])
        require((update["changed_parameter_tensors"] > 0) == (arm == "joint"), "Wrong encoder update receipt")
        require((update["before_sha256"] != update["after_sha256"]) == (arm == "joint"), "Encoder update hash receipt differs")
        require((training[arm]["final_encoder_tensor_sha256"] != source_models["initial_eo_sha256"]) == (arm == "joint"), "Encoder final state receipt differs")
        reload_report = summary["checkpoint_reload"][arm]
        require(reload_report["trained_state_roundtrip"]["logits_bit_exact"] is True and reload_report["trained_state_roundtrip"]["split"] == "train", "No trained-state checkpoint roundtrip receipt")
        require(reload_report["same_checkpoint_repeatability"]["logits_bit_exact"] is True, "Checkpoint repeatability failed")
    require(len(before_hashes) == 1 and summary["llm_frozen_tensor_hash_unchanged"] is True, "Initial encoders or LLM freeze receipts differ")
    require(summary["valid"] is True and summary["prediction_rows"] == len(raw_records), "Final count/validity disagrees")
    require(summary["donor_coverage"] == {"n": len(donors), "of": len(dev_ids)}, "Summary donor coverage differs")
    return {"schema": "oe1-independent-result-receipt-audit-v0", "valid_within_scope": True,
        "scope": "Local receipt integrity and independent response accounting; no tensor reload or regenerated inference",
        "verified_result_files_sha256": verified, "checkpoint_files_absent_locally": absent_checkpoints,
        "prepared_source_receipts_verified": source_receipts_checked,
        "prediction_rows": len(raw_records), "rescored": metrics, "paired_same_ids": paired, "per_group": grouped,
        "exposure_receipts": {"common_schedule_sha256": sha256(run_dir / "batches.json"), "steps_per_arm": len(batches), "exposures_per_arm": sum(exposures.values()), "unique_training_question_ids": len(exposures), "per_question_exposure_counts": dict(Counter(exposures.values())), "all_three_training_logs_match_step_count": True},
        "llm_frozen_and_checkpoint_reload_receipts_pass": True,
        "limitations": ["Declared common batch schedule is checked; training logs do not independently record item IDs for every arm.",
            "Model, raw-image, and original parquet tensors are not locally reloaded.",
            "Donor subset differs from all-dev; matched-ID comparisons are reported separately.",
            "Donor reuse creates dependence; no IID significance claim or new threshold is introduced.",
            "Development-only results do not establish pretraining novelty or spatial/numeric grounding."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.data_dir)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid_within_scope": result["valid_within_scope"], "prediction_rows": result["prediction_rows"], "rescored": result["rescored"], "checkpoint_files_absent_locally": result["checkpoint_files_absent_locally"]}, allow_nan=False))


if __name__ == "__main__":
    main()
