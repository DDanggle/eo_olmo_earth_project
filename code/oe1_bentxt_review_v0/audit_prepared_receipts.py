#!/usr/bin/env python3
"""Audit downloaded OE1 metadata and QA receipts without loading image pixels."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def expected_donors(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input"], row["output"])].append(row)
    result = {}
    for row in rows:
        opposite = "no" if row["output"] == "yes" else "yes"
        candidates = [other for other in grouped[(row["input"], opposite)]
                      if other["patch_id"] != row["patch_id"]]
        if candidates:
            result[row["id"]] = min(other["id"] for other in candidates)
    return result


def score(golds, predictions):
    require(len(golds) == len(predictions), "Metric inputs have unequal lengths")
    require(all(gold in ("yes", "no") for gold in golds), "Unknown metric gold")
    support = Counter(golds)
    correct = Counter(gold for gold, prediction in zip(golds, predictions) if gold == prediction)
    recalls = {gold: correct[gold] / support[gold] for gold in ("yes", "no") if support[gold]}
    return {"n": len(golds), "support": dict(support),
            "accuracy": sum(correct.values()) / len(golds) if golds else None,
            "balanced_accuracy": sum(recalls.values()) / 2 if len(recalls) == 2 else None,
            "recall_by_label": recalls,
            "invalid_count": sum(prediction not in ("yes", "no") for prediction in predictions)}


def distance_km(first, second):
    first_lat, second_lat = map(math.radians, (first["latitude"], second["latitude"]))
    delta_lon = math.radians(first["longitude"] - second["longitude"])
    haversine = math.sin((first_lat - second_lat) / 2) ** 2
    haversine += math.cos(first_lat) * math.cos(second_lat) * math.sin(delta_lon / 2) ** 2
    return 12742.0176 * math.asin(math.sqrt(min(1, max(0, haversine))))


def majority_probe(train, dev, field):
    counts = defaultdict(Counter)
    for row in train:
        counts[row.get(field) or "unparsed"][row["output"]] += 1
    predictions = []
    for row in dev:
        observed = counts[row.get(field) or "unparsed"]
        predictions.append("yes" if observed["yes"] >= observed["no"] else "no")
    return {"tie_and_unseen_fallback": "yes", "train_only_fit": True,
            "dev": score([row["output"] for row in dev], predictions),
            "training_answer_counts": {key: dict(value) for key, value in sorted(counts.items())}}


def audit(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    require(manifest["status"] == "complete", "Preparation incomplete")
    rows = {split: read_rows(data_dir / (split + ".jsonl")) for split in ("train", "dev")}
    patches = {item["patch_id"]: item for item in manifest["patches"]}
    require(len(patches) == len(manifest["patches"]), "Duplicate manifest patch")
    all_ids = [row["id"] for split in rows.values() for row in split]
    require(len(all_ids) == len(set(all_ids)), "Duplicate QA ID")
    hashes = [item["image_array_sha256"] for item in patches.values()]
    require(len(hashes) == len(set(hashes)), "Repeated declared pixel array hash")
    split_report = {}
    for split, current in rows.items():
        expected_split = "train" if split == "train" else "validation"
        expected = manifest["splits"][split]
        require(sha256(data_dir / (split + ".jsonl")) == expected["question_file_sha256"], "QA SHA mismatch: " + split)
        patch_answers = defaultdict(list)
        for row in current:
            require(row["patch_id"] in patches, "Unknown QA patch")
            patch = patches[row["patch_id"]]
            require(row["split"] == patch["split"] == split, "Prepared split mismatch")
            require(row["official_split"] == patch["official_split"] == patch["geobench_split"] == expected_split, "Official split mismatch")
            require(row["bands"] == patch["bands"] == manifest["bands"], "Band order mismatch")
            require(row["date"] == row["timestamp_utc"] == patch["timestamp_utc"], "Date mismatch")
            require(row["mgrs"] == patch["mgrs"] and row["image_path"] == patch["image_path"], "Observation mapping mismatch")
            require(row["input"] == row["question"] and row["output"] == row["answer"] and row["id"] == row["question_id"], "Question aliases disagree")
            match = re.fullmatch(r"S2[AB]_MSIL2A_(\d{8}T\d{6})_N\d+_R\d+_(T\d{2}[A-Z]{3})_\d+_\d+", row["patch_id"])
            require(match is not None, "Invalid patch ID")
            timestamp = datetime.strptime(match[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            require(timestamp == row["date"] and match[2] == row["mgrs"], "Patch-derived date/tile mismatch")
            patch_answers[row["patch_id"]].append(row["output"])
        require(all(sorted(answers) == ["no", "yes"] for answers in patch_answers.values()), "Per-patch yes/no balance broken")
        require(len(current) == expected["questions"] and len(patch_answers) == expected["patches"], "Split counts differ")
        require(dict(Counter(row["output"] for row in current)) == expected["answers"], "Answer counts differ")
        class_counts = defaultdict(Counter)
        question_counts = defaultdict(Counter)
        for row in current:
            class_counts[row.get("query_class") or "unparsed"][row["output"]] += 1
            question_counts[row["input"]][row["output"]] += 1
        split_report[split] = {"questions": len(current), "patches": len(patch_answers),
            "mgrs_count": len({row["mgrs"] for row in current}),
            "answers": dict(Counter(row["output"] for row in current)),
            "class_answer_counts": {key: dict(value) for key, value in sorted(class_counts.items())},
            "exact_question_count": len(question_counts),
            "single_answer_exact_questions": sum(len(value) == 1 for value in question_counts.values()),
            "countries_by_patch": dict(Counter(patches[patch_id]["country"] for patch_id in patch_answers))}
    train_ids = {row["patch_id"] for row in rows["train"]}
    dev_ids = {row["patch_id"] for row in rows["dev"]}
    require(train_ids.isdisjoint(dev_ids) and train_ids | dev_ids == set(patches), "Patch partitions overlap or omit metadata")
    require({row["mgrs"] for row in rows["train"]}.isdisjoint(row["mgrs"] for row in rows["dev"]), "MGRS overlap")
    minimum = min(distance_km(patches[first], patches[second]) for first in train_ids for second in dev_ids)
    require(minimum >= 2 and math.isclose(minimum, manifest["cross_split_minimum_center_km"], abs_tol=1e-8), "Cross-split distance mismatch")
    donors = expected_donors(rows["dev"])
    dev_by_id = {row["id"]: row for row in rows["dev"]}
    return {"schema": "oe1-prepared-receipt-audit-v0", "valid_within_scope": True,
        "scope": "Metadata and QA receipts only; no original parquet, tortilla, or image pixels re-read",
        "source_hashes": {name: sha256(data_dir / name) for name in ("manifest.json", "train.jsonl", "dev.jsonl")},
        "splits": split_report, "minimum_center_km_recomputed": minimum,
        "unique_declared_image_hashes": len(hashes),
        "unparsed_classes": {split: sum(row.get("query_class") is None for row in current) for split, current in rows.items()},
        "train_fitted_question_only_probes": {field: majority_probe(rows["train"], rows["dev"], field) for field in ("input", "query_class")},
        "donors": {"eligible": len(donors), "of": len(rows["dev"]), "distinct_presented_patches": len({dev_by_id[donor]["patch_id"] for donor in donors.values()}),
            "source_answer_counts": dict(Counter(dev_by_id[source]["output"] for source in donors)),
            "presented_answer_counts": dict(Counter(dev_by_id[donor]["output"] for donor in donors.values())), "mapping": donors},
        "limitations": ["Original BENtxt answer correctness and full source SHA are not independently verified here.",
            "Patch-level answer balance does not imply question/class balance.",
            "Declared image uniqueness is not a local pixel rehash.",
            "Development split only; public pretraining exposure remains unknown."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid_within_scope": result["valid_within_scope"], "splits": {key: {name: value[name] for name in ("questions", "patches", "mgrs_count")} for key, value in result["splits"].items()}, "donors": {key: value for key, value in result["donors"].items() if key != "mapping"}, "probes": {key: value["dev"] for key, value in result["train_fitted_question_only_probes"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
