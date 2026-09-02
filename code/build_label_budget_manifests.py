#!/usr/bin/env python3
"""Seal nested source-label subsets without training or reading target metrics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


FOLDS = (
    "holdout_hiroshima",
    "holdout_hokkaido",
    "holdout_indonesia",
    "holdout_itogon",
    "holdout_kyrgyzstan1",
    "holdout_kyrgyzstan2",
    "holdout_newzealand",
    "holdout_thrissur",
)
FRACTIONS = (0.01, 0.05, 0.10)
SUBSET_SEEDS = (20260827, 20260828, 20260829)
BATCH_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> tuple[str, str]:
    raw = f"{seed}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), sample_id


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            records[row["sample_id"]] = row
    fold_rows = {row["fold"]: row for row in json.loads(
        args.folds.read_text(encoding="utf-8"))["folds"]}

    output_folds: dict[str, dict] = {}
    all_actual_train_counts = []
    all_one_percent_counts = []
    all_one_percent_positive = []

    for fold_name in FOLDS:
        fold = fold_rows[fold_name]
        mask_dir = args.cache_root / fold_name / "mask_u8"
        if not mask_dir.is_dir():
            raise SystemExit(f"mask cache missing: {mask_dir}")
        train_regions = set(fold["train_regions"])
        train_rows = [
            row for row in records.values()
            if row["region"] in train_regions
            and not row.get("error")
            and row.get("s15_eligible", True)
            and (mask_dir / f"{row['sample_id']}.npy").is_file()
        ]
        train_rows.sort(key=lambda row: row["sample_id"])
        all_actual_train_counts.append(len(train_rows))
        by_stratum: dict[tuple[str, bool], list[str]] = {}
        for row in train_rows:
            key = (row["region"], bool(row["label_positive"]))
            by_stratum.setdefault(key, []).append(row["sample_id"])
        expected_strata = {(region, positive) for region in train_regions for positive in (False, True)}
        if set(by_stratum) != expected_strata:
            missing = sorted(expected_strata - set(by_stratum))
            raise SystemExit(f"empty train strata in {fold_name}: {missing}")

        seed_payloads: dict[str, dict] = {}
        for subset_seed in SUBSET_SEEDS:
            ordered = {
                key: sorted(ids, key=lambda sample_id: stable_key(subset_seed, sample_id))
                for key, ids in by_stratum.items()
            }
            fraction_payloads: dict[str, dict] = {}
            previous: set[str] = set()
            for fraction in FRACTIONS:
                selected: list[str] = []
                for key in sorted(ordered):
                    ids = ordered[key]
                    selected.extend(ids[: math.ceil(fraction * len(ids))])
                selected = sorted(selected)
                selected_set = set(selected)
                if not previous.issubset(selected_set):
                    raise SystemExit(f"non-nested subset: {fold_name} seed={subset_seed}")
                previous = selected_set
                positive_count = sum(bool(records[sid]["label_positive"]) for sid in selected)
                label_pixels = sum(int(records[sid].get("mask_positive_pixels", 0)) for sid in selected)
                fraction_payloads[f"{fraction:.2f}"] = {
                    "fraction": fraction,
                    "sample_ids": selected,
                    "sample_ids_sha256": canonical_sha256(selected),
                    "labeled_tiles": len(selected),
                    "positive_tiles": positive_count,
                    "positive_pixels": label_pixels,
                    "batch_size": BATCH_SIZE,
                    "batches_drop_last_true": len(selected) // BATCH_SIZE,
                    "examples_dropped_per_epoch_if_drop_last_true": len(selected) % BATCH_SIZE,
                    "batches_drop_last_false": math.ceil(len(selected) / BATCH_SIZE),
                }
                if fraction == 0.01:
                    all_one_percent_counts.append(len(selected))
                    all_one_percent_positive.append(positive_count)
            seed_payloads[str(subset_seed)] = fraction_payloads

        cache_ids = sorted(row["sample_id"] for row in train_rows)
        output_folds[fold_name] = {
            "train_regions": fold["train_regions"],
            "val_region": fold["val_region"],
            "test_region": fold["test_region"],
            "actual_cached_train_tiles": len(train_rows),
            "actual_cached_positive_train_tiles": sum(
                bool(row["label_positive"]) for row in train_rows
            ),
            "actual_cached_train_ids_sha256": canonical_sha256(cache_ids),
            "subsets": seed_payloads,
        }

    payload = {
        "schema": "source-label-budget-subsets-v1",
        "status": "SUBSET_IDS_SEALED_NO_MODEL_METRICS_READ",
        "selection_rule": (
            "Within each source-region x label-positive stratum, sort by "
            "SHA256(str(subset_seed) || NUL || sample_id) and take ceil(fraction*n) prefix."
        ),
        "source": {
            "contract": str(args.contract),
            "contract_sha256": sha256_file(args.contract),
            "folds": str(args.folds),
            "folds_sha256": sha256_file(args.folds),
            "cache_root": str(args.cache_root),
        },
        "fractions": FRACTIONS,
        "subset_seeds": SUBSET_SEEDS,
        "folds": output_folds,
        "preflight_summary": {
            "actual_cached_train_tiles_range": [min(all_actual_train_counts), max(all_actual_train_counts)],
            "one_percent_labeled_tiles_range": [min(all_one_percent_counts), max(all_one_percent_counts)],
            "one_percent_positive_tiles_range": [
                min(all_one_percent_positive), max(all_one_percent_positive)
            ],
            "zero_batch_fold_seed_pairs": 0,
            "target_region_labels_in_subsets": 0,
        },
    }
    payload["content_sha256_without_this_field"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.compact_output is not None:
        compact = copy.deepcopy(payload)
        compact["status"] = "COMPACT_SUBSET_SEAL_NO_MODEL_METRICS_READ"
        compact["full_manifest"] = {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "note": "Full sample-ID manifest is server-resident; this compact seal plus the generator and source contracts reproduce it."
        }
        for fold in compact["folds"].values():
            for seed in fold["subsets"].values():
                for fraction in seed.values():
                    fraction.pop("sample_ids")
        compact.pop("content_sha256_without_this_field", None)
        compact["content_sha256_without_this_field"] = canonical_sha256(compact)
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload["preflight_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
