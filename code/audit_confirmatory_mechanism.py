#!/usr/bin/env python3
"""Audit *why* P4 beat raw baselines in the completed eight-region release.

This is a CPU-only, read-only analysis of the sealed per-sample JSONL files.  It
does not train a model and it does not alter the confirmatory outputs.

The audit intentionally stays at the stored 0.5 operating point.  It measures:

* positive-tile macro IoU for P2/P3/P4;
* false-positive pixels on empty tiles;
* paired P4-vs-P2 tile wins and an oracle upper bound.

The oracle uses target labels and therefore is *not* a deployable router.  It is
only a necessary-condition screen for a future context/detail fusion method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


ARMS = ("P2", "P3", "P4")
SEEDS = (1, 2, 3)
FUSION_MIN_HEADROOM = 0.02
FUSION_MIN_REGIONS = 4
FP_MIN_REGIONS = 6
FP_MIN_MEDIAN_RATIO = 2.0


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}:{line_no}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty per-sample file: {path}")
    return rows


def find_file(region_dir: Path, arm: str, seed: int) -> Path:
    candidates = sorted(
        (region_dir / f"{arm}_seed{seed}" / "per_sample").glob(
            f"*/{arm}_test.jsonl"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {arm} seed {seed} test JSONL in {region_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def finite_float(value: object, *, field: str, sample_id: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field} for {sample_id}: {value!r}")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_region(region_dir: Path) -> tuple[dict, list[dict]]:
    records: dict[tuple[str, int], dict[str, dict]] = {}
    sources: list[dict] = []
    reference_ids: set[str] | None = None

    for arm in ARMS:
        for seed in SEEDS:
            path = find_file(region_dir, arm, seed)
            rows = read_jsonl(path)
            sample_ids = set(rows)
            if reference_ids is None:
                reference_ids = sample_ids
            elif sample_ids != reference_ids:
                raise ValueError(
                    f"sample-set mismatch in {region_dir.name}/{arm}_seed{seed}"
                )
            records[(arm, seed)] = rows
            sources.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(rows),
                }
            )

    assert reference_ids is not None
    sample_ids = sorted(reference_ids)
    positives = [
        sid
        for sid in sample_ids
        if int(records[("P2", 1)][sid]["mask_positive_pixels"]) > 0
    ]
    negatives = [sid for sid in sample_ids if sid not in set(positives)]
    if not positives or not negatives:
        raise ValueError(f"{region_dir.name} needs both positive and empty tiles")

    for sid in sample_ids:
        label_counts = {
            int(records[(arm, seed)][sid]["mask_positive_pixels"])
            for arm in ARMS
            for seed in SEEDS
        }
        if len(label_counts) != 1:
            raise ValueError(f"label mismatch for {region_dir.name}/{sid}")

    arm_summary: dict[str, dict] = {}
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            rows = records[(arm, seed)]
            pos_iou = [
                finite_float(rows[sid]["iou_at_0_5"], field="iou_at_0_5", sample_id=sid)
                for sid in positives
            ]
            empty_fp = [int(rows[sid]["fp"]) for sid in negatives]
            per_seed.append(
                {
                    "seed": seed,
                    "positive_tile_macro_iou": mean(pos_iou),
                    "empty_tile_fp_pixels": sum(empty_fp),
                    "empty_tiles_with_any_fp_fraction": sum(v > 0 for v in empty_fp)
                    / len(empty_fp),
                }
            )
        arm_summary[arm] = {
            "per_seed": per_seed,
            "seed_mean_positive_tile_macro_iou": mean(
                row["positive_tile_macro_iou"] for row in per_seed
            ),
            "seed_median_empty_tile_fp_pixels": median(
                row["empty_tile_fp_pixels"] for row in per_seed
            ),
        }

    p2_iou = {
        sid: mean(
            finite_float(
                records[("P2", seed)][sid]["iou_at_0_5"],
                field="iou_at_0_5",
                sample_id=sid,
            )
            for seed in SEEDS
        )
        for sid in positives
    }
    p4_iou = {
        sid: mean(
            finite_float(
                records[("P4", seed)][sid]["iou_at_0_5"],
                field="iou_at_0_5",
                sample_id=sid,
            )
            for seed in SEEDS
        )
        for sid in positives
    }
    differences = {sid: p4_iou[sid] - p2_iou[sid] for sid in positives}
    p2_mean = mean(p2_iou.values())
    p4_mean = mean(p4_iou.values())
    oracle = mean(max(p2_iou[sid], p4_iou[sid]) for sid in positives)
    best_single = max(p2_mean, p4_mean)
    p4_fp = arm_summary["P4"]["seed_median_empty_tile_fp_pixels"]
    p2_fp = arm_summary["P2"]["seed_median_empty_tile_fp_pixels"]

    return (
        {
            "region": region_dir.name,
            "n_tiles": len(sample_ids),
            "n_positive_tiles": len(positives),
            "n_empty_tiles": len(negatives),
            "arms": arm_summary,
            "p4_vs_p2": {
                "positive_tile_win_fraction": sum(v > 0 for v in differences.values())
                / len(differences),
                "positive_tile_tie_fraction": sum(v == 0 for v in differences.values())
                / len(differences),
                "positive_tile_loss_fraction": sum(v < 0 for v in differences.values())
                / len(differences),
                "mean_absolute_tile_iou_disagreement": mean(
                    abs(v) for v in differences.values()
                ),
                "oracle_positive_tile_macro_iou": oracle,
                "best_single_positive_tile_macro_iou": best_single,
                "oracle_headroom": oracle - best_single,
                "empty_tile_fp_ratio_p2_over_p4": (
                    p2_fp / p4_fp if p4_fp > 0 else None
                ),
                "p4_has_lower_empty_tile_fp": p4_fp < p2_fp,
            },
        },
        sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    region_dirs = sorted(path for path in args.root.glob("holdout_*") if path.is_dir())
    if not region_dirs:
        raise SystemExit(f"no holdout_* directories under {args.root}")

    regions = []
    sources = []
    for region_dir in region_dirs:
        region, region_sources = audit_region(region_dir)
        regions.append(region)
        sources.extend(region_sources)

    headrooms = [r["p4_vs_p2"]["oracle_headroom"] for r in regions]
    fp_ratios = [
        r["p4_vs_p2"]["empty_tile_fp_ratio_p2_over_p4"]
        for r in regions
        if r["p4_vs_p2"]["empty_tile_fp_ratio_p2_over_p4"] is not None
    ]
    fusion_regions = sum(value >= FUSION_MIN_HEADROOM for value in headrooms)
    fp_lower_regions = sum(
        r["p4_vs_p2"]["p4_has_lower_empty_tile_fp"] for r in regions
    )
    aggregate_headroom = mean(headrooms)
    median_fp_ratio = median(fp_ratios) if fp_ratios else None

    result = {
        "schema": "confirmatory-mechanism-audit-v1",
        "status": "DESCRIPTIVE_EXISTING_OUTPUTS_ONLY",
        "operating_point": 0.5,
        "n_regions": len(regions),
        "regions": regions,
        "aggregate": {
            "region_macro_oracle_headroom": aggregate_headroom,
            "regions_with_oracle_headroom_ge_0_02": fusion_regions,
            "regions_where_p4_has_lower_empty_tile_fp": fp_lower_regions,
            "median_empty_tile_fp_ratio_p2_over_p4": median_fp_ratio,
        },
        "preregistered_screen": {
            "fusion_candidate": {
                "rule": (
                    "region-macro oracle headroom >= 0.02 and at least 4 regions "
                    "have oracle headroom >= 0.02"
                ),
                "pass": aggregate_headroom >= FUSION_MIN_HEADROOM
                and fusion_regions >= FUSION_MIN_REGIONS,
            },
            "false_alarm_mechanism_candidate": {
                "rule": (
                    "P4 has lower empty-tile FP in at least 6 regions and median "
                    "P2/P4 FP ratio >= 2 at threshold 0.5"
                ),
                "pass": fp_lower_regions >= FP_MIN_REGIONS
                and median_fp_ratio is not None
                and median_fp_ratio >= FP_MIN_MEDIAN_RATIO,
            },
        },
        "claim_boundary": [
            "This audit reuses already-open confirmatory predictions; it is not a new confirmatory experiment.",
            "The tile oracle uses target labels and cannot be reported as a deployable router.",
            "False-positive ratios are measured at threshold 0.5; a causal or operational claim requires FP-budget-matched curves.",
            "A passed fusion screen is necessary but not sufficient; it only authorizes a development prototype.",
        ],
        "source_files": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_regions": len(regions),
                "aggregate": result["aggregate"],
                "screens": result["preregistered_screen"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
