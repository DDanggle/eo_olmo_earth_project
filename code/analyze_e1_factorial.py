#!/usr/bin/env python3
"""Analyze the sealed E1 2x2 context × decoder experiment without ad-hoc arithmetic.

All four cells must have identical runner code SHA and paired sample IDs. Pixel-micro IoU contrasts are
recomputed from per-sample confusion counts and receive paired spatial-block bootstrap sensitivity intervals.
AUPRC and other non-decomposable metrics are read from the sealed pilot JSON and reported descriptively.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CELLS = {
    "y00": ("tiled_small", "P4"),
    "y01": ("tiled_big", "P4c"),
    "y10": ("full_small", "P4"),
    "y11": ("full_big", "P4c"),
}
BLOCK_KM = [2.56, 5.12, 10.24, 20.48]
N_BOOT = 10_000
SEED = 20260826
P2_IOU = 0.159254
P2_AUPRC = 0.174585


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("/home/work/data/olmoearth/e1_factorial_v2"),
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/home/work/data/sen12landslides/extracted"),
    )
    parser.add_argument("--fold", default="holdout_chimanimani")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def micro_iou(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, selected=None) -> float:
    if selected is not None:
        tp, fp, fn = tp[selected], fp[selected], fn[selected]
    denominator = tp.sum() + fp.sum() + fn.sum()
    return float(tp.sum() / denominator) if denominator else float("nan")


def contrasts(values: dict[str, float]) -> dict[str, float]:
    y00, y01, y10, y11 = (values[key] for key in ("y00", "y01", "y10", "y11"))
    c_small = y10 - y00
    c_large = y11 - y01
    d_tiled = y01 - y00
    d_full = y11 - y10
    return {
        "C_small": c_small,
        "C_large": c_large,
        "C_context_mean": (c_small + c_large) / 2,
        "D_tiled": d_tiled,
        "D_full": d_full,
        "D_decoder_mean": (d_tiled + d_full) / 2,
        "I_interaction": y11 - y10 - y01 + y00,
    }


def main() -> None:
    import xarray as xr

    args = parse_args()
    cell_rows = {}
    cell_pilots = {}
    input_files = {}
    code_hashes = set()
    for cell, (directory, arm) in CELLS.items():
        pilot_path = args.root / directory / f"{args.fold}_pilot.json"
        rows_path = args.root / directory / "per_sample" / args.fold / f"{arm}_test.jsonl"
        if not pilot_path.is_file() or not rows_path.is_file():
            raise SystemExit(f"cell {cell} 산출물 없음: {pilot_path} / {rows_path}")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        cell_pilots[cell] = pilot
        cell_rows[cell] = {row["sample_id"]: row for row in rows}
        input_files[str(pilot_path.relative_to(args.root))] = sha256_file(pilot_path)
        input_files[str(rows_path.relative_to(args.root))] = sha256_file(rows_path)
        code_hashes.add(pilot["code_sha256"])
    if len(code_hashes) != 1:
        raise SystemExit(f"네 cell runner code SHA 불일치: {sorted(code_hashes)}")

    sample_ids = sorted(cell_rows["y00"])
    for cell in CELLS:
        if sorted(cell_rows[cell]) != sample_ids:
            raise SystemExit(f"cell {cell} sample ID 불일치")

    arrays = {}
    cells = {}
    for cell, (_, arm) in CELLS.items():
        rows = cell_rows[cell]
        tp = np.array([rows[sample_id]["tp"] for sample_id in sample_ids], dtype=float)
        fp = np.array([rows[sample_id]["fp"] for sample_id in sample_ids], dtype=float)
        fn = np.array([rows[sample_id]["fn"] for sample_id in sample_ids], dtype=float)
        arrays[cell] = (tp, fp, fn)
        reported = cell_pilots[cell]["arms"][arm]["test"]
        recomputed = micro_iou(tp, fp, fn)
        if abs(recomputed - reported["iou"]) > 5e-7:
            raise SystemExit(f"cell {cell} IoU 재계산 불일치: {recomputed} vs {reported['iou']}")
        cells[cell] = {
            "cache": "tiled_4x64" if cell[1] == "0" else "full_1x128",
            "decoder": "small" if cell in {"y00", "y10"} else "large_convolutional",
            "arm": arm,
            "trainable_params": cell_pilots[cell]["arms"][arm]["trainable_params"],
            "fit_plus_epoch_val_seconds": cell_pilots[cell]["arms"][arm][
                "fit_plus_epoch_val_seconds"
            ],
            "best_val_epoch": cell_pilots[cell]["arms"][arm]["best_val_epoch"],
            "test": reported,
            "iou_recomputed": round(recomputed, 6),
        }

    observed = contrasts({cell: cells[cell]["iou_recomputed"] for cell in CELLS})

    center_x, center_y = [], []
    for sample_id in sample_ids:
        with xr.open_dataset(args.data_root / f"{sample_id}.nc", decode_times=False, cache=False) as ds:
            center_x.append(float(ds["x"].values.mean()))
            center_y.append(float(ds["y"].values.mean()))
    center_x, center_y = np.asarray(center_x), np.asarray(center_y)

    intervals = {}
    for block_km in BLOCK_KM:
        block_m = block_km * 1000
        block_key = (
            np.floor(center_x / block_m).astype(np.int64) * 1_000_003
            + np.floor(center_y / block_m).astype(np.int64)
        )
        unique_blocks, inverse = np.unique(block_key, return_inverse=True)
        indices_by_block = [np.where(inverse == index)[0] for index in range(len(unique_blocks))]
        rng = np.random.default_rng(SEED)
        draws = {name: np.empty(N_BOOT, dtype=float) for name in observed}
        for draw_index in range(N_BOOT):
            chosen_blocks = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
            selected = np.concatenate([indices_by_block[index] for index in chosen_blocks])
            cell_values = {
                cell: micro_iou(*arrays[cell], selected=selected) for cell in CELLS
            }
            draw_contrasts = contrasts(cell_values)
            for name, value in draw_contrasts.items():
                draws[name][draw_index] = value
        intervals[f"{block_km}km"] = {
            "n_blocks": len(unique_blocks),
            "contrasts": {
                name: {
                    "ci95_percentile": [
                        round(float(np.percentile(values, 2.5)), 6),
                        round(float(np.percentile(values, 97.5)), 6),
                    ],
                    "bootstrap_tail_fraction_le_0": round(float((values <= 0).mean()), 6),
                }
                for name, values in draws.items()
            },
        }

    def excludes_zero(block: dict, name: str) -> bool:
        low, high = block["contrasts"][name]["ci95_percentile"]
        return low > 0 or high < 0

    context_support_count = sum(
        excludes_zero(block, "C_context_mean") for block in intervals.values()
    )
    decoder_support_count = sum(
        excludes_zero(block, "D_decoder_mean") for block in intervals.values()
    )
    decisions = {
        "context_supported": bool(
            observed["C_small"] > 0 and observed["C_large"] > 0
            and context_support_count >= 3
        ),
        "context_ci_excludes_zero_n_of_4": context_support_count,
        "capacity_supported": bool(
            observed["D_tiled"] > 0 and observed["D_full"] > 0
            and decoder_support_count >= 3
        ),
        "capacity_ci_excludes_zero_n_of_4": decoder_support_count,
        "exploratory_parity_y11": bool(
            cells["y11"]["test"]["iou"] >= 0.95 * P2_IOU
            and cells["y11"]["test"]["auprc_exact"] >= 0.95 * P2_AUPRC
        ),
        "parity_thresholds": {
            "iou": round(0.95 * P2_IOU, 8),
            "auprc": round(0.95 * P2_AUPRC, 8),
        },
    }
    result = {
        "schema": "e1-context-decoder-factorial-v1",
        "evidence_status": "development_only_not_confirmatory",
        "analysis_contract": "docs/E1_CONTEXT_DECODER_ANALYSIS_PLAN.md",
        "runner_code_sha256": next(iter(code_hashes)),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "input_files_sha256": input_files,
        "n_paired_test_tiles": len(sample_ids),
        "cells": cells,
        "observed_iou_contrasts": {name: round(value, 6) for name, value in observed.items()},
        "paired_spatial_block_sensitivity": {
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "blocks": intervals,
            "warning": "bootstrap tail fractions are not formal p-values; one exposed region only",
        },
        "decisions": decisions,
        "limitations": [
            "Chimanimani test was already exposed and all results are development-only.",
            "One seed and one region cannot establish regional or optimization generalization.",
            "Large decoder is not a U-Net and does not use intermediate encoder features.",
            "The factorial does not identify a 40 m support or representation-adaptation effect.",
        ],
    }
    output = args.root / "e1_factorial_analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
