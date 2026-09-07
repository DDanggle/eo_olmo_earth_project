#!/usr/bin/env python3
"""Fail-closed summary for the preregistered token-resolution screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FOLDS = ("hiroshima", "hokkaido", "indonesia", "itogon", "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur")


def read_arm(root: Path, arm: str) -> dict[str, float]:
    out = {}
    for fold in FOLDS:
        path = root / arm / f"holdout_{fold}_seed1.json"
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        obj = json.loads(path.read_text())
        if obj.get("seed") != 1 or obj.get("fold") != f"holdout_{fold}":
            raise SystemExit(f"identity mismatch: {path}")
        value = obj.get("test", {}).get("positive_patch_macro_iou")
        if value is None or not np.isfinite(value):
            raise SystemExit(f"invalid primary metric: {path}")
        out[fold] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--p4-summary", type=Path, required=True, help="sealed confirmatory_8region_summary.json (three-seed P4 means)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sealed = json.loads(args.p4_summary.read_text())
    by_fold = {row["fold"].removeprefix("holdout_"): float(row["primary_mean"]["reuse"]) for row in sealed["regions"]}
    missing = sorted(set(FOLDS) - set(by_fold))
    if missing:
        raise SystemExit(f"sealed P4 summary missing folds: {missing}")
    p4 = {fold: by_fold[fold] for fold in FOLDS}
    arms = {name: read_arm(args.root, name) for name in ("p4_native_control", "p4_upsample2", "p2_native", "p2_avgpool2")}
    means = {"p4_native": float(np.mean(list(p4.values())))}
    means.update({k: float(np.mean(list(v.values()))) for k, v in arms.items()})
    deltas = {
        "registered_p2_minus_p4": means["p2_native"] - means["p4_native"],
        "dense_representation_effect": means["p2_native"] - means["p4_upsample2"],
        "fine_grid_information_effect": means["p2_native"] - means["p2_avgpool2"],
        "same_trainer_calibration_gap": means["p4_native_control"] - means["p4_native"],
        "decoder_grid_artifact": means["p4_upsample2"] - means["p4_native_control"],
    }
    wins = sum(arms["p2_native"][f] > p4[f] for f in FOLDS)
    registered = deltas["registered_p2_minus_p4"] >= 0.03 and wins >= 6
    report = {
        "schema": "resolution-contract-summary-v1",
        "status": "DEVELOPMENT_SCREEN_ONLY",
        "p4_reference": str(args.p4_summary),
        "folds": list(FOLDS),
        "means": means,
        "deltas": deltas,
        "p2_native_wins_vs_p4_native": wins,
        "registered_gate_pass": registered,
        "interpretation_gate": {
            "same_trainer_calibration_pass": abs(deltas["same_trainer_calibration_gap"]) <= 0.03,
            "token_resolution_lever": registered and abs(deltas["same_trainer_calibration_gap"]) <= 0.03 and deltas["dense_representation_effect"] > 0.0 and deltas["fine_grid_information_effect"] > 0.0,
            "warning": "Sen12 folds were previously exposed; external task/Korea preregistration is required for confirmation.",
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    (args.out or (args.root / "summary.json")).write_text(text + "\n")


if __name__ == "__main__":
    main()
