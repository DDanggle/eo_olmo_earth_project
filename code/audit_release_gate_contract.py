#!/usr/bin/env python3
"""Recompute release-migration gates with the preregistered frozen-threshold IoU.

This is an audit, not a replacement experiment. It reads an immutable migration
report and compares two summaries:

* registered: ``eval.iou_frozen_thr`` (threshold selected before target test)
* legacy: ``eval.iou_fp_matched`` (uses a target-test false-positive budget)

The output never overwrites the original report or its legacy gate summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


BRIDGE_ARMS = (
    "R2_mean_shift",
    "R3_procrustes",
    "R4_affine_ridge",
    "R5_spatial_stitch",
)
REFERENCE_ARM = "R0_old_reference"
IDENTITY_ARM = "R1_identity"


def _aggregate(report: dict[str, Any], metric: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in report["runs"]:
        value = run["eval"].get(metric)
        if value is None:
            raise ValueError(
                f"missing {metric}: region={run['region']} arm={run['arm']} seed={run.get('seed')}"
            )
        values[(run["region"], run["arm"])].append(float(value))
    return {key: fmean(group) for key, group in values.items()}


def _summarize(report: dict[str, Any], metric: str) -> dict[str, Any]:
    ap = _aggregate(report, "tie_ap")
    iou = _aggregate(report, metric)
    regions = sorted(region for region, arm in ap if arm == REFERENCE_ARM)
    if not regions:
        raise ValueError("report has no R0_old_reference runs")

    arms = [arm for arm in BRIDGE_ARMS if all((region, arm) in ap for region in regions)]
    gates: dict[str, Any] = {}
    for arm in arms:
        compatibility = []
        beats_identity = []
        for region in regions:
            ap_ok = abs(ap[(region, arm)] - ap[(region, REFERENCE_ARM)]) <= 0.02
            iou_ok = iou[(region, arm)] >= 0.95 * iou[(region, REFERENCE_ARM)]
            compatibility.append(bool(ap_ok and iou_ok))
            beats_identity.append(ap[(region, arm)] - ap[(region, IDENTITY_ARM)] >= 0.01)
        gates[arm] = {
            "compatibility_pass_count": sum(compatibility),
            "compatibility_required": 6,
            "compatibility_gate_pass": sum(compatibility) >= 6,
            "beats_identity_count": sum(beats_identity),
            "per_region_compatibility": dict(zip(regions, compatibility, strict=True)),
            "macro_ap": fmean(ap[(region, arm)] for region in regions),
            "macro_iou": fmean(iou[(region, arm)] for region in regions),
            "ap_retention": fmean(ap[(region, arm)] for region in regions)
            / fmean(ap[(region, REFERENCE_ARM)] for region in regions),
        }
    return {"metric": metric, "regions": regions, "gates": gates}


def audit(report_path: Path) -> dict[str, Any]:
    source_bytes = report_path.read_bytes()
    report = json.loads(source_bytes)
    return {
        "schema": "release-migration-gate-contract-audit-v1",
        "source_report": str(report_path),
        "source_report_bytes": report_path.stat().st_size,
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "new_cache": report.get("new_cache"),
        "n_runs": len(report["runs"]),
        "registered_summary": _summarize(report, "iou_frozen_thr"),
        "legacy_target_test_dependent_summary": _summarize(report, "iou_fp_matched"),
        "interpretation": (
            "Only registered_summary is eligible for the preregistered compatibility claim. "
            "The legacy summary is retained to quantify the effect of the implementation mismatch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {"audits": [audit(path) for path in args.reports]}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
