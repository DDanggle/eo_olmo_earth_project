#!/usr/bin/env python3
"""Validate existing T0/T1 extraction evidence without changing any artifacts.

Zero error is valid. None/NaN, absent audit samples, incomplete extraction and
summary/sample disagreement are not. This proves extraction evidence only,
not causal sampling, downstream validity or experiment approval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(report: dict, minimum_examples: int, tolerance: float = 0.05) -> list[str]:
    errors = []
    if minimum_examples < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive minimum_examples and finite positive tolerance required")
    n = report.get("n_ids")
    if type(n) is not int or n < 1:
        errors.append("n_ids must be a positive integer")
    if type(report.get("n_valid")) is not int or report.get("n_valid") != n:
        errors.append("n_valid must equal n_ids")
    if type(report.get("n_skipped")) is not int or report.get("n_skipped") != 0:
        errors.append("n_skipped must be integer zero")
    if report.get("all_gates_pass") is not True:
        errors.append("upstream extraction gate did not pass")

    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    summary = report.get("audit_max")
    if not numeric(summary):
        errors.append("audit_max must be finite and nonnegative (zero is valid)")
    elif summary >= tolerance:
        errors.append("audit_max exceeds the existing strict tolerance")
    if report.get("schema") == "olmo-temporal-cache-audit-v0":
        rows, key = report.get("audit_mean_vs_sealed"), "max_abs_diff_mean_vs_sealed"
    elif report.get("schema") == "olmo-streaming-dev-audit-v0":
        rows, key = report.get("audit"), "max_abs_diff_c12_vs_sealed"
    else:
        return errors + ["unrecognized audit schema"]
    if not isinstance(rows, list) or len(rows) < minimum_examples:
        return errors + [f"at least {minimum_examples} replay/comparison samples required"]
    ids, values = [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            errors.append("invalid audit sample id")
            continue
        ids.append(row["id"])
        value = row.get(key)
        if not numeric(value):
            errors.append(f"invalid comparison for {row['id']}")
        else:
            values.append(value)
    if len(ids) != len(set(ids)):
        errors.append("audit sample ids must be unique")
    if values and numeric(summary) and not math.isclose(max(values), summary, rel_tol=1e-9, abs_tol=1e-12):
        errors.append("audit_max does not match the recorded sample maximum")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-examples", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    errors = validate(report, args.minimum_examples, args.tolerance)
    summary = report.get("audit_max")
    # Keep rejected NaN/Inf diagnostics machine-readable rather than crashing the JSON writer.
    if type(summary) not in (int, float) or not math.isfinite(summary):
        summary = None
    print(json.dumps({"scope": "EXTRACTION_EVIDENCE_ONLY", "accepted": not errors,
                      "audit_max": summary, "errors": errors}, allow_nan=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
