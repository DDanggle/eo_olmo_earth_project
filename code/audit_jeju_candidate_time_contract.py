#!/usr/bin/env python3
"""Audit whether historical Jeju candidates had a valid time contract.

This is a lineage audit, not a ground-truth relabeling. A candidate exposed to
an overlapping window or the season-confounded four-period path is ineligible
for an annual-change claim even when later RGB review happens to show a real
surface change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


WINDOWS = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026": ("2025-07-01", "2026-07-01"),
}
OVERLAP_TRANSITION = "2025->2026"
LEGACY_SOURCE = "v3_top"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_days(left: tuple[str, str], right: tuple[str, str]) -> int:
    left_start, left_end = (dt.date.fromisoformat(value) for value in left)
    right_start, right_end = (dt.date.fromisoformat(value) for value in right)
    return max(0, (min(left_end, right_end) - max(left_start, right_start)).days)


def build_audit(time_axis: dict[str, Any], candidates: dict[str, Any]) -> dict[str, Any]:
    rows = candidates.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate manifest must contain a non-empty candidates list")
    gates = time_axis.get("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("time-axis summary is missing gates")

    audited = []
    for row in rows:
        algorithm = row.get("algorithm", {})
        reasons = []
        if algorithm.get("when") == OVERLAP_TRANSITION:
            reasons.append("overlapping_2025_rolling_2026_windows")
        if algorithm.get("source") == LEGACY_SOURCE:
            reasons.append("season_confounded_four_period_source")
        audited.append(
            {
                "candidate_id": row.get("candidate_id"),
                "source": algorithm.get("source"),
                "transition": algorithm.get("when"),
                "eligible_for_annual_change_claim": not reasons,
                "contract_failure_reasons": reasons,
            }
        )

    overlap_count = sum(
        OVERLAP_TRANSITION == row["transition"] for row in audited
    )
    legacy_count = sum(LEGACY_SOURCE == row["source"] for row in audited)
    exposed_count = sum(bool(row["contract_failure_reasons"]) for row in audited)
    overlap = overlap_days(WINDOWS["2025"], WINDOWS["2026"])
    return {
        "schema": "jeju-candidate-time-contract-audit-v1",
        "status": "historical_candidates_invalid_for_annual_change_claim",
        "window_contract": {
            year: {"start": interval[0], "end": interval[1]}
            for year, interval in WINDOWS.items()
        },
        "overlap_2025_rolling_2026_days": overlap,
        "time_axis_gates": {
            "model_first4_season_aligned_across_years": gates.get(
                "model_first4_season_aligned_across_years"
            ),
            "all12_cover_same_calendar_month_set": gates.get(
                "all12_cover_same_calendar_month_set"
            ),
        },
        "candidate_summary": {
            "records": len(audited),
            "overlap_transition_records": overlap_count,
            "legacy_four_period_source_records": legacy_count,
            "union_contract_exposed_records": exposed_count,
            "not_exposed_by_these_two_checks": len(audited) - exposed_count,
        },
        "candidate_records": audited,
        "interpretation": {
            "allowed": "Nine records have invalid lineage for an annual-change claim.",
            "forbidden": "Do not infer that all nine are visual false positives; this audit is not ground truth.",
            "embedding_constraint": "The 768 output bands are fused feature dimensions, not a recoverable month axis. Correcting the time window requires encoder inference on a new valid input contract.",
            "next_gate": "No new candidate ranking until a non-overlapping, season-aligned input manifest passes before inference.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-axis-summary",
        type=Path,
        default=Path("artifacts/results/jeju_time_axis_summary.json"),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("artifacts/human_review_v1/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/jeju_candidate_time_contract_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_axis = json.loads(args.time_axis_summary.read_text())
    candidates = json.loads(args.candidate_manifest.read_text())
    result = build_audit(time_axis, candidates)
    result["evidence"] = {
        "time_axis_summary_sha256": file_sha256(args.time_axis_summary),
        "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["candidate_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

