#!/usr/bin/env python3
"""Build the first executable, audit-only K-EvidenceShift Jeju pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kearth_benchmark.pilot import PilotInputs, build_pilot, write_pilot


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/k_evidence_shift_jeju_pilot_v0.json")
    parser.add_argument("--candidate-manifest", type=Path, default=ROOT / "artifacts/human_review_v1/manifest.json")
    parser.add_argument("--assistant-review", type=Path, default=ROOT / "artifacts/human_review_v1/assistant_review.json")
    parser.add_argument(
        "--candidate-evidence",
        type=Path,
        default=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/candidate_evidence.json",
    )
    parser.add_argument(
        "--observation-context",
        type=Path,
        default=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/observation_context.json",
    )
    parser.add_argument(
        "--api-run-summary",
        type=Path,
        default=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/run_summary.json",
    )
    parser.add_argument(
        "--api-requests",
        type=Path,
        default=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/requests.json",
    )
    parser.add_argument(
        "--api-complete-marker",
        type=Path,
        default=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/COMPLETE.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/benchmarks/k_evidence_shift_jeju_pilot_v0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = PilotInputs(
        config=args.config,
        candidate_manifest=args.candidate_manifest,
        assistant_review=args.assistant_review,
        candidate_evidence=args.candidate_evidence,
        observation_context=args.observation_context,
        api_run_summary=args.api_run_summary,
        api_requests=args.api_requests,
        api_complete_marker=args.api_complete_marker,
    )
    result = write_pilot(build_pilot(inputs), args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
