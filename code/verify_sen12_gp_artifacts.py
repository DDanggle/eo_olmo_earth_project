#!/usr/bin/env python3
"""Verify a Sen12 G-P pilot bundle without trusting its headline JSON.

This verifier checks every recorded checkpoint/per-sample SHA-256 and recomputes
all thresholded aggregate metrics that can be recovered from the JSONL rows.
Exact pixel AP/ECE/Brier/NLL require pixel probabilities, so those are verified
by deterministic evaluation replay rather than falsely claiming JSONL coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_path(root: Path, value: str) -> Path:
    """Resolve a server-recorded absolute path inside a pulled result bundle."""
    path = Path(value)
    if path.is_file():
        return path
    for anchor in ("checkpoints", "per_sample"):
        if anchor in path.parts:
            candidate = root.joinpath(*path.parts[path.parts.index(anchor):])
            if candidate.is_file():
                return candidate
    return path


def close(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return abs(actual - expected) <= tolerance


def verify(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root = summary_path.parent
    failures: list[str] = []
    checked: dict[str, dict] = {}

    for arm, result in summary["arms"].items():
        arm_checked: dict[str, object] = {}
        checkpoint = recorded_path(root, result["checkpoint"]["path"])
        if not checkpoint.is_file():
            failures.append(f"{arm}: checkpoint missing: {checkpoint}")
        elif sha256_file(checkpoint) != result["checkpoint"]["sha256"]:
            failures.append(f"{arm}: checkpoint SHA-256 mismatch")
        else:
            arm_checked["checkpoint_sha256"] = "pass"

        for split in ("val", "test"):
            record = result["per_sample"][split]
            rows_path = recorded_path(root, record["path"])
            if not rows_path.is_file():
                failures.append(f"{arm}/{split}: per-sample file missing: {rows_path}")
                continue
            if sha256_file(rows_path) != record["sha256"]:
                failures.append(f"{arm}/{split}: per-sample SHA-256 mismatch")
                continue
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            if len(rows) != record["rows"]:
                failures.append(f"{arm}/{split}: row count {len(rows)} != {record['rows']}")

            tp = sum(row["tp"] for row in rows)
            fp = sum(row["fp"] for row in rows)
            fn = sum(row["fn"] for row in rows)
            pixels = len(rows) * 128 * 128
            positive = sum(row["mask_positive_pixels"] for row in rows)
            positive_rows = [row for row in rows if row["mask_positive_pixels"] > 0]
            ld_rows = [row for row in rows if row["mask_positive_pixels"] > 50]
            ld_tp = sum(row["tp"] for row in ld_rows)
            ld_fp = sum(row["fp"] for row in ld_rows)
            ld_fn = sum(row["fn"] for row in ld_rows)
            recomputed = {
                "iou": tp / max(tp + fp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "positive_pixel_frac": positive / pixels,
                "positive_patch_macro_iou": (
                    sum(row["iou_at_0_5"] for row in positive_rows) / len(positive_rows)
                    if positive_rows else None
                ),
                "ld_iou": ld_tp / max(ld_tp + ld_fp + ld_fn, 1),
                "ld_f1": 2 * ld_tp / max(2 * ld_tp + ld_fp + ld_fn, 1),
            }
            reported = result[split]
            for key, actual in recomputed.items():
                # ld_* is serialized to 5 decimals; the remaining aggregate
                # metrics are serialized to 6 (positive prevalence to 8).
                tolerance = 5.1e-6 if key.startswith("ld_") else 1.1e-6
                if actual is not None and not close(actual, reported[key], tolerance):
                    failures.append(
                        f"{arm}/{split}: {key} recomputed={actual:.9f} reported={reported[key]}"
                    )
            confusion = reported["confusion_pixels"]
            for key, actual in (("tp", tp), ("fp", fp), ("fn", fn)):
                if actual != confusion[key]:
                    failures.append(
                        f"{arm}/{split}: {key} sum={actual} reported={confusion[key]}"
                    )
            if len(positive_rows) != reported["positive_patch_n"]:
                failures.append(f"{arm}/{split}: positive_patch_n mismatch")
            if len(ld_rows) != reported["ld_subset_n"]:
                failures.append(f"{arm}/{split}: ld_subset_n mismatch")
            arm_checked[split] = {
                "per_sample_sha256": "pass",
                "rows": len(rows),
                "thresholded_aggregates_recomputed": "pass",
            }
        checked[arm] = arm_checked

    return {
        "schema": "sen12-gp-artifact-verification-v1",
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "all_checks_pass": not failures,
        "failures": failures,
        "checked": checked,
        "not_covered_without_pixel_scores": [
            "auprc_exact",
            "ece_15bin_pixel_micro",
            "brier_pixel_micro",
            "nll_pixel_micro",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.summary)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
