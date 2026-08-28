#!/usr/bin/env python3
"""Validate and aggregate the frozen eight-region confirmatory release.

This program intentionally consumes the small, human-readable region summaries and
the post-run release manifests.  It does not re-open prediction maps or recompute
metrics; those belong to the per-region reader.  Its job is to make accidental
mixing of folds, recipes, failed gates, or incomplete releases impossible when the
preregistered region-macro headline is calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


REGION_ORDER = [
    "holdout_thrissur",
    "holdout_hiroshima",
    "holdout_hokkaido",
    "holdout_indonesia",
    "holdout_itogon",
    "holdout_kyrgyzstan1",
    "holdout_kyrgyzstan2",
    "holdout_newzealand",
]
RECIPE_SHA256 = "95becb32ab2df2c73537a4d19550dfd2c93d426671c15703e59cf4d8d44d2f5a"
ARM_KEYS = ("reuse", "raw_strong", "raw_efficient")
EXPECTED_ARM_IDS = {"reuse": "P4", "raw_strong": "P2", "raw_efficient": "P3"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


def find_sources(repo: Path, recovered: Path | None, fold: str) -> tuple[Path, Path]:
    summary_candidates = [repo / "evidence" / "confirmatory" / fold / "read_summary.json"]
    post_candidates = [repo / "evidence" / "confirmatory_manifests" / f"{fold}_post.json"]
    if recovered is not None:
        summary_candidates.append(recovered / "summaries" / fold / "read_summary.json")
        post_candidates.append(recovered / "confirmatory_manifests" / f"{fold}_post.json")
    summary = next((p for p in summary_candidates if p.is_file()), None)
    post = next((p for p in post_candidates if p.is_file()), None)
    if summary is None or post is None:
        raise FileNotFoundError(
            f"missing release source for {fold}: summary={summary_candidates}, post={post_candidates}"
        )
    return summary, post


def close(a: float, b: float, tol: float = 2e-6) -> bool:
    # Region readers persist six decimals; two independently rounded operands can
    # therefore differ from their persisted difference by one unit in the last place.
    return abs(a - b) <= tol


def validate_region(fold: str, summary: dict[str, Any], post: dict[str, Any]) -> None:
    if summary.get("schema") != "confirmatory-region-read-v1":
        raise ValueError(f"{fold}: unexpected summary schema")
    if summary.get("fold") != fold or summary.get("gate_verdict") != "PASS":
        raise ValueError(f"{fold}: summary fold/gate mismatch")
    if post.get("schema") != "confirmatory-release-gate-v2":
        raise ValueError(f"{fold}: unexpected post schema")
    if post.get("fold") != fold or post.get("verdict") != "PASS" or post.get("failed_checks"):
        raise ValueError(f"{fold}: post-release gate did not pass cleanly")

    recipe = post.get("recipe", {})
    if recipe.get("declared_self_sha256") != RECIPE_SHA256:
        raise ValueError(f"{fold}: declared recipe SHA mismatch")
    if recipe.get("recomputed_self_sha256") != RECIPE_SHA256:
        raise ValueError(f"{fold}: recomputed recipe SHA mismatch")

    checks = post.get("checks", {})
    required = {
        "all_nine_runs_complete",
        "identical_code_sha_across_runs",
        "identical_sample_sets",
        "identical_split",
        "prob_maps_present",
        "recipe_self_sha_matches",
        "seeds_declared_match",
        "test_region_matches_fold",
        "test_set_matches_sealed_contract",
    }
    failed_required = sorted(k for k in required if not checks.get(k, {}).get("pass"))
    if failed_required:
        raise ValueError(f"{fold}: required release checks failed/missing: {failed_required}")

    # Thrissur is the disclosed M57 provenance deviation.  It predates the source
    # snapshot mechanism; all later folds must carry and verify the immutable copy.
    if fold != "holdout_thrissur":
        snapshot_checks = {
            "snapshot_before_first_checkpoint",
            "snapshot_required_files",
            "snapshot_sha256sums_match",
        }
        bad_snapshot = sorted(k for k in snapshot_checks if not checks.get(k, {}).get("pass"))
        if bad_snapshot:
            raise ValueError(f"{fold}: snapshot checks failed/missing: {bad_snapshot}")

    arms = summary.get("arms", {})
    for arm_key in ARM_KEYS:
        arm = arms.get(arm_key, {})
        if arm.get("arm_id") != EXPECTED_ARM_IDS[arm_key]:
            raise ValueError(f"{fold}: arm mapping mismatch for {arm_key}")
        values = arm.get("primary_per_seed")
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{fold}: {arm_key} does not contain exactly 3 seeds")
        if not close(mean(float(v) for v in values), float(arm["primary_mean"])):
            raise ValueError(f"{fold}: {arm_key} mean does not match its seed values")

    win = summary.get("preregistered_win_reuse_vs_raw_strong", {})
    recomputed_gaps = [
        round(float(a) - float(b), 6)
        for a, b in zip(
            arms["reuse"]["primary_per_seed"], arms["raw_strong"]["primary_per_seed"]
        )
    ]
    if any(not close(a, b) for a, b in zip(recomputed_gaps, win.get("per_seed_gap", []))):
        raise ValueError(f"{fold}: stored per-seed gaps do not match arm values")
    gap_mean = mean(recomputed_gaps)
    if not close(gap_mean, float(win["mean_gap"])):
        raise ValueError(f"{fold}: stored mean gap does not match seed gaps")
    expected_win = gap_mean > 0 and all(g > 0 for g in recomputed_gaps)
    if bool(win.get("per_region_win")) != expected_win:
        raise ValueError(f"{fold}: preregistered win rule was not applied correctly")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--recovered-root",
        type=Path,
        default=None,
        help="Optional recovered server bundle containing summaries/ and confirmatory_manifests/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/confirmatory_8region_summary.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    recovered = args.recovered_root.resolve() if args.recovered_root else None
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []

    for fold in REGION_ORDER:
        summary_path, post_path = find_sources(repo, recovered, fold)
        summary, post = load_json(summary_path), load_json(post_path)
        validate_region(fold, summary, post)
        arms = summary["arms"]
        win = summary["preregistered_win_reuse_vs_raw_strong"]
        rows.append(
            {
                "fold": fold,
                "n_tiles": int(arms["reuse"]["n_tiles"]),
                "n_positive_tiles": int(arms["reuse"]["n_positive_tiles"]),
                "primary_mean": {
                    arm: float(arms[arm]["primary_mean"]) for arm in ARM_KEYS
                },
                "reuse_minus_raw_strong": float(win["mean_gap"]),
                "per_seed_gap": [float(x) for x in win["per_seed_gap"]],
                "per_region_win": bool(win["per_region_win"]),
                "strong_win": bool(summary["strong_win"]),
                "protocol_deviation": summary.get("protocol_deviation"),
            }
        )
        for kind, path in (("read_summary", summary_path), ("post_manifest", post_path)):
            canonical_server_path = (
                f"/home/work/data/olmoearth/confirmatory/{fold}/read_summary.json"
                if kind == "read_summary"
                else f"/home/work/data/olmoearth/confirmatory_manifests/{fold}_post.json"
            )
            try:
                retrieved_from = str(path.relative_to(repo))
            except ValueError:
                suffix = (
                    f"summaries/{fold}/read_summary.json"
                    if kind == "read_summary"
                    else f"confirmatory_manifests/{fold}_post.json"
                )
                retrieved_from = f"recovered_bundle://{suffix}"
            source_files.append(
                {
                    "fold": fold,
                    "kind": kind,
                    "retrieved_from": retrieved_from,
                    "canonical_server_path": canonical_server_path,
                    "sha256": sha256_file(path),
                }
            )

    aggregate = {
        arm: round(mean(r["primary_mean"][arm] for r in rows), 9) for arm in ARM_KEYS
    }
    aggregate["reuse_minus_raw_strong"] = round(
        mean(r["reuse_minus_raw_strong"] for r in rows), 9
    )
    without_thrissur = rows[1:]
    sensitivity = {
        arm: round(mean(r["primary_mean"][arm] for r in without_thrissur), 9)
        for arm in ARM_KEYS
    }
    sensitivity["reuse_minus_raw_strong"] = round(
        mean(r["reuse_minus_raw_strong"] for r in without_thrissur), 9
    )

    result: dict[str, Any] = {
        "schema": "confirmatory-8region-aggregate-v1",
        "status": "COMPLETE_VALIDATED",
        "recipe_self_sha256": RECIPE_SHA256,
        "primary_metric": "positive-tile macro IoU",
        "aggregation": "equal-weight mean across the 8 preregistered held-out regions",
        "source_precision_note": "aggregate is computed from six-decimal per-region read summaries",
        "regions": rows,
        "headline": {
            "region_macro_primary_mean": aggregate,
            "per_region_wins_reuse_vs_raw_strong": sum(r["per_region_win"] for r in rows),
            "strong_wins_reuse_vs_raw_strong": sum(r["strong_win"] for r in rows),
            "n_regions": len(rows),
            "non_win_regions": [r["fold"] for r in rows if not r["per_region_win"]],
            "negative_mean_gap_regions": [
                r["fold"] for r in rows if r["reuse_minus_raw_strong"] < 0
            ],
        },
        "sensitivity_excluding_disclosed_thrissur_provenance_deviation": {
            "region_macro_primary_mean": sensitivity,
            "n_regions": len(without_thrissur),
        },
        "claim_boundary": [
            "This closes the frozen v2 OLMo-vs-raw eight-region headline only.",
            "It does not establish OLMo-specific superiority before a second frozen GeoFM control.",
            "It does not establish label-free region routing; Indonesia loses and Itogon fails the all-seed win rule.",
            "Strong-win count uses the stored multiscale spatial-CI rule; Hokkaido is not strong because its 10.24 km CI is undefined.",
        ],
        "source_files": source_files,
    }
    unsigned = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["content_sha256_without_this_field"] = hashlib.sha256(unsigned.encode()).hexdigest()

    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["headline"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
