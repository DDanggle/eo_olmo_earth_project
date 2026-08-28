from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "summarize_confirmatory_8region.py"
SPEC = importlib.util.spec_from_file_location("summarize_confirmatory_8region", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_pair(fold: str = "holdout_hiroshima"):
    summary = {
        "schema": "confirmatory-region-read-v1",
        "fold": fold,
        "gate_verdict": "PASS",
        "arms": {
            "reuse": {
                "arm_id": "P4",
                "primary_per_seed": [0.2, 0.3, 0.4],
                "primary_mean": 0.3,
            },
            "raw_strong": {
                "arm_id": "P2",
                "primary_per_seed": [0.1, 0.2, 0.3],
                "primary_mean": 0.2,
            },
            "raw_efficient": {
                "arm_id": "P3",
                "primary_per_seed": [0.1, 0.1, 0.1],
                "primary_mean": 0.1,
            },
        },
        "preregistered_win_reuse_vs_raw_strong": {
            "per_seed_gap": [0.1, 0.1, 0.1],
            "mean_gap": 0.1,
            "per_region_win": True,
        },
    }
    passed = {
        key: {"pass": True}
        for key in {
            "all_nine_runs_complete",
            "identical_code_sha_across_runs",
            "identical_sample_sets",
            "identical_split",
            "prob_maps_present",
            "recipe_self_sha_matches",
            "seeds_declared_match",
            "test_region_matches_fold",
            "test_set_matches_sealed_contract",
            "snapshot_before_first_checkpoint",
            "snapshot_required_files",
            "snapshot_sha256sums_match",
        }
    }
    post = {
        "schema": "confirmatory-release-gate-v2",
        "fold": fold,
        "verdict": "PASS",
        "failed_checks": [],
        "checks": passed,
        "recipe": {
            "declared_self_sha256": MODULE.RECIPE_SHA256,
            "recomputed_self_sha256": MODULE.RECIPE_SHA256,
        },
    }
    return summary, post


def test_validate_region_accepts_complete_sealed_release():
    summary, post = valid_pair()
    MODULE.validate_region("holdout_hiroshima", summary, post)


def test_validate_region_rejects_snapshot_failure_after_thrissur():
    summary, post = valid_pair()
    post["checks"]["snapshot_sha256sums_match"]["pass"] = False
    with pytest.raises(ValueError, match="snapshot checks"):
        MODULE.validate_region("holdout_hiroshima", summary, post)


def test_validate_region_rejects_incorrect_preregistered_win():
    summary, post = valid_pair()
    summary["preregistered_win_reuse_vs_raw_strong"]["per_region_win"] = False
    with pytest.raises(ValueError, match="win rule"):
        MODULE.validate_region("holdout_hiroshima", summary, post)
