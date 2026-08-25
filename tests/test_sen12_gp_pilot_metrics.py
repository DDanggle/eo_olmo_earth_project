from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from audit_sen12_fold_cache import array_sha256  # noqa: E402
from pilot_sen12_gp_heads import exact_average_precision, fixed_bin_ece  # noqa: E402
from verify_sen12_gp_artifacts import verify  # noqa: E402


def test_exact_average_precision_groups_tied_scores():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    labels = np.array([1, 0, 1, 0])
    # recall 0→0.5 at precision 1, then 0.5→1 at precision 2/3.
    assert exact_average_precision(scores, labels) == pytest.approx(5 / 6)


def test_exact_average_precision_returns_none_without_positive_label():
    assert exact_average_precision([0.2, 0.1], [0, 0]) is None


def test_exact_average_precision_matches_sklearn_with_many_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(17)
    scores = np.round(rng.random(2000), 1)
    labels = rng.integers(0, 2, 2000, dtype=np.uint8)
    assert exact_average_precision(scores, labels) == pytest.approx(
        metrics.average_precision_score(labels, scores), abs=1e-12)


def test_fixed_bin_ece_uses_every_pixel():
    assert fixed_bin_ece([0.1, 0.9], [0, 1], bins=10) == pytest.approx(0.1)


def test_array_content_hash_changes_with_one_value_not_memory_layout():
    base = np.arange(12, dtype=np.uint16).reshape(3, 4)
    same = np.asfortranarray(base)
    changed = base.copy()
    changed[0, 0] = 99
    assert array_sha256(base) == array_sha256(same)
    assert array_sha256(base) != array_sha256(changed)


def test_artifact_verifier_recomputes_thresholded_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "fold" / "P4_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    row = {
        "sample_id": "s1", "region": "r", "ann_id": "a", "event_date": "d",
        "mask_positive_pixels": 100, "prediction_positive_pixels": 80,
        "tp": 50, "fp": 30, "fn": 50, "iou_at_0_5": round(50 / 130, 8),
        "mean_probability": 0.1,
    }
    per_sample = {}
    for split in ("val", "test"):
        path = tmp_path / "per_sample" / "fold" / f"P4_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        per_sample[split] = {
            "path": f"/remote/per_sample/fold/P4_{split}.jsonl",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 1,
        }
    metrics = {
        "iou": round(50 / 130, 6), "f1": round(100 / 180, 6),
        "precision": round(50 / 80, 6), "recall": 0.5,
        "positive_pixel_frac": round(100 / (128 * 128), 8),
        "positive_patch_macro_iou": round(50 / 130, 6), "positive_patch_n": 1,
        "ld_iou": round(50 / 130, 5), "ld_f1": round(100 / 180, 5),
        "ld_subset_n": 1,
        "confusion_pixels": {"tp": 50, "fp": 30, "fn": 50, "tn": 0},
    }
    summary = {
        "arms": {"P4": {
            "checkpoint": {"path": "/remote/checkpoints/fold/P4_best.pt", "sha256": checkpoint_sha},
            "per_sample": per_sample, "val": metrics, "test": metrics,
        }}
    }
    summary_path = tmp_path / "result.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = verify(summary_path)
    assert result["all_checks_pass"] is True
    assert result["failures"] == []
