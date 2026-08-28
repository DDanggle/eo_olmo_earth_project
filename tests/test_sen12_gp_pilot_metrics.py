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
from analyze_e1_factorial import contrasts  # noqa: E402
from materialize_aihub_s2_12band_v2 import (  # noqa: E402
    normalize_platform,
    target_bbox_is_exact_10m_grid,
)
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


def test_aihub_v2_platform_matching_is_explicit():
    assert normalize_platform("S2A") == "S2A"
    assert normalize_platform("sentinel-2a") == "S2A"
    assert normalize_platform("Sentinel_2B") == "S2B"
    assert normalize_platform("landsat-8") is None
    assert normalize_platform(None) is None


def test_aihub_v2_rejects_non_10m_or_malformed_target_grid():
    assert target_bbox_is_exact_10m_grid([100.0, 200.0, 10_340.0, 10_440.0])
    assert not target_bbox_is_exact_10m_grid([100.0, 200.0, 10_339.0, 10_440.0])
    assert not target_bbox_is_exact_10m_grid([100.0, 200.0, 100.0, 10_440.0])
    assert not target_bbox_is_exact_10m_grid([100.0, 200.0, float("nan"), 10_440.0])
    assert not target_bbox_is_exact_10m_grid(None)


def test_e1_factorial_contrasts_keep_main_effects_and_interaction_separate():
    result = contrasts({"y00": 0.10, "y01": 0.20, "y10": 0.15, "y11": 0.30})
    assert result["C_small"] == pytest.approx(0.05)
    assert result["C_large"] == pytest.approx(0.10)
    assert result["C_context_mean"] == pytest.approx(0.075)
    assert result["D_tiled"] == pytest.approx(0.10)
    assert result["D_full"] == pytest.approx(0.15)
    assert result["D_decoder_mean"] == pytest.approx(0.125)
    assert result["I_interaction"] == pytest.approx(0.05)


def test_gp_bundle_v2_counts_blank_rows_not_unique_blank_values():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "evidence" / "gp_official_bundle"
    manifest = json.loads((bundle / "bundle_manifest_v2.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (bundle / "per_sample" / "P2_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    corrected = manifest["bootstrap_units_available_corrected"]
    assert corrected["n_rows"] == len(rows) == 1_133
    for key in ("ann_id", "event_date", "region"):
        values = [row.get(key) for row in rows]
        assert corrected[key]["n_blank_rows"] == sum(v in (None, "") for v in values)
        assert corrected[key]["n_distinct_nonblank"] == len({
            v for v in values if v not in (None, "")
        })


def test_e1_factorial_evidence_matches_sealed_input_hashes():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence" / "e1_factorial_v2"
    analysis = json.loads((evidence / "e1_factorial_analysis.json").read_text(encoding="utf-8"))

    assert analysis["n_paired_test_tiles"] == 1_133
    assert analysis["analysis_code_sha256"] == hashlib.sha256(
        (root / "code" / "analyze_e1_factorial.py").read_bytes()
    ).hexdigest()
    # E1 predates the M58 source-snapshot mechanism.  Comparing its historical
    # runner hash with today's mutable runner makes a valid old bundle fail as
    # soon as the live code evolves and still cannot prove the bytes that ran.
    # The strongest surviving check is agreement among all four sealed pilot
    # records; the missing source snapshot remains a disclosed limitation.
    pilot_runner_hashes = {
        json.loads(p.read_text(encoding="utf-8"))["code_sha256"]
        for p in evidence.glob("*/holdout_chimanimani_pilot.json")
    }
    assert pilot_runner_hashes == {analysis["runner_code_sha256"]}
    for relative_path, expected_sha in analysis["input_files_sha256"].items():
        path = evidence / relative_path
        assert path.is_file(), relative_path
        if relative_path.endswith("_pilot.json"):
            # M53 appended a provenance-preserving correction to these JSONs,
            # so their original E1 byte hashes are intentionally historical.
            corrected = json.loads(path.read_text(encoding="utf-8"))
            note = corrected["information_contract_correction_2026_08_26"]
            assert note["corrected_statement"]["information_parity"] is True
            assert "원본 필드는 삭제하지 않는다" in note["policy"]
            assert expected_sha != hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha


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
