from __future__ import annotations

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from build_sen12_gp_contract import (  # noqa: E402
    HEADLINE_REGIONS,
    assign_task_eligibility,
    build_loco_folds,
    parse_bool,
    parse_float,
    parse_pre_post_dates,
)
from smoke_sen12_olmo_v1_embeddings import (  # noqa: E402
    select_stratified,
    select_timestep_indices,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "sen12_gp_contract"
SMOKE_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "sen12_olmo_v1_smoke_64" / "summary.json"
)
FROZEN_SAMPLE_SHA256 = "dcdfef9a6c9bde5ccb5e81e29dff32f67727a04ec320118b08c991a2bbec8733"
FROZEN_INDEX_SHA256 = "b6fe0c854808485ae87a9b6e3c6f7bd39db2d4004e511b30b1188928a697f086"
FROZEN_ANOMALY_SHA256 = "bf086042f20039e8cab929e1d757da05925c86b4d6462de2ed72de35be23afc8"


def test_parse_pre_post_dates_accepts_dict_and_serialized_dict():
    assert parse_pre_post_dates({"pre": 3, "post": 7}) == (3, 7)
    assert parse_pre_post_dates("{'pre': 8, 'post': 9}") == (8, 9)
    assert parse_pre_post_dates("{}") == (None, None)
    assert parse_pre_post_dates("not a dict") == (None, None)


def test_parse_bool_is_explicit_not_python_truthiness():
    assert parse_bool("False") is False
    assert parse_bool("true") is True
    assert parse_bool(0) is False
    assert parse_bool("unknown") is None


def test_parse_float_preserves_empty_attributes_as_missing():
    assert parse_float(0.75) == 0.75
    assert parse_float("1") == 1.0
    assert parse_float("") is None
    assert parse_float("unknown") is None


def test_loco_never_splits_a_region_between_roles():
    records = []
    for region in HEADLINE_REGIONS:
        records.extend([
            {"region": region, "sample_id": f"{region}_1"},
            {"region": region, "sample_id": f"{region}_2"},
        ])
    folds = build_loco_folds(records)
    assert len(folds) == 10
    assert {f["test_region"] for f in folds} == set(HEADLINE_REGIONS)
    for fold in folds:
        assert fold["test_region"] != fold["val_region"]
        assert fold["test_region"] not in fold["train_regions"]
        assert fold["val_region"] not in fold["train_regions"]
        assert fold["sample_counts"] == {"train": 16, "val": 2, "test": 2}


def test_loco_hashes_are_stable_under_input_reordering():
    records = [
        {"region": region, "sample_id": f"{region}_{i}"}
        for region in HEADLINE_REGIONS
        for i in (2, 1)
    ]
    forward = build_loco_folds(records)
    backward = build_loco_folds(list(reversed(records)))
    assert forward == backward


def test_task_eligibility_excludes_ambiguous_label_without_mutating_source():
    bad = assign_task_eligibility({
        "sample_id": "x",
        "annotation_consistent": False,
        "label_positive": False,
        "pre_post_valid": True,
    })
    assert bad["s15_eligible"] is False
    assert bad["r_event_eligible"] is False
    assert bad["s_cutoff_positive_eligible"] is False

    good = assign_task_eligibility({
        "sample_id": "y",
        "annotation_consistent": True,
        "label_positive": True,
        "pre_post_valid": True,
    })
    assert good["s15_eligible"] is True
    assert good["s_cutoff_positive_eligible"] is True


def test_frozen_full_audit_contract():
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["retrospective_contract_ready"] is True
    assert summary["samples_readable"] == 13_628
    assert summary["s15_eligible_samples"] == 13_626
    assert summary["headline_s15_eligible_samples"] == 6_834
    assert summary["negative_only_regions"] == {"lanaodelnorte": 71}
    assert summary["sample_contract_sha256"] == FROZEN_SAMPLE_SHA256
    assert summary["source_index_sha256"] == FROZEN_INDEX_SHA256
    assert summary["s15_excluded_label_anomalies_sha256"] == FROZEN_ANOMALY_SHA256
    assert all(summary["schema_gates"].values())


def test_frozen_loco_has_ten_region_disjoint_folds():
    contract = json.loads((ARTIFACT / "loco_folds.json").read_text(encoding="utf-8"))
    folds = contract["folds"]
    assert contract["sample_contract_sha256"] == FROZEN_SAMPLE_SHA256
    assert len(folds) == 10
    assert {f["test_region"] for f in folds} == set(HEADLINE_REGIONS)
    for fold in folds:
        assert sum(fold["sample_counts"].values()) == 6_834
        assert fold["test_region"] != fold["val_region"]
        assert fold["test_region"] not in fold["train_regions"]
        assert fold["val_region"] not in fold["train_regions"]


def test_embedding_smoke_selection_covers_region_and_label_strata():
    records = [
        {
            "sample_id": f"{region}_{positive}_{i}",
            "region": region,
            "s15_eligible": True,
            "label_positive": positive,
        }
        for region in HEADLINE_REGIONS
        for positive in (True, False)
        for i in range(4)
    ]
    selected = select_stratified(records, 64)
    assert len(selected) == 64
    assert {r["region"] for r in selected} == set(HEADLINE_REGIONS)
    assert {r["label_positive"] for r in selected} == {True, False}


def test_timestep_selection_is_quality_based_but_chronological():
    record = {"sample_id": "x", "scl_clear_fraction": [float(i) for i in range(15)]}
    # 최고 품질 12개는 index 3..14이고 모델에는 시간순으로 들어간다.
    assert select_timestep_indices(record) == list(range(3, 15))


def test_frozen_olmo_v1_embedding_smoke_passed():
    smoke = json.loads(SMOKE_ARTIFACT.read_text(encoding="utf-8"))
    assert smoke["all_gates_pass"] is True
    assert smoke["input_contract_sha256"] == FROZEN_SAMPLE_SHA256
    assert smoke["samples"] == 64
    assert smoke["crops"] == 256
    assert smoke["model_timesteps"] == 12
    assert smoke["replay_max_abs_diff"] == 0.0
    assert smoke["cache_bytes_per_sample"] == 1_572_992
    assert len({r["sample_id"] for r in smoke["outputs"]}) == 64
    assert {r["region"] for r in smoke["outputs"]} == set(HEADLINE_REGIONS)
    assert sum(r["label_positive"] for r in smoke["outputs"]) == 32
