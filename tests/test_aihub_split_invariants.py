"""동결된 AI-Hub 71363 spatial holdout의 불변식 테스트.

M10에서 split을 동결했다. 이후 누가 builder를 고쳐도 아래가 깨지면 CI가 잡아야 한다.
동결의 의미는 "바뀌지 않는다"가 아니라 "바뀌면 즉시 드러난다"이다.

검사 범위는 `artifacts/aihub71363_*.json(l)` 산출물이다. 서버 접속을 요구하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
HOLDOUT = ART / "aihub71363_spatial_holdout.json"
ASSIGN = ART / "aihub71363_tile_assignment.jsonl"
LOCO = ART / "aihub71363_loco_folds.json"

# M10에서 동결된 값. 이 상수를 고치는 것은 곧 split을 바꾸는 것이다.
FROZEN_SHA256 = {
    "train": "50fdcb4b6b404d41296935854e60ef14f0cfefbdc6da00ac1798c0accd434cec",
    "val": "8e133c51db9b2eb577d06be3e1c466af11aae4911f109f8bbbf41461e92b56c1",
    "test": "3f44498758600c3f56c005c2af20ff6d6a69ae14855eed130d398742d77eb168",
    "excluded": "e67c09a5b013cf738e3ba9fe6871aabeb08973163390d84ba0246625653c927f",
}
FROZEN_TILES = {"train": 393, "val": 84, "test": 113, "excluded": 4}
TOTAL_TILES = 594
RARE = ("70:벌목지", "80:산사태및토석류피해지")
MIN_RARE_TILES = 10
MIN_GAP_M = 10240.0


@pytest.fixture(scope="module")
def holdout() -> dict:
    if not HOLDOUT.exists():
        pytest.skip("holdout 산출물이 없다")
    return json.loads(HOLDOUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def assignment() -> list[dict]:
    if not ASSIGN.exists():
        pytest.skip("assignment 산출물이 없다")
    return [json.loads(l) for l in ASSIGN.read_text(encoding="utf-8").splitlines() if l]


def test_assignment_tiles_are_unique_and_complete(assignment):
    ids = [r["tile_id"] for r in assignment]
    assert len(ids) == TOTAL_TILES, f"타일 수가 {len(ids)}로 바뀌었다"
    assert len(set(ids)) == len(ids), "중복 타일이 있다"


def test_split_sizes_match_frozen(assignment):
    counts = Counter(r["split"] for r in assignment)
    assert dict(counts) == FROZEN_TILES, f"split 크기가 바뀌었다: {dict(counts)}"


def test_split_hashes_match_frozen(assignment):
    """타일 목록 해시를 재계산해 동결값과 대조한다. 이것이 동결의 핵심이다."""
    for split, expected in FROZEN_SHA256.items():
        tiles = sorted(r["tile_id"] for r in assignment if r["split"] == split)
        got = hashlib.sha256("\n".join(tiles).encode()).hexdigest()
        assert got == expected, f"{split} split이 바뀌었다\n  기대 {expected}\n  실제 {got}"


def test_clusters_do_not_span_splits(assignment):
    """한 군집이 두 split에 걸치면 공간 분리가 깨진다.

    예외: `excluded`는 test 군집 안에서 빼낸 것이므로 test와 군집을 공유한다.
    """
    by_cluster = {}
    for r in assignment:
        by_cluster.setdefault(r["cluster"], set()).add(r["split"])
    for cluster, splits in by_cluster.items():
        assert splits <= {"test", "excluded"} or len(splits) == 1, \
            f"군집 {cluster}가 여러 split에 걸쳐 있다: {sorted(splits)}"


def test_excluded_tiles_have_reason(assignment):
    for r in assignment:
        if r["split"] == "excluded":
            assert r.get("reason"), f"{r['tile_id']}에 제외 사유가 없다"


def test_gates_all_pass(holdout):
    gates = holdout["gates"]
    failed = [k for k, v in gates.items() if not v]
    assert not failed, f"게이트 실패: {failed}"


def test_inter_cluster_gap_is_at_least_one_tile(holdout):
    gap = holdout["min_inter_cluster_gap_m"]
    assert gap >= MIN_GAP_M, f"군집 간 이격 {gap} m < {MIN_GAP_M} m"


def test_val_and_test_have_enough_rare_tiles(holdout):
    for split in ("val", "test"):
        for rare in RARE:
            n = holdout["splits"][split]["tiles_per_class"].get(rare, 0)
            assert n >= MIN_RARE_TILES, f"{split}의 {rare}가 {n}타일 (< {MIN_RARE_TILES})"


def test_rare_positive_cluster_counts_are_recorded(holdout, assignment):
    """희소 클래스의 실질 독립 표본은 타일이 아니라 **군집** 수다.

    test의 산사태 22타일은 소수 군집에서 나온다. 타일 단위 bootstrap은 불확실성을
    과소평가하므로, 군집 수를 명시적으로 확인해 보고서에 강제로 드러나게 한다.
    """
    tile_split = {r["tile_id"]: r["split"] for r in assignment}
    tile_cluster = {r["tile_id"]: r["cluster"] for r in assignment}
    stats = holdout["cluster_stats"]
    for rare in RARE:
        per_split = Counter()
        for cluster, s in stats.items():
            if s["tiles_per_class"].get(rare, 0) <= 0:
                continue
            splits = {tile_split[t] for t, c in tile_cluster.items() if c == cluster}
            for sp in splits - {"excluded"}:
                per_split[sp] += 1
        # 최소 2개 군집은 있어야 지역 간 변동을 볼 수 있다.
        for sp in ("train", "val", "test"):
            assert per_split[sp] >= 2, \
                f"{sp}의 {rare} 양성 군집이 {per_split[sp]}개 — 지역 일반화를 말할 수 없다"


def test_loco_folds_cover_every_cluster(holdout):
    if not LOCO.exists():
        pytest.skip("LOCO 산출물이 없다")
    folds = json.loads(LOCO.read_text(encoding="utf-8"))["folds"]
    assert len(folds) == holdout["cluster_count"], "LOCO 폴드 수가 군집 수와 다르다"
    assert len({f["held_out_cluster"] for f in folds}) == len(folds), "중복 폴드가 있다"
