from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sen12_retrieval_probe import average_precision_at_k, recall_at_k  # noqa: E402


def test_ap_at_k_penalizes_missing_relevant_items() -> None:
    # The previous implementation returned 1.0 because it divided by the one
    # retrieved positive. Standard AP@2 divides by min(10 relevant, 2) = 2.
    hit = np.array([True, False])
    assert average_precision_at_k(hit, total_relevant=10, k=2) == pytest.approx(0.5)


def test_ap_at_k_uses_precision_at_positive_ranks() -> None:
    hit = np.array([False, True, True, False])
    expected = ((1 / 2) + (2 / 3)) / 3
    assert average_precision_at_k(hit, total_relevant=3, k=4) == pytest.approx(expected)


def test_ap_and_recall_at_k_handle_no_relevant_items() -> None:
    hit = np.array([False, False])
    assert average_precision_at_k(hit, total_relevant=0, k=2) == 0.0
    assert recall_at_k(hit, total_relevant=0, k=2) == 0.0


def test_recall_at_k_uses_all_gallery_relevant_items() -> None:
    hit = np.array([True, False, True])
    assert recall_at_k(hit, total_relevant=8, k=3) == pytest.approx(0.25)
