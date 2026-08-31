from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "audit_confirmatory_mechanism.py"


def write_arm(root: Path, region: str, arm: str, seed: int, ious: tuple[float, float], fp: int) -> None:
    path = (
        root
        / region
        / f"{arm}_seed{seed}"
        / "per_sample"
        / region
        / f"{arm}_test.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": f"{region}_positive_1",
            "mask_positive_pixels": 10,
            "iou_at_0_5": ious[0],
            "fp": 1,
        },
        {
            "sample_id": f"{region}_positive_2",
            "mask_positive_pixels": 20,
            "iou_at_0_5": ious[1],
            "fp": 2,
        },
        {
            "sample_id": f"{region}_empty",
            "mask_positive_pixels": 0,
            "iou_at_0_5": 0.0,
            "fp": fp,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_mechanism_audit_computes_paired_oracle_and_fp(tmp_path: Path) -> None:
    root = tmp_path / "confirmatory"
    for region in ("holdout_a", "holdout_b"):
        for seed in (1, 2, 3):
            write_arm(root, region, "P2", seed, (0.8, 0.2), fp=100)
            write_arm(root, region, "P3", seed, (0.4, 0.4), fp=50)
            write_arm(root, region, "P4", seed, (0.4, 0.8), fp=10)

    out = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n_regions"] == 2
    assert result["aggregate"]["region_macro_oracle_headroom"] == pytest.approx(0.2)
    assert result["aggregate"]["regions_where_p4_has_lower_empty_tile_fp"] == 2
    assert result["aggregate"]["median_empty_tile_fp_ratio_p2_over_p4"] == 10
    assert result["preregistered_screen"]["fusion_candidate"]["pass"] is False
    first = result["regions"][0]["p4_vs_p2"]
    assert first["positive_tile_win_fraction"] == 0.5
    assert first["positive_tile_loss_fraction"] == 0.5
