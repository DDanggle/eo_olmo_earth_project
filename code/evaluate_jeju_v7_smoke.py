#!/usr/bin/env python3
"""Evaluate the pre-registered one-window Jeju v7 SCL smoke test."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


V1_WINDOW = Path(
    "/home/work/data/olmoearth/embed_search/dataset/windows/default/jeju25_30720_-372736"
)
V7_WINDOW = Path(
    "/home/work/data/olmoearth/embed_jeju_v7_smoke/dataset/windows/default/"
    "smoke25_30720_-372736"
)
OUT = Path("/home/work/data/olmoearth/embed_jeju_v7_smoke/audit")
B02_INDEX = 2
RGB_INDEXES = (4, 3, 2)
CLOUD_DN = 1800
BLOCK = 4
TARGET_BLOCK_ROW = 237
TARGET_BLOCK_COL = 27
TARGET_PERIOD = 3


def layer_index(path: Path) -> int:
    match = re.fullmatch(r"sentinel2_l2a(?:\.(\d+))?", path.name)
    return int(match.group(1) or 0) if match else 10_000


def layer_tiffs(window_dir: Path) -> list[Path]:
    result = []
    for layer in sorted(
        (path for path in (window_dir / "layers").glob("sentinel2_l2a*") if path.is_dir()),
        key=layer_index,
    ):
        matches = sorted(layer.glob("*/geotiff.tif"))
        if matches:
            result.append(matches[0])
    return result


def groups(window_dir: Path) -> list[list[str]]:
    payload = json.loads((window_dir / "items.json").read_text(encoding="utf-8"))
    layer = next(item for item in payload if item["layer_name"] == "sentinel2_l2a")
    return [
        [serialized["name"] for serialized in group]
        for group in layer["serialized_item_groups"]
    ]


def quality(path: Path) -> tuple[dict, np.ndarray]:
    with rasterio.open(path) as src:
        array = src.read()
        b02 = array[B02_INDEX - 1]
        invalid = src.read_masks(B02_INDEX) == 0
    zero = (b02 == 0) | invalid
    cloud = (b02 > CLOUD_DN) & ~zero
    bad = cloud | zero
    return {
        "cloud_proxy": float(cloud.mean()),
        "zero_proxy": float(zero.mean()),
        "bad_proxy": float(bad.mean()),
    }, array


def target_bad(path: Path) -> float:
    window = Window(TARGET_BLOCK_COL * BLOCK, TARGET_BLOCK_ROW * BLOCK, BLOCK, BLOCK)
    with rasterio.open(path) as src:
        b02 = src.read(B02_INDEX, window=window)
        invalid = src.read_masks(B02_INDEX, window=window) == 0
    return float(((b02 > CLOUD_DN) | (b02 == 0) | invalid).mean())


def read_rgb(path: Path, radius: int = 64) -> np.ndarray:
    center_row = TARGET_BLOCK_ROW * BLOCK + BLOCK // 2
    center_col = TARGET_BLOCK_COL * BLOCK + BLOCK // 2
    with rasterio.open(path) as src:
        r0, c0 = max(0, center_row - radius), max(0, center_col - radius)
        r1, c1 = min(src.height, center_row + radius), min(src.width, center_col + radius)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        rgb = np.stack([src.read(index, window=window) for index in RGB_INDEXES]).astype(
            np.float32
        )
    return np.moveaxis(np.clip(rgb / 3000.0, 0, 1) ** 0.8, 0, -1)


def relative_reduction(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_paths = layer_tiffs(V1_WINDOW)[:4]
    v7_paths = layer_tiffs(V7_WINDOW)[:4]
    if len(v1_paths) != 4 or len(v7_paths) != 4:
        raise RuntimeError(f"expected four periods, got v1={len(v1_paths)} v7={len(v7_paths)}")

    v1_groups = groups(V1_WINDOW)[:4]
    v7_groups = groups(V7_WINDOW)[:4]
    periods = []
    for index, (v1_path, v7_path) in enumerate(zip(v1_paths, v7_paths)):
        v1_quality, v1_array = quality(v1_path)
        v7_quality, v7_array = quality(v7_path)
        periods.append(
            {
                "period_index": index,
                "v1_source_items": len(v1_groups[index]),
                "v7_source_items": len(v7_groups[index]),
                "ordered_group_hash_equal": hashlib.sha256("\n".join(v1_groups[index]).encode()).hexdigest()
                == hashlib.sha256("\n".join(v7_groups[index]).encode()).hexdigest(),
                "all_band_array_equal": bool(np.array_equal(v1_array, v7_array)),
                "all_band_max_abs_difference": int(
                    np.max(np.abs(v1_array.astype(np.int64) - v7_array.astype(np.int64)))
                ),
                "v1": v1_quality,
                "v7": v7_quality,
                "target_v1_bad": target_bad(v1_path),
                "target_v7_bad": target_bad(v7_path),
            }
        )

    mean_v1_bad = float(np.mean([period["v1"]["bad_proxy"] for period in periods]))
    mean_v7_bad = float(np.mean([period["v7"]["bad_proxy"] for period in periods]))
    mean_v1_zero = float(np.mean([period["v1"]["zero_proxy"] for period in periods]))
    mean_v7_zero = float(np.mean([period["v7"]["zero_proxy"] for period in periods]))
    comparisons = {
        "first4_bad_relative_reduction": relative_reduction(mean_v1_bad, mean_v7_bad),
        "first4_zero_delta_percentage_points": 100 * (mean_v7_zero - mean_v1_zero),
        "target_period_v1_bad": periods[TARGET_PERIOD]["target_v1_bad"],
        "target_period_v7_bad": periods[TARGET_PERIOD]["target_v7_bad"],
    }
    gates = {
        "source_groups_and_output_pixels_changed": (
            any(not period["ordered_group_hash_equal"] for period in periods)
            and any(not period["all_band_array_equal"] for period in periods)
        ),
        "first4_bad_reduction_at_least_10pct": comparisons[
            "first4_bad_relative_reduction"
        ]
        >= 0.10,
        "target_period_bad_at_most_0_5": comparisons["target_period_v7_bad"] <= 0.5,
        "zero_proxy_not_worse_by_more_than_1pp": comparisons[
            "first4_zero_delta_percentage_points"
        ]
        <= 1.0,
        "manual_rgb_target_cloud_reduced": None,
    }

    fig, axes = plt.subplots(4, 2, figsize=(8, 13))
    for index, (v1_path, v7_path) in enumerate(zip(v1_paths, v7_paths)):
        for column, (name, path) in enumerate((("v1", v1_path), ("v7 SCL", v7_path))):
            axes[index, column].imshow(read_rgb(path))
            axes[index, column].axis("off")
            bad = periods[index]["target_v1_bad"] if column == 0 else periods[index]["target_v7_bad"]
            axes[index, column].set_title(f"period {index} {name} | target bad={bad:.2f}")
    fig.suptitle(
        "Jeju v7 golden window — fixed target selected before v7\n"
        "SCL scored with nearest; reflectance rendered with fixed stretch",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "v7_rgb_pairs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    output = {
        "status": "numeric_complete_manual_rgb_pending",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "v1_window": str(V1_WINDOW),
            "v7_window": str(V7_WINDOW),
            "target_block": [TARGET_BLOCK_ROW, TARGET_BLOCK_COL],
            "target_period": TARGET_PERIOD,
            "cloud_proxy": f"B02 > {CLOUD_DN}",
        },
        "periods": periods,
        "comparisons": comparisons,
        "gates": gates,
        "limitations": [
            "This is one pre-registered golden window, not a Jeju-wide result.",
            "SCL BestClear selects one scene per period; it is not pixel-wise cloud gap filling.",
            "B02 brightness is a proxy and still requires RGB inspection.",
        ],
    }
    (OUT / "v7_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "gates": gates}, indent=2), flush=True)
    print(f"V7_SMOKE_EVAL_DONE out={OUT}", flush=True)


if __name__ == "__main__":
    main()
