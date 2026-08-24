#!/usr/bin/env python3
"""Audit Jeju v5 input quality against the preserved v1 input recipe.

This is an input-quality audit, not a cloud classifier. It compares the same
216 year-window cells under:

* v1: MOSAIC (one selected scene per period)
* v5: PER_PERIOD_MOSAIC / FIRST_VALID composite

The B02 thresholds deliberately match cloud_mask_v3.py/v4.py so the new result
is comparable with the recorded failure lineage. B02 > 1800 is only a bright
cloud/haze proxy. B02 == 0 or the raster mask being invalid is only a nodata
proxy; neither is treated as ecological truth.

Outputs are written below ``--out``:

* summary.json: aggregate metrics, gates, provenance
* per_window.csv: every dataset/year/window/scope metric
* quality_summary.png: v1-v5 quantitative comparison
* rgb_blind_pairs.png: deterministic locations selected using v1 quality only
* rgb_blind_pairs.json: the selections and per-pair proxy values
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


YEAR_PREFIX = {
    "2023": {"v1": "jeju23_", "v5": "jeju23_"},
    "2024": {"v1": "jeju_", "v5": "jeju24_"},
    "2025": {"v1": "jeju25_", "v5": "jeju25_"},
    "2026": {"v1": "jeju26r_", "v5": "jeju26r_"},
}
YEARS = list(YEAR_PREFIX)
SCOPES = {"model_used_4": 4, "all_12": 12}
B02_INDEX = 2
RGB_INDEXES = (4, 3, 2)
BLOCK = 4
CLOUD_DN = 1800
STRICT_BAD_MAX = 0.35
RNG_SEED = 20260822


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-root",
        default="/home/work/data/olmoearth/embed_search/dataset/windows/default",
    )
    parser.add_argument(
        "--v5-root",
        default="/home/work/data/olmoearth/embed_jeju_v2/dataset/windows/default",
    )
    parser.add_argument(
        "--out",
        default="/home/work/data/olmoearth/embed_jeju_v2/audit_v1_vs_v5",
    )
    return parser.parse_args()


def natural_layer_key(path: Path) -> int:
    match = re.fullmatch(r"sentinel2_l2a(?:\.(\d+))?", path.name)
    if not match:
        return 10_000
    return int(match.group(1) or 0)


def layer_tiffs(window_dir: Path) -> list[Path]:
    layers = sorted(
        (p for p in (window_dir / "layers").glob("sentinel2_l2a*") if p.is_dir()),
        key=natural_layer_key,
    )
    result: list[Path] = []
    for layer in layers:
        matches = sorted(layer.glob("*/geotiff.tif"))
        if matches:
            result.append(matches[0])
    return result


def discover(root: Path, dataset: str) -> dict[str, dict[str, list[Path]]]:
    found: dict[str, dict[str, list[Path]]] = {}
    for year in YEARS:
        prefix = YEAR_PREFIX[year][dataset]
        found[year] = {}
        for window_dir in sorted(root.glob(f"{prefix}*")):
            if not window_dir.is_dir():
                continue
            key = window_dir.name.removeprefix(prefix)
            found[year][key] = layer_tiffs(window_dir)
    return found


def block_mean(array: np.ndarray, factor: int = BLOCK) -> np.ndarray:
    height = array.shape[0] // factor * factor
    width = array.shape[1] // factor * factor
    return (
        array[:height, :width]
        .reshape(height // factor, factor, width // factor, factor)
        .mean(axis=(1, 3), dtype=np.float32)
    )


def read_period_quality(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        b02 = src.read(B02_INDEX)
        invalid_mask = src.read_masks(B02_INDEX) == 0
    zero = (b02 == 0) | invalid_mask
    cloud = (b02 > CLOUD_DN) & ~zero
    bad = cloud | zero
    return (
        block_mean(cloud.astype(np.float32)),
        block_mean(zero.astype(np.float32)),
        block_mean(bad.astype(np.float32)),
    )


def scan_dataset(
    dataset: str,
    discovered: dict[str, dict[str, list[Path]]],
    matched: dict[str, list[str]],
) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    worst: dict[str, dict[str, dict[str, np.ndarray]]] = {
        scope: {year: {} for year in YEARS} for scope in SCOPES
    }
    layouts: dict[str, dict[str, dict]] = {year: {} for year in YEARS}

    total = sum(len(matched[year]) for year in YEARS)
    done = 0
    for year in YEARS:
        for key in matched[year]:
            paths = discovered[year][key]
            period_cloud: list[np.ndarray] = []
            period_zero: list[np.ndarray] = []
            period_bad: list[np.ndarray] = []
            for path in paths:
                cloud, zero, bad = read_period_quality(path)
                period_cloud.append(cloud)
                period_zero.append(zero)
                period_bad.append(bad)

            cloud_stack = np.stack(period_cloud)
            zero_stack = np.stack(period_zero)
            bad_stack = np.stack(period_bad)
            with rasterio.open(paths[0]) as src:
                layouts[year][key] = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "paths": paths,
                }

            for scope, n_periods in SCOPES.items():
                cloud_view = cloud_stack[:n_periods]
                zero_view = zero_stack[:n_periods]
                bad_view = bad_stack[:n_periods]
                worst_bad = bad_view.max(axis=0).astype(np.float32)
                worst[scope][year][key] = worst_bad
                rows.append(
                    {
                        "dataset": dataset,
                        "year": year,
                        "key": key,
                        "scope": scope,
                        "periods_available": len(paths),
                        "periods_used": n_periods,
                        "cloud_proxy_mean": float(cloud_view.mean()),
                        "zero_proxy_mean": float(zero_view.mean()),
                        "bad_proxy_mean": float(bad_view.mean()),
                        "worst_period_bad_mean": float(worst_bad.mean()),
                    }
                )
            done += 1
            if done % 12 == 0 or done == total:
                print(f"{dataset}: {done}/{total} year-windows scanned", flush=True)
    return rows, worst, layouts


def strict_clean_ratio(worst: dict, scope: str, common_keys: list[str]) -> float:
    clean_count = 0
    pixel_count = 0
    for key in common_keys:
        annual_worst = np.stack([worst[scope][year][key] for year in YEARS]).max(axis=0)
        clean = annual_worst <= STRICT_BAD_MAX
        clean_count += int(clean.sum())
        pixel_count += int(clean.size)
    return clean_count / pixel_count


def aggregate(
    rows: list[dict], worst: dict, common_keys: list[str]
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for scope in SCOPES:
        selected = [row for row in rows if row["scope"] == scope]
        summary[scope] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in (
                "cloud_proxy_mean",
                "zero_proxy_mean",
                "bad_proxy_mean",
                "worst_period_bad_mean",
            )
        }
        summary[scope]["strict_clean_ratio"] = strict_clean_ratio(
            worst, scope, common_keys
        )
        summary[scope]["year"] = {
            year: {
                metric: float(
                    np.mean(
                        [
                            row[metric]
                            for row in selected
                            if row["year"] == year
                        ]
                    )
                )
                for metric in (
                    "cloud_proxy_mean",
                    "zero_proxy_mean",
                    "bad_proxy_mean",
                    "worst_period_bad_mean",
                )
            }
            for year in YEARS
        }
    return summary


def relative_reduction(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def quality_at_block(path: Path, block_row: int, block_col: int) -> float:
    window = Window(block_col * BLOCK, block_row * BLOCK, BLOCK, BLOCK)
    with rasterio.open(path) as src:
        b02 = src.read(B02_INDEX, window=window)
        invalid = src.read_masks(B02_INDEX, window=window) == 0
    bad = (b02 > CLOUD_DN) | (b02 == 0) | invalid
    return float(bad.mean())


def block_lonlat(layout: dict, block_row: int, block_col: int) -> tuple[float, float]:
    row = block_row * BLOCK + BLOCK / 2
    col = block_col * BLOCK + BLOCK / 2
    x, y = rasterio.transform.xy(layout["transform"], row, col)
    transformer = Transformer.from_crs(layout["crs"], "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def select_blind_pairs(
    v1_worst: dict,
    v1_layouts: dict,
    common_keys: list[str],
    count: int = 5,
) -> list[dict]:
    """Select contaminated locations using only v1, before consulting v5."""
    rng = np.random.default_rng(RNG_SEED)
    candidates: list[dict] = []
    for year in YEARS:
        for key in common_keys:
            arr = v1_worst["model_used_4"][year][key]
            eligible = np.flatnonzero(arr >= STRICT_BAD_MAX)
            if not eligible.size:
                continue
            flat = int(rng.choice(eligible))
            block_row, block_col = np.unravel_index(flat, arr.shape)
            lon, lat = block_lonlat(v1_layouts[year][key], block_row, block_col)
            candidates.append(
                {
                    "year": year,
                    "key": key,
                    "block_row": int(block_row),
                    "block_col": int(block_col),
                    "lon": float(lon),
                    "lat": float(lat),
                    "v1_model_scope_worst_bad": float(arr[block_row, block_col]),
                }
            )
    rng.shuffle(candidates)

    selected: list[dict] = []
    per_year = {year: 0 for year in YEARS}
    for candidate in candidates:
        if per_year[candidate["year"]] >= 2:
            continue
        if any(
            (candidate["lat"] - other["lat"]) ** 2
            + (candidate["lon"] - other["lon"]) ** 2
            < 0.025**2
            for other in selected
        ):
            continue
        selected.append(candidate)
        per_year[candidate["year"]] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"could only select {len(selected)} blind RGB pairs")
    return selected


def read_rgb_chip(path: Path, center_row: int, center_col: int, radius: int = 64) -> np.ndarray:
    with rasterio.open(path) as src:
        r0 = max(0, center_row - radius)
        c0 = max(0, center_col - radius)
        r1 = min(src.height, center_row + radius)
        c1 = min(src.width, center_col + radius)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        rgb = np.stack([src.read(index, window=window) for index in RGB_INDEXES]).astype(
            np.float32
        )
    # Fixed stretch preserves brightness differences instead of beautifying each panel.
    rgb = np.clip(rgb / 3000.0, 0, 1) ** 0.8
    return np.moveaxis(rgb, 0, -1)


def create_rgb_pairs(
    out: Path,
    selected: list[dict],
    v1_layouts: dict,
    v5_layouts: dict,
) -> list[dict]:
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3.25 * len(selected)))
    records: list[dict] = []
    for row_index, pick in enumerate(selected):
        year, key = pick["year"], pick["key"]
        v1_paths = v1_layouts[year][key]["paths"][: SCOPES["model_used_4"]]
        v5_paths = v5_layouts[year][key]["paths"][: SCOPES["model_used_4"]]
        v1_period_bad = [
            quality_at_block(path, pick["block_row"], pick["block_col"])
            for path in v1_paths
        ]
        period = int(np.argmax(v1_period_bad))
        v5_bad = quality_at_block(
            v5_paths[period], pick["block_row"], pick["block_col"]
        )
        center_row = pick["block_row"] * BLOCK + BLOCK // 2
        center_col = pick["block_col"] * BLOCK + BLOCK // 2
        images = (
            read_rgb_chip(v1_paths[period], center_row, center_col),
            read_rgb_chip(v5_paths[period], center_row, center_col),
        )
        for column, (dataset, image) in enumerate(zip(("v1 MOSAIC", "v5 composite"), images)):
            axis = axes[row_index, column]
            axis.imshow(image)
            axis.axis("off")
            bad = v1_period_bad[period] if column == 0 else v5_bad
            axis.set_title(f"{dataset} | bad proxy={bad:.2f}")
        axes[row_index, 0].set_ylabel(
            f"blind {row_index + 1}\n{year} p{period + 1}\n"
            f"{pick['lat']:.4f}, {pick['lon']:.4f}",
            fontsize=9,
        )
        record = dict(pick)
        record.update(
            {
                "period_index_zero_based": period,
                "v1_period_bad_proxy": float(v1_period_bad[period]),
                "v5_period_bad_proxy": float(v5_bad),
                "proxy_improved": bool(v5_bad < v1_period_bad[period]),
            }
        )
        records.append(record)
    fig.suptitle(
        "Jeju v1 vs v5 — locations selected from v1 only\n"
        "fixed RGB stretch (0–0.30 reflectance); inspect cloud/nodata, not color beauty",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "rgb_blind_pairs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    (out / "rgb_blind_pairs.json").write_text(
        json.dumps({"selection_rule": "v1-only deterministic", "pairs": records}, indent=2),
        encoding="utf-8",
    )
    return records


def plot_summary(out: Path, summaries: dict, rows_by_dataset: dict) -> None:
    colors = {"v1": "#8a93a6", "v5": "#16a085"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    metrics = ["cloud_proxy_mean", "zero_proxy_mean", "bad_proxy_mean", "worst_period_bad_mean"]
    labels = ["bright-cloud", "zero/mask", "combined bad", "worst period"]
    x = np.arange(len(metrics))
    for offset, dataset in zip((-0.18, 0.18), ("v1", "v5")):
        values = [summaries[dataset]["model_used_4"][metric] for metric in metrics]
        axes[0, 0].bar(x + offset, values, width=0.36, label=dataset, color=colors[dataset])
    axes[0, 0].set_xticks(x, labels, rotation=15)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Model-input scope: first 4 periods")
    axes[0, 0].legend()

    for dataset in ("v1", "v5"):
        values = [
            summaries[dataset]["model_used_4"]["year"][year]["bad_proxy_mean"]
            for year in YEARS
        ]
        axes[0, 1].plot(YEARS, values, marker="o", label=dataset, color=colors[dataset])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Combined bad proxy by year (first 4 periods)")
    axes[0, 1].legend()

    scope_labels = ["first 4", "all 12"]
    for offset, dataset in zip((-0.18, 0.18), ("v1", "v5")):
        values = [
            summaries[dataset][scope]["strict_clean_ratio"]
            for scope in ("model_used_4", "all_12")
        ]
        axes[1, 0].bar(
            np.arange(2) + offset,
            values,
            width=0.36,
            label=dataset,
            color=colors[dataset],
        )
    axes[1, 0].set_xticks(np.arange(2), scope_labels)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title(f"Strict clean coverage (worst bad ≤ {STRICT_BAD_MAX})")
    axes[1, 0].legend()

    for dataset in ("v1", "v5"):
        selected = [
            row
            for row in rows_by_dataset[dataset]
            if row["scope"] == "model_used_4"
        ]
        axes[1, 1].hist(
            [row["worst_period_bad_mean"] for row in selected],
            bins=np.linspace(0, 1, 31),
            alpha=0.55,
            label=dataset,
            color=colors[dataset],
        )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_title("Window-year worst-period bad distribution")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Jeju input quality audit: MOSAIC v1 vs composite v5", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "quality_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    roots = {"v1": Path(args.v1_root), "v5": Path(args.v5_root)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    discovered = {dataset: discover(root, dataset) for dataset, root in roots.items()}
    matched: dict[str, list[str]] = {}
    for year in YEARS:
        matched[year] = sorted(
            set(discovered["v1"][year]) & set(discovered["v5"][year])
        )
        print(
            f"{year}: v1={len(discovered['v1'][year])}, "
            f"v5={len(discovered['v5'][year])}, matched={len(matched[year])}",
            flush=True,
        )
    common_keys = sorted(set.intersection(*(set(matched[year]) for year in YEARS)))

    structural = {
        dataset: {
            year: {
                "windows": len(discovered[dataset][year]),
                "period_counts": sorted(
                    {len(paths) for paths in discovered[dataset][year].values()}
                ),
            }
            for year in YEARS
        }
        for dataset in ("v1", "v5")
    }
    structural_gate = (
        len(common_keys) == 54
        and all(
            len(matched[year]) == 54
            and structural[dataset][year]["period_counts"] == [12]
            for dataset in ("v1", "v5")
            for year in YEARS
        )
    )
    if not structural_gate:
        raise RuntimeError(f"structural gate failed: {structural}")

    rows_by_dataset: dict[str, list[dict]] = {}
    worst_by_dataset: dict[str, dict] = {}
    layouts_by_dataset: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for dataset in ("v1", "v5"):
        rows, worst, layouts = scan_dataset(dataset, discovered[dataset], matched)
        rows_by_dataset[dataset] = rows
        worst_by_dataset[dataset] = worst
        layouts_by_dataset[dataset] = layouts
        summaries[dataset] = aggregate(rows, worst, common_keys)

    all_rows = rows_by_dataset["v1"] + rows_by_dataset["v5"]
    write_csv(out / "per_window.csv", all_rows)
    plot_summary(out, summaries, rows_by_dataset)

    selected = select_blind_pairs(
        worst_by_dataset["v1"], layouts_by_dataset["v1"], common_keys
    )
    rgb_records = create_rgb_pairs(
        out,
        selected,
        layouts_by_dataset["v1"],
        layouts_by_dataset["v5"],
    )

    v1_model = summaries["v1"]["model_used_4"]
    v5_model = summaries["v5"]["model_used_4"]
    strict_v1 = summaries["v1"]["all_12"]["strict_clean_ratio"]
    strict_v5 = summaries["v5"]["all_12"]["strict_clean_ratio"]
    comparisons = {
        "model_scope_cloud_relative_reduction": relative_reduction(
            v1_model["cloud_proxy_mean"], v5_model["cloud_proxy_mean"]
        ),
        "model_scope_bad_relative_reduction": relative_reduction(
            v1_model["bad_proxy_mean"], v5_model["bad_proxy_mean"]
        ),
        "model_scope_zero_delta_percentage_points": 100
        * (v5_model["zero_proxy_mean"] - v1_model["zero_proxy_mean"]),
        "all12_strict_clean_multiplier": strict_v5 / strict_v1 if strict_v1 else None,
        "rgb_proxy_improved_count": int(
            sum(record["proxy_improved"] for record in rgb_records)
        ),
    }
    gates = {
        "structural_216_windows_12_periods": structural_gate,
        "cloud_and_bad_reduction_at_least_25pct": (
            comparisons["model_scope_cloud_relative_reduction"] >= 0.25
            and comparisons["model_scope_bad_relative_reduction"] >= 0.25
        ),
        "all12_strict_clean_at_least_6pct_and_5x": (
            strict_v5 >= 0.06
            and comparisons["all12_strict_clean_multiplier"] is not None
            and comparisons["all12_strict_clean_multiplier"] >= 5
        ),
        "zero_proxy_not_worse_by_more_than_1pp": (
            comparisons["model_scope_zero_delta_percentage_points"] <= 1.0
        ),
        "rgb_numeric_proxy_improved_4_of_5": (
            comparisons["rgb_proxy_improved_count"] >= 4
        ),
        "rgb_manual_review": None,
    }

    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "status": "numeric_complete_manual_rgb_pending",
        "provenance": {
            "script_sha256": code_sha,
            "v1_root": str(roots["v1"]),
            "v5_root": str(roots["v5"]),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "cloud_proxy": f"B02 > {CLOUD_DN}",
            "zero_proxy": "B02 == 0 or raster mask invalid",
            "block": f"{BLOCK}x{BLOCK} 10m pixels -> 40m",
            "strict_bad_max": STRICT_BAD_MAX,
            "model_used_periods": 4,
            "all_available_periods": 12,
            "rgb_selection_seed": RNG_SEED,
        },
        "structural": structural,
        "matched_windows_by_year": {year: len(matched[year]) for year in YEARS},
        "common_spatial_keys": len(common_keys),
        "summaries": summaries,
        "comparisons": comparisons,
        "gates": gates,
        "limitations": [
            "B02 brightness is a cloud/haze proxy, not a validated cloud mask.",
            "Zero/mask is a nodata proxy; valid dark pixels may need band-wise review.",
            "The model configuration consumes only the first 4 of 12 materialized periods.",
            "RGB pairs were selected using v1 contamination only; manual review remains required.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "gates": gates}, indent=2), flush=True)
    print(f"AUDIT_NUMERIC_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
