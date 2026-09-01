#!/usr/bin/env python3
"""Post-hoc robustness audit for the Nepal NP-88 token-scale result.

NP-88 was a prospective-score / retrospective-label comparison, but its first
report pooled overlapping 40 m tokens and compared OlmoEarth only with weak
classical change scores.  This read-only audit adds:

* AUROC and AUPRC, including each external-label provider separately;
* water-sensitive and spectral classical controls on the exact same tokens;
* 5.12 km spatial-block summaries and a paired block-bootstrap interval;
* river-distance restrictions and same-window, same-distance-bin conditional
  AUROC to diagnose whether the score merely rediscovers the river corridor;
* hashes for every compact source used by the audit.

This is a post-hoc diagnostic, not a new preregistered confirmation.  It never
rewrites the Nepal sidecar or its sealed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fiona
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as shapely_transform, unary_union
from shapely.prepared import prep
from sklearn.metrics import average_precision_score, roc_auc_score


LABEL_PATHS = {
    "iwm_planetscope_vap02": (
        "external_data/nepal_olmo_live_v1/external_labels_20260831/"
        "IWM_2026_08_26_Nepal_Flood_VAP02/IWM_2026_08_26_Nepal_Flood_VAP02.shp"
    ),
    "tasa_formosat5_0816_0828": (
        "external_data/nepal_olmo_live_v1/external_labels_20260831/"
        "Affected_Flood_in_Rasuwa_District_Nepal/Affected_0816_0828.shp"
    ),
    "jaxa_alos2_fldext_20260828": (
        "external_data/nepal_olmo_live_v1/external_labels_20260831/"
        "JAXA_20250828_FPM_ALOS2_Nepal_FLDEXT/JAXA_20250828_FPM_ALOS2_Nepal_FLDEXT/"
        "2026-00033-WLD_202608280620_FLDEXT.geojson"
    ),
}
SCORES = (
    "olmo_delta",
    "abs_delta_ndvi",
    "abs_delta_ndwi",
    "abs_delta_mndwi",
    "post_ndwi",
    "post_mndwi",
    "abs_band_difference",
    "spectral_angle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vector(path: Path):
    geometries = []
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True).transform
    with fiona.open(path) as source:
        source_crs = source.crs_wkt or source.crs
        transformer = None
        if source_crs:
            transformer = Transformer.from_crs(
                source_crs, "EPSG:32645", always_xy=True
            ).transform
        for feature in source:
            geometry = shape(feature["geometry"])
            geometry = shapely_transform(transformer or to_utm, geometry)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            geometries.append(geometry)
    if not geometries:
        raise ValueError(f"no geometry in {path}")
    return unary_union(geometries)


def metric(scores: np.ndarray, labels: np.ndarray) -> dict | None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    return {
        "n": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def conditional_auc(records: list[dict], score: str, distance_bin_m: int = 80) -> dict:
    numerator = 0.0
    denominator = 0
    window_ids = sorted({record["window_id"] for record in records})
    for window_id in window_ids:
        window = [record for record in records if record["window_id"] == window_id]
        max_distance = max(record["river_distance_m"] for record in window)
        for lower in range(0, int(max_distance) + distance_bin_m, distance_bin_m):
            stratum = [
                record
                for record in window
                if lower <= record["river_distance_m"] < lower + distance_bin_m
            ]
            positive = np.asarray(
                [record[score] for record in stratum if record["label_union"]],
                dtype=float,
            )
            negative = np.asarray(
                [record[score] for record in stratum if not record["label_union"]],
                dtype=float,
            )
            if not len(positive) or not len(negative):
                continue
            numerator += float(np.sum(positive[:, None] > negative[None, :]))
            numerator += 0.5 * float(np.sum(positive[:, None] == negative[None, :]))
            denominator += len(positive) * len(negative)
    return {
        "distance_bin_m": distance_bin_m,
        "comparable_pairs": int(denominator),
        "conditional_auroc": float(numerator / denominator) if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepal-repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()

    repo = args.nepal_repo.resolve()
    artifact_root = args.artifact_root.resolve()
    scan_path = artifact_root / "corridor_s2_candidates/embed_scan_v2/report.json"
    manifest_path = artifact_root / "corridor_s2_candidates/prepare_v2/windows_manifest.json"
    delta_root = artifact_root / "corridor_s2_candidates/embed_scan_v2/deltas"
    cube_root = artifact_root / "corridor_s2_candidates/prepare_v2"
    river_path = repo / "web/public/data/hydrography.geojson"

    labels = {
        name: load_vector(artifact_root / relative)
        for name, relative in LABEL_PATHS.items()
    }
    labels["union"] = unary_union(list(labels.values()))
    prepared_labels = {name: prep(geometry) for name, geometry in labels.items()}

    river_data = json.loads(river_path.read_text(encoding="utf-8"))
    route = river_data["simulation_route"]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    route_x, route_y = to_utm.transform(
        [point[0] for point in route], [point[1] for point in route]
    )
    river = LineString(zip(route_x, route_y))

    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_kind = {window["id"]: window.get("kind") for window in manifest["windows"]}
    records: list[dict] = []
    sources = [
        scan_path,
        manifest_path,
        river_path,
        repo / "code/corridor_s2_candidates_embed.py",
        repo / "code/score_external_tokens.py",
        repo / "code/score_external_tokens_classical.py",
    ]

    def pool4(array: np.ndarray) -> np.ndarray:
        return array.reshape(64, 4, 64, 4).mean(axis=(1, 3))

    def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) / (a + b + 1e-6)

    ranked_windows = [window for window in scan["windows"] if window.get("status") == "ranked"]
    for window in ranked_windows:
        delta_path = delta_root / f"{window['id']}_delta.npz"
        cube_path = cube_root / f"{window['id']}.npz"
        if not delta_path.exists() or not cube_path.exists():
            raise FileNotFoundError(f"missing NP-88 input for {window['id']}")
        sources.extend([delta_path, cube_path])
        delta = np.load(delta_path)
        cube = np.load(cube_path)["cube"].astype("float32")
        valid = delta["valid_event"].astype(bool)
        base = cube[:, 0:3].mean(axis=1)
        post = cube[:, 4]
        base_vector = base.reshape(12, 64, 4, 64, 4).mean(axis=(2, 4))
        post_vector = post.reshape(12, 64, 4, 64, 4).mean(axis=(2, 4))
        dot = (base_vector * post_vector).sum(axis=0)
        spectral_angle = np.arccos(
            np.clip(
                dot
                / (
                    np.linalg.norm(base_vector, axis=0)
                    * np.linalg.norm(post_vector, axis=0)
                    + 1e-6
                ),
                -1,
                1,
            )
        )
        ndvi = lambda array: normalized_difference(array[3], array[2])
        ndwi = lambda array: normalized_difference(array[1], array[3])
        mndwi = lambda array: normalized_difference(array[1], array[8])
        score_arrays = {
            "olmo_delta": delta["d_event"],
            "abs_delta_ndvi": np.abs(pool4(ndvi(post)) - pool4(ndvi(base))),
            "abs_delta_ndwi": np.abs(pool4(ndwi(post)) - pool4(ndwi(base))),
            "abs_delta_mndwi": np.abs(pool4(mndwi(post)) - pool4(mndwi(base))),
            "post_ndwi": pool4(ndwi(post)),
            "post_mndwi": pool4(mndwi(post)),
            "abs_band_difference": pool4(np.abs(post - base).mean(axis=0)),
            "spectral_angle": spectral_angle,
        }
        x0, y0, x1, y1 = map(float, delta["bounds_utm"])
        step = (x1 - x0) / 64
        xs = x0 + (np.arange(64) + 0.5) * step
        ys = y1 - (np.arange(64) + 0.5) * step
        for row_index, y in enumerate(ys):
            for column_index, x in enumerate(xs):
                if not valid[row_index, column_index]:
                    continue
                point = Point(float(x), float(y))
                record = {
                    "window_id": window["id"],
                    "window_kind": window_kind.get(window["id"]),
                    "x": float(x),
                    "y": float(y),
                    "river_distance_m": float(river.distance(point)),
                }
                for name, array in score_arrays.items():
                    record[name] = float(array[row_index, column_index])
                for name, prepared in prepared_labels.items():
                    record[f"label_{name}"] = int(prepared.contains(point))
                records.append(record)

    arrays = {
        key: np.asarray([record[key] for record in records])
        for key in [*SCORES, "river_distance_m", *[f"label_{name}" for name in labels]]
    }
    pooled = {
        score: metric(arrays[score], arrays["label_union"])
        for score in SCORES
    }
    provider_sensitivity = {
        name: metric(arrays["olmo_delta"], arrays[f"label_{name}"])
        for name in LABEL_PATHS
    }
    river_restricted = {}
    for radius in (40, 80, 160, 320, 640):
        selected = arrays["river_distance_m"] <= radius
        river_restricted[str(radius)] = {
            score: metric(arrays[score][selected], arrays["label_union"][selected])
            for score in SCORES
        }
    conditional = {score: conditional_auc(records, score) for score in SCORES}

    blocks: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = (int(record["x"] // 5120), int(record["y"] // 5120))
        blocks.setdefault(key, []).append(index)
    block_rows = []
    for key, indices in sorted(blocks.items()):
        labels_block = arrays["label_union"][indices]
        if labels_block.sum() == 0 or labels_block.sum() == len(labels_block):
            continue
        row = {"block": list(key), "n": len(indices), "positive_fraction": float(labels_block.mean())}
        for score in SCORES:
            row[f"auroc_{score}"] = float(roc_auc_score(labels_block, arrays[score][indices]))
        block_rows.append(row)
    spatial_block_macro = {}
    for score in SCORES:
        values = np.asarray([row[f"auroc_{score}"] for row in block_rows])
        spatial_block_macro[score] = {
            "n_blocks": len(values),
            "mean_auroc": float(values.mean()),
            "median_auroc": float(np.median(values)),
            "iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
        }

    rng = np.random.default_rng(20260901)
    differences = np.asarray(
        [row["auroc_olmo_delta"] - row["auroc_post_ndwi"] for row in block_rows]
    )
    bootstrap = np.empty(args.bootstrap_repetitions, dtype=float)
    for index in range(args.bootstrap_repetitions):
        bootstrap[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    paired_block_difference = {
        "contrast": "olmo_delta_minus_post_ndwi",
        "observed_mean": float(differences.mean()),
        "bootstrap_seed": 20260901,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "percentile_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }

    source_manifest = []
    for path in sorted(set(sources)):
        source_manifest.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    label_sources = []
    for relative in LABEL_PATHS.values():
        label_path = artifact_root / relative
        components = (
            sorted(label_path.parent.glob(f"{label_path.stem}.*"))
            if label_path.suffix.lower() == ".shp"
            else [label_path]
        )
        for path in components:
            label_sources.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )

    result = {
        "schema": "nepal-np88-robustness-audit-v1",
        "status": "POST_HOC_DIAGNOSTIC_NOT_CONFIRMATORY",
        "legacy_measurement": "M88",
        "namespaced_measurement": "NP-89",
        "event_count": 1,
        "window_count": len({record["window_id"] for record in records}),
        "token_count_descriptive_not_independent_n": len(records),
        "pooled": pooled,
        "provider_sensitivity": provider_sensitivity,
        "river_restricted": river_restricted,
        "same_window_same_river_distance_bin": conditional,
        "spatial_block_size_m": 5120,
        "spatial_block_macro": spatial_block_macro,
        "paired_block_difference": paired_block_difference,
        "block_rows": block_rows,
        "claim_boundary": [
            "The independent event sample size is one; 122,558 overlapping tokens are not independent replicates.",
            "External products are flood proxies derived from 28 August imagery, not field-confirmed damage labels.",
            "This audit was designed after NP-88 was opened and is descriptive robustness analysis.",
            "OlmoEarth superiority requires surviving strong water-index baselines under spatial-block uncertainty and another event.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "source_manifest": source_manifest,
        "label_source_manifest": label_sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "event_count", "window_count", "token_count_descriptive_not_independent_n", "pooled", "provider_sensitivity", "spatial_block_macro", "paired_block_difference")}, indent=2))


if __name__ == "__main__":
    main()
