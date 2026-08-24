from __future__ import annotations

import sys
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from analyze_olmo_release_smoke import (  # noqa: E402
    analyze,
    canonical_bytes,
    file_sha256,
    linear_cka,
    paired_valid_tokens,
    representation_metrics,
    row_l2_normalized_cka,
    spatial_cka_metrics,
    spearman_correlation,
)


class ReleaseMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        self.features = generator.normal(size=(8, 12))
        self.clusters = ["a", "b", "c", "d", "e", "f", "g", "c"]

    def test_linear_cka_is_one_under_orthogonal_change_of_basis(self) -> None:
        generator = np.random.default_rng(11)
        q, _ = np.linalg.qr(generator.normal(size=(12, 12)))
        self.assertAlmostEqual(linear_cka(self.features, self.features @ q), 1.0)

    def test_row_normalized_cka_is_scale_invariant(self) -> None:
        scales = np.arange(1, 9, dtype=np.float64)[:, None]
        self.assertAlmostEqual(
            row_l2_normalized_cka(self.features, self.features * scales), 1.0
        )

    def test_identical_representations_preserve_geometry(self) -> None:
        result = representation_metrics(
            self.features, self.features.copy(), spatial_cluster_ids=self.clusters
        )
        self.assertAlmostEqual(result["pairwise_euclidean_distance_spearman"], 1.0)
        self.assertAlmostEqual(result["pooled_linear_cka"], 1.0)
        for neighbor_k in ("1", "2"):
            self.assertAlmostEqual(
                result["neighbor_overlap"][neighbor_k]["mean_fraction"], 1.0
            )
        self.assertEqual(result["spatial_clusters"], 7)
        self.assertEqual(len(result["leave_one_spatial_cluster_out"]), 7)

    def test_chance_overlap_is_pre_registered_from_sample_count(self) -> None:
        result = representation_metrics(self.features, self.features.copy())
        self.assertAlmostEqual(result["neighbor_overlap"]["1"]["random_expectation"], 1 / 7)
        self.assertAlmostEqual(result["neighbor_overlap"]["2"]["random_expectation"], 2 / 7)

    def test_spearman_supports_ties_and_rejects_constant_ranks(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation(np.array([1, 1, 2, 3]), np.array([2, 2, 4, 6])),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "constant"):
            spearman_correlation(np.ones(4), np.arange(4))

    def test_invalid_neighbor_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "neighbor k"):
            representation_metrics(self.features, self.features, neighbor_ks=(8,))

    def test_validity_mask_mismatch_is_a_hard_failure(self) -> None:
        left = np.array([[1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
        right = np.array([[1.0, 0.0], [2.0, 0.0], [np.nan, np.nan]])
        with self.assertRaisesRegex(ValueError, "valid/nodata masks"):
            paired_valid_tokens(left, right)

    def test_feature_dimensions_may_differ_when_token_grid_matches(self) -> None:
        generator = np.random.default_rng(19)
        left = generator.normal(size=(5, 7))
        right = generator.normal(size=(5, 11))
        left_valid, right_valid = paired_valid_tokens(left, right)
        self.assertEqual(left_valid.shape, (5, 7))
        self.assertEqual(right_valid.shape, (5, 11))
        self.assertTrue(0.0 <= linear_cka(left_valid, right_valid) <= 1.0)

    def test_spatial_metric_records_toroidal_shift_control(self) -> None:
        generator = np.random.default_rng(23)
        grid = generator.normal(size=(8, 8, 6))
        result = spatial_cka_metrics(grid, grid.copy(), maximum_tokens=32)
        self.assertAlmostEqual(result["linear_cka"], 1.0)
        self.assertAlmostEqual(result["row_l2_normalized_linear_cka"], 1.0)
        self.assertGreaterEqual(len(result["toroidal_shift_null"]), 3)
        self.assertGreater(result["excess_over_shift_null_median"], 0.0)


@unittest.skipUnless(importlib.util.find_spec("rasterio"), "rasterio is server-only")
class ReleaseMetricRasterIntegrationTests(unittest.TestCase):
    def test_exact_identity_grid_and_output_contract_end_to_end(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        generator = np.random.default_rng(29)
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            exact_records = []
            outputs = {"olmoearth_v1_base": [], "olmoearth_v1_2_base": []}
            for index in range(8):
                sample_id = f"sample-{index}"
                window_name = f"window-{index}"
                cluster_id = f"cluster-{index if index < 7 else 2}"
                x_min, y_min = 1000 + 32 * index, -2032
                x_max, y_max = x_min + 32, -2000
                window_metadata = root / f"{sample_id}-window.json"
                window_metadata.write_text(
                    json.dumps(
                        {
                            "projection": {
                                "crs": "EPSG:32652",
                                "x_resolution": 10.0,
                                "y_resolution": -10.0,
                            },
                            "bounds": [x_min, y_min, x_max, y_max],
                        }
                    ),
                    encoding="utf-8",
                )
                exact_records.append(
                    {
                        "sample_id": sample_id,
                        "window_name": window_name,
                        "input_bundle_identity": f"bundle-{index}",
                        "spatial_cluster_id": cluster_id,
                        "smoke_stratum": "clear_proxy" if index % 2 == 0 else "contaminated_proxy",
                        "window_metadata": {
                            "path": window_metadata.as_posix(),
                            "bytes": window_metadata.stat().st_size,
                            "sha256": file_sha256(window_metadata),
                        },
                    }
                )
                tensor = generator.normal(size=(5, 8, 8)).astype("float32")
                for release in outputs:
                    target = root / release / f"{sample_id}.tif"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(
                        target,
                        "w",
                        driver="GTiff",
                        height=8,
                        width=8,
                        count=5,
                        dtype="float32",
                        crs="EPSG:32652",
                        transform=from_origin(x_min * 10, y_min * -10, 40, 40),
                    ) as dataset:
                        dataset.write(tensor)
                    outputs[release].append(
                        {
                            "sample_id": sample_id,
                            "window": window_name,
                            "input_bundle_identity": f"bundle-{index}",
                            "spatial_cluster_id": cluster_id,
                            "path": target.as_posix(),
                            "bytes": target.stat().st_size,
                            "sha256": file_sha256(target),
                            "mtime_ns": target.stat().st_mtime_ns,
                        }
                    )

            exact_path = root / "exact.json"
            exact_path.write_bytes(
                canonical_bytes(
                    {
                        "exact_tensor_file_pairing_ready": True,
                        "records": exact_records,
                    }
                )
            )
            run_path = root / "run.json"
            run_path.write_bytes(
                canonical_bytes(
                    {
                        "status": "complete",
                        "input_pairing": {
                            "same_manifest_for_both_releases": True,
                            "exact_inputs_sha256": file_sha256(exact_path),
                        },
                        "runs": [
                            {
                                "release_id": release,
                                "started_at": "2020-01-01T00:00:00+00:00",
                                "outputs": values,
                            }
                            for release, values in outputs.items()
                        ],
                    }
                )
            )
            complete_path = root / "COMPLETE.json"
            complete_path.write_bytes(
                canonical_bytes({"run_summary_sha256": file_sha256(run_path)})
            )
            summary, rows = analyze(run_path, complete_path, exact_path, 32)

        self.assertEqual(summary["sample_contract"]["n_records"], 8)
        self.assertEqual(summary["sample_contract"]["n_spatial_clusters"], 7)
        self.assertEqual(len(rows), 8)
        self.assertAlmostEqual(summary["per_window_spatial_cka"]["linear_cka"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
