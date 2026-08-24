from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from analyze_olmo_release_full import (  # noqa: E402
    EXPECTED_RELEASE_RUNS,
    NEAR_TIE_COSINE_TOLERANCE,
    FrozenEvidence,
    _cosine_distance,
    _numpy_configuration_identity,
    _observed_raster_contract,
    _thread_environment_identity,
    _validate_release_execution,
    assert_analysis_runtime_unchanged,
    canonical_bytes,
    consecutive_manifest_window_contrast_continuity,
    create_preanalysis_lock,
    cross_bind_paired_outputs_to_release_runs,
    cross_site_neighbor_overlap,
    exact_identity_retrieval,
    fit_affine_ridge,
    fit_mean_shift_translation,
    fit_translated_orthogonal_procrustes,
    lattice_axis,
    query_flat_indices,
    representation_retrieval_gate,
    select_ridge_multiplier,
    validate_frozen_evidence,
)
from finalize_olmo_release_full import (  # noqa: E402
    FINALIZER_POST_CODE_SCHEMA,
    finalizer_code_contract,
)
from run_olmo_release_full import (  # noqa: E402
    FULL_RUNNER_POST_CODE_SCHEMA,
    full_runner_code_contract,
    validate_full_release_command,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_bytes(canonical_bytes(payload))


class FullReleaseMetricTests(unittest.TestCase):
    def test_live_raster_contract_normalizes_repeated_band_dtypes(self) -> None:
        class FakeCRS:
            @staticmethod
            def to_string() -> str:
                return "EPSG:32652"

        class FakeDataset:
            height = 256
            width = 256
            count = 768
            dtypes = ("float32",) * 768
            crs = FakeCRS()
            transform = (40.0, 0.0, 0.0, 0.0, -40.0, 10240.0, 0.0, 0.0, 1.0)
            bounds = (0.0, 0.0, 10240.0, 10240.0)
            nodata = None

        observed = _observed_raster_contract(FakeDataset())
        self.assertEqual(observed["count"], 768)
        self.assertEqual(observed["dtypes"], ["float32"])

    def test_lattice_is_centered_unique_and_query_is_exact_subset(self) -> None:
        np.testing.assert_array_equal(
            lattice_axis(64), np.arange(2, 64, 4, dtype=np.int64)
        )
        query = query_flat_indices()
        self.assertEqual(query.shape, (64,))
        self.assertEqual(len(set(query.tolist())), 64)
        self.assertTrue(np.all((0 <= query) & (query < 256)))
        expected = [row * 16 + column for row in range(1, 16, 2) for column in range(1, 16, 2)]
        self.assertEqual(query.tolist(), expected)
        with self.assertRaisesRegex(ValueError, "cannot place"):
            lattice_axis(15)

    def test_translated_orthogonal_procrustes_recovers_held_out_map(self) -> None:
        generator = np.random.default_rng(42)
        source = generator.normal(size=(80, 5))
        orthogonal, _ = np.linalg.qr(generator.normal(size=(5, 5)))
        translation = np.array([[2.0, -1.0, 0.5, 3.0, -2.0]])
        target = source @ orthogonal + translation
        bridge = fit_translated_orthogonal_procrustes(source[:60], target[:60])
        np.testing.assert_allclose(bridge.transform(source[60:]), target[60:], atol=1e-11)
        np.testing.assert_allclose(bridge.matrix.T @ bridge.matrix, np.eye(5), atol=1e-11)
        self.assertEqual(bridge.digest(), bridge.digest())

    def test_mean_shift_translation_is_a_distinct_calibration_only_baseline(self) -> None:
        generator = np.random.default_rng(43)
        source = generator.normal(size=(80, 5))
        translation = np.array([[2.0, -1.0, 0.5, 3.0, -2.0]])
        target = source + translation
        bridge = fit_mean_shift_translation(source[:60], target[:60])
        self.assertEqual(bridge.method, "mean_shift_translation_only")
        np.testing.assert_array_equal(bridge.matrix, np.eye(5))
        np.testing.assert_allclose(
            bridge.transform(source[60:]), target[60:], atol=1e-12
        )

    def test_affine_ridge_and_selection_use_spatial_calibration_folds(self) -> None:
        generator = np.random.default_rng(7)
        record_count, rows_per_record, dimensions = 12, 8, 4
        source = generator.normal(size=(record_count * rows_per_record, dimensions))
        weights = generator.normal(size=(dimensions, dimensions))
        target = source @ weights + np.array([[0.4, -0.2, 0.1, 1.0]])
        clusters = [f"cluster-{index}" for index in range(record_count)]
        spatial_keys = [
            f"{100 + index % 3}_-{200 + index}" for index in range(record_count)
        ]
        selection = select_ridge_multiplier(
            source,
            target,
            clusters,
            spatial_keys,
            np.array([1, 3, 5, 7]),
            multipliers=(1e-8, 1e-4, 1.0),
        )
        self.assertEqual(selection["spatial_folds"], [100, 101, 102])
        self.assertFalse(selection["sealed_test_used_for_selection"])
        self.assertIn(selection["selected_alpha_multiplier"], (1e-8, 1e-4, 1.0))
        bridge, alpha = fit_affine_ridge(
            source, target, selection["selected_alpha_multiplier"]
        )
        self.assertGreater(alpha, 0)
        self.assertLess(np.mean((bridge.transform(source) - target) ** 2), 1e-6)

    def test_exact_retrieval_uses_stable_identity_ties(self) -> None:
        gallery = np.eye(5, dtype=np.float64)
        gallery[1] = gallery[0]
        queries = np.stack((gallery[1], gallery[4]))
        metrics, per_cluster = exact_identity_retrieval(
            queries,
            gallery,
            np.array([1, 4]),
            ["cluster-a", "cluster-b"],
            chunk_size=1,
            k_values=(1, 2),
        )
        self.assertEqual(metrics["recall_at_k"]["1"], 0.5)
        self.assertEqual(metrics["recall_at_k"]["2"], 1.0)
        self.assertEqual(metrics["queries_with_similarity_ties"], 1)
        self.assertEqual([row["median_rank"] for row in per_cluster], [2.0, 1.0])

    def test_retrieval_reports_fixed_tolerance_near_ties_and_margins(self) -> None:
        gallery = np.array(
            [[1.0, 0.0], [1.0, 1e-3], [0.0, 1.0]], dtype=np.float64
        )
        metrics, rows = exact_identity_retrieval(
            gallery[[0]],
            gallery,
            np.array([0]),
            ["cluster-a"],
            chunk_size=1,
            k_values=(1,),
        )
        self.assertEqual(metrics["queries_with_similarity_ties"], 0)
        self.assertEqual(metrics["near_tie_cosine_tolerance"], NEAR_TIE_COSINE_TOLERANCE)
        self.assertEqual(metrics["queries_with_near_tie_competitors"], 1)
        self.assertGreater(
            metrics["correct_to_best_competitor_similarity_margin"]["minimum"],
            0.0,
        )
        self.assertLessEqual(
            metrics["correct_to_best_competitor_similarity_margin"]["maximum"],
            NEAR_TIE_COSINE_TOLERANCE,
        )
        self.assertEqual(metrics["recall_at_k"]["1"], 1.0)
        self.assertEqual(
            metrics["tolerance_recall_bounds_at_k"]["1"]["pessimistic"],
            0.0,
        )
        self.assertEqual(rows[0]["queries_with_near_tie_competitors"], 1)

    def test_cross_near_ties_fail_tolerance_pessimistic_gate(self) -> None:
        native_gallery = np.eye(4, dtype=np.float64)
        native, _ = exact_identity_retrieval(
            native_gallery,
            native_gallery.copy(),
            np.arange(4),
            ["a", "b", "c", "d"],
            chunk_size=2,
            k_values=(1,),
        )
        cross_gallery = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1e-3, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cross, _ = exact_identity_retrieval(
            cross_gallery.copy(),
            cross_gallery,
            np.arange(4),
            ["a", "b", "c", "d"],
            chunk_size=2,
            k_values=(1,),
        )
        gate = representation_retrieval_gate(cross, native)
        self.assertFalse(gate["pass"])
        self.assertFalse(
            gate["checks"][
                "cross_tolerance_pessimistic_recall_at_1_at_least_0_95"
            ]
        )
        self.assertFalse(
            gate["checks"][
                "minimum_leave_one_location_cluster_out_tolerance_pessimistic_recall_at_1_at_least_0_95"
            ]
        )

    def test_collapsed_native_gallery_fails_uniqueness_and_pessimistic_gate(self) -> None:
        gallery = np.ones((4, 3), dtype=np.float64)
        metrics, _ = exact_identity_retrieval(
            gallery.copy(),
            gallery,
            np.arange(4),
            ["a", "b", "c", "d"],
            chunk_size=2,
            k_values=(1,),
        )
        self.assertEqual(
            metrics["exact_tie_recall_bounds_at_k"]["1"],
            {"optimistic": 1.0, "pessimistic": 0.0},
        )
        gate = representation_retrieval_gate(metrics, metrics)
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["checks"]["native_exact_similarity_ties_zero"])
        self.assertFalse(gate["checks"]["native_near_tie_competitors_zero"])

    def test_retrieval_gate_uses_cluster_deletion_minimum_not_normal_interval(self) -> None:
        gallery = np.eye(4, dtype=np.float64)
        metrics, _ = exact_identity_retrieval(
            gallery,
            gallery.copy(),
            np.arange(4),
            ["a", "b", "c", "d"],
            chunk_size=2,
            k_values=(1,),
        )
        metrics["cluster_jackknife"]["recall_at_1"]["normal_95_interval"] = [
            0.0,
            1.0,
        ]
        metrics["cluster_jackknife"]["pessimistic_recall_at_1"][
            "normal_95_interval"
        ] = [0.0, 1.0]
        gate = representation_retrieval_gate(metrics, metrics)
        self.assertTrue(gate["pass"])
        self.assertEqual(
            gate["minimum_leave_one_location_cluster_out_recall_at_1"], 1.0
        )
        self.assertEqual(
            gate["descriptive_location_clustered_normal_95_lower_bound"], 0.0
        )
        metrics["cluster_jackknife"]["recall_at_1"][
            "leave_one_cluster_out_range"
        ] = [0.94, 1.0]
        self.assertFalse(representation_retrieval_gate(metrics, metrics)["pass"])

    def test_runtime_revalidation_is_exact_and_thread_env_is_allowlisted(self) -> None:
        expected = {
            "numpy": {"version": "2.0", "configuration_sha256": "a" * 64},
            "rasterio_gdal": {"version": "1.4", "gdal_version": "3.9"},
            "numerical_thread_environment": {"OMP_NUM_THREADS": "4"},
            "fingerprint_sha256": "b" * 64,
        }
        self.assertEqual(
            assert_analysis_runtime_unchanged(expected, observed=dict(expected)),
            expected,
        )
        changed = json.loads(json.dumps(expected))
        changed["rasterio_gdal"]["gdal_version"] = "3.10"
        with self.assertRaisesRegex(ValueError, "runtime drifted"):
            assert_analysis_runtime_unchanged(expected, observed=changed)
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}, clear=False):
            self.assertEqual(_thread_environment_identity()["OMP_NUM_THREADS"], "4")
        with patch.dict(
            os.environ, {"OMP_NUM_THREADS": "not-a-thread-count"}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "unsafe or non-numeric"):
                _thread_environment_identity()

    def test_numpy_runtime_identity_binds_config_blas_and_extension_binaries(self) -> None:
        identity = _numpy_configuration_identity()
        self.assertEqual(identity["version"], np.__version__)
        self.assertEqual(len(identity["configuration_sha256"]), 64)
        self.assertEqual(len(identity["blas_lapack_configuration_sha256"]), 64)
        for field in ("core_extension", "linalg_extension"):
            self.assertTrue(Path(identity[field]["path"]).is_file())
            self.assertGreater(identity[field]["bytes"], 0)
            self.assertEqual(len(identity[field]["sha256"]), 64)

    def test_cosine_distance_is_clipped_and_nonnegative(self) -> None:
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        distance = _cosine_distance(vector, vector.copy())
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 2.0)

    def test_pair_inventory_is_cross_bound_to_each_release_run(self) -> None:
        identity = {
            "window": "window-a",
            "spatial_cluster_id": "cluster-a",
            "input_bundle_identity": "f" * 64,
        }
        v1 = {
            "sample-a": {
                "sample_id": "sample-a",
                **identity,
                "path": "/data/v1.tif",
                "bytes": 10,
                "sha256": "1" * 64,
                "mtime_ns": 100,
            }
        }
        v1_2 = {
            "sample-a": {
                "sample_id": "sample-a",
                **identity,
                "path": "/data/v1_2.tif",
                "bytes": 11,
                "sha256": "2" * 64,
                "mtime_ns": 101,
            }
        }
        health = {
            "sample-a": {
                "sample_id": "sample-a",
                "height": 64,
                "width": 64,
                "count": 768,
                "dtypes": ["float32"],
                "crs": "EPSG:32652",
                "transform": [1, 0, 0, 0, -1, 64],
                "bounds": [0, 0, 64, 64],
                "nodata": None,
                "usable_tokens": 4096,
                "nonzero_usable_tokens": 4096,
                "finite_values": 64 * 64 * 768,
                "total_values": 64 * 64 * 768,
                "all_values_finite": True,
            }
        }
        pairs = {
            "sample-a": {
                "sample_id": "sample-a",
                **identity,
                "v1_output": {
                    key: v1["sample-a"][key]
                    for key in ("path", "bytes", "sha256", "mtime_ns")
                },
                "v1_2_output": {
                    key: v1_2["sample-a"][key]
                    for key in ("path", "bytes", "sha256", "mtime_ns")
                },
                "raster_contract": {
                    key: health["sample-a"][key]
                    for key in (
                        "height",
                        "width",
                        "count",
                        "dtypes",
                        "crs",
                        "transform",
                        "bounds",
                        "nodata",
                    )
                },
                "v1_value_health": {
                    key: health["sample-a"][key]
                    for key in (
                        "usable_tokens",
                        "nonzero_usable_tokens",
                        "finite_values",
                        "total_values",
                        "all_values_finite",
                    )
                },
                "v1_2_value_health": {
                    key: health["sample-a"][key]
                    for key in (
                        "usable_tokens",
                        "nonzero_usable_tokens",
                        "finite_values",
                        "total_values",
                        "all_values_finite",
                    )
                },
            }
        }
        cross_bind_paired_outputs_to_release_runs(pairs, v1, v1_2, health, health)
        pairs["sample-a"]["v1_2_output"]["sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            cross_bind_paired_outputs_to_release_runs(pairs, v1, v1_2, health, health)
        pairs["sample-a"]["v1_2_output"]["sha256"] = "2" * 64
        pairs["sample-a"]["raster_contract"]["count"] = 767
        with self.assertRaisesRegex(ValueError, "raster-health structure mismatch"):
            cross_bind_paired_outputs_to_release_runs(pairs, v1, v1_2, health, health)

    def test_preanalysis_lock_is_one_time_and_binds_metric_contract(self) -> None:
        evidence = FrozenEvidence(
            exact_records={},
            pairs={},
            split_by_sample={},
            split_by_cluster={},
            evidence_hashes={"exact_inputs_sha256": "a" * 64},
            execution_contract={
                "status": "both_releases_match_one_promoted_execution_contract"
            },
            code_contracts={
                "full_runner": full_runner_code_contract(),
                "finalizer": finalizer_code_contract(),
            },
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            output_dir = Path(temporary_name) / "analysis"
            path, digest, payload = create_preanalysis_lock(
                output_dir,
                evidence,
                output_hash_workers=2,
                retrieval_chunk_size=128,
                argv=["analyze", "--output-dir", output_dir.as_posix()],
                runtime_identity={"python": "test", "numpy": "test", "rasterio": "test"},
            )
            self.assertEqual(sha(path), digest)
            self.assertEqual(payload["metric_contract"]["native_recall_at_1_floor"], 1.0)
            self.assertEqual(
                payload["metric_contract"]["near_tie_cosine_tolerance"],
                NEAR_TIE_COSINE_TOLERANCE,
            )
            self.assertEqual(
                payload["metric_contract"]["bridge_methods"],
                [
                    "identity_no_bridge",
                    "mean_shift_translation_only",
                    "translated_orthogonal_procrustes",
                    "affine_ridge",
                ],
            )
            self.assertEqual(payload["frozen_evidence"], evidence.evidence_hashes)
            self.assertEqual(
                payload["upstream_code_contracts"], evidence.code_contracts
            )
            with self.assertRaisesRegex(FileExistsError, "one-time"):
                create_preanalysis_lock(
                    output_dir,
                    evidence,
                    output_hash_workers=2,
                    retrieval_chunk_size=128,
                    argv=["analyze"],
                    runtime_identity={"python": "test"},
                )

    def test_cross_site_neighbor_overlap_excludes_all_same_site_years(self) -> None:
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.99, 0.01],
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 0.99],
            ]
        )
        sample_ids = [f"sample-{index}" for index in range(6)]
        clusters = ["a", "a", "b", "b", "c", "c"]
        metrics, per_cluster = cross_site_neighbor_overlap(
            vectors,
            vectors.copy(),
            sample_ids,
            clusters,
            k_values=(1, 2),
        )
        self.assertTrue(metrics["same_spatial_site_all_years_excluded"])
        self.assertEqual(metrics["metrics"]["overlap_at_1"]["estimate"], 1.0)
        self.assertEqual(len(per_cluster), 3)

    def test_manifest_window_contrast_has_48_sealed_contrasts(self) -> None:
        sample_ids, clusters, years, vectors = [], [], [], []
        for cluster_index in range(16):
            angle = 0.0
            for year_offset, year in enumerate((2023, 2024, 2025, 2026)):
                if year_offset:
                    angle += 0.005 * (1 + cluster_index * 3 + year_offset)
                sample_ids.append(f"{cluster_index}-{year}")
                clusters.append(f"cluster-{cluster_index:02d}")
                years.append(year)
                vectors.append([np.cos(angle), np.sin(angle), cluster_index * 1e-4])
        matrix = np.asarray(vectors)
        result = consecutive_manifest_window_contrast_continuity(
            matrix, matrix.copy(), sample_ids, clusters, years
        )
        self.assertEqual(result["contrasts"], 48)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertAlmostEqual(result["kendall_tau_b"], 1.0)
        self.assertEqual(result["top_k_overlap"]["10"]["fraction"], 1.0)
        self.assertEqual(
            result["transition_results"]["2025_to_2026"]["window_role"],
            "rolling_2026_window_contrast_not_a_prospective_change_event",
        )
        self.assertFalse(result["temporal_change_detection_claim_allowed"])


class FrozenEvidenceContractTests(unittest.TestCase):
    def test_release_execution_is_bound_to_promoted_contract_and_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            case_root = Path(temporary_name)
            result_root = case_root / "result"
            result_root.mkdir()
            checkpoint_marker = result_root / "POST_RUN_CHECKPOINTS_VERIFIED.json"
            runtime_marker = result_root / "POST_RUN_RSLEARN_RUNTIME_VERIFIED.json"
            runner_code_marker = (
                result_root / "POST_RUN_FULL_RUNNER_CODE_VERIFICATION.json"
            )
            resolved_config = case_root / "resolved_config.yaml"
            resolved_config.write_text("model: test\n", encoding="utf-8")
            rslearn_path = case_root / "rslearn"
            rslearn_path.write_text("#!/bin/sh\n", encoding="utf-8")
            bindings = {
                "exact_inputs_sha256": "a" * 64,
                "exact_complete_sha256": "b" * 64,
                "split_manifest_sha256": "c" * 64,
                "split_complete_sha256": "d" * 64,
                "selected_gpu_uuid": "GPU-test",
            }
            tuning = {"candidate_id": "b004_w02", "batch_size": 4, "num_workers": 2}
            semantic_core = {"inputs": ["sentinel2_l2a"], "timestamps": "legacy"}
            runtime = {"python": "3.11", "torch": "2"}
            rslearn = {
                "schema": "test-runtime",
                "sha256": "e" * 64,
                "entrypoint": {"path": rslearn_path.resolve().as_posix()},
            }
            runner_code = full_runner_code_contract()
            write_json(
                runner_code_marker,
                {
                    "schema": FULL_RUNNER_POST_CODE_SCHEMA,
                    "status": "verified",
                    "initial_full_runner_code_contract": runner_code,
                    "live_full_runner_code_contract": runner_code,
                    "error": None,
                },
            )
            checkpoint_files = [
                {"name": "config.json", "bytes": 10, "sha256": "1" * 64},
                {"name": "weights.pth", "bytes": 20, "sha256": "2" * 64},
            ]
            revision = "3" * 40
            write_json(
                checkpoint_marker,
                {
                    "schema": "olmoearth-release-post-run-checkpoint-verification-v1",
                    "status": "verified",
                    "checkpoint_manifest_sha256": "4" * 64,
                    "repo_id": "allenai/OlmoEarth-v1-Base",
                    "revision": revision,
                    "checkpoint_files": checkpoint_files,
                },
            )
            write_json(
                runtime_marker,
                {
                    "schema": "olmoearth-release-post-run-rslearn-runtime-verification-v1",
                    "status": "verified",
                    "rslearn_runtime_fingerprint": rslearn,
                },
            )
            execution_contract = {
                "schema": "olmoearth-release-execution-contract-v1",
                "selected_tuning": tuning,
                "semantic_config_core": semantic_core,
                "runtime_versions": runtime,
                "rslearn_runtime_fingerprint": rslearn,
                "physical_gpu": {"index": "0", "uuid": "GPU-test"},
                "checkpoint_manifest_sha256": "4" * 64,
                "releases": {
                    "allenai/OlmoEarth-v1-Base": {
                        "release_id": "olmoearth_v1_base",
                        "repo_id": "allenai/OlmoEarth-v1-Base",
                        "revision": revision,
                        "files": checkpoint_files,
                    }
                },
                "validation": {"numerical_equivalence": True},
            }
            promotion_evidence = []
            for index in range(3):
                path = result_root / f"promotion-evidence-{index}.json"
                write_json(path, {"index": index})
                promotion_evidence.append({"path": path.as_posix(), "sha256": sha(path)})
            batch_contract = {
                "schema": "olmoearth-release-batch-contract-v1",
                "status": "promoted",
                "selected": tuning,
                "full_run_allowed": True,
                "promotion_pending": [],
                "gate_checks": {"numerical_equivalence": True},
                "evidence_files": promotion_evidence,
                "execution_contract": execution_contract,
            }
            preflight = {
                "schema": "olmoearth-release-full-preflight-v1",
                "status": "ready",
                "records": 216,
                "spatial_clusters": 54,
                "selected_gpu": "0",
                "selected_gpu_uuid": "GPU-test",
                **{key: value for key, value in bindings.items() if key != "selected_gpu_uuid"},
                "batch_contract_sha256": hashlib.sha256(
                    canonical_bytes(batch_contract)
                ).hexdigest(),
                "batch_contract_complete_sha256": "0" * 64,
                "checkpoint_manifest_sha256": "4" * 64,
                "rslearn_runtime_fingerprint": rslearn,
                "full_runner_code_contract": runner_code,
                "batch_contract": batch_contract,
                "promoted_execution_contract_check": {
                    "schema": "olmoearth-release-full-execution-contract-check-v1",
                    "status": "matched",
                    "repo_id": "allenai/OlmoEarth-v1-Base",
                    "revision": revision,
                    "physical_gpu": {"index": "0", "uuid": "GPU-test"},
                    "semantic_config_core": semantic_core,
                    "runtime_versions": runtime,
                    "rslearn_runtime_fingerprint": rslearn,
                    "checkpoint_manifest_sha256": "4" * 64,
                    "execution_binding": {
                        "output_layer": "embeddings_full_v1_legacy",
                        "batch_size": 4,
                        "num_workers": 2,
                        "dataset_path_environment": "DATASET_PATH",
                    },
                },
            }
            preflight_path = result_root / "preflight.json"
            write_json(preflight_path, preflight)
            health = [
                {
                    "sample_id": f"sample-{index}",
                    "height": 64,
                    "width": 64,
                    "count": 768,
                    "dtypes": ["float32"],
                    "crs": "EPSG:32652",
                    "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 64.0],
                    "bounds": [0.0, 0.0, 64.0, 64.0],
                    "nodata": None,
                    "usable_tokens": 4096,
                    "nonzero_usable_tokens": 4096,
                    "finite_values": 64 * 64 * 768,
                    "total_values": 64 * 64 * 768,
                    "all_values_finite": True,
                }
                for index in range(216)
            ]
            executed_command = [
                rslearn_path.resolve().as_posix(),
                "model",
                "predict",
                "--config",
                resolved_config.resolve().as_posix(),
            ]
            run = {
                "schema": "olmoearth-release-full-result-v1",
                "status": "complete",
                "failure_reasons": [],
                "records": 216,
                "release": {
                    "release_id": "olmoearth_v1_base",
                    "output_layer": "embeddings_full_v1_legacy",
                },
                "repo_id": "allenai/OlmoEarth-v1-Base",
                "revision": revision,
                "checkpoint_files": checkpoint_files,
                "outputs": [
                    {
                        "sample_id": value["sample_id"],
                        "window": f"window-{value['sample_id']}",
                        "spatial_cluster_id": f"cluster-{index}",
                        "input_bundle_identity": hashlib.sha256(
                            value["sample_id"].encode()
                        ).hexdigest(),
                        "path": f"/data/{value['sample_id']}.tif",
                        "bytes": 10,
                        "sha256": hashlib.sha256(
                            f"output-{value['sample_id']}".encode()
                        ).hexdigest(),
                        "mtime_ns": index + 1,
                    }
                    for index, value in enumerate(health)
                ],
                "output_health": health,
                "batch_size": 4,
                "num_workers": 2,
                "post_run_inputs_verified": True,
                "post_run_checkpoints_verified": True,
                "post_run_rslearn_runtime_verified": True,
                "post_run_checkpoint_verification": {
                    "path": checkpoint_marker.as_posix(),
                    "sha256": sha(checkpoint_marker),
                },
                "post_run_rslearn_runtime_verification": {
                    "path": runtime_marker.as_posix(),
                    "sha256": sha(runtime_marker),
                },
                "resolved_config_sha256": sha(resolved_config),
                "executed_command": executed_command,
                "executed_command_contract": validate_full_release_command(
                    executed_command,
                    rslearn_entrypoint=rslearn_path,
                    resolved_config=resolved_config,
                ),
                "full_runner_code_contract": runner_code,
                "post_run_full_runner_code_verified": True,
                "post_run_full_runner_code_error": None,
                "post_run_full_runner_code_verification": {
                    "path": runner_code_marker.resolve().as_posix(),
                    "sha256": sha(runner_code_marker),
                },
                "preflight_sha256": sha(preflight_path),
            }
            run_path = result_root / "run_summary.json"
            complete_path = result_root / "RELEASE_COMPLETE.json"
            write_json(run_path, run)
            write_json(
                complete_path,
                {
                    "schema": "olmoearth-release-full-completion-v1",
                    "status": "complete",
                    "run_summary_sha256": sha(run_path),
                    "post_run_full_runner_code_verified": True,
                    "full_runner_code_contract_sha256": runner_code[
                        "inventory_sha256"
                    ],
                    "post_run_full_runner_code_verification_sha256": sha(
                        runner_code_marker
                    ),
                },
            )
            result = _validate_release_execution(
                run_path,
                complete_path,
                EXPECTED_RELEASE_RUNS["v1"],
                sha(run_path),
                sha(complete_path),
                bindings,
            )
            self.assertEqual(
                result["execution_contract_sha256"],
                hashlib.sha256(canonical_bytes(execution_contract)).hexdigest(),
            )
            write_json(
                runtime_marker,
                {
                    "schema": "olmoearth-release-post-run-rslearn-runtime-verification-v1",
                    "status": "verified",
                    "rslearn_runtime_fingerprint": {"schema": "tampered"},
                },
            )
            run["post_run_rslearn_runtime_verification"]["sha256"] = sha(runtime_marker)
            write_json(run_path, run)
            complete = json.loads(complete_path.read_text())
            complete["run_summary_sha256"] = sha(run_path)
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "rslearn marker content drift"):
                _validate_release_execution(
                    run_path,
                    complete_path,
                    EXPECTED_RELEASE_RUNS["v1"],
                    sha(run_path),
                    sha(complete_path),
                    bindings,
                )

    def _bundle(self, root: Path) -> dict[str, Path]:
        x_values = [22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720]
        y_values = [-372736, -371712, -370688, -369664, -368640, -367616]
        split_x = {
            22528: "calibration",
            23552: "calibration",
            24576: "calibration",
            25600: "calibration",
            26624: "calibration",
            27648: "embargo",
        }
        disclosed = {"28672_-372736", "29696_-367616"}
        records, assignments, pairs = [], [], []
        for x_value in x_values:
            for y_value in y_values:
                key = f"{x_value}_{y_value}"
                cluster = f"jeju-window-{key}"
                if x_value in split_x:
                    split_name = split_x[x_value]
                elif key in disclosed:
                    split_name = "disclosed_audit"
                else:
                    split_name = "sealed_test"
                sample_ids = []
                for year in (2023, 2024, 2025, 2026):
                    sample_id = f"legacy__{year}__{key}"
                    sample_ids.append(sample_id)
                    identity = hashlib.sha256(sample_id.encode()).hexdigest()
                    records.append(
                        {
                            "sample_id": sample_id,
                            "window_name": f"window-{sample_id}",
                            "spatial_cluster_id": cluster,
                            "spatial_key": key,
                            "year": year,
                            "hash_policy": "sha256",
                            "input_bundle_identity": identity,
                            "input_layers": [
                                {"period_index": index} for index in range(12)
                            ],
                        }
                    )
                    contract = {
                        "height": 64,
                        "width": 64,
                        "count": 768,
                        "dtypes": ["float32"],
                        "crs": "EPSG:32652",
                        "transform": [1, 0, 0, 0, -1, 0],
                        "bounds": [0, 0, 64, 64],
                        "nodata": -9999.0,
                    }
                    pairs.append(
                        {
                            "sample_id": sample_id,
                            "window": f"window-{sample_id}",
                            "spatial_cluster_id": cluster,
                            "input_bundle_identity": identity,
                            "v1_output": {
                                "path": f"/tmp/v1-{sample_id}.tif",
                                "bytes": 10,
                                "sha256": "1" * 64,
                                "mtime_ns": 1,
                            },
                            "v1_2_output": {
                                "path": f"/tmp/v12-{sample_id}.tif",
                                "bytes": 10,
                                "sha256": "2" * 64,
                                "mtime_ns": 1,
                            },
                            "raster_contract": contract,
                            "v1_value_health": {
                                "usable_tokens": 4096,
                                "nonzero_usable_tokens": 4096,
                                "finite_values": 64 * 64 * 768,
                                "total_values": 64 * 64 * 768,
                                "all_values_finite": True,
                            },
                            "v1_2_value_health": {
                                "usable_tokens": 4096,
                                "nonzero_usable_tokens": 4096,
                                "finite_values": 64 * 64 * 768,
                                "total_values": 64 * 64 * 768,
                                "all_values_finite": True,
                            },
                            "validity_masks_exact": True,
                            "value_health_passed_both_releases": True,
                        }
                    )
                assignments.append(
                    {
                        "spatial_key": key,
                        "spatial_cluster_id": cluster,
                        "split": split_name,
                        "years": [2023, 2024, 2025, 2026],
                        "sample_ids": sorted(sample_ids),
                    }
                )
        exact = root / "exact.json"
        exact_complete = root / "EXACT_INPUTS_COMPLETE.json"
        split = root / "split.json"
        split_complete = root / "SPLIT_COMPLETE.json"
        prior_disclosure = root / "prior_disclosure.json"
        paired = root / "paired_outputs.jsonl"
        evidence = root / "evidence_summary.json"
        evidence_complete = root / "FULL_EVIDENCE_COMPLETE.json"
        finalizer_marker = root / "POST_RUN_FINALIZER_CODE_VERIFICATION.json"
        runner_code = full_runner_code_contract()
        finalizer_code = finalizer_code_contract()
        write_json(
            finalizer_marker,
            {
                "schema": FINALIZER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_finalizer_code_contract": finalizer_code,
                "live_finalizer_code_contract": finalizer_code,
                "error": None,
            },
        )
        write_json(
            exact,
            {
                "schema": "olmoearth-release-exact-input-selection-v1",
                "exact_tensor_file_pairing_ready": True,
                "records": records,
            },
        )
        write_json(
            exact_complete,
            {
                "schema": "olmoearth-release-exact-input-completion-v1",
                "exact_inputs_sha256": sha(exact),
                "records": 216,
                "spatial_clusters": 54,
                "years": [2023, 2024, 2025, 2026],
                "unique_files": 5616,
            },
        )
        write_json(
            prior_disclosure,
            {
                "records": [
                    {"spatial_key": "28672_-372736"},
                    {"spatial_key": "29696_-367616"},
                    {"spatial_key": "22528_-367616"},
                ]
            },
        )
        write_json(
            split,
            {
                "schema": "olmoearth-release-spatial-split-v1",
                "frozen_before_full_output_inspection": True,
                "exact_inputs": {"sha256": sha(exact)},
                "prior_disclosure_inputs": {
                    "path": prior_disclosure.as_posix(),
                    "sha256": sha(prior_disclosure),
                },
                "split_rule": {
                    "calibration_x": [22528, 23552, 24576, 25600, 26624],
                    "embargo_x": [27648],
                    "east_x": [28672, 29696, 30720],
                    "disclosed_east_clusters_removed_from_test": [
                        "28672_-372736",
                        "29696_-367616",
                    ],
                },
                "counts": {
                    "calibration": {
                        "spatial_clusters": 30,
                        "site_years": 120,
                        "adjacent_year_events": 90,
                    },
                    "embargo": {
                        "spatial_clusters": 6,
                        "site_years": 24,
                        "adjacent_year_events": 18,
                    },
                    "sealed_test": {
                        "spatial_clusters": 16,
                        "site_years": 64,
                        "adjacent_year_events": 48,
                    },
                    "disclosed_audit": {
                        "spatial_clusters": 2,
                        "site_years": 8,
                        "adjacent_year_events": 6,
                    },
                },
                "analysis_contract": {
                    "bridge_fit": "calibration only",
                    "hyperparameter_selection": "grouped inner validation within calibration only",
                    "all_years_of_each_location_share_one_split": True,
                },
                "assignments": assignments,
            },
        )
        write_json(
            split_complete,
            {
                "schema": "olmoearth-release-spatial-split-completion-v1",
                "split_manifest_sha256": sha(split),
                "frozen_before_full_output_inspection": True,
            },
        )
        paired.write_bytes(b"".join(canonical_bytes(value) for value in sorted(pairs, key=lambda value: value["sample_id"])))
        write_json(
            evidence,
            {
                "schema": "olmoearth-release-full-paired-evidence-v1",
                "status": "complete",
                "paired_outputs": 216,
                "paired_outputs_jsonl_sha256": sha(paired),
                "exact_inputs_sha256": sha(exact),
                "exact_complete_sha256": sha(exact_complete),
                "split_manifest_sha256": sha(split),
                "split_complete_sha256": sha(split_complete),
                "raster_contracts_exact": True,
                "validity_masks_exact": True,
                "same_exact_inputs_both_releases": True,
                "value_health_passed_all_432_outputs": True,
                "selected_gpu_uuid": "GPU-test",
                "input_post_run_closure": {
                    "files": 5616,
                    "all_sha256_match_frozen_manifest": True,
                },
                "output_post_run_closure": {
                    "files": 432,
                    "all_sha256_match_release_manifests": True,
                },
                "v1_run_summary_sha256": "3" * 64,
                "v1_complete_sha256": "4" * 64,
                "v1_2_run_summary_sha256": "5" * 64,
                "v1_2_complete_sha256": "6" * 64,
                "finalizer_code_sha256": finalizer_code["owner"]["sha256"],
                "finalizer_code_contract": finalizer_code,
                "post_run_finalizer_code_verified": True,
                "post_run_finalizer_code_verification": {
                    "path": finalizer_marker.resolve().as_posix(),
                    "sha256": sha(finalizer_marker),
                },
                "full_runner_code_contract": runner_code,
                "claims_forbidden": [
                    "task_accuracy",
                    "negative_transfer_reduction",
                    "cloud_robustness",
                    "korean_population_generalization",
                ],
            },
        )
        write_json(
            evidence_complete,
            {
                "schema": "olmoearth-release-full-paired-evidence-completion-v1",
                "status": "complete",
                "evidence_summary_sha256": sha(evidence),
                "paired_outputs_jsonl_sha256": sha(paired),
                "finalizer_code_contract_sha256": finalizer_code[
                    "inventory_sha256"
                ],
                "post_run_finalizer_code_verified": True,
                "post_run_finalizer_code_verification_sha256": sha(
                    finalizer_marker
                ),
                "full_runner_code_contract_sha256": runner_code[
                    "inventory_sha256"
                ],
            },
        )
        return {
            "evidence_summary_path": evidence,
            "evidence_complete_path": evidence_complete,
            "paired_outputs_path": paired,
            "exact_inputs_path": exact,
            "exact_complete_path": exact_complete,
            "split_manifest_path": split,
            "split_complete_path": split_complete,
        }

    def test_frozen_contract_accepts_complete_chain_and_rejects_marker_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = self._bundle(Path(temporary_name))
            result = validate_frozen_evidence(**paths)
            self.assertEqual(len(result.pairs), 216)
            self.assertEqual(
                Counter(result.split_by_cluster.values()),
                Counter(
                    {
                        "calibration": 30,
                        "embargo": 6,
                        "sealed_test": 16,
                        "disclosed_audit": 2,
                    }
                ),
            )
            with paths["paired_outputs_path"].open("ab") as output:
                output.write(b"\n")
            with self.assertRaisesRegex(ValueError, "does not bind paired_outputs"):
                validate_frozen_evidence(**paths)

    def test_split_validator_rejects_coordinate_rule_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = self._bundle(Path(temporary_name))
            split = json.loads(paths["split_manifest_path"].read_text())
            assignment = next(
                value
                for value in split["assignments"]
                if value["spatial_key"] == "30720_-367616"
            )
            assignment["split"] = "calibration"
            write_json(paths["split_manifest_path"], split)
            split_complete = json.loads(paths["split_complete_path"].read_text())
            split_complete["split_manifest_sha256"] = sha(paths["split_manifest_path"])
            write_json(paths["split_complete_path"], split_complete)
            evidence = json.loads(paths["evidence_summary_path"].read_text())
            evidence["split_manifest_sha256"] = sha(paths["split_manifest_path"])
            evidence["split_complete_sha256"] = sha(paths["split_complete_path"])
            write_json(paths["evidence_summary_path"], evidence)
            marker = json.loads(paths["evidence_complete_path"].read_text())
            marker["evidence_summary_sha256"] = sha(paths["evidence_summary_path"])
            write_json(paths["evidence_complete_path"], marker)
            with self.assertRaisesRegex(ValueError, "coordinate rule"):
                validate_frozen_evidence(**paths)

    def test_split_validator_rejects_prior_disclosure_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = self._bundle(Path(temporary_name))
            split = json.loads(paths["split_manifest_path"].read_text())
            prior_path = Path(split["prior_disclosure_inputs"]["path"])
            write_json(prior_path, {"records": [{"spatial_key": "30720_-367616"}]})
            with self.assertRaisesRegex(ValueError, "prior-disclosure source evidence drifted"):
                validate_frozen_evidence(**paths)


if __name__ == "__main__":
    unittest.main()
