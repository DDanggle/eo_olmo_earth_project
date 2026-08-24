from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_olmo_release_audit_manifest import select_smoke  # noqa: E402
from resolve_olmo_release_checkpoints import snapshot_revision  # noqa: E402
from hash_olmo_release_inputs import (  # noqa: E402
    hash_referenced_files,
    read_records,
    upgrade_record,
    validate_contract,
    write_if_absent_or_identical,
)
from run_olmo_release_smoke import filter_gpu_processes, output_inventory  # noqa: E402


class SmokeSelectionTests(unittest.TestCase):
    def test_selected_input_is_upgraded_to_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            geotiff = root / "geotiff.tif"
            metadata = root / "metadata.json"
            items = root / "items.json"
            window_metadata = root / "window.json"
            geotiff.write_bytes(b"tensor")
            metadata.write_text("{}", encoding="utf-8")
            items.write_text("[]", encoding="utf-8")
            window_metadata.write_text("{}", encoding="utf-8")
            record = {
                "window_name": "window",
                "input_layers": [
                    {
                        "period_index": 0,
                        "layer_name": "sentinel2_l2a",
                        "geotiff": {"path": geotiff.as_posix(), "bytes": 6},
                        "metadata": {"path": metadata.as_posix(), "bytes": 2},
                    }
                ],
                "items_json": {"path": items.as_posix(), "bytes": 2},
                "window_metadata": {"path": window_metadata.as_posix(), "bytes": 2},
                "hash_policy": "metadata",
                "input_bundle_identity": "old",
            }
            upgraded = upgrade_record(record)
        self.assertEqual(upgraded["hash_policy"], "sha256")
        self.assertEqual(len(upgraded["input_bundle_identity"]), 64)
        self.assertEqual(len(upgraded["input_layers"][0]["geotiff"]["sha256"]), 64)

    def test_snapshot_revision_is_taken_before_symlink_resolution(self) -> None:
        path = Path("/cache/models--owner--repo/snapshots/abc123/config.json")
        self.assertEqual(snapshot_revision(path), "abc123")

    def test_full_manifest_reader_and_contract_are_jsonl_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            source = Path(temp_name) / "inputs.jsonl"
            records = []
            for year in (2023, 2024):
                record = {
                    "sample_id": f"sample-{year}",
                    "window_name": f"window-{year}",
                    "spatial_cluster_id": "cluster",
                    "year": year,
                    "input_layers": [
                        {
                            "period_index": index,
                            "layer_name": "sentinel2_l2a"
                            if index == 0
                            else f"sentinel2_l2a.{index}",
                        }
                        for index in range(12)
                    ],
                }
                records.append(record)
            source.write_text(
                "".join(f"{json.dumps(record)}\n" for record in records),
                encoding="utf-8",
            )
            loaded = read_records(source)
            validate_contract(loaded, 2, 1, {2023, 2024})
        self.assertEqual(loaded, records)

    def test_contract_rejects_spatial_leakage_shape_errors(self) -> None:
        record = {
            "sample_id": "sample",
            "window_name": "window",
            "spatial_cluster_id": "cluster",
            "year": 2023,
            "input_layers": [{"period_index": 0}],
        }
        with self.assertRaisesRegex(ValueError, "ordered periods"):
            validate_contract([record], 1, 1, {2023})

    def test_parallel_inventory_hashes_duplicate_path_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            geotiff = root / "geotiff.tif"
            metadata = root / "metadata.json"
            geotiff.write_bytes(b"tensor")
            metadata.write_bytes(b"{}")
            record = {
                "input_layers": [
                    {
                        "geotiff": {"path": geotiff.as_posix(), "bytes": 6},
                        "metadata": {"path": metadata.as_posix(), "bytes": 2},
                    }
                ],
                "items_json": {"path": metadata.as_posix(), "bytes": 2},
                "window_metadata": {"path": metadata.as_posix(), "bytes": 2},
            }
            result = hash_referenced_files([record], workers=2)
        self.assertEqual(set(result), {geotiff.as_posix(), metadata.as_posix()})
        self.assertEqual(result[geotiff.as_posix()]["bytes"], 6)

    def test_evidence_writer_refuses_different_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            target = Path(temp_name) / "evidence.json"
            write_if_absent_or_identical(target, b"same")
            write_if_absent_or_identical(target, b"same")
            with self.assertRaises(FileExistsError):
                write_if_absent_or_identical(target, b"different")

    def test_selection_is_extremal_deterministic_and_label_free(self) -> None:
        records = [
            {
                "sample_id": f"sample-{year}-{index}",
                "year": year,
                "bad_proxy_mean": float(index),
            }
            for year in (2023, 2026)
            for index in range(6)
        ]
        selected = select_smoke(records, [2023, 2026], 2, 2)
        self.assertEqual(len(selected), 8)
        by_year = {year: [value for value in selected if value["year"] == year] for year in (2023, 2026)}
        for values in by_year.values():
            clear = sorted(value["bad_proxy_mean"] for value in values if value["smoke_stratum"] == "clear_proxy")
            contaminated = sorted(
                value["bad_proxy_mean"]
                for value in values
                if value["smoke_stratum"] == "contaminated_proxy"
            )
            self.assertEqual(clear, [0.0, 1.0])
            self.assertEqual(contaminated, [4.0, 5.0])

    def test_release_configs_pin_paths_and_timestamp_mode(self) -> None:
        v1 = (ROOT / "config/olmo_release_v1_legacy.yaml").read_text(encoding="utf-8")
        v12 = (ROOT / "config/olmo_release_v1_2_legacy.yaml").read_text(encoding="utf-8")
        self.assertIn("model_path: ${OLMO_V1_MODEL_PATH}", v1)
        self.assertIn("model_path: ${OLMO_V1_2_MODEL_PATH}", v12)
        self.assertIn("use_legacy_timestamps: true", v1)
        self.assertIn("use_legacy_timestamps: true", v12)
        self.assertNotIn("model_id:", v1)
        self.assertNotIn("model_id:", v12)

    def test_output_inventory_preserves_pairing_identity_and_rejects_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            output = (
                root
                / "windows/default/window/layers/embedding/channels/geotiff.tif"
            )
            output.parent.mkdir(parents=True)
            output.write_bytes(b"embedding")
            record = {
                "sample_id": "sample",
                "window_name": "window",
                "input_bundle_identity": "bundle",
                "spatial_cluster_id": "cluster",
            }
            inventory = output_inventory(root, "embedding", [record], time.time() - 1)
            self.assertEqual(inventory[0]["sample_id"], "sample")
            self.assertEqual(inventory[0]["input_bundle_identity"], "bundle")
            with self.assertRaisesRegex(ValueError, "stale output"):
                output_inventory(root, "embedding", [record], time.time() + 1)

    def test_gpu_process_filter_is_scoped_to_selected_gpu(self) -> None:
        rows = "\n".join(
            [
                "GPU-0, 101, python3, 100 MiB",
                "GPU-1, 202, train.py, 60000 MiB",
            ]
        )
        self.assertEqual(filter_gpu_processes("GPU-0", rows), ["101, python3, 100 MiB"])
        self.assertEqual(filter_gpu_processes("GPU-free", rows), [])


if __name__ == "__main__":
    unittest.main()
