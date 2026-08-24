from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from freeze_olmo_release_spatial_split import build_split  # noqa: E402


class SpatialSplitTests(unittest.TestCase):
    def test_split_keeps_years_together_and_removes_disclosed_test_sites(self) -> None:
        x_values = [22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720]
        y_values = [-372736, -371712, -370688, -369664, -368640, -367616]
        records = []
        for x_value in x_values:
            for y_value in y_values:
                key = f"{x_value}_{y_value}"
                for year in (2023, 2024, 2025, 2026):
                    records.append(
                        {
                            "sample_id": f"{year}-{key}",
                            "spatial_key": key,
                            "spatial_cluster_id": f"cluster-{key}",
                            "year": year,
                            "hash_policy": "sha256",
                        }
                    )
        exact = {"exact_tensor_file_pairing_ready": True, "records": records}
        disclosed = {
            "records": [
                {"spatial_key": "28672_-372736"},
                {"spatial_key": "29696_-367616"},
                {"spatial_key": "22528_-367616"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            exact_path = root / "exact.json"
            disclosed_path = root / "disclosed.json"
            exact_path.write_text(json.dumps(exact), encoding="utf-8")
            disclosed_path.write_text(json.dumps(disclosed), encoding="utf-8")
            result = build_split(exact_path, disclosed_path)
        self.assertEqual(result["counts"]["calibration"]["spatial_clusters"], 30)
        self.assertEqual(result["counts"]["sealed_test"]["site_years"], 64)
        by_key = {value["spatial_key"]: value["split"] for value in result["assignments"]}
        self.assertEqual(by_key["28672_-372736"], "disclosed_audit")
        self.assertEqual(by_key["29696_-367616"], "disclosed_audit")
        self.assertEqual(by_key["30720_-367616"], "sealed_test")


if __name__ == "__main__":
    unittest.main()
