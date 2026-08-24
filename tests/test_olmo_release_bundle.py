from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from verify_olmo_release_bundle import EvidenceChecks, parse_first_json  # noqa: E402


class ReleaseEvidenceVerifierTests(unittest.TestCase):
    def test_first_launcher_json_is_parsed_before_following_payloads(self) -> None:
        first = {"ready": True, "selected_gpu": "0"}
        text = '\n  {"ready": true, "selected_gpu": "0"}\n{"status": "complete"}\n'
        self.assertEqual(parse_first_json(text), first)

    def test_raw_file_inventory_checks_bytes_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "raw.bin"
            path.write_bytes(b"verified")
            checks = EvidenceChecks(require_raw=True)
            checks.inventory(
                "raw",
                path,
                {
                    "bytes": 8,
                    "sha256": "1c34f88707b55e6104c4eb20e71ffa3d33e414b71ef689a15fad0640d0ac58cb",
                },
                raw=True,
            )
        self.assertEqual(checks.failures, [])
        self.assertEqual(checks.raw_files_checked, 1)
        self.assertEqual(checks.raw_bytes_checked, 8)

    def test_missing_raw_is_partial_unless_raw_is_required(self) -> None:
        missing = Path("/definitely/missing/olmo-output.tif")
        optional = EvidenceChecks(require_raw=False)
        optional.inventory("output", missing, {"sha256": "0" * 64}, raw=True)
        self.assertEqual(optional.failures, [])
        self.assertEqual(optional.missing_raw, [missing.as_posix()])

        required = EvidenceChecks(require_raw=True)
        required.inventory("output", missing, {"sha256": "0" * 64}, raw=True)
        self.assertEqual(required.failures, ["output.exists"])


if __name__ == "__main__":
    unittest.main()
