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

from run_olmo_release_full import render_config, validate_batch_contract  # noqa: E402


class FullReleaseRunnerTests(unittest.TestCase):
    def test_batch_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            path = root / "contract.json"
            evidence = []
            for index in range(3):
                evidence_path = root / f"evidence-{index}.json"
                evidence_path.write_text(f"{index}\n", encoding="utf-8")
                evidence.append(
                    {
                        "path": evidence_path.as_posix(),
                        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                    }
                )
            path.write_text(
                json.dumps(
                    {
                        "schema": "olmoearth-release-batch-contract-v1",
                        "status": "promoted",
                        "selected": {"batch_size": 1, "num_workers": 2},
                        "full_run_allowed": True,
                        "promotion_pending": [],
                        "gate_checks": {"numerical_equivalence": True},
                        "execution_contract": {
                            "schema": "olmoearth-release-execution-contract-v1"
                        },
                        "evidence_files": evidence,
                    }
                ),
                encoding="utf-8",
            )
            marker = root / "BATCH_CONTRACT_COMPLETE.json"
            marker.write_text(
                json.dumps(
                    {
                        "status": "promoted",
                        "batch_contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            validate_batch_contract(path, marker, 1, 2)
            with self.assertRaisesRegex(ValueError, "differ"):
                validate_batch_contract(path, marker, 4, 2)

    def test_full_config_is_resolved_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            template = root / "template.yaml"
            output = root / "resolved.yaml"
            template.write_text(
                "model: __MODEL_PATH__\nbatch: __BATCH_SIZE__\nworkers: __NUM_WORKERS__\nlayer: __OUTPUT_LAYER__\n",
                encoding="utf-8",
            )
            render_config(
                template,
                output,
                model_env="MODEL_PATH",
                output_layer="embedding",
                batch_size=1,
                workers=2,
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("${MODEL_PATH}", rendered)
            self.assertIn("batch: 1", rendered)
            with self.assertRaises(FileExistsError):
                render_config(
                    template,
                    output,
                    model_env="MODEL_PATH",
                    output_layer="embedding",
                    batch_size=4,
                    workers=2,
                )


if __name__ == "__main__":
    unittest.main()
