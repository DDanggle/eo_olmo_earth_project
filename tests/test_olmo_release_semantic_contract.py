from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from olmo_release_semantic_contract import (  # noqa: E402
    RELEASE_SPECS,
    fingerprint_python_sources,
    fingerprint_rslearn_runtime,
    normalize_checkpoint_manifest,
    validate_launcher_runtime_binding,
    validate_promoted_execution_contract,
    validate_resolved_config,
)
from run_olmo_release_full import render_config  # noqa: E402


RUNTIME = {
    "python": "3.11.15 test",
    "rslearn": "0.1.13",
    "torch": "2.7.1",
    "lightning": "2.5.2",
    "torch_cuda": "12.8",
    "cudnn": 91002,
}
GPU = {"index": "0", "uuid": "GPU-58459350-e802-b3ee-03be-fd3451eda731"}
MANIFEST_SHA = "a" * 64


def rslearn_fingerprint(source_sha: str = "c" * 64) -> dict:
    return {
        "schema": "olmoearth-rslearn-runtime-fingerprint-v1",
        "entrypoint": {
            "path": "/runtime/bin/rslearn",
            "bytes": 100,
            "sha256": "d" * 64,
            "shebang": "#!/runtime/bin/python",
        },
        "interpreter": {
            "path": "/runtime/bin/python",
            "bytes": 1000,
            "sha256": "e" * 64,
            "invocation_path": "/runtime/venv/bin/python",
            "version": "3.11.15 test",
            "implementation": "cpython",
        },
        "rslearn_package": {
            "version": "0.1.13",
            "root": "/runtime/lib/rslearn",
            "python_sources": {
                "files": 10,
                "bytes": 10000,
                "inventory_sha256": source_sha,
            },
            "git": None,
        },
    }


def checkpoint_manifest() -> dict:
    models = []
    for index, repo_id in enumerate(sorted(RELEASE_SPECS), start=1):
        revision = str(index) * 40
        models.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "snapshot_path": f"/cache/snapshots/{revision}",
                "files": [
                    {"name": "config.json", "bytes": 10, "sha256": str(index) * 64},
                    {"name": "weights.pth", "bytes": 20, "sha256": str(index + 2) * 64},
                ],
            }
        )
    return {"schema": "olmoearth-checkpoint-resolution-v1", "models": models}


class SemanticContractTests(unittest.TestCase):
    def resolved_v12(self, root: Path) -> Path:
        path = root / "resolved.yaml"
        render_config(
            ROOT / "config/olmo_release_full.template.yaml",
            path,
            model_env="OLMO_V1_2_MODEL_PATH",
            output_layer="embeddings_full_v1_2_legacy",
            batch_size=4,
            workers=2,
        )
        return path

    def test_actual_yaml_is_parsed_and_band_order_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            path = self.resolved_v12(root)
            contract = validate_resolved_config(
                path,
                model_env="OLMO_V1_2_MODEL_PATH",
                output_layer="embeddings_full_v1_2_legacy",
                batch_size=4,
                num_workers=2,
            )
            self.assertEqual(contract["semantic_core"]["encoder"]["patch_size"], 4)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            bands = payload["data"]["init_args"]["inputs"]["sentinel2_l2a"]["bands"]
            bands[0], bands[1] = bands[1], bands[0]
            drifted = root / "drifted.yaml"
            drifted.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "band order"):
                validate_resolved_config(
                    drifted,
                    model_env="OLMO_V1_2_MODEL_PATH",
                    output_layer="embeddings_full_v1_2_legacy",
                    batch_size=4,
                    num_workers=2,
                )

    def test_checkpoint_manifest_binds_both_revisions_and_files(self) -> None:
        normalized = normalize_checkpoint_manifest(checkpoint_manifest())
        self.assertEqual(set(normalized), set(RELEASE_SPECS))
        broken = checkpoint_manifest()
        broken["models"][0]["files"][1]["sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            normalize_checkpoint_manifest(broken)

    def test_python_source_inventory_is_deterministic_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            (root / "nested").mkdir()
            (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "nested/module.py").write_text("VALUE = 2\n", encoding="utf-8")
            first = fingerprint_python_sources(root)
            second = fingerprint_python_sources(root)
            self.assertEqual(first, second)
            (root / "nested/module.py").write_text("VALUE = 3\n", encoding="utf-8")
            self.assertNotEqual(
                first["inventory_sha256"],
                fingerprint_python_sources(root)["inventory_sha256"],
            )

    def test_rslearn_fingerprint_uses_shebang_interpreter_and_import_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            package = root / "rslearn"
            package.mkdir()
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            entrypoint = root / "rslearn-cli"
            entrypoint.write_text(
                f"#!{Path(sys.executable).resolve()}\nprint('stub')\n",
                encoding="utf-8",
            )
            probe = {
                "sys_executable": Path(sys.executable).resolve().as_posix(),
                "python_version": sys.version,
                "implementation": sys.implementation.name,
                "rslearn_version": "0.1.13",
                "package_roots": [package.as_posix()],
            }

            def run(command, **_kwargs):
                if len(command) >= 2 and command[1] == "-c":
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(probe) + "\n", stderr=""
                    )
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="not a repo")

            with mock.patch(
                "olmo_release_semantic_contract.subprocess.run", side_effect=run
            ):
                result = fingerprint_rslearn_runtime(entrypoint)
            self.assertEqual(result["entrypoint"]["path"], entrypoint.resolve().as_posix())
            self.assertEqual(
                result["interpreter"]["path"], Path(sys.executable).resolve().as_posix()
            )
            self.assertEqual(result["rslearn_package"]["root"], package.resolve().as_posix())
            self.assertEqual(result["rslearn_package"]["python_sources"]["files"], 1)

    def test_launcher_must_share_the_rslearn_interpreter_and_package_version(self) -> None:
        fingerprint = rslearn_fingerprint()
        fingerprint["interpreter"]["path"] = Path(sys.executable).resolve().as_posix()
        runtime = dict(RUNTIME)
        self.assertEqual(
            validate_launcher_runtime_binding(
                runtime, Path(sys.executable), fingerprint
            )["rslearn"],
            "0.1.13",
        )
        runtime["rslearn"] = "0.0.27"
        with self.assertRaisesRegex(ValueError, "rslearn version differs"):
            validate_launcher_runtime_binding(runtime, Path(sys.executable), fingerprint)

    def test_promoted_contract_strictly_matches_full_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            resolved = validate_resolved_config(
                self.resolved_v12(root),
                model_env="OLMO_V1_2_MODEL_PATH",
                output_layer="embeddings_full_v1_2_legacy",
                batch_size=4,
                num_workers=2,
            )
            checkpoints = normalize_checkpoint_manifest(checkpoint_manifest())
            promoted = {
                "schema": "olmoearth-release-execution-contract-v1",
                "semantic_config_core": resolved["semantic_core"],
                "selected_tuning": {
                    "candidate_id": "b004_w02",
                    "batch_size": 4,
                    "num_workers": 2,
                },
                "runtime_versions": RUNTIME,
                "physical_gpu": GPU,
                "rslearn_runtime_fingerprint": rslearn_fingerprint(),
                "exact_smoke_inputs_sha256": "b" * 64,
                "checkpoint_manifest_sha256": MANIFEST_SHA,
                "releases": checkpoints,
                "validation": {"actual_resolved_yaml_validated": True},
            }
            result = validate_promoted_execution_contract(
                promoted,
                repo_id="allenai/OlmoEarth-v1_2-Base",
                resolved_config_contract=resolved,
                batch_size=4,
                num_workers=2,
                runtime_versions=RUNTIME,
                gpu_index=GPU["index"],
                gpu_uuid=GPU["uuid"],
                checkpoint_manifest_sha256=MANIFEST_SHA,
                checkpoint_models=checkpoints,
                rslearn_runtime_fingerprint=rslearn_fingerprint(),
            )
            self.assertEqual(result["status"], "matched")
            drifted_runtime = copy.deepcopy(RUNTIME)
            drifted_runtime["torch"] = "different"
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                validate_promoted_execution_contract(
                    promoted,
                    repo_id="allenai/OlmoEarth-v1_2-Base",
                    resolved_config_contract=resolved,
                    batch_size=4,
                    num_workers=2,
                    runtime_versions=drifted_runtime,
                    gpu_index=GPU["index"],
                    gpu_uuid=GPU["uuid"],
                    checkpoint_manifest_sha256=MANIFEST_SHA,
                    checkpoint_models=checkpoints,
                    rslearn_runtime_fingerprint=rslearn_fingerprint(),
                )
            with self.assertRaisesRegex(ValueError, "rslearn executable/runtime/source"):
                validate_promoted_execution_contract(
                    promoted,
                    repo_id="allenai/OlmoEarth-v1_2-Base",
                    resolved_config_contract=resolved,
                    batch_size=4,
                    num_workers=2,
                    runtime_versions=RUNTIME,
                    gpu_index=GPU["index"],
                    gpu_uuid=GPU["uuid"],
                    checkpoint_manifest_sha256=MANIFEST_SHA,
                    checkpoint_models=checkpoints,
                    rslearn_runtime_fingerprint=rslearn_fingerprint("f" * 64),
                )


if __name__ == "__main__":
    unittest.main()
