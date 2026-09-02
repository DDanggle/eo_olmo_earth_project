#!/usr/bin/env python3
"""Create a compact, content-addressed summary of the 24 C1b native-grid runs.

The full checkpoints and probability maps remain on the server.  This artifact
keeps the exact source-result hashes, code snapshot hashes, seed-level primary
metrics, and the aggregation needed by the paper ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


FOLDS = (
    "holdout_hiroshima",
    "holdout_hokkaido",
    "holdout_indonesia",
    "holdout_itogon",
    "holdout_kyrgyzstan1",
    "holdout_kyrgyzstan2",
    "holdout_newzealand",
    "holdout_thrissur",
)
SEEDS = (1, 2, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.runs_root / "code_snapshot"
    pilot_snapshot = snapshot / "pilot_sen12_gp_heads.py"
    baseline_snapshot = snapshot / "sen12_official_baselines.py"
    started_path = snapshot / "started_at_utc.txt"
    for required in (pilot_snapshot, baseline_snapshot, started_path):
        if not required.is_file():
            raise SystemExit(f"missing C1b snapshot file: {required}")

    pilot_sha = sha256_file(pilot_snapshot)
    source_manifest: dict[str, str] = {}
    per_region: dict[str, dict] = {}
    all_params: set[int] = set()
    all_code_sha: set[str] = set()
    fit_seconds = 0.0

    for fold in FOLDS:
        seeds = []
        for seed in SEEDS:
            result_path = args.runs_root / f"{fold}_seed{seed}" / f"{fold}_pilot.json"
            if not result_path.is_file():
                raise SystemExit(f"missing C1b result: {result_path}")
            source_manifest[str(result_path.relative_to(args.runs_root))] = sha256_file(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("schema") != "sen12-gp-pilot-v2" or result.get("fold") != fold:
                raise SystemExit(f"schema/fold mismatch: {result_path}")
            recorded_seed = result.get("development_protocol_v2", {}).get("seed")
            if recorded_seed != seed:
                raise SystemExit(f"seed mismatch in {result_path}: {recorded_seed} != {seed}")
            all_code_sha.add(result.get("code_sha256", ""))

            arm = result.get("arms", {}).get("P4native")
            if not arm:
                raise SystemExit(f"P4native arm missing: {result_path}")
            if arm.get("embedding_input_shape") != [128, 128, 128]:
                raise SystemExit(f"native input shape mismatch: {result_path}")
            if arm.get("parameter_parity", {}).get("passed") is not True:
                raise SystemExit(f"parameter parity failed: {result_path}")

            params = int(arm["trainable_params"])
            all_params.add(params)
            fit_seconds += float(arm["fit_plus_epoch_val_seconds"])
            metric = arm["test"]["positive_patch_macro_iou"]
            if metric is None:
                raise SystemExit(f"primary metric missing: {result_path}")
            seeds.append(
                {
                    "seed": seed,
                    "positive_patch_macro_iou": float(metric),
                    "auprc_exact": float(arm["test"]["auprc_exact"]),
                    "test_fp_pixels_at_threshold_0_5": int(
                        arm["test"]["confusion_pixels"]["fp"]
                    ),
                    "fit_plus_epoch_val_seconds": float(arm["fit_plus_epoch_val_seconds"]),
                    "best_val_epoch": int(arm["best_val_epoch"]),
                    "checkpoint_sha256": arm["checkpoint"]["sha256"],
                    "test_per_sample_sha256": arm["per_sample"]["test"]["sha256"],
                    "test_rows": int(arm["per_sample"]["test"]["rows"]),
                }
            )

        values = [entry["positive_patch_macro_iou"] for entry in seeds]
        per_region[fold.removeprefix("holdout_")] = {
            "seeds": seeds,
            "mean_positive_patch_macro_iou": statistics.fmean(values),
            "sample_sd_across_optimizer_seeds": statistics.stdev(values),
        }

    if all_code_sha != {pilot_sha}:
        raise SystemExit(
            f"recorded code hash does not equal executed snapshot: {all_code_sha} vs {pilot_sha}"
        )
    if len(all_params) != 1:
        raise SystemExit(f"trainable parameter count changed across runs: {sorted(all_params)}")

    region_means = [row["mean_positive_patch_macro_iou"] for row in per_region.values()]
    payload = {
        "schema": "c1b-presto-native-compact-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": [
            "C1b is a retrospective native-product sensitivity; C1a common-grid remains primary.",
            "This evaluates one off-domain second GeoFM and does not establish universal OlmoEarth superiority.",
            "The compact artifact preserves result hashes but not the server-resident checkpoints or probability maps.",
        ],
        "provenance": {
            "runs_root": str(args.runs_root),
            "run_count": len(FOLDS) * len(SEEDS),
            "started_at_utc": started_path.read_text(encoding="utf-8").strip(),
            "pilot_snapshot_sha256": pilot_sha,
            "baseline_snapshot_sha256": sha256_file(baseline_snapshot),
            "source_result_sha256": source_manifest,
        },
        "contract_checks": {
            "all_24_results_present": len(source_manifest) == 24,
            "all_recorded_code_hashes_equal_snapshot": True,
            "all_native_shapes_128x128x128": True,
            "all_parameter_parity_checks_pass": True,
            "trainable_params": next(iter(all_params)),
        },
        "per_region": per_region,
        "aggregate": {
            "region_macro_positive_patch_iou": statistics.fmean(region_means),
            "sample_sd_across_region_means": statistics.stdev(region_means),
            "total_fit_plus_epoch_val_seconds": fit_seconds,
        },
    }
    payload["content_sha256_without_this_field"] = content_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
