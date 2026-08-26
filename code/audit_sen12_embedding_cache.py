#!/usr/bin/env python3
"""Audit an alternate Sen12 embedding cache against a sealed base cache and LOCO contract.

E1 full-context cache contains only `emb_fp16`; raw, mask, months and the original cache audit live in
the tiled base cache. This audit verifies the alternate embedding file set/shape/dtype/finite content and
binds it to the already-passed base audit without duplicating the raw/mask audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SHAPE = (768, 32, 32)
EXPECTED_DTYPE = "float16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def array_sha256(array) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb-cache", type=Path,
        default=Path("/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani"),
    )
    parser.add_argument(
        "--base-cache", type=Path,
        default=Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani"),
    )
    parser.add_argument(
        "--folds", type=Path,
        default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"),
    )
    parser.add_argument(
        "--contract", type=Path,
        default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"),
    )
    parser.add_argument("--fold", default="holdout_chimanimani")
    return parser.parse_args()


def main() -> None:
    import numpy as np

    args = parse_args()
    base_audit_path = args.base_cache / "cache_audit.json"
    if not base_audit_path.is_file():
        raise SystemExit(f"base cache audit 없음: {base_audit_path}")
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))

    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold = next((row for row in folds["folds"] if row["fold"] == args.fold), None)
    if fold is None:
        raise SystemExit(f"fold 없음: {args.fold}")
    records = {
        row["sample_id"]: row
        for row in (
            json.loads(line) for line in args.contract.read_text(encoding="utf-8").splitlines()
            if line
        )
    }

    expected = []
    split_hashes_match = True
    split_counts = {}
    for role in ("train", "val", "test"):
        regions = (
            fold["train_regions"] if role == "train"
            else [fold["val_region"]] if role == "val"
            else [fold["test_region"]]
        )
        sample_ids = sorted(
            sample_id for sample_id, record in records.items()
            if record["region"] in regions and not record.get("error")
            and record.get("s15_eligible", True)
        )
        split_hashes_match &= sha256_lines(sample_ids) == fold["sample_sha256"][role]
        split_counts[role] = len(sample_ids)
        expected.extend(sample_ids)
    expected_set = set(expected)

    embedding_dir = args.emb_cache / "emb_fp16"
    actual_set = {path.stem for path in embedding_dir.glob("*.npy")}
    failures = []
    aggregate = hashlib.sha256()
    rows = []
    for index, sample_id in enumerate(expected, 1):
        path = embedding_dir / f"{sample_id}.npy"
        if not path.is_file():
            failures.append(f"{sample_id}:missing")
            continue
        array = np.load(path, allow_pickle=False)
        shape_ok = tuple(array.shape) == EXPECTED_SHAPE
        dtype_ok = str(array.dtype) == EXPECTED_DTYPE
        finite = bool(np.isfinite(array).all())
        content_sha256 = array_sha256(array)
        if not (shape_ok and dtype_ok and finite):
            failures.append(f"{sample_id}:contract")
        row = {
            "sample_id": sample_id,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": finite,
            "bytes": path.stat().st_size,
            "content_sha256": content_sha256,
        }
        rows.append(row)
        aggregate.update(
            f"{sample_id}\t{EXPECTED_SHAPE}\t{EXPECTED_DTYPE}\t{content_sha256}\n".encode()
        )
        if index % 500 == 0 or index == len(expected):
            print(f"[{index}/{len(expected)}] failures={len(failures)}", flush=True)

    manifest_path = args.emb_cache / "cache_audit_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    gates = {
        "base_cache_audit_passed": base_audit.get("all_gates_pass") is True,
        "base_contract_matches_fold_contract": (
            base_audit.get("sample_contract_sha256") == folds.get("sample_contract_sha256")
        ),
        "split_sha256_verified": split_hashes_match,
        "exact_embedding_file_set": actual_set == expected_set,
        "all_embedding_contracts": not failures,
        "all_expected_samples_audited": len(rows) == len(expected),
    }
    summary = {
        "schema": "sen12-alternate-embedding-cache-audit-v1",
        "audit_code_sha256": sha256_file(Path(__file__)),
        "fold": args.fold,
        "cache_sources": {
            "embedding": str(args.emb_cache),
            "base_mask_raw_month": str(args.base_cache),
        },
        "base_cache_audit": {
            "path": str(base_audit_path),
            "sha256": sha256_file(base_audit_path),
        },
        "sample_counts": split_counts,
        "samples_audited": len(rows),
        "file_set": {
            "expected": len(expected_set),
            "actual": len(actual_set),
            "missing": sorted(expected_set - actual_set),
            "extra": sorted(actual_set - expected_set),
        },
        "embedding_content_sha256": aggregate.hexdigest(),
        "manifest_sha256": sha256_file(manifest_path),
        "failures": failures,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    output = args.emb_cache / "cache_audit.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
