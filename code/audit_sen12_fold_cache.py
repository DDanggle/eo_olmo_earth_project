#!/usr/bin/env python3
"""Sen12 G-P fold cache를 내용까지 전수 감사하고 학습용 seal을 만든다.

검사 범위
  - 봉인된 LOCO split의 sample ID와 cache 세 디렉터리의 파일 집합이 정확히 일치
  - emb/raw/mask의 shape·dtype, finite/range/binary 계약
  - cache mask의 양성 픽셀 수가 원본 NetCDF 전수 계약과 표본별로 일치
  - 모든 배열 content SHA-256과 전체 manifest SHA-256

원본을 수정하지 않는다. 하나라도 실패하면 cache_audit.json의 all_gates_pass=false이고
pilot_sen12_gp_heads.py v2는 실행을 거부한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KINDS = {
    "emb_fp16": ((768, 32, 32), "float16"),
    "raw_u16": ((10, 12, 128, 128), "uint16"),
    "mask_u8": ((128, 128), "uint8"),
}


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array) -> str:
    """Shape/dtype는 manifest에 별도로 넣고 C-order content bytes를 해시한다."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani"))
    p.add_argument("--folds", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"))
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--fold", default="holdout_chimanimani")
    return p.parse_args()


def main() -> None:
    import numpy as np

    args = parse_args()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    contract_sha256 = sha256_file(args.contract)
    expected_contract_sha256 = folds.get("sample_contract_sha256")
    fold = next((row for row in folds["folds"] if row["fold"] == args.fold), None)
    if fold is None:
        raise SystemExit(f"fold 없음: {args.fold}")
    records = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            records[row["sample_id"]] = row

    role_ids = {}
    for role in ("train", "val", "test"):
        regions = (fold["train_regions"] if role == "train"
                   else [fold["val_region"]] if role == "val" else [fold["test_region"]])
        ids = sorted(sid for sid, row in records.items()
                     if row["region"] in regions and not row.get("error")
                     and row.get("s15_eligible", True))
        if sha256_lines(ids) != fold["sample_sha256"][role]:
            raise SystemExit(f"{role} split SHA 불일치")
        role_ids[role] = ids
    expected = sorted(sid for role in ("train", "val", "test") for sid in role_ids[role])
    expected_set = set(expected)

    file_set_gates = {}
    for kind in KINDS:
        actual = {path.stem for path in (args.cache / kind).glob("*.npy")}
        file_set_gates[kind] = {
            "exact": actual == expected_set,
            "missing": sorted(expected_set - actual),
            "extra": sorted(actual - expected_set),
        }

    rows = []
    failures = []
    aggregate = {kind: hashlib.sha256() for kind in KINDS}
    for index, sid in enumerate(expected, 1):
        row = {"sample_id": sid}
        for kind, (shape, dtype) in KINDS.items():
            path = args.cache / kind / f"{sid}.npy"
            if not path.is_file():
                failures.append(f"{kind}:{sid}:missing")
                continue
            array = np.load(path, allow_pickle=False)
            shape_ok = tuple(array.shape) == shape
            dtype_ok = str(array.dtype) == dtype
            content_sha = array_sha256(array)
            entry = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "content_sha256": content_sha, "bytes": int(path.stat().st_size),
            }
            if kind == "emb_fp16":
                entry["finite"] = bool(np.isfinite(array).all())
                semantic_ok = entry["finite"]
            elif kind == "raw_u16":
                entry["min"] = int(array.min())
                entry["max"] = int(array.max())
                semantic_ok = entry["min"] >= 0 and entry["max"] <= 10_000
                entry["reflectance_range_ok"] = semantic_ok
            else:
                values = [int(v) for v in np.unique(array)]
                positives = int(np.count_nonzero(array))
                entry["values"] = values
                entry["positive_pixels"] = positives
                entry["contract_positive_pixels"] = int(records[sid]["mask_positive_pixels"])
                semantic_ok = (set(values) <= {0, 1}
                               and positives == entry["contract_positive_pixels"])
                entry["binary_and_contract_match"] = semantic_ok
            entry["shape_ok"] = shape_ok
            entry["dtype_ok"] = dtype_ok
            if not (shape_ok and dtype_ok and semantic_ok):
                failures.append(f"{kind}:{sid}:contract")
            row[kind] = entry
            aggregate[kind].update(
                f"{sid}\t{shape}\t{dtype}\t{content_sha}\n".encode("utf-8"))
        rows.append(row)
        if index % 500 == 0 or index == len(expected):
            print(f"[{index}/{len(expected)}] failures={len(failures)}", flush=True)

    manifest_path = args.cache / "cache_audit_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    gates = {
        "sample_contract_sha256_verified": contract_sha256 == expected_contract_sha256,
        "split_sha256_verified": True,
        "exact_file_sets": all(v["exact"] for v in file_set_gates.values()),
        "all_array_contracts": not failures,
        "all_expected_samples_audited": len(rows) == len(expected),
    }
    summary = {
        "schema": "sen12-fold-cache-audit-v1",
        "fold": args.fold,
        "sample_contract_sha256": contract_sha256,
        "expected_sample_contract_sha256": expected_contract_sha256,
        "sample_counts": {key: len(value) for key, value in role_ids.items()},
        "samples_audited": len(rows),
        "file_set_gates": file_set_gates,
        "content_sha256": {kind: h.hexdigest() for kind, h in aggregate.items()},
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "failures": failures,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    out = args.cache / "cache_audit.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
