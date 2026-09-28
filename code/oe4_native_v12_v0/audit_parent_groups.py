#!/usr/bin/env python3
"""Read-only H5 coordinate-group audit and a candidate split of an existing cohort.

Exact stored lat/lon equality is a proxy for inherited parent tiles, not proof of
geographic independence. Only new audit/candidate artifacts are written. No
existing manifest, data, runtime controller or launch configuration is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_key(coords):
    """No decimal rounding: key records exact finite float values read from H5."""
    values = [float(v) for v in coords]
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("latlon must contain exactly two finite values")
    if not (-90 <= values[0] <= 90 and -180 <= values[1] <= 180):
        raise ValueError(f"latlon outside coordinate bounds: {values}")
    key = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return key, values


def select_group_subset(group_sizes, count, seed):
    """Deterministic exact subset sum; no pixel content or loss enters selection."""
    ordered = sorted(group_sizes, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    options = {0: ()}
    for key in ordered:
        for subtotal, chosen in list(options.items()):
            value = subtotal + group_sizes[key]
            if value <= count and value not in options:
                options[value] = chosen + (key,)
        if count in options:
            return set(options[count]), sorted(options)
    return None, sorted(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deps-root", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New audit output directory")
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--max-files", type=int, default=10000)
    args = parser.parse_args()
    if args.deps_root:
        sys.path.insert(0, str(args.deps_root.resolve()))
    import h5py
    import numpy as np
    import hdf5plugin  # noqa: F401 -- metadata may still use compressed datasets.

    corpus_root = args.h5_root.resolve()
    paths = sorted(set(corpus_root.rglob("*.h5")) | set(corpus_root.rglob("*.hdf5")))
    if not paths or len(paths) > args.max_files:
        raise ValueError(f"Expected 1..{args.max_files} H5 files, found {len(paths)}")
    manifest = json.loads(args.manifest.read_text())
    manifest_rows = manifest["selected"]
    by_key = defaultdict(list)
    by_path = {}
    errors = []
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                raw = np.asarray(f["latlon"][()])
                key, values = coordinate_key(raw.reshape(-1))
                entry = {"file": str(path), "relative_file": str(path.relative_to(corpus_root)),
                         "coordinate_key": key, "latlon": values, "stored_dtype": str(raw.dtype),
                         "stored_shape": list(raw.shape), "group_id": hashlib.sha256(key.encode()).hexdigest()[:20]}
                by_key[key].append(entry)
                by_path[str(path)] = entry
        except (OSError, KeyError, ValueError) as error:
            errors.append({"file": str(path), "error": f"{type(error).__name__}: {error}"})
    selected_by_split = defaultdict(list)
    selected_by_group = defaultdict(list)
    selected_errors = []
    seen_paths = set()
    for i, row in enumerate(manifest_rows):
        path = str(Path(row["file"]).resolve())
        if path in seen_paths:
            selected_errors.append({"row": i, "file": path, "error": "Duplicate selected filename"})
            continue
        seen_paths.add(path)
        if path not in by_path:
            selected_errors.append({"row": i, "file": path, "error": "No valid coordinate record in scanned corpus"})
            continue
        if row.get("split") not in ["train_diagnostic", "dev_diagnostic"]:
            selected_errors.append({"row": i, "file": path, "error": "Unexpected split name"})
            continue
        meta = dict(by_path[path], original_manifest_index=i, split=row["split"])
        if row.get("latlon") is not None:
            recorded_key, _ = coordinate_key(np.asarray(row["latlon"]).reshape(-1))
            if recorded_key != meta["coordinate_key"]:
                selected_errors.append({"row": i, "file": path, "error": "Manifest latlon differs from current H5 latlon"})
                continue
        selected_by_split[row["split"]].append(meta)
        selected_by_group[meta["coordinate_key"]].append(meta)
    split_groups = {split: {r["coordinate_key"] for r in rows} for split, rows in selected_by_split.items()}
    train_groups = split_groups.get("train_diagnostic", set())
    dev_groups = split_groups.get("dev_diagnostic", set())
    overlap = sorted(train_groups & dev_groups)
    overlap_details = []
    for key in overlap:
        overlap_details.append({"coordinate_key": key, "latlon": json.loads(key),
                                "train_files": [r["file"] for r in selected_by_group[key] if r["split"] == "train_diagnostic"],
                                "dev_files": [r["file"] for r in selected_by_group[key] if r["split"] == "dev_diagnostic"]})
    groups = [{"coordinate_key": key, "latlon": json.loads(key), "group_id": rows[0]["group_id"],
               "file_count": len(rows), "files": [r["relative_file"] for r in rows]}
              for key, rows in sorted(by_key.items())]
    split_summary = {}
    for split in ["train_diagnostic", "dev_diagnostic"]:
        count = Counter(r["coordinate_key"] for r in selected_by_split.get(split, []))
        split_summary[split] = {"files": sum(count.values()), "distinct_coordinate_groups": len(count),
                                "same_coordinate_multiplicity_histogram": dict(sorted(Counter(count.values()).items()))}
    audit = {
        "status": "read_only_coordinate_group_audit", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__), "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "h5_root": str(corpus_root), "h5py_version": importlib.metadata.version("h5py"),
        "group_definition": "Exact stored finite lat/lon values serialized without rounding; coordinates are inherited parent centers in the official converter.",
        "limits": ["Exact shared coordinates identify a conservative grouping proxy, not verified independent geographic areas.",
                   "Different coordinate keys may describe spatial neighbors, overlapping footprints, or repeated nearby scenes.",
                   "Official pretraining corpus exposure is not undone by a new development split.",
                   "This audit reads latlon only; it does not revalidate raw pixels, timestamps, crop hashes or original parent IDs."],
        "corpus": {"h5_files_seen": len(paths), "valid_coordinate_files": len(by_path), "distinct_coordinate_keys": len(by_key),
                   "same_coordinate_multiplicity_histogram": dict(sorted(Counter(len(v) for v in by_key.values()).items())),
                   "invalid_coordinate_files": errors, "groups": groups},
        "existing_manifest": {"rows": len(manifest_rows), "valid_selected_files": sum(len(v) for v in selected_by_split.values()),
                              "distinct_coordinate_keys": len(selected_by_group), "split_summary": split_summary,
                              "train_dev_overlapping_coordinate_groups": len(overlap), "overlap_groups": overlap_details,
                              "selected_row_errors": selected_errors, "selected_records": [r for split in selected_by_split.values() for r in split]},
    }
    requested_dev = sum(row.get("split") == "dev_diagnostic" for row in manifest_rows)
    requested_train = sum(row.get("split") == "train_diagnostic" for row in manifest_rows)
    candidate = {"status": "candidate_only_not_approved_or_launched", "source_manifest": str(args.manifest.resolve()),
                 "source_manifest_sha256": audit["manifest_sha256"], "seed": args.seed,
                 "scope": "Reassign the existing selected development cohort only. Reuses previously visible data; not a sealed test set.",
                 "constraints": {"same_selected_files": True, "same_raw_crops": True, "same_total_counts": True,
                                 "train_dev_exact_coordinate_overlap": 0},
                 "selection_rule": "Seed-hash order exact coordinate groups, deterministic subset sum to original dev count; no imagery or loss selection.",
                 "no_runtime_controller_changed": True, "no_existing_manifest_changed": True,
                 "interpretation": "Coordinate-group-disjoint development candidate only; no independent geography or new pretraining-exposure claim."}
    if selected_errors or requested_train < 1 or requested_dev < 1:
        candidate["status"] = "candidate_unavailable_selected_manifest_invalid"
        candidate["errors"] = selected_errors
    else:
        dev_keys, reachable = select_group_subset({k: len(v) for k, v in selected_by_group.items()}, requested_dev, args.seed)
        if dev_keys is None:
            candidate["status"] = "candidate_unavailable_exact_counts_infeasible"
            candidate["requested_dev_files"] = requested_dev
            candidate["reachable_dev_counts_up_to_requested"] = reachable
        else:
            candidate_rows = []
            for index, row in enumerate(manifest_rows):
                meta = by_path[str(Path(row["file"]).resolve())]
                revised = dict(row)
                revised["original_split"] = row["split"]
                revised["original_manifest_index"] = index
                revised["coordinate_group_id"] = meta["group_id"]
                revised["coordinate_key"] = meta["coordinate_key"]
                revised["split"] = "dev_diagnostic" if meta["coordinate_key"] in dev_keys else "train_diagnostic"
                candidate_rows.append(revised)
            candidate["selected"] = candidate_rows
            candidate["split_counts"] = dict(Counter(row["split"] for row in candidate_rows))
            candidate["distinct_coordinate_groups"] = {split: len({r["coordinate_key"] for r in candidate_rows if r["split"] == split})
                                                        for split in ["train_diagnostic", "dev_diagnostic"]}
            candidate["reassigned_files"] = sum(r["original_split"] != r["split"] for r in candidate_rows)
            if candidate["split_counts"].get("train_diagnostic") != requested_train or candidate["split_counts"].get("dev_diagnostic") != requested_dev:
                raise AssertionError("Candidate split count invariant failed")
            candidate_train = {r["coordinate_key"] for r in candidate_rows if r["split"] == "train_diagnostic"}
            candidate_dev = {r["coordinate_key"] for r in candidate_rows if r["split"] == "dev_diagnostic"}
            if candidate_train & candidate_dev:
                raise AssertionError("Candidate coordinate disjointness invariant failed")
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, obj in [("parent_group_audit.json", audit), ("group_disjoint_manifest_candidate.json", candidate)]:
        (args.out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "h5_files": len(paths), "corpus_coordinate_groups": len(by_key),
                      "selected_coordinate_groups": len(selected_by_group), "existing_train_dev_coordinate_overlap": len(overlap),
                      "candidate_status": candidate["status"], "candidate_split_counts": candidate.get("split_counts"),
                      "candidate_coordinate_groups": candidate.get("distinct_coordinate_groups")}), flush=True)


if __name__ == "__main__":
    main()
