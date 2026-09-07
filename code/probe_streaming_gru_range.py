#!/usr/bin/env python3
"""Read-only, source-only CPU bound check for the unprojected v0 GRU state.

m_next=(1-z)*m + z*tanh(h), 0<=z<=1, implies every coordinate always stays
inside [min(m_initial,-1), max(m_initial,1)]. Teachers outside that interval
cannot be exactly represented by this GRU, regardless of training.
This does not predict downstream AP or prove every GRU design is unsuitable.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("positive sample count required")
    manifest = json.loads((args.root / "sen12_gp_contract/t1_manifest.json").read_text())
    out = []
    for fold, splits in manifest.items():
        ids = splits["train"]
        indices = np.linspace(0, len(ids) - 1, min(args.samples, len(ids))).astype(int)
        count = total = 0; squared_lower = squared_teacher = 0.0
        selected = []
        for i in indices:
            sid = ids[i]
            teacher = np.load(args.root / "olmo_streaming_dev/teacher_fp16" / f"{sid}.npy").astype("float32")
            initial = teacher[0]
            low, high = np.minimum(initial, -1), np.maximum(initial, 1)
            future = teacher[1:]
            deviation = np.maximum(np.maximum(low - future, future - high), 0)
            count += int((deviation > 0).sum()); total += deviation.size
            squared_lower += float(np.square(deviation, dtype="float64").sum())
            squared_teacher += float(np.square(future, dtype="float64").sum())
            selected.append(sid)
        out.append({"fold": fold, "source_ids": selected,
                    "unreachable_teacher_coordinate_fraction": count / total,
                    "feature_rel_mse_lower_bound": squared_lower / squared_teacher})
    print(json.dumps({"scope": "source-only representability diagnostic, not downstream performance", "rows": out}, indent=2))


if __name__ == "__main__":
    main()
