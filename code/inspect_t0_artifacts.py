#!/usr/bin/env python3
"""Read-only reaggregation of a downloaded T0 report snapshot; incomplete is explicit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.reports.glob("*.json")):
        raw = path.read_bytes(); report = json.loads(raw)
        groups[(report["fold"], report["readout"])].append({
            "seed": report["seed"], "ap": report["test"]["auprc_exact"],
            "macro_iou": report["test"]["positive_patch_macro_iou"],
            "best_epoch": report["best_val_epoch"], "split": report["split"],
            "params": report["trainable_params"],
            "report_sha256": hashlib.sha256(raw).hexdigest()})
    out = []
    for (fold, arm), rows in sorted(groups.items()):
        complete = sorted(r["seed"] for r in rows) == [1, 2, 3]
        out.append({"fold": fold, "arm": arm, "complete_three_seeds": complete,
                    "ap_mean": statistics.mean(r["ap"] for r in rows) if complete else None,
                    "ap_sample_sd": statistics.stdev(r["ap"] for r in rows) if complete else None,
                    "macro_iou_mean": statistics.mean(r["macro_iou"] for r in rows) if complete else None,
                    "rows": rows})
    print(json.dumps({"scope": "snapshot only; unobserved arms are not failures", "groups": out}, indent=2))


if __name__ == "__main__":
    main()
