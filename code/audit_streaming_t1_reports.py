#!/usr/bin/env python3
"""Read-only T1 audit; partial seed sets never become completed gate decisions.

Saved JSON is endpoint evidence, not a prediction archive or pre-run provenance.
The CLI prints JSON to stdout; it does not mutate downloaded results.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

FOLDS = ("holdout_chimanimani", "holdout_hiroshima")
MODULES = ("ema", "gru", "residual")
SEEDS = (1, 2, 3)


def recovery(student, frozen, teacher):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (student, frozen, teacher)):
        raise ValueError("finite numeric AP required")
    if any(x < 0 or x > 1 for x in (student, frozen, teacher)):
        raise ValueError("AP outside [0,1]")
    gap = teacher - frozen
    return (student - frozen) / gap if gap > 1e-9 else None


def analyze(documents, manifest=None, references=None):
    references = references or {}
    errors, entries, seen = [], [], set()
    for name, report, digest in documents:
        try:
            if report.get("schema") != "streaming-update-train-v0":
                raise ValueError("unknown schema")
            fold, module, seed = report["fold"], report["module"], report["seed"]
            key = (fold, module, seed)
            if fold not in FOLDS or module not in MODULES or type(seed) is not int or seed not in SEEDS:
                raise ValueError("unexpected fold/module/seed")
            if key in seen:
                raise ValueError("duplicate fold/module/seed")
            seen.add(key)
            ds = report["downstream_c12"]
            teacher, frozen, student, singles = [ds[k]["auprc_exact"] for k in
                ("teacher_full_reencode", "frozen_m4", "student", "singles_mean")]
            rec = recovery(student, frozen, teacher)
            recovery(singles, frozen, teacher)
            counts = report["n"]
            if manifest is not None and counts != {s: len(v) for s, v in manifest[fold].items()}:
                raise ValueError("reported split counts differ from manifest")
            if type(report["params"]) is not int or report["params"] < 1:
                raise ValueError("invalid parameter count")
            entries.append({"file": name, "sha256": digest, "fold": fold, "module": module,
                "seed": seed, "teacher_ap": teacher, "frozen_ap": frozen, "student_ap": student,
                "singles_ap": singles, "recovery": rec, "teacher_minus_student_ap": teacher-student,
                "student_cosine": report["agreement_c12"]["student"]["cosine"],
                "student_rel_mse": report["agreement_c12"]["student"]["rel_mse"],
                "student_macro_iou": ds["student"]["positive_patch_macro_iou"],
                "params": report["params"], "counts": counts, "best_val_epoch": report["best_val_epoch"],
                "reported_train_s_not_inference_latency": report["train_s"],
                "missing_provenance_fields": [k for k in ("sample_ids", "code_sha256", "decoder_sha256", "manifest_sha256") if k not in report]})
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"file": name, "error": str(exc)})
    groups = []
    for fold in FOLDS:
        for module in MODULES:
            rows = sorted((r for r in entries if r["fold"] == fold and r["module"] == module), key=lambda r: r["seed"])
            complete = [r["seed"] for r in rows] == list(SEEDS)
            observed = {}
            for key in ("teacher_ap", "frozen_ap", "student_ap", "singles_ap", "recovery",
                        "teacher_minus_student_ap", "student_cosine", "student_rel_mse", "student_macro_iou"):
                values = [r[key] for r in rows]
                observed[key] = statistics.mean(values) if values and all(v is not None for v in values) else None
            groups.append({"fold": fold, "module": module, "observed_seeds": [r["seed"] for r in rows],
                "complete_three_seeds": complete, "status": "COMPLETE" if complete else "INCOMPLETE",
                "observed_mean_not_completion": observed, "params_observed": sorted(set(r["params"] for r in rows)),
                "student_ap_sample_sd": statistics.stdev(r["student_ap"] for r in rows) if len(rows)>1 else None,
                "rows": rows})
    refchecks = []
    for fold in FOLDS:
        rows = [r for r in entries if r["fold"] == fold]
        for key in ("teacher_ap", "frozen_ap", "singles_ap"):
            vals = [r[key] for r in rows]
            if vals and max(vals)-min(vals) > 1e-8:
                errors.append({"fold": fold, "error": f"{key} differs across runs"})
        if rows and fold in references:
            refchecks.append({"fold": fold, "reference_ap": references[fold]["test"]["auprc_exact"],
                "teacher_ap": rows[0]["teacher_ap"],
                "teacher_minus_saved_decoder_ap": rows[0]["teacher_ap"]-references[fold]["test"]["auprc_exact"]})
    if manifest:
        for fold, splits in manifest.items():
            if any(len(v) != len(set(v)) for v in splits.values()):
                errors.append({"fold": fold, "error": "duplicate manifest IDs"})
            if any(set(splits[a]) & set(splits[b]) for a,b in (("train","val"),("train","test"),("val","test"))):
                errors.append({"fold": fold, "error": "overlapping manifest splits"})
    complete = all(g["complete_three_seeds"] for g in groups) and not errors
    residual_groups = [g for g in groups if g["module"] == "residual"]
    prerequisite = None
    if all(g["complete_three_seeds"] for g in residual_groups) and not errors:
        prerequisite = all(g["observed_mean_not_completion"]["recovery"] is not None and
            g["observed_mean_not_completion"]["recovery"] >= .90 for g in residual_groups)
    gate = {"status": "INCOMPLETE_NO_FINAL_GATE", "residual_v0_pass": None,
            "residual_90pct_in_both_folds_prerequisite": prerequisite, "utility_90pct_by_module": None}
    if complete:
        means = {(g["fold"],g["module"]): g["observed_mean_not_completion"] for g in groups}
        utility = {m: all(means[(f,m)]["recovery"] is not None and means[(f,m)]["recovery"] >= .90 for f in FOLDS) for m in MODULES}
        gate = {"status": "COMPLETE_DEVELOPMENT_NOT_CONFIRMATORY", "residual_v0_pass": utility["residual"] and
            all(means[(f,"residual")]["student_ap"]-means[(f,m)]["student_ap"] >= .01 for f in FOLDS for m in ("gru","ema")),
            "residual_90pct_in_both_folds_prerequisite": prerequisite,
            "utility_90pct_by_module": utility}
    split_audit = {}
    if manifest:
        for fold, splits in manifest.items():
            split_audit[fold] = {"counts": {k: len(v) for k,v in splits.items()},
                "duplicate_counts": {k: len(v)-len(set(v)) for k,v in splits.items()},
                "train_val_intersection": len(set(splits["train"]) & set(splits["val"])),
                "train_test_intersection": len(set(splits["train"]) & set(splits["test"])),
                "val_test_intersection": len(set(splits["val"]) & set(splits["test"])),
                "source_region_counts_from_id": dict(Counter(s.rsplit("_s2_",1)[0] for s in splits["train"]))}
    return {"scope": "endpoint development snapshot; no spatial CI, inference timing, or causal attribution",
            "n_valid_reports": len(entries), "n_expected_reports": 18, "errors": errors,
            "groups": groups, "v0_gate": gate, "reference_checks": refchecks, "manifest_audit": split_audit,
            "matched_workload_timestep_bookkeeping_not_speedup": {
                "all_cutoffs_including_initial": {"full": 40, "stream": 12},
                "updates_after_common_initial": {"full": 36, "stream": 8},
                "final_only_cold_start": {"full": 12, "stream": 12},
                "final_only_after_common_initial": {"full": 12, "stream": 8}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--additional-reports", type=Path, help="Additive later snapshot; duplicates are rejected")
    args = parser.parse_args()
    documents = []
    paths = list((args.snapshot/"streaming_t1").glob("*.json"))
    if args.additional_reports:
        paths.extend(args.additional_reports.glob("*.json"))
    for path in sorted(paths):
        raw = path.read_bytes()
        documents.append((path.name, json.loads(raw), hashlib.sha256(raw).hexdigest()))
    path = args.snapshot/"t1_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else None
    refs = {fold: json.loads((args.snapshot/f"{fold}_seed1.json").read_text()) for fold in FOLDS if (args.snapshot/f"{fold}_seed1.json").exists()}
    result = analyze(documents, manifest, refs)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
