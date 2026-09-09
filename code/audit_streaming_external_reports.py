#!/usr/bin/env python3
"""Retrospective audit of copied MS-117--120 reports; no GPU, images or labels.

Arithmetic gates are not provenance certification. Event means are descriptive;
decoder/updater seeds sharing the same events are not independent replications.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean


def ap_value(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError(f"invalid AP: {x!r}")
    return x


def recovery(value, teacher, stale):
    for x in (value, teacher, stale):
        ap_value(x)
    return (value-stale)/(teacher-stale) if teacher-stale > 1e-9 else None


def summarize_kuro(reports):
    expected = {"teacher3", "stale2", "singles_mean"} | {
        f"{module}_seed{s}" for module in ("gru", "gru_noobs", "ema") for s in (1, 2, 3)}
    rows, event_rows, errors, seed_ids = [], [], [], []
    event_set, n_test = None, None
    for r in sorted(reports, key=lambda r: r["decoder_seed"]):
        ds = r["decoder_seed"]
        seed_ids.append(ds)
        if set(r["arms"]) != expected:
            errors.append(f"decoder {ds}: incomplete/unexpected arm matrix")
            continue
        if n_test is not None and r["n_test"] != n_test:
            errors.append("test count mismatch")
        n_test = r["n_test"]
        arms = r["arms"]
        t, st = [ap_value(arms[k]["flood_ap"]) for k in ("teacher3", "stale2")]
        row = {"decoder_seed": ds, "n_test": n_test, "teacher_ap": t, "stale_ap": st,
               "teacher_event_macro_ap": arms["teacher3"]["activation_macro_ap"]}
        events = set(arms["teacher3"]["per_activation_ap"])
        if event_set is not None and events != event_set:
            errors.append("event set mismatch between decoders")
        event_set = events
        for name, a in arms.items():
            value = ap_value(a["flood_ap"])
            rec = recovery(value, t, st)
            if rec is None or not math.isfinite(a["recovery"]) or abs(rec-a["recovery"]) > 1e-7:
                errors.append(f"decoder {ds}/{name}: undefined or inconsistent recovery")
            if set(a["per_activation_ap"]) != events:
                errors.append(f"decoder {ds}/{name}: event set mismatch")
            ev = [ap_value(v) for v in a["per_activation_ap"].values() if v is not None]
            if not ev or abs(mean(ev)-a["activation_macro_ap"]) > 1e-7:
                errors.append(f"decoder {ds}/{name}: event macro mismatch")
        for module in ("gru", "gru_noobs", "ema"):
            vals = [arms[f"{module}_seed{s}"] for s in (1, 2, 3)]
            row[module] = {"ap_mean": mean(v["flood_ap"] for v in vals),
                "recovery_mean": recovery(mean(v["flood_ap"] for v in vals), t, st),
                "ap_by_updater_seed": [v["flood_ap"] for v in vals],
                "event_macro_ap_mean": mean(v["activation_macro_ap"] for v in vals),
                "reported_far_mean": mean(v["false_alarm_no_water"] for v in vals)}
        row["reported_teacher_far"] = arms["teacher3"]["false_alarm_no_water"]
        row["arithmetic_gate"] = (row["gru"]["recovery_mean"] is not None
            and row["gru"]["recovery_mean"] >= .8 and row["gru_noobs"]["recovery_mean"] <= .2)
        rows.append(row)
        for event in sorted(events):
            teacher = arms["teacher3"]["per_activation_ap"][event]
            vals = [arms[f"gru_seed{s}"]["per_activation_ap"][event] for s in (1, 2, 3)]
            if teacher is not None and all(v is not None for v in vals):
                event_rows.append({"decoder_seed": ds, "event": event, "teacher_ap": teacher,
                    "gru_ap": mean(vals), "delta_ap": mean(vals)-teacher})
    if sorted(seed_ids) != [1, 2, 3]:
        errors.append("expected exactly decoder seeds 1,2,3")
    by_event = defaultdict(list)
    for r in event_rows:
        by_event[r["event"]].append(r["delta_ap"])
    deltas = {e: mean(v) for e, v in by_event.items() if len(v) == 3}
    return {"errors": errors, "complete": not errors, "rows": rows,
        "all_arithmetic_gates_pass": bool(rows) and not errors and all(r["arithmetic_gate"] for r in rows),
        "event_delta_gru_minus_teacher_seed_mean": deltas,
        "events_positive": sum(v > 0 for v in deltas.values()),
        "events_negative": sum(v < 0 for v in deltas.values()),
        "event_macro_delta": mean(deltas.values()) if deltas else None,
        "provenance_certified": False,
        "note": "Reported AP and arithmetic screen only; selection, excluded-input and cost audits remain separate."}


def summarize_italy(root):
    rows, errors = [], []
    for arm in ("decoder", "p4_upsample2", "p2_native", "p2_avgpool2"):
        reports = [json.loads(p.read_text()) for p in sorted((root/arm).glob("holdout_italy_seed*.json"))]
        if sorted(r["seed"] for r in reports) != [1, 2, 3]:
            errors.append(f"incomplete {arm}")
            continue
        if any(r["best_val_epoch"] <= 0 or any(not math.isfinite(h[k]) for h in r["history"]
               for k in ("loss", "val_iou")) for r in reports):
            errors.append(f"invalid history {arm}")
        rows.append({"arm": arm, "macro_iou": mean(ap_value(r["test"]["positive_patch_macro_iou"]) for r in reports),
            "ap": mean(ap_value(r["test"]["auprc_exact"]) for r in reports),
            "micro_iou": mean(r["test"]["iou"] for r in reports),
            "splits": [r["split"] for r in reports]})
    lookup = {r["arm"]: r for r in rows}
    gate = None
    contrasts = {}
    if len(lookup) == 4 and not errors:
        n = lookup["p2_native"]["macro_iou"]
        contrasts = {"native20_minus_native40": n-lookup["decoder"]["macro_iou"],
                     "native20_minus_up40": n-lookup["p4_upsample2"]["macro_iou"],
                     "native20_minus_pooled20": n-lookup["p2_avgpool2"]["macro_iou"]}
        gate = all(contrasts[k] >= v for k,v in zip(contrasts, (.03,.02,.02)))
    size_file = root/"size_stratified.json"
    strata = []
    if size_file.exists():
        for arm, reports in json.loads(size_file.read_text()).items():
            if sorted(r["seed"] for r in reports) != [1, 2, 3]:
                errors.append(f"incomplete size strata {arm}")
                continue
            strata.append({"arm": arm,
                "large_component_tile_macro": mean(r["macro_iou_tiles_with_component_ge16px"] for r in reports),
                "all_small_tile_macro": mean(r["macro_iou_tiles_all_small"] for r in reports),
                "n_large_tiles": reports[0]["n_tiles_ge16"], "n_small_tiles": reports[0]["n_tiles_small_only"],
                "any_overlap_component_recall": {k:mean(r["component_recall"][k][0] for r in reports)
                    for k in ("<16", "16-64", ">=64")}})
    return {"errors": errors, "rows": rows, "registered_contrasts": contrasts, "resolution_gate": gate,
            "size_strata_descriptive_only": strata}


def date_stats(dates):
    gaps = [x for r in dates for x in r["gap_days"]]
    return {"files":len(dates),"assignments":len(gaps),
        "after_target_s2_date":sum(g>0 for g in gaps),
        "absolute_gap_mean_days":mean(abs(g) for g in gaps) if gaps else None,
        "max_absolute_gap_days":max(map(abs,gaps)) if gaps else None,
        "tiles_reusing_same_s1_date":sum(len(set(r["s1_dates"])) < len(r["s1_dates"]) for r in dates)}


def summarize_cross(snapshot):
    groups, rows = defaultdict(list), []
    for path in sorted((snapshot/"artifacts/streaming_t1xs").glob("*.json")):
        r = json.loads(path.read_text())
        ds = r["downstream_c12"]
        t,st,v = [ap_value(ds[k]["auprc_exact"]) for k in ("teacher_full_reencode", "frozen_m4", "student")]
        groups[(r["fold"], r["obs_source"])].append({"seed":r["seed"], "ap":v,
            "recovery":recovery(v,t,st), "n":r["n"], "teacher":t,"stale":st})
    for (fold,obs), reports in sorted(groups.items()):
        rows.append({"fold":fold,"obs_source":obs,"complete":sorted(r["seed"] for r in reports)==[1,2,3],
            "ap":mean(r["ap"] for r in reports),"recovery":mean(r["recovery"] for r in reports),
            "counts":reports[0]["n"],"teacher_ap":reports[0]["teacher"],"stale_ap":reports[0]["stale"]})
    dates = {p.stem:json.loads(p.read_text()) for p in sorted((snapshot/"olmo_streaming_dev/single_s1_dates").glob("*.json"))}
    manifest_path = snapshot/"sen12_gp_contract/t1_manifest.json"
    per_fold = {}
    if manifest_path.exists():
        for fold, splits in json.loads(manifest_path.read_text()).items():
            per_fold[fold] = {split:date_stats([dates[s] for s in ids if s in dates]) for split,ids in splits.items()}
    failed = {}
    for fold in ("thrissur","newzealand"):
        p=snapshot/f"logs/t1xs_holdout_{fold}_s1_s1.log"
        txt=p.read_text() if p.exists() else ""
        failed[fold]={"error_present":"ValueError: need at least one array to stack" in txt,
            "filter_line":next((l for l in txt.splitlines() if l.startswith("after obs-source filter")),None)}
    return {"rows":rows,"failed_execution_not_scientific_negative":failed,
        "date_alignment":date_stats(list(dates.values())), "date_alignment_by_manifest":per_fold,
        "all_four_fold_failure_established":False}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("snapshot",type=Path); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); root=args.snapshot
    kuro=summarize_kuro([json.loads(p.read_text()) for p in sorted((root/"artifacts/kurosiwo").glob("eval_dec*.json"))])
    dropped=json.loads((root/"artifacts/kurosiwo/dropped_nonfinite.json").read_text())
    logs={}
    for p in sorted((root/"logs").glob("ks_update_*.log")):
        text=p.read_text(); found=re.search(r"DONE update ([\d.eE+-]+) skipped (\d+)",text)
        logs[p.name]={"done_finite":bool(found and math.isfinite(float(found[1]))),
            "skipped_batches":int(found[2]) if found else None,"nan_epochs":sum("val nan" in l for l in text.splitlines())}
    result={"scope":"2026-09-09 retrospective report arithmetic; no new model evaluation, no raw test labels",
        "kuro":kuro,"kuro_dropped":dropped,"updater_logs":logs,"italy":summarize_italy(root/"artifacts/italy_sealed"),
        "cross_sensor":summarize_cross(root),
        "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/"code").glob("*")) if p.is_file()}}
    manifest = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
        and p not in (args.out, root/"input_sha256.json", root/"architecture_summary.json", root/"README.md")}
    (root/"input_sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))
    return bool(result["kuro"]["errors"] or result["italy"]["errors"])


if __name__=="__main__":
    raise SystemExit(main())
