#!/usr/bin/env python3
"""Read-only design audit: synthetic annotations and analytic probabilities only.

No real evaluation images, annotations, or model predictions are loaded.
This does not execute stage H, L, or R, or change their registered gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def wilson(k, n, z=1.959963984540054):
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def episode(i):
    return {
        "id": f"synthetic-{i:02d}", "aoi": f"synthetic-aoi-{i // 2}",
        "region": "NW" if i % 2 == 0 else "SE", "cutoff": "2021-01",
        "reference_id": "F000",
        "frames": [{"id": f"F{j:03d}",
                    "date": f"{2020 + j // 12}-{j % 12 + 1:02d}",
                    "path": f"frames/synthetic-{i:02d}-F{j:03d}.png"}
                   for j in range(13)],
    }


def annotation(ep, who, kind):
    states = {f["id"]: "no_visible_change" for f in ep["frames"]}
    if kind == "change":
        states.update({f"F{j:03d}": "visible_change" for j in range(3, 13)})
    elif kind == "insufficient":
        states["F000"] = "unreadable"
    return {"episode_id": ep["id"], "annotator_id": who,
            "status": "complete", "states": states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "code"))
    from sn7_visible_contract_v05 import derive_target, consensus_target
    from sn7_visible_pack_v05 import review_exports
    from sn7_evidence_loss_v0 import tier_flags
    from sn7_d1_reader_run_v0 import _episode_diff

    pack = {"pack_id": "synthetic-design-audit-only",
            "episodes": [episode(i) for i in range(12)]}

    def review(kinds):
        exports = [{"schema": "sn7-visible-annotations-v0.5",
                    "pack_id": pack["pack_id"], "annotator_id": who,
                    "annotations": [annotation(ep, who, kind)
                                    for ep, kind in zip(pack["episodes"], kinds)]}
                   for who in ("synthetic-a", "synthetic-b")]
        result = review_exports(pack, exports)
        keys = ("n_total", "n_agreed", "agreement_rate", "agreed_answer_counts",
                "scientific_gate_passed", "h_gate_reason", "diagnostic_manifest_ready")
        return {**{key: result[key] for key in keys},
                "n_control_pairs": len(result["control_pairs"])}

    ep = pack["episodes"][0]
    a = annotation(ep, "synthetic-a", "change")
    b = annotation(ep, "synthetic-b", "change")
    b["states"]["F002"] = "ambiguous"
    ta, tb = derive_target(ep, a), derive_target(ep, b)
    try:
        consensus_target(ep, [a, b])
        consensus_error = None
    except ValueError as exc:
        consensus_error = str(exc)

    c = annotation(ep, "synthetic-b", "change")
    c["states"]["F004"] = "ambiguous"
    postfirst_consensus = consensus_target(ep, [a, c])
    frame_ids = [f["id"] for f in ep["frames"]]
    negative = derive_target(ep, annotation(ep, "synthetic-a", "no_change"))
    bootstrap = _episode_diff([
        {"episode_id": "same-aoi-left", "donor": True, "source": False},
        {"episode_id": "same-aoi-right", "donor": True, "source": False},
    ], "donor", "source")

    source_files = ["code/sn7_visible_contract_v05.py", "code/sn7_visible_pack_v05.py",
                    "code/sn7_evidence_loss_v0.py", "code/sn7_d1_reader_run_v0.py",
                    "config/decision_experiment_d1_prereg_v0.json",
                    "config/decision_experiment_d1_prereg_v0_amendment_20260924.json"]
    out = {
        "scope": "synthetic design counterexamples; not actual H/L/R results",
        "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                          for name in source_files},
        "statistics": {
            "wilson_95_iid_illustration": {f"{k}/12": wilson(k, 12) for k in (7, 8, 9, 10, 11, 12)},
            "agreement_rule_only_iid_P_at_least_8_of_12": {
                str(p): sum(math.comb(12, k) * p**k * (1-p)**(12-k) for k in range(8, 13))
                for p in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)},
            "missing_class_if_prevalence_5pct_12_iid": 0.95**12,
            "note": "IID assumptions are illustrative, not satisfied by the selected six-AOI sample; class coverage is excluded from binomial pass probabilities.",
        },
        "perfect_agreement_clear_positive_and_negative_but_no_insufficient":
            review(["change", "no_change"] * 6),
        "H_pass_with_one_changed_episode":
            review(["change", "insufficient"] + ["no_change"] * 10),
        "same_answer_and_first_but_rejected_on_last_clear": {
            "answers": [ta["answer"], tb["answer"]],
            "first_ids": [ta["first_change_id"], tb["first_change_id"]],
            "last_clear_ids": [ta["last_clear_no_change_id"], tb["last_clear_no_change_id"]],
            "consensus_error": consensus_error,
        },
        "full_target_consensus_despite_one_frame_label_disagreement": {
            "disagreed_frame": "F004", "labels": [a["states"]["F004"], c["states"]["F004"]],
            "consensus_answer": postfirst_consensus["answer"],
        },
        "evidence_tiers_are_not_equivalent": {
            "derived_evidence_ids": ta["evidence_ids"],
            "last_clear": ta["last_clear_no_change_id"],
            "keep_F000_F003": tier_flags(frame_ids, ["F000", "F003"], ta),
            "keep_F002_F003": tier_flags(frame_ids, ["F002", "F003"], ta),
            "no_change_derived_evidence_count": len(negative["evidence_ids"]),
            "no_change_K6": tier_flags(frame_ids, frame_ids[:6], negative),
        },
        "one_bidirectional_pair_bootstrap": {
            "independent_aoi_count": 1, "episode_rows": 2,
            "mean": bootstrap[1], "ci95": [bootstrap[2], bootstrap[3]],
            "note": "Synthetic all-success sample: nonparametric bootstrap cannot invent failures or new AOIs; this is not strong population evidence.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
