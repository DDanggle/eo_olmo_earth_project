#!/usr/bin/env python3
"""Read-only OE1 saved-logit policy audit. Outputs go to a fresh separate directory."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from yes_no_scoring import POLICY_VERSION, analyze_historical_record, score_answers

ARMS = ("frozen", "joint", "blind")
CONTROLS = ("real", "zero", "observation_swap")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def same_structure(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_structure(a[key], b[key]) for key in a)
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
    return a == b


def comparable_metrics(golds, predictions):
    return {k: v for k, v in score_answers(golds, predictions).items() if k not in ("n", "support")}


def audit(run_dir, out):
    run_dir, out = Path(run_dir).resolve(strict=True), Path(out)
    require(not out.exists(), "Refuse to overwrite audit directory")
    source_names = ("status.json", "manifest.json", "prompts.json", "summary.json", "predictions.jsonl",
                    "items.json", "donor_plan.json", "run_config.json")
    pins = {name: sha(run_dir / name) for name in source_names}
    status, manifest = read(run_dir / "status.json"), read(run_dir / "manifest.json")
    require(status["status"] == "completed" and status["valid"] is True and manifest["valid"] is True, "Source experiment is not complete and valid")
    require(status["manifest_sha256"] == pins["manifest.json"], "Original completion manifest changed")
    for name in source_names:
        if name not in ("status.json", "manifest.json"):
            require(manifest["files_sha256"][name] == pins[name], "Original receipt SHA mismatch: " + name)
    token_ids = read(run_dir / "prompts.json")["answer_ids"]
    config, summary = read(run_dir / "run_config.json"), read(run_dir / "summary.json")
    original_items = read(run_dir / "items.json")
    items = {item["id"]: item for item in original_items}
    require(len(items) == len(original_items), "Duplicate source items")
    dev = {key for key, value in items.items() if value["partition"] == "dev"}
    require(len(dev) == config["n_dev"] == 256, "Unexpected source development population")
    donors = read(run_dir / "donor_plan.json")["mapping"]
    require(len(donors) == 68 and set(donors).issubset(dev) and set(donors.values()).issubset(dev), "Unexpected donor population")
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    require(len(rows) == summary["prediction_rows"] == 1740, "Unexpected original response coverage")
    groups, changes, residuals = defaultdict(list), [], []
    seen = set()
    for record_number, row in enumerate(rows, 1):
        key = (row["arm"], row["control"], row["id"])
        require(key not in seen and key[0] in ARMS and key[1] in CONTROLS and key[2] in dev, "Duplicate or unknown prediction")
        seen.add(key)
        item = items[row["id"]]
        shown_id = donors[row["id"]] if row["control"] == "observation_swap" else row["id"]
        shown = items[shown_id]
        require(row["presented_id"] == shown_id and row["source_gold"] == item["output"] and row["eval_gold"] == shown["output"], "Original question/gold association differs")
        require(row["patch_id"] == item["patch_id"] and row["presented_patch_id"] == shown["patch_id"] and row["presented_date"] == shown["date"], "Original observation association differs")
        if row["control"] == "observation_swap":
            require(item["input"] == shown["input"] and item["output"] != shown["output"], "Swap lacks an exact-question opposite-answer source")
        analysis = analyze_historical_record(row, token_ids)
        entry = {"original_record_number": record_number, "original": row, "policy_comparison": analysis}
        groups[row["arm"] + "/" + row["control"]].append(entry)
        if analysis["constrained_label_changes"]:
            changes.append(entry)
        if analysis["prospective_binary_differs_from_primary"]:
            residuals.append(entry)
    report_groups = {}
    for arm in ARMS:
        for control in CONTROLS:
            name = arm + "/" + control
            entries = groups[name]
            require({e["original"]["id"] for e in entries} == (set(donors) if control == "observation_swap" else dev), "Original condition coverage differs: " + name)
            golds = [e["original"]["eval_gold"] for e in entries]
            primary = [e["policy_comparison"]["primary_unchanged"] for e in entries]
            old = [e["policy_comparison"]["historical_constrained"] for e in entries]
            proposed = [e["policy_comparison"]["prospective_constrained"] for e in entries]
            original_metrics = {"n": len(entries), "support": dict(Counter(golds)),
                                "parsed": comparable_metrics(golds, primary),
                                "constrained_prediction": comparable_metrics(golds, old)}
            require(same_structure(original_metrics, summary["arms"][name]), "Historical score no longer matches preserved summary: " + name)
            report_groups[name] = {
                "historical_primary_unchanged": score_answers(golds, primary),
                "historical_secondary_unchanged": score_answers(golds, old),
                "prospective_rule_on_saved_logits_diagnostic_only": score_answers(golds, proposed),
                "exact_logit_ties": sum(e["policy_comparison"]["exact_logit_tie"] for e in entries),
                "changed_secondary_rows": sum(e["policy_comparison"]["constrained_label_changes"] for e in entries),
                "probability_half_without_exact_tie": sum(e["policy_comparison"]["saved_probability_half_without_logit_tie"] for e in entries),
                "remaining_output_space_disagreements": sum(e["policy_comparison"]["prospective_binary_differs_from_primary"] for e in entries),
                "correct_gains_under_prospective_rule": sum(o != g and p == g for g, o, p in zip(golds, old, proposed)),
                "correct_losses_under_prospective_rule": sum(o == g and p != g for g, o, p in zip(golds, old, proposed)),
            }
    require(all(sha(run_dir / name) == value for name, value in pins.items()), "Original receipts changed during audit")
    result = {
        "schema": "oe1-historical-tie-policy-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only retrospective scoring diagnostic; original experiment results unchanged",
        "source_run": str(run_dir), "source_files_sha256": pins, "answer_token_ids": token_ids,
        "proposed_policy": POLICY_VERSION, "records": len(rows), "changed_secondary_records": len(changes),
        "historical_primary_and_secondary_scores_reproduced": True,
        "source_bytes_unchanged": True, "groups": report_groups,
        "limitations": [
            "These rerule diagnostics do not replace the registered/historical primary or secondary results.",
            "Only the saved first argmax token and canonical yes/no logits are available; full-vocabulary logits are not reconstructed.",
            "Casting saved BF16-derived logits to float cannot recover precision lost before saving; a future precision comparison requires new inference.",
            "Canonical yes/no decoding and case-normalized full-vocabulary decoding remain different tasks; alternate-case tokens and invalid tokens can cause legitimate residual disagreement.",
            "A probability rounded to 0.5 is distinct from equal saved logits. The new policy compares logits directly and does not use a probability threshold.",
            "Observation swaps replace both image and acquisition date and do not isolate pixel-only effects.",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    json_write(out / "audit.json", result)
    for name, entries in (("changed_secondary_rows.jsonl", changes), ("remaining_output_space_disagreements.jsonl", residuals)):
        with (out / name).open("x") as stream:
            for entry in entries:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    json_write(out / "audit_manifest.json", {"schema": "oe1-tie-audit-artifacts-v1", "source_files_sha256": pins,
                "files_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
                "code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("yes_no_scoring.py")]}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.out)
    print(json.dumps({"records": result["records"], "changed_secondary_records": result["changed_secondary_records"],
                      "historical_scores_reproduced": result["historical_primary_and_secondary_scores_reproduced"],
                      "source_bytes_unchanged": result["source_bytes_unchanged"]}))
