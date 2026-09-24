#!/usr/bin/env python3
"""Stage R of the D1 decision experiment (config/decision_experiment_d1_prereg_v0.json).

Runs a frozen VLM over the diagnostic matrix written by `sn7_visible_pack_v05.py finalize` and
scores the registered matched-content control.

The swap condition pairs a SOURCE prompt with DONOR images whose agreed answer differs, and the
matrix already carries the donor's own target. So:
    swap_vs_donor_target  -> correct when the model answers from the pixels
    swap_vs_source_target -> correct when the model answers from the prompt/date order
`content_use = swap_vs_donor - swap_vs_source`, with an episode-level bootstrap CI.

Launch gate: the pack builder writes allowed_for_model_run=false on every row on purpose. This
runner only proceeds when the review report says diagnostic_manifest_ready, and it records the
prereg id plus the sha256 of the exact matrix it consumed.

Usage
  python3 code/sn7_d1_reader_run_v0.py run   --matrix <dir> --reader qwen|molmo --out <dir>
  python3 code/sn7_d1_reader_run_v0.py score --matrix <dir> --answers <file> --out <dir>
  python3 code/sn7_d1_reader_run_v0.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PREREG = "config/decision_experiment_d1_prereg_v0.json"
AMENDMENT = "config/decision_experiment_d1_prereg_v0_amendment_20260924.json"
ANSWERS = ("change_supported", "no_visible_change", "insufficient_evidence")
STATES = ("no_visible_change", "visible_change", "unreadable", "ambiguous")
READERS = {
    "qwen": {"path": "models/Qwen3-VL-8B-Instruct", "loader": "qwen"},
    "molmo": {"path": "models/Molmo2-O-7B", "loader": "molmo"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_reply(text: str) -> dict | None:
    """Tolerant JSON extraction. Unparsable output is an explicit None, counted as wrong."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    answer = obj.get("answer")
    out = {
        "answer": answer if answer in ANSWERS else None,
        "first_change_id": obj.get("first_change_id") or None,
        "last_clear_no_change_id": obj.get("last_clear_no_change_id") or None,
        "evidence_ids": [e for e in (obj.get("evidence_ids") or []) if isinstance(e, str)],
        "current_state": obj.get("current_state") if obj.get("current_state") in STATES else None,
    }
    return out


def score_one(pred: dict | None, target: dict) -> dict:
    if pred is None:
        return {"answer_ok": False, "first_ok": False, "state_ok": False,
                "evidence_jaccard": 0.0, "parse_fail": True}
    answer_ok = pred["answer"] == target.get("answer")
    # first_change_id only counts on change_supported targets; elsewhere both should be null
    if target.get("answer") == "change_supported":
        first_ok = pred["first_change_id"] == target.get("first_change_id")
    else:
        first_ok = pred["first_change_id"] in (None, "", "null")
    gold = set(target.get("evidence_ids") or [])
    got = set(pred["evidence_ids"])
    jac = len(gold & got) / len(gold | got) if (gold or got) else 1.0
    return {"answer_ok": bool(answer_ok), "first_ok": bool(first_ok),
            "state_ok": pred["current_state"] == target.get("current_state"),
            "evidence_jaccard": float(jac), "parse_fail": False}


# ---------------------------------------------------------------- pure helpers (amendment 2026-09-24)

def swap_row_kind(row: dict, real_answer) -> str:
    """Classify a swap row as 'content' or 'temporal'.

    New matrices carry the pack builder's explicit control_pair_kind; legacy
    matrices are classified by comparing the swap target answer with the
    episode's real-image target answer.
    """
    kind = row.get("control_pair_kind")
    if kind in ("content", "temporal"):
        return kind
    return "content" if row.get("target", {}).get("answer") != real_answer else "temporal"


def answer_coverage(answer_ids, matrix_ids):
    """Exact-set coverage between answer ids and matrix row ids (pure).

    Returns (missing, unknown, duplicates) as sorted lists; all three empty
    means the answer file covers the matrix exactly once.
    """
    seen = set()
    duplicates = set()
    for aid in answer_ids:
        if aid in seen:
            duplicates.add(aid)
        seen.add(aid)
    matrix_set = set(matrix_ids)
    return sorted(matrix_set - seen), sorted(seen - matrix_set), sorted(duplicates)


def withheld_input_rates(rows, answers_by_id):
    """Proper-abstention and coincidental-match rates over metadata_only+blank rows (pure).

    proper_abstention_rate: pred answer equals the withheld target answer, i.e.
    the reader abstains when pixels are missing (diagnostic, not a gate).
    coincidental_match_rate: pred answer equals the real-image reference answer;
    a None prediction (parse fail) counts as a mismatch.
    """
    n = 0
    proper = 0
    coincidence = 0
    for row in rows:
        if row.get("condition") not in ("metadata_only", "blank"):
            continue
        n += 1
        ans = answers_by_id.get(row["id"])
        pred = ans.get("pred") if ans else None
        pred_answer = pred.get("answer") if isinstance(pred, dict) else None
        if pred_answer == row.get("target", {}).get("answer"):
            proper += 1
        reference = row.get("real_image_reference_target") or {}
        if pred_answer is not None and pred_answer == reference.get("answer"):
            coincidence += 1
    if n == 0:
        return None, None
    return proper / n, coincidence / n


# ---------------------------------------------------------------- readers

def load_reader(name: str, root: Path):
    import torch
    from PIL import Image

    cfg = READERS[name]
    model_dir = root / cfg["path"]
    if cfg["loader"] == "qwen":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(model_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda").eval()

        def generate(prompt: str, image_paths: list[str]) -> str:
            content = [{"type": "image", "image": p} for p in image_paths]
            content.append({"type": "text", "text": prompt})
            chat = proc.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
            images = [Image.open(p).convert("RGB") for p in image_paths] or None
            inputs = proc(text=[chat], images=images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    else:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()

        def generate(prompt: str, image_paths: list[str]) -> str:
            content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
            content.append({"type": "text", "text": prompt})
            inputs = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                              add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True)
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def frame_paths(row: dict, pack_dir: Path, blank_dir: Path) -> list[str]:
    """Image list per registered condition. metadata_only sends no pixels; blank sends grey frames."""
    mode = row["input"]["image_mode"]
    if mode == "metadata_only":
        return []
    if mode == "blank":
        from PIL import Image
        blank_dir.mkdir(parents=True, exist_ok=True)
        blank = blank_dir / "blank_512.png"
        if not blank.exists():
            Image.new("RGB", (512, 512), (128, 128, 128)).save(blank)
        return [str(blank)] * len(row["input"]["frames"])
    return [str((pack_dir / f["path"]).resolve()) for f in row["input"]["frames"]]


# ---------------------------------------------------------------- commands

def cmd_run(a: argparse.Namespace) -> None:
    root = Path(a.root)
    matrix_dir = Path(a.matrix)
    matrix_path = matrix_dir / "diagnostic_matrix.jsonl"
    review = json.loads((matrix_dir / "review_report.json").read_text())
    gate_reasons = []
    if not review.get("scientific_gate_passed"):
        gate_reasons.append(f"scientific_gate_passed is false ({review.get('h_gate_reason') or 'no reason recorded'})")
    if not review.get("diagnostic_manifest_ready"):
        gate_reasons.append("diagnostic_manifest_ready is false")
    if gate_reasons:
        raise SystemExit("stage H gate not passed: " + "; ".join(gate_reasons))
    rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
    real_answer_by_ep = {r["episode_id"]: r["target"]["answer"] for r in rows if r.get("condition") == "real"}
    n_content_pairs = sum(1 for r in rows
                          if r.get("condition") == "different_outcome_swap"
                          and swap_row_kind(r, real_answer_by_ep.get(r["episode_id"])) == "content")
    if n_content_pairs == 0:
        raise SystemExit("preparation_incomplete: no content-different donor pair")
    pack_dir = Path(a.pack)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run_context_{a.reader}.json").write_text(json.dumps({
        "prereg": PREREG, "amendment": AMENDMENT, "reader": a.reader,
        "matrix_sha256": sha256_file(matrix_path),
        "pack_id": review.get("pack_id"), "n_rows": len(rows), "n_content_pairs": n_content_pairs,
        "h_agreement_rate": review.get("agreement_rate"), "h_gate_pass": review.get("h_gate_pass"),
        "unlocked_because": "review_report.diagnostic_manifest_ready and review_report.scientific_gate_passed are both true",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    generate = load_reader(a.reader, root)
    dest = out / f"answers_{a.reader}.jsonl"
    done = {json.loads(l)["id"] for l in dest.read_text().splitlines() if l.strip()} if dest.exists() else set()
    handle = dest.open("a")
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        images = frame_paths(row, pack_dir, out / "blank")
        try:
            raw = generate(row["input"]["prompt"], images)
        except Exception as exc:  # recorded, never silently skipped
            raw = f"ERROR {str(exc)[:200]}"
        handle.write(json.dumps({"id": row["id"], "episode_id": row["episode_id"], "aoi": row["aoi"],
                                 "condition": row["condition"], "n_images": len(images),
                                 "raw": raw[:2000], "pred": parse_reply(raw)}) + "\n")
        handle.flush()
        if i % 10 == 0:
            print(i, f"{time.perf_counter() - t0:.0f}s", flush=True)
    handle.close()
    print("READER RUN DONE")


def _episode_diff(swap_rows, donor_key, source_key):
    """Mean per-episode (donor - source) gap with a 5000-episode bootstrap CI."""
    import numpy as np

    by_ep: dict[str, list[float]] = {}
    for r in swap_rows:
        by_ep.setdefault(r["episode_id"], []).append(float(r[donor_key]) - float(r[source_key]))
    vals = np.array([np.mean(v) for v in by_ep.values()])
    rng = np.random.default_rng(20260923)
    boot = np.array([rng.choice(vals, len(vals)).mean() for _ in range(5000)])
    diff = float(vals.mean())
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return by_ep, diff, lo, hi


def cmd_score(a: argparse.Namespace) -> None:
    import numpy as np

    matrix_dir = Path(a.matrix)
    matrix_path = matrix_dir / "diagnostic_matrix.jsonl"
    rows = {r["id"]: r for r in (json.loads(l) for l in matrix_path.read_text().splitlines() if l.strip())}
    answers_path = Path(a.answers)
    answers = [json.loads(l) for l in answers_path.read_text().splitlines() if l.strip()]
    matrix_hash = sha256_file(matrix_path)
    answers_hash = sha256_file(answers_path)

    missing, unknown, duplicates = answer_coverage([ans["id"] for ans in answers], rows.keys())
    if missing or unknown or duplicates:
        raise SystemExit(f"incomplete answers: missing {len(missing)}, unknown {len(unknown)}, duplicate {len(duplicates)}")
    for ctx_path in sorted(answers_path.parent.glob("run_context_*.json")):
        recorded = json.loads(ctx_path.read_text()).get("matrix_sha256")
        if recorded and recorded != matrix_hash:
            raise SystemExit(f"answers resume from a different matrix: {ctx_path.name} recorded "
                             f"{recorded[:16]}..., current matrix is {matrix_hash[:16]}...")

    per_condition: dict[str, list] = {}
    scored_rows: dict[str, dict] = {}
    content_rows = []   # swap rows whose donor has a DIFFERENT answer
    temporal_rows = []  # swap rows whose donor has the SAME answer but a different first change
    for ans in answers:
        row = rows[ans["id"]]
        target = row["target"]
        sc = score_one(ans["pred"], target)
        scored_rows[ans["id"]] = {**sc, "episode_id": row["episode_id"], "condition": row["condition"]}
        per_condition.setdefault(row["condition"], []).append(scored_rows[ans["id"]])
        if row["condition"] == "different_outcome_swap":
            real_row = rows[f"{row['episode_id']}|real"]
            source_sc = score_one(ans["pred"], real_row["target"])
            entry = {"episode_id": row["episode_id"],
                     "vs_donor": sc["answer_ok"], "vs_source": source_sc["answer_ok"],
                     "joint_donor": bool(sc["answer_ok"] and sc["first_ok"]),
                     "joint_source": bool(source_sc["answer_ok"] and source_sc["first_ok"])}
            if swap_row_kind(row, real_row["target"].get("answer")) == "content":
                content_rows.append(entry)
            else:
                temporal_rows.append(entry)

    summary = {}
    for cond, items in sorted(per_condition.items()):
        summary[cond] = {
            "n": len(items),
            "answer_acc": float(np.mean([x["answer_ok"] for x in items])),
            "first_acc": float(np.mean([x["first_ok"] for x in items])),
            "state_acc": float(np.mean([x["state_ok"] for x in items])),
            "evidence_jaccard": float(np.mean([x["evidence_jaccard"] for x in items])),
            "parse_fail": float(np.mean([x["parse_fail"] for x in items])),
        }

    change_first = [sc["first_ok"] for rid, sc in scored_rows.items()
                    if rows[rid]["target"].get("answer") == "change_supported"]
    first_acc_change_subset = float(np.mean(change_first)) if change_first else None

    content = None
    if content_rows:
        by_ep, diff, lo, hi = _episode_diff(content_rows, "vs_donor", "vs_source")
        content = {"n_content_rows": len(content_rows), "n_episodes": len(by_ep),
                   "swap_vs_donor": float(np.mean([r["vs_donor"] for r in content_rows])),
                   "swap_vs_source": float(np.mean([r["vs_source"] for r in content_rows])),
                   "diff": diff, "ci95": [lo, hi], "criterion": "diff >= .30 and CI excludes 0",
                   "content_use_pass": bool(diff >= 0.30 and lo > 0)}

    temporal = None
    if temporal_rows:
        by_ep, diff, lo, hi = _episode_diff(temporal_rows, "joint_donor", "joint_source")
        temporal = {"n_temporal_rows": len(temporal_rows), "n_episodes": len(by_ep),
                    "swap_vs_donor_joint": float(np.mean([r["joint_donor"] for r in temporal_rows])),
                    "swap_vs_source_joint": float(np.mean([r["joint_source"] for r in temporal_rows])),
                    "diff": diff, "ci95": [lo, hi], "low_support": len(temporal_rows) < 4,
                    "note": "descriptive only; no pass/fail threshold is defined for temporal_use "
                            "and the 0.30 content threshold is not transplanted"}

    proper_abstention_rate, coincidental_match_rate = withheld_input_rates(
        rows.values(), {ans["id"]: ans for ans in answers})

    status = "ok"
    reading = None
    reading_basis = None
    real_summary = summary.get("real")
    if content is None:
        status = "preparation_incomplete"
    elif real_summary is not None and coincidental_match_rate is not None:
        margin = real_summary["answer_acc"] - coincidental_match_rate
        reading_basis = {"real_answer_acc": real_summary["answer_acc"],
                         "coincidental_match_rate": coincidental_match_rate,
                         "proper_abstention_rate": proper_abstention_rate,
                         "margin": margin}
        reading = ("reads_content" if (content["content_use_pass"] and margin > 0.15)
                   else "does_not_read")

    out = {"schema": "sn7-d1-reader-scores-v1", "prereg": PREREG, "amendment": AMENDMENT,
           "status": status, "answers_file": str(answers_path.name),
           "answers_sha256": answers_hash, "matrix_sha256": matrix_hash,
           "per_condition": summary, "first_acc_change_subset": first_acc_change_subset,
           "content_check": content, "temporal_check": temporal,
           "proper_abstention_rate": proper_abstention_rate,
           "coincidental_match_rate": coincidental_match_rate,
           "registered_reading": reading, "reading_basis": reading_basis,
           "not_a_claim": "12-episode exposed development sample; no memory claim follows from this alone"}
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"scores_{answers_path.stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"per_condition": summary, "content_check": content, "temporal_check": temporal,
                      "registered_reading": reading, "status": status}, indent=1))
    print("READER SCORE DONE")


def selftest() -> None:
    assert parse_reply('noise {"answer": "change_supported", "first_change_id": "F004", '
                       '"last_clear_no_change_id": "F003", "evidence_ids": ["F003","F004"], '
                       '"current_state": "visible_change"} tail')["first_change_id"] == "F004"
    assert parse_reply("no json here") is None
    assert parse_reply('{"answer": "maybe"}')["answer"] is None
    assert parse_reply('{"answer": "no_visible_change", "current_state": "bogus"}')["current_state"] is None

    target = {"answer": "change_supported", "first_change_id": "F004",
              "evidence_ids": ["F003", "F004"], "current_state": "visible_change"}
    perfect = score_one(parse_reply('{"answer":"change_supported","first_change_id":"F004",'
                                    '"evidence_ids":["F003","F004"],"current_state":"visible_change"}'), target)
    assert perfect["answer_ok"] and perfect["first_ok"] and perfect["state_ok"]
    assert perfect["evidence_jaccard"] == 1.0 and not perfect["parse_fail"]
    wrong = score_one(parse_reply('{"answer":"no_visible_change","first_change_id":null,'
                                  '"evidence_ids":[],"current_state":"no_visible_change"}'), target)
    assert not wrong["answer_ok"] and not wrong["first_ok"] and wrong["evidence_jaccard"] == 0.0
    unparsable = score_one(None, target)
    assert unparsable["parse_fail"] and not unparsable["answer_ok"]
    # on a no-change target a null first_change_id is correct
    nc = {"answer": "no_visible_change", "first_change_id": None, "evidence_ids": [], "current_state": "no_visible_change"}
    assert score_one(parse_reply('{"answer":"no_visible_change","first_change_id":null,'
                                 '"evidence_ids":[],"current_state":"no_visible_change"}'), nc)["first_ok"]

    # --- amendment 2026-09-24 repairs (pure checks) ---
    # swap rows classify by explicit control_pair_kind; legacy matrices fall
    # back to comparing the swap target answer with the real target answer
    assert swap_row_kind({"target": {"answer": "no_visible_change"}}, "change_supported") == "content"
    assert swap_row_kind({"target": {"answer": "change_supported"}}, "change_supported") == "temporal"
    assert swap_row_kind({"target": {"answer": "change_supported"}, "control_pair_kind": "temporal"},
                         "no_visible_change") == "temporal"
    assert swap_row_kind({"target": {"answer": "no_visible_change"}, "control_pair_kind": "content"},
                         "change_supported") == "content"
    # exact answer coverage reports missing/unknown/duplicate counts
    assert answer_coverage(["a|real", "a|real", "zzz|real"], ["a|real", "b|real"]) == (
        ["b|real"], ["zzz|real"], ["a|real"])
    assert answer_coverage(["a", "b"], ["b", "a"]) == ([], [], [])
    # withheld-input diagnostics: proper abstention vs the withheld target,
    # coincidence vs the real-image reference; None predictions never coincide
    withheld_rows = [
        {"id": "a|metadata_only", "condition": "metadata_only",
         "target": {"answer": "insufficient_evidence"}, "real_image_reference_target": {"answer": "change_supported"}},
        {"id": "a|blank", "condition": "blank",
         "target": {"answer": "insufficient_evidence"}, "real_image_reference_target": {"answer": "change_supported"}},
        {"id": "a|real", "condition": "real",
         "target": {"answer": "change_supported"}, "real_image_reference_target": {"answer": "change_supported"}},
    ]
    withheld_answers = {"a|metadata_only": {"pred": {"answer": "insufficient_evidence"}},
                        "a|blank": {"pred": None},
                        "a|real": {"pred": {"answer": "change_supported"}}}
    proper_rate, coincidence_rate = withheld_input_rates(withheld_rows, withheld_answers)
    assert proper_rate == 0.5
    assert coincidence_rate == 0.0
    print("SELFTEST OK (18 checks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--matrix", required=True)
    run.add_argument("--pack", required=True)
    run.add_argument("--reader", required=True, choices=sorted(READERS))
    run.add_argument("--root", default="/home/work/data/olmoearth")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    sc = sub.add_parser("score")
    sc.add_argument("--matrix", required=True)
    sc.add_argument("--answers", required=True)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=cmd_score)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not getattr(a, "func", None):
        ap.error("run, score or --selftest")
    a.func(a)


if __name__ == "__main__":
    main()
