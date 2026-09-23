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
    if not review.get("diagnostic_manifest_ready"):
        raise SystemExit("stage H not complete: diagnostic_manifest_ready is false")
    rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
    pack_dir = Path(a.pack)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_context.json").write_text(json.dumps({
        "prereg": PREREG, "reader": a.reader, "matrix_sha256": sha256_file(matrix_path),
        "pack_id": review.get("pack_id"), "n_rows": len(rows),
        "unlocked_because": "review_report.diagnostic_manifest_ready is true",
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


def cmd_score(a: argparse.Namespace) -> None:
    import numpy as np

    matrix_dir = Path(a.matrix)
    rows = {r["id"]: r for r in (json.loads(l) for l in
                                 (matrix_dir / "diagnostic_matrix.jsonl").read_text().splitlines() if l.strip())}
    answers = [json.loads(l) for l in Path(a.answers).read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(20260923)

    per_condition: dict[str, list] = {}
    swap_rows = []
    for ans in answers:
        row = rows.get(ans["id"])
        if row is None:
            raise SystemExit(f"answer {ans['id']} is not in this matrix")
        target = row["target"]
        sc = score_one(ans["pred"], target)
        per_condition.setdefault(row["condition"], []).append({**sc, "episode_id": row["episode_id"]})
        if row["condition"] == "different_outcome_swap":
            source_target = rows[f"{row['episode_id']}|real"]["target"]
            swap_rows.append({"episode_id": row["episode_id"],
                              "vs_donor": score_one(ans["pred"], target)["answer_ok"],
                              "vs_source": score_one(ans["pred"], source_target)["answer_ok"]})

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

    content = None
    if swap_rows:
        by_ep: dict[str, list[float]] = {}
        for r in swap_rows:
            by_ep.setdefault(r["episode_id"], []).append(float(r["vs_donor"]) - float(r["vs_source"]))
        vals = np.array([np.mean(v) for v in by_ep.values()])
        boot = np.array([rng.choice(vals, len(vals)).mean() for _ in range(5000)])
        diff = float(vals.mean())
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        content = {"n_swap_rows": len(swap_rows), "n_episodes": len(by_ep),
                   "swap_vs_donor": float(np.mean([r["vs_donor"] for r in swap_rows])),
                   "swap_vs_source": float(np.mean([r["vs_source"] for r in swap_rows])),
                   "diff": diff, "ci95": [lo, hi], "criterion": "diff >= .30 and CI excludes 0",
                   "content_use_pass": bool(diff >= 0.30 and lo > 0)}

    real = summary.get("different_outcome_swap") and summary.get("real")
    meta = summary.get("metadata_only", {}).get("answer_acc")
    reading = None
    if content is not None and real is not None and meta is not None:
        beats_meta = summary["real"]["answer_acc"] > meta + 0.15
        reading = "reads_content" if (content["content_use_pass"] and beats_meta) else "does_not_read"

    out = {"schema": "sn7-d1-reader-scores-v0", "prereg": PREREG,
           "answers_file": str(Path(a.answers).name), "per_condition": summary,
           "content_check": content, "registered_reading": reading,
           "not_a_claim": "12-episode exposed development sample; no memory claim follows from this alone"}
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"scores_{Path(a.answers).stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"per_condition": summary, "content_check": content,
                      "registered_reading": reading}, indent=1))
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
    print("SELFTEST OK (10 checks)")


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
