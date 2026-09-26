#!/usr/bin/env python3
"""E8t: the E8/E8b protocol on TEOChat (ICLR 2025 temporal EO-VLM), zero-shot, landslide Q1.

Prereg: config/e8t_teochat_prereg_v0.json. Inference follows code/sentinel_qa_teochat.py exactly (same model dir,
fp16 load, prompt, run_inference_single with timestamps=dates, max_new_tokens 64, regex parser), so the real
arm can be checked against its saved answers (sentinel_qa_v0_1/answers_teochat.jsonl).
TEOChat receives dates twice (prompt text and timestamps), so a date_swap arm tests date use directly.
Arms: real; swap_within_tile (dates of the item, images of the other item on the tile); date_swap (images of the
item, dates of the other item); pair_post_post; pair_pre_pre; pair_reversed; blank (two grey images).

  python3 -B code/e8t_teochat_protocol_v0.py --selftest
  CUDA_VISIBLE_DEVICES=1 python -B code/e8t_teochat_protocol_v0.py --items sentinel_qa_v0_1 --out e8t_teochat_hiroshima_v0
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import bootstrap, d_stat, within_tile_pairs  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
ARMS = ("real", "swap_within_tile", "date_swap", "pair_post_post", "pair_pre_pre", "pair_reversed", "blank")
BLANK = "<blank>"


def prompt(dates):
    return (f"These are 2 Sentinel-2 satellite images of the same area in chronological order, taken on {', '.join(dates)}: <video> "
            "Did a landslide occur between the two images? Answer with yes or no.")


def parse(a):
    m = re.search(r"\b(yes|no)\b", a.strip().lower())
    return m.group(1) if m else None


def plan(q1):
    """job = (src item, images, dates, image_gold, date_gold)."""
    pairs = within_tile_pairs(q1)
    jobs = {a: [] for a in ARMS}
    for t in sorted(pairs):
        pos, neg = pairs[t]
        for it, other in ((pos, neg), (neg, pos)):
            pre, post = it["images"]
            jobs["real"].append((it, list(it["images"]), it["dates"], it["answer"], it["answer"]))
            jobs["swap_within_tile"].append((it, list(other["images"]), it["dates"], other["answer"], it["answer"]))
            jobs["date_swap"].append((it, list(it["images"]), other["dates"], it["answer"], other["answer"]))
            jobs["pair_post_post"].append((it, [post, post], it["dates"], "no", it["answer"]))
            jobs["pair_pre_pre"].append((it, [pre, pre], it["dates"], "no", it["answer"]))
            jobs["pair_reversed"].append((it, [post, pre], it["dates"], it["answer"], it["answer"]))
            jobs["blank"].append((it, [BLANK, BLANK], it["dates"], None, it["answer"]))
    return jobs, pairs


def row(arm, job, raw):
    it, images, dates, image_gold, date_gold = job
    # "emb_gold" = gold of the shown images (as in E0/E8 scoring), "date_gold" = gold of the item whose dates were shown
    return {"id": it["id"], "tile": it["tile"], "arm": arm, "images": images, "dates": dates, "text_gold": date_gold,
            "emb_gold": image_gold, "item_gold": it["answer"], "raw": raw, "parsed": parse(raw)}


def paired_diff(a, b, n=2000, seed=20260927):
    tiles = sorted(set(a) & set(b))
    f = lambda ts: sum(a[t] - b[t] for t in ts) / len(ts)
    rng = random.Random(seed)
    boots = sorted(f([rng.choice(tiles) for _ in tiles]) for _ in range(n))
    return {"n_tiles": len(tiles), "value": f(tiles), "ci95": [boots[int(.025 * n)], boots[int(.975 * n) - 1]]}


def analyse(R, saved_real=None, keep=lambda r: True):
    R = {a: [r for r in v if keep(r)] for a, v in R.items()}
    pos_yes = lambda rows: {r["tile"]: int(r["parsed"] == "yes") for r in rows if r["item_gold"] == "yes"}
    out = {"parse_fail": {a: sum(r["parsed"] is None for r in v) / len(v) for a, v in R.items()},
           "p_yes": {a: sum(r["parsed"] == "yes" for r in v) / len(v) for a, v in R.items()},
           "d_real": {"value": d_stat(R["real"], "emb"), "ci95": bootstrap(R["real"], "emb")},
           "d_swap_images": {"value": d_stat(R["swap_within_tile"], "emb"), "ci95": bootstrap(R["swap_within_tile"], "emb")},
           "d_date_swap_by_dates": {"value": d_stat(R["date_swap"], "text"), "ci95": bootstrap(R["date_swap"], "text")},
           "compare_real_vs_postpost_pos": paired_diff(pos_yes(R["real"]), pos_yes(R["pair_post_post"])),
           "compare_real_vs_reversed_pos": paired_diff(pos_yes(R["real"]), pos_yes(R["pair_reversed"]))}
    if saved_real is not None:
        same = [saved_real.get(r["id"]) == r["parsed"] for r in R["real"] if r["id"] in saved_real]
        out["reproduction_vs_saved"] = (sum(same) / len(same)) if same else None
    rep = out.get("reproduction_vs_saved")
    if max(out["parse_fail"].values()) > 0.10 or (rep is not None and rep < 0.95):
        out["verdict"] = "invalid_run"
        return out
    d, (lo, hi) = out["d_swap_images"]["value"], out["d_swap_images"]["ci95"]
    if out["d_real"]["value"] < 0.10:
        out["verdict"] = "uninformative_low_signal"
    elif d >= 0.10 and lo > 0:
        out["verdict"] = "reads_images"
    elif d <= -0.10 and hi < 0:
        out["verdict"] = "follows_text"
    else:
        out["verdict"] = "not_demonstrated"
    ds = out["d_date_swap_by_dates"]
    out["date_use"] = "uses_dates" if ds["value"] >= 0.10 and ds["ci95"][0] > 0 else "no_date_use_detected"
    c = out["compare_real_vs_postpost_pos"]
    out["pair_use"] = "compares_pair" if c["value"] >= 0.10 and c["ci95"][0] > 0 else "post_scar_cue" if abs(c["value"]) < 0.05 else "unclear"
    return out


def selftest():
    items = []
    for t in range(40):
        base = {"tile": f"t{t}", "fold": "holdout_a", "type": "Q1"}
        items += [{**base, "id": f"t{t}_pos", "answer": "yes", "dates": ["2018-01-01", "2018-03-01"], "images": [f"t{t}_pre", f"t{t}_post"]},
                  {**base, "id": f"t{t}_neg", "answer": "no", "dates": ["2018-03-01", "2018-05-01"], "images": [f"t{t}_post", f"t{t}_later"]}]
    jobs, pairs = plan(items)
    assert len(pairs) == 40 and all(len(jobs[a]) == 80 for a in ARMS)
    scar = lambda p: p.endswith(("post", "later"))
    comparer = lambda job: "yes" if (not scar(job[1][0]) and scar(job[1][1])) else "no"
    dater = lambda job: "yes" if job[2][0] == "2018-01-01" else "no"
    for fn, verdict, date_use in ((comparer, "reads_images", "no_date_use_detected"), (dater, "follows_text", "uses_dates")):
        R = {a: [row(a, j, fn(j)) for j in jobs[a]] for a in ARMS}
        s = analyse(R, {r["id"]: r["parsed"] for r in R["real"]})
        assert s["verdict"] == verdict and s["date_use"] == date_use, (verdict, s["verdict"], s["date_use"])
    print("selftest ok: 7 arms, image/date swap scoring, verdict + date-use branches")


def run(a):
    from PIL import Image
    sys.path.insert(0, str(ROOT / "teochat/code"))
    from videollava.eval.eval import load_model
    from videollava.eval.inference import run_inference_single

    items_dir = ROOT / a.items
    q1 = [x for x in (json.loads(l) for l in (items_dir / "items.jsonl").read_text().splitlines() if l.strip()) if x["type"] == "Q1"]
    jobs, pairs = plan(q1)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    grey = out / "blank_grey.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(grey)
    tok, model, proc = load_model(model_path=str(ROOT / "teochat/TEOChat"), model_base=None, load_8bit=False, device="cuda")
    manifest = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "items": str(items_dir),
                "items_sha256": hashlib.sha256((items_dir / "items.jsonl").read_bytes()).hexdigest(),
                "n_pairs": len(pairs), "arms": list(ARMS), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)
    R, t0 = {}, time.perf_counter()
    for arm in ARMS:
        rows = []
        for job in jobs[arm]:
            images = [str(grey) if p == BLANK else p for p in job[1]]
            try:
                raw = run_inference_single(model, proc, tok, prompt(job[2]), images, timestamps=job[2], max_new_tokens=64)
            except Exception as e:  # recorded as a parse failure, never retried with other settings
                raw = f"ERROR {e}"
            rows.append(row(arm, job, raw))
        R[arm] = rows
        (out / f"answers_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{arm}: {len(rows)} done {time.perf_counter() - t0:.0f}s e.g. {rows[0]['raw'][:30]!r}", flush=True)
    saved = items_dir / "answers_teochat.jsonl"
    saved_real = {json.loads(l)["id"]: json.loads(l)["parsed"] for l in saved.read_text().splitlines() if l.strip()} if saved.exists() else None
    scores = {"all": analyse(R, saved_real)}
    if a.primary_tiles_from:
        keep_tiles = {json.loads(l)["tile"] for l in (ROOT / a.primary_tiles_from).read_text().splitlines() if l.strip()}
        scores["primary"] = analyse(R, saved_real, lambda r: r["tile"] in keep_tiles)
    scores["manifest"] = manifest
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("verdict", "date_use", "pair_use", "d_real", "d_swap_images", "d_date_swap_by_dates",
                                                  "compare_real_vs_postpost_pos", "reproduction_vs_saved") if kk in v}
                      for k, v in scores.items() if k != "manifest"}, indent=1))
    print("E8t DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--items", default="sentinel_qa_v0_1")
    ap.add_argument("--primary-tiles-from", default="earthtalk_content_controls_v0/answers_real.jsonl")
    ap.add_argument("--out", default="e8t_teochat_hiroshima_v0")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
