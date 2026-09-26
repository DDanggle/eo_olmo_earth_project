#!/usr/bin/env python3
"""E8b: same two-image question, controlled image pairs, on EarthDial (zero-shot).

Prereg: config/e8b_pair_controls_prereg_v0.json. Wraps code/e8_earthdial_protocol_v0.py (unchanged,
sha256 529d1422...) and only adds arms, so the question wording is identical across arms:
  pair_post_post  [post, post] of each item      (no change; a post-scar reader says yes on pos tiles)
  pair_pre_pre    [pre, pre]                     (no change)
  pair_reversed   [post, pre]                    (a direction-blind difference detector says yes on pos tiles)
  real            rerun, for a reproduction check against E8 (MS-159)

  python3 -B code/e8b_pair_controls_v0.py --selftest
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.../third_party/earthdial_py python -B code/e8b_pair_controls_v0.py --ckpt RGB --out e8b_pair_controls_v0
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_earthdial_protocol_v0 as e8  # noqa: E402

EXTRA = ("pair_post_post", "pair_pre_pre", "pair_reversed")
_orig_plan = e8.plan_arms


def plan_arms(q1):
    jobs, pairs = _orig_plan(q1)
    for a in EXTRA:
        jobs[a] = []
    for t in sorted(pairs):
        for it in pairs[t]:
            pre, post = it["images"]
            jobs["pair_post_post"].append(dict(src=it, images=[post, post], n_images=2, emb_gold="no", emb_item=it["id"]))
            jobs["pair_pre_pre"].append(dict(src=it, images=[pre, pre], n_images=2, emb_gold="no", emb_item=it["id"]))
            jobs["pair_reversed"].append(dict(src=it, images=[post, pre], n_images=2, emb_gold=it["answer"], emb_item=it["id"]))
    return jobs, pairs


def yes_by_tile(rows, kind):
    """{tile: 1/0} for the pos ('yes') or neg ('no') item of each tile."""
    return {r["tile"]: int(r["parsed"] == "yes") for r in rows if r["text_gold"] == kind}


def paired_diff(a, b, n=2000, seed=20260927):
    tiles = sorted(set(a) & set(b))
    f = lambda ts: sum(a[t] - b[t] for t in ts) / len(ts)
    rng = random.Random(seed)
    boots = sorted(f([rng.choice(tiles) for _ in tiles]) for _ in range(n))
    return {"n_tiles": len(tiles), "value": f(tiles), "ci95": [boots[int(.025 * n)], boots[int(.975 * n) - 1]]}


def analyse(rows_by_arm, e8_real, keep=lambda r: True):
    R = {a: [r for r in v if keep(r)] for a, v in rows_by_arm.items()}
    same = [e8_real.get(r["id"]) == r["parsed"] for r in R["real"] if r["id"] in e8_real]
    pos = {a: yes_by_tile(R[a], "yes") for a in R}
    neg = {a: yes_by_tile(R[a], "no") for a in R}
    out = {"reproduction_vs_e8": sum(same) / len(same) if same else None,
           "parse_fail": {a: sum(r["parsed"] is None for r in v) / len(v) for a, v in R.items()},
           "p_yes_pos_items": {a: sum(v.values()) / len(v) for a, v in pos.items()},
           "p_yes_neg_items": {a: sum(v.values()) / len(v) for a, v in neg.items()},
           "compare_real_vs_postpost_pos": paired_diff(pos["real"], pos["pair_post_post"]),
           "compare_real_vs_reversed_pos": paired_diff(pos["real"], pos["pair_reversed"])}
    c, d = out["compare_real_vs_postpost_pos"], out["compare_real_vs_reversed_pos"]
    if out["reproduction_vs_e8"] is None or out["reproduction_vs_e8"] < 0.95 or max(out["parse_fail"].values()) > 0.10:
        out["verdict"] = "invalid_run"
    else:
        out["verdict"] = "compares_pair" if c["value"] >= 0.10 and c["ci95"][0] > 0 else \
            "post_scar_cue" if abs(c["value"]) < 0.05 else "unclear"
        out["direction"] = "direction_sensitive" if d["value"] >= 0.10 and d["ci95"][0] > 0 else "direction_blind_or_unclear"
    return out


def selftest():
    items = []
    for t in range(40):
        base = {"tile": f"t{t}", "fold": "holdout_a", "type": "Q1", "dates": ["2018-01-01", "2018-03-01"]}
        items += [{**base, "id": f"t{t}_pos", "answer": "yes", "images": [f"t{t}_pre.png", f"t{t}_post.png"]},
                  {**base, "id": f"t{t}_neg", "answer": "no", "images": [f"t{t}_post.png", f"t{t}_later.png"]}]
    jobs, pairs = plan_arms(items)
    assert all(len(jobs[a]) == 80 for a in EXTRA) and len(pairs) == 40
    scar = lambda p: p.endswith(("post.png", "later.png"))

    def run(fn):
        return {a: [e8.row(a, j, fn(j["images"])) for j in jobs[a]] for a in ("real",) + EXTRA}

    comparer = lambda ims: "yes" if (not scar(ims[0]) and scar(ims[1])) else "no"   # new scar appears
    post_cue = lambda ims: "yes" if ims[1].endswith("post.png") else "no"            # looks at the 2nd image only
    diff_blind = lambda ims: "yes" if scar(ims[0]) != scar(ims[1]) else "no"          # any difference
    for fn, verdict, direction in ((comparer, "compares_pair", "direction_sensitive"),
                                   (post_cue, "post_scar_cue", None),
                                   (diff_blind, "compares_pair", "direction_blind_or_unclear")):
        rows = run(fn)
        e8_real = {r["id"]: r["parsed"] for r in rows["real"]}
        s = analyse(rows, e8_real)
        assert s["verdict"] == verdict, (verdict, s)
        if direction:
            assert s["direction"] == direction, (direction, s["direction"])
    print("selftest ok: extra arms, reproduction gate, compares/post-cue/direction branches")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default="RGB")
    ap.add_argument("--out", default="e8b_pair_controls_v0")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    arms = ("real",) + EXTRA
    e8.plan_arms = plan_arms       # adds the extra arms; e8.ARMS stays as in v0 so v0's own scorer is skipped
    e8.run(argparse.Namespace(selftest=False, probe=False, ckpt=a.ckpt, model_dir=None, earthdial_src=str(e8.EARTHDIAL_SRC),
                              template="earthdial", sensor="hr_temp", arms=",".join(arms), out=a.out))
    root = e8.ROOT / a.out
    jl = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    rows = {x: jl(root / f"answers_{x}.jsonl") for x in arms}
    e8_real = {r["id"]: r["parsed"] for r in jl(e8.ROOT / "e8_earthdial_rgb_v0/answers_real.jsonl")}
    e0_tiles = {r["tile"] for r in jl(e8.ROOT / "earthtalk_content_controls_v0/answers_real.jsonl")}
    scores = {"primary_192": analyse(rows, e8_real, lambda r: r["tile"] in e0_tiles), "all_203": analyse(rows, e8_real)}
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores, indent=1))
    print("E8b DONE")
