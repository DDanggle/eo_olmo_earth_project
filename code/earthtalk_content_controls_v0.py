#!/usr/bin/env python3
"""E0: does the frozen EarthTalk projector (MS-131, seed 1) answer from the OlmoEarth embedding?

Prereg: config/earthtalk_content_controls_prereg_v0.json. No training. Arms: real, swap_within_tile
(prompt of one item, embedding of the other item on the same tile), zero_embedding, swap_cross_tile,
plus a descriptive Q3 cross-tile swap. Prompt, parser and token construction mirror
code/earthtalk_projector_train.py so the real arm reproduces answers_seed1.jsonl.

  python3 -B code/earthtalk_content_controls_v0.py --selftest           # CPU, pure logic only
  CUDA_VISIBLE_DEVICES=1 python3 -B code/earthtalk_content_controls_v0.py --out earthtalk_content_controls_v0
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
PROJECTOR = ROOT / "earthtalk_projector_v0/projector_seed1.pt"
PROJECTOR_SHA256 = "792a2485049164e080f5dd74f59122c403e5b17486d247807fedb58649e61ac9"
SAVED_REAL = ROOT / "earthtalk_projector_v0/answers_seed1.jsonl"
SEED = "20260925"
QTEXT = {"Q1": "Did a landslide occur between the two observations? Answer with yes or no.",
         "Q3": "A landslide occurred between the two observations. In which quadrant(s) is the landslide located? Answer with one or more of NW, NE, SW, SE."}


# ---------------------------------------------------------------- pure logic (tested by --selftest)

def rank(*parts):
    return hashlib.sha256("|".join((SEED,) + parts).encode()).hexdigest()


def parse(t, ans):
    ans = ans.strip().lower()
    if t == "Q1":
        m = re.search(r"\b(yes|no)\b", ans)
        return m.group(1) if m else None
    if t == "Q3":
        q = sorted(set(re.findall(r"\b(nw|ne|sw|se)\b", ans)))
        return [x.upper() for x in q] or None


def within_tile_pairs(q1):
    """{tile: (pos_item, neg_item)} for tiles that have exactly one of each."""
    by = defaultdict(dict)
    for it in q1:
        by[it["tile"]].setdefault(it["answer"], []).append(it)
    return {t: (d["yes"][0], d["no"][0]) for t, d in by.items()
            if len(d.get("yes", [])) == 1 and len(d.get("no", [])) == 1}


def cross_tile_donor(src, pool, differs):
    cands = [c for c in pool if c["fold"] == src["fold"] and c["tile"] != src["tile"] and differs(src, c)]
    if not cands:
        return None
    return min(cands, key=lambda c: rank(src["id"], c["id"]))


def plan_arms(q1, q3):
    """Return {arm: [job]} where job = (source item, embedding item or None, emb_gold)."""
    pairs = within_tile_pairs(q1)
    paired = [it for t in sorted(pairs) for it in pairs[t]]
    jobs = {"real": [(it, it, it["answer"]) for it in paired],
            "zero_embedding": [(it, None, None) for it in paired],
            "swap_within_tile": [], "swap_cross_tile": [], "q3_swap_cross_tile": []}
    for pos, neg in (pairs[t] for t in sorted(pairs)):
        jobs["swap_within_tile"] += [(pos, neg, "no"), (neg, pos, "yes")]
    for it in paired:
        d = cross_tile_donor(it, paired, lambda s, c: c["answer"] != s["answer"])
        if d:
            jobs["swap_cross_tile"].append((it, d, d["answer"]))
    for it in q3:
        d = cross_tile_donor(it, q3, lambda s, c: set(c["answer"]) != set(s["answer"]))
        if d:
            jobs["q3_swap_cross_tile"].append((it, d, d["answer"]))
    return jobs, pairs


def d_stat(rows, by):
    """P(yes | gold yes) - P(yes | gold no), gold taken from the embedding ('emb') or the prompt ('text')."""
    key = "emb_gold" if by == "emb" else "text_gold"
    yes = [r["parsed"] == "yes" for r in rows if r[key] == "yes"]
    no = [r["parsed"] == "yes" for r in rows if r[key] == "no"]
    if not yes or not no:
        return None
    return sum(yes) / len(yes) - sum(no) / len(no)


def bootstrap(rows, by, n=2000, seed=20260925):
    tiles = sorted({r["tile"] for r in rows})
    by_tile = defaultdict(list)
    for r in rows:
        by_tile[r["tile"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [r for t in (rng.choice(tiles) for _ in tiles) for r in by_tile[t]]
        v = d_stat(sample, by)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def balanced_acc(rows, key):
    pos = [r for r in rows if r[key] == "yes"]
    neg = [r for r in rows if r[key] == "no"]
    if not pos or not neg:
        return None
    return 0.5 * (sum(r["parsed"] == "yes" for r in pos) / len(pos) + sum(r["parsed"] == "no" for r in neg) / len(neg))


def jaccard(pred, gold):
    p, g = set(pred or []), set(gold)
    return len(p & g) / len(p | g) if p | g else 1.0


def verdict(scores):
    rep = scores["validity"]["reproduction_rate"]
    if rep is None or rep < 0.95 or scores["validity"]["max_parse_fail"] > 0.10:
        return "invalid_run"
    if scores["d_real"]["value"] < 0.10:
        return "uninformative_low_signal"
    d, (lo, hi) = scores["d_swap"]["value"], scores["d_swap"]["ci95"]
    if d >= 0.10 and lo > 0:
        return "reads_embedding"
    if d <= -0.10 and hi < 0:
        return "follows_text"
    return "not_demonstrated"


def summarize(rows_by_arm, saved_real):
    s = {"n": {a: len(r) for a, r in rows_by_arm.items()}}
    parse_fail = {a: sum(r["parsed"] is None for r in rows) / max(len(rows), 1) for a, rows in rows_by_arm.items()}
    real = rows_by_arm["real"]
    same = [saved_real.get(r["id"]) == r["parsed"] for r in real if r["id"] in saved_real]
    s["validity"] = {"reproduction_rate": (sum(same) / len(same)) if same else None,
                     "reproduction_n": len(same), "parse_fail": parse_fail,
                     "max_parse_fail": max(parse_fail.values())}
    for name, arm, by in (("d_real", "real", "emb"), ("d_swap", "swap_within_tile", "emb"),
                          ("d_zero", "zero_embedding", "text"), ("d_cross", "swap_cross_tile", "emb")):
        rows = rows_by_arm[arm]
        s[name] = {"value": d_stat(rows, by), "ci95": bootstrap(rows, by)}
    s["balanced_acc"] = {a: {"vs_prompt_gold": balanced_acc(rows, "text_gold"),
                             "vs_embedding_gold": balanced_acc(rows, "emb_gold")}
                         for a, rows in rows_by_arm.items() if a != "q3_swap_cross_tile"}
    s["p_yes"] = {a: sum(r["parsed"] == "yes" for r in rows) / max(len(rows), 1)
                  for a, rows in rows_by_arm.items() if a != "q3_swap_cross_tile"}
    q3 = rows_by_arm.get("q3_swap_cross_tile", [])
    if q3:
        s["q3_secondary"] = {"n": len(q3),
                             "jaccard_vs_donor": sum(jaccard(r["parsed"], r["emb_gold"]) for r in q3) / len(q3),
                             "jaccard_vs_source": sum(jaccard(r["parsed"], r["text_gold"]) for r in q3) / len(q3)}
    s["verdict"] = verdict(s)
    return s


def selftest():
    q1 = []
    for t in range(40):
        fold = "holdout_a" if t < 20 else "holdout_b"
        q1 += [{"id": f"t{t}_pos", "tile": f"t{t}", "fold": fold, "type": "Q1", "dates": ["2018-01-01", "2018-03-01"], "answer": "yes"},
               {"id": f"t{t}_neg", "tile": f"t{t}", "fold": fold, "type": "Q1", "dates": ["2018-03-01", "2018-05-01"], "answer": "no"}]
    q1.append({"id": "lonely", "tile": "t99", "fold": "holdout_a", "type": "Q1", "dates": ["a", "b"], "answer": "yes"})
    q3 = [{"id": f"t{t}_q3", "tile": f"t{t}", "fold": "holdout_a", "type": "Q3", "dates": ["x", "y"],
           "answer": [["NW"], ["SE"], ["NE", "SW"]][t % 3]} for t in range(12)]
    jobs, pairs = plan_arms(q1, q3)
    assert len(pairs) == 40 and "t99" not in pairs, "unpaired tile excluded"
    assert len(jobs["swap_within_tile"]) == 80
    assert all(src["tile"] == emb["tile"] and src["answer"] != emb["answer"] for src, emb, _ in jobs["swap_within_tile"])
    assert all(src["fold"] == emb["fold"] and src["tile"] != emb["tile"] and src["answer"] != emb["answer"]
               for src, emb, _ in jobs["swap_cross_tile"]) and len(jobs["swap_cross_tile"]) == 80
    assert all(set(s["answer"]) != set(e["answer"]) for s, e, _ in jobs["q3_swap_cross_tile"])
    assert plan_arms(q1, q3)[0]["swap_cross_tile"] == jobs["swap_cross_tile"], "deterministic donors"

    def rows_for(arm, answer_fn):
        out = []
        for src, emb, emb_gold in jobs[arm]:
            out.append({"id": src["id"], "tile": src["tile"], "text_gold": src["answer"],
                        "emb_gold": emb_gold, "parsed": answer_fn(src, emb_gold)})
        return out

    reader = lambda src, eg: eg if eg is not None else "no"          # follows the embedding
    texter = lambda src, eg: src["answer"]                           # follows the prompt
    for fn, expect in ((reader, "reads_embedding"), (texter, "follows_text")):
        arms = {a: rows_for(a, fn) for a in ("real", "zero_embedding", "swap_within_tile", "swap_cross_tile")}
        arms["q3_swap_cross_tile"] = []
        saved = {r["id"]: r["parsed"] for r in arms["real"]}
        s = summarize(arms, saved)
        assert s["verdict"] == expect, (expect, s["verdict"], s["d_swap"])
    # a model that always says "no" has no signal
    arms = {a: rows_for(a, lambda s_, e: "no") for a in ("real", "zero_embedding", "swap_within_tile", "swap_cross_tile")}
    arms["q3_swap_cross_tile"] = []
    assert summarize(arms, {r["id"]: r["parsed"] for r in arms["real"]})["verdict"] == "uninformative_low_signal"
    # reproduction below 95% invalidates the run
    arms = {a: rows_for(a, reader) for a in ("real", "zero_embedding", "swap_within_tile", "swap_cross_tile")}
    arms["q3_swap_cross_tile"] = []
    assert summarize(arms, {r["id"]: "maybe" for r in arms["real"]})["verdict"] == "invalid_run"
    assert parse("Q1", "Yes, it did.") == "yes" and parse("Q3", "NW and se") == ["NW", "SE"]
    print("selftest ok: pairing, donors, d/CI, verdict branches (reads/follows/low-signal/invalid)")


# ---------------------------------------------------------------- model run (GPU)

def run(a):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    digest = hashlib.sha256(PROJECTOR.read_bytes()).hexdigest()
    if digest != PROJECTOR_SHA256:
        raise SystemExit(f"projector sha256 mismatch: {digest}")
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=False)
    src_dir = ROOT / "olmo_streaming_dev/single_fp16"
    dev = torch.device("cuda")
    torch.manual_seed(1)
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}

    def kept_dates(tile):
        r = rec[tile]
        q = r["scl_clear_fraction"]
        k = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        return [str(r["times"][i])[:10] for i in k]

    llm_dir = str(ROOT / "olmo_llm/Olmo-3-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(llm_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    emb_layer = llm.get_input_embeddings()
    H = emb_layer.weight.shape[1]
    emb_rms = float(emb_layer.weight.detach().float().pow(2).mean().sqrt())

    class Projector(nn.Module):
        def __init__(s):
            super().__init__()
            s.mlp = nn.Sequential(nn.Linear(768, 2048), nn.GELU(), nn.Linear(2048, H))
            s.ttype = nn.Embedding(4, H)
            s.norm = nn.LayerNorm(768)
            s.out = nn.LayerNorm(H)
            s.gain = nn.Parameter(torch.tensor(1.0))

        def forward(s, tokens, types):
            return s.out(s.mlp(s.norm(tokens))) * (emb_rms * s.gain) + s.ttype(types) * emb_rms

    proj = Projector().to(dev)
    proj.load_state_dict(torch.load(PROJECTOR, map_location=dev))
    proj.eval()

    def eo_tokens(tile, dates):
        S = np.load(src_dir / f"{tile}.npy").astype("float32")
        kd = kept_dates(tile)
        idx = [kd.index(d) for d in dates]
        T = torch.from_numpy(S[idx])
        T = F.avg_pool2d(T, 4).flatten(2).permute(0, 2, 1)
        parts = [T[i] for i in range(len(idx))]
        types = [torch.full((64,), min(i, 2), dtype=torch.long) for i in range(len(idx))]
        if len(idx) >= 2:
            parts.append(T[-1] - T[0])
            types.append(torch.full((64,), 3, dtype=torch.long))
        return torch.cat(parts).to(dev), torch.cat(types).to(dev)

    def generate(src, emb_item, zero):
        n = len(src["dates"])
        user = (f"These are {n} Sentinel-2 observations of the same area in chronological order, "
                f"taken on {', '.join(src['dates'])}: <EO> {QTEXT[src['type']]}")
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        pre, post = text.split("<EO>")
        ids_pre = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        ids_post = tok(post, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        item = src if zero else emb_item
        t, ty = eo_tokens(item["tile"], item["dates"])
        if zero:
            t = torch.zeros_like(t)
        e = torch.cat([emb_layer(ids_pre), proj(t, ty).to(torch.bfloat16), emb_layer(ids_post)])
        out = llm.generate(inputs_embeds=e[None], attention_mask=torch.ones(1, e.shape[0], dtype=torch.long, device=dev),
                           max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    items = [json.loads(l) for l in (ROOT / "sentinel_qa_v0_1/items.jsonl").read_text().splitlines() if l]
    items = [x for x in items if (src_dir / f"{x['tile']}.npy").exists()]
    q1 = [x for x in items if x["type"] == "Q1"]
    q3 = [x for x in items if x["type"] == "Q3"]
    jobs, pairs = plan_arms(q1, q3)
    saved_real = {}
    for l in SAVED_REAL.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            saved_real[r["id"]] = r["parsed"]
    manifest = {"projector_sha256": digest, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "n_q1_items": len(q1), "n_pairs": len(pairs), "jobs": {k: len(v) for k, v in jobs.items()},
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)
    rows_by_arm = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for arm, arm_jobs in jobs.items():
            rows = []
            for src, emb_item, emb_gold in arm_jobs:
                raw = generate(src, emb_item, zero=(arm == "zero_embedding"))
                rows.append({"id": src["id"], "tile": src["tile"], "fold": src["fold"],
                             "emb_item": emb_item["id"] if emb_item else None,
                             "text_gold": src["answer"], "emb_gold": emb_gold,
                             "answer_raw": raw, "parsed": parse(src["type"], raw)})
            rows_by_arm[arm] = rows
            (out_dir / f"answers_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{arm}: {len(rows)} done {time.perf_counter() - t0:.0f}s", flush=True)
    scores = summarize(rows_by_arm, saved_real)
    scores["manifest"] = manifest
    scores["elapsed_s"] = time.perf_counter() - t0
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps({k: scores[k] for k in ("verdict", "d_real", "d_swap", "d_zero", "d_cross", "validity")}, indent=1))
    print("E0 DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="earthtalk_content_controls_v0")
    args = ap.parse_args()
    selftest() if args.selftest else run(args)
