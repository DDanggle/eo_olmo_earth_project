#!/usr/bin/env python3
"""E2: one projector-only EO reader for landslide (S2) + flood (S1), with a same-budget blind baseline.

Prereg: config/e2_multi_reader_prereg_v0.json. Architecture and decoding follow MS-131
(code/earthtalk_projector_train.py); swap/zero/bootstrap logic is imported from the E0 script.
One process loads the LLM once, runs a tiny probe, then 3 reader + 3 blind trainings and their
evaluations. Finished runs (scores present) are skipped on restart; settings never change.

  python3 -B code/e2_multi_reader_v0.py --selftest
  CUDA_VISIBLE_DEVICES=1 python3 -B code/e2_multi_reader_v0.py --out e2_multi_reader_v0
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import (balanced_acc, bootstrap, cross_tile_donor, d_stat, parse,  # noqa: E402
                                           within_tile_pairs)

ROOT = Path("/home/work/data/olmoearth")
SEEDS = (1, 2, 3)
PHEN = {"landslide": ("Sentinel-2", "landslide"), "flood": ("Sentinel-1", "flood")}
FLOOD_SLOTS = {"pre_1": 0, "pre_2": 1, "post": 2}


# ---------------------------------------------------------------- pure logic

def eval_plan(test):
    """Reader arms per phenomenon: real_all on everything, controls on within-tile pairs."""
    plan = {"real_all": [(it, it, it["answer"]) for it in test],
            "swap_within_tile": [], "swap_cross_tile": [], "zero_embedding": []}
    for phen in PHEN:
        items = [it for it in test if it["phen"] == phen]
        pairs = within_tile_pairs(items)
        paired = [it for t in sorted(pairs) for it in pairs[t]]
        for pos, neg in (pairs[t] for t in sorted(pairs)):
            plan["swap_within_tile"] += [(pos, neg, "no"), (neg, pos, "yes")]
        for it in paired:
            plan["zero_embedding"].append((it, None, None))
            d = cross_tile_donor(it, paired, lambda s, c: c["answer"] != s["answer"])
            if d:
                plan["swap_cross_tile"].append((it, d, d["answer"]))
    return plan


def phen_scores(rows_by_arm, phen):
    R = {a: [r for r in rows if r["phen"] == phen] for a, rows in rows_by_arm.items()}
    real = R["real_all"]
    out = {"n_real": len(real), "balanced_acc": balanced_acc(real, "text_gold"),
           "parse_fail": {a: sum(r["parsed"] is None for r in rows) / max(len(rows), 1) for a, rows in R.items() if rows}}
    pos = [r for r in real if r["kind"] == "pos"]
    out["recall"] = sum(r["parsed"] == "yes" for r in pos) / max(len(pos), 1)
    for kind in ("neg", "hard_neg"):
        k = [r for r in real if r["kind"] == kind]
        out[f"fpr_{kind}"] = (sum(r["parsed"] == "yes" for r in k) / len(k)) if k else None
    paired_real = [r for r in real if r["kind"] in ("pos", "neg")]
    if "swap_within_tile" in R and R["swap_within_tile"]:
        out["d_real"] = {"value": d_stat(paired_real, "emb"), "ci95": bootstrap(paired_real, "emb")}
        out["d_swap"] = {"value": d_stat(R["swap_within_tile"], "emb"), "ci95": bootstrap(R["swap_within_tile"], "emb")}
        out["d_cross"] = {"value": d_stat(R["swap_cross_tile"], "emb"), "ci95": bootstrap(R["swap_cross_tile"], "emb")}
        out["d_zero"] = {"value": d_stat(R["zero_embedding"], "text")}
    return out


def final_verdict(scores):
    """scores[(arm, seed)][phen] -> prereg verdict."""
    for key, per in scores.items():
        for phen, s in per.items():
            if s is None or max(s["parse_fail"].values()) > 0.10:
                return "invalid", {}
    passes = {}
    for phen in PHEN:
        ok = 0
        for seed in SEEDS:
            r, b = scores[("reader", seed)][phen], scores[("blind", seed)][phen]
            d = r["d_swap"]
            if d["value"] is not None and d["value"] >= 0.10 and d["ci95"][0] > 0 and \
                    r["balanced_acc"] - b["balanced_acc"] >= 0.05:
                ok += 1
        passes[phen] = ok
    n_ok = sum(v >= 2 for v in passes.values())
    return {2: "reads_both", 1: "reads_one", 0: "not_demonstrated"}[n_ok], passes


def selftest():
    test = []
    for phen in PHEN:
        for t in range(30):
            base = {"tile": f"{phen}{t}", "fold": "test", "type": "Q1", "phen": phen, "dates": ["2020-01-01", "2020-02-01"]}
            test += [{**base, "id": f"{phen}{t}_pos", "answer": "yes", "kind": "pos"},
                     {**base, "id": f"{phen}{t}_neg", "answer": "no", "kind": "neg"}]
        test.append({"tile": f"{phen}_dry", "fold": "test", "type": "Q1", "phen": phen, "dates": ["a", "b"],
                     "id": f"{phen}_hard", "answer": "no", "kind": "hard_neg"})
    plan = eval_plan(test)
    assert len(plan["real_all"]) == len(test) and len(plan["swap_within_tile"]) == 120
    assert all(s["phen"] == e["phen"] and s["tile"] == e["tile"] for s, e, _ in plan["swap_within_tile"])
    assert all(s["phen"] == e["phen"] and s["tile"] != e["tile"] for s, e, _ in plan["swap_cross_tile"])

    def rows(fn, arms):
        return {a: [{"id": s["id"], "tile": s["tile"], "phen": s["phen"], "kind": s["kind"], "text_gold": s["answer"],
                     "emb_gold": eg, "parsed": fn(s, eg)} for s, e, eg in plan[a]] for a in arms}

    reader = lambda s, eg: eg if eg is not None else "no"
    blind = lambda s, eg: "no"
    all_arms = list(plan)
    sc = {}
    for seed in SEEDS:
        rr, bb = rows(reader, all_arms), rows(blind, ["real_all"])
        sc[("reader", seed)] = {p: phen_scores(rr, p) for p in PHEN}
        sc[("blind", seed)] = {p: phen_scores(bb, p) for p in PHEN}
    assert final_verdict(sc)[0] == "reads_both", final_verdict(sc)
    # flood reader follows text only -> reads_one
    texter = lambda s, eg: s["answer"]
    for seed in SEEDS:
        rr = rows(lambda s, eg: reader(s, eg) if s["phen"] == "landslide" else texter(s, eg), all_arms)
        sc[("reader", seed)] = {p: phen_scores(rr, p) for p in PHEN}
    assert final_verdict(sc)[0] == "reads_one", final_verdict(sc)
    # a blind model as good as the reader -> not demonstrated
    for seed in SEEDS:
        sc[("blind", seed)] = sc[("reader", seed)]
    assert final_verdict(sc)[0] == "not_demonstrated"
    print("selftest ok: plan per phenomenon, scores, verdict branches (both/one/none)")


# ---------------------------------------------------------------- GPU run

def run(a):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_root = ROOT / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}
    S2 = ROOT / "olmo_streaming_dev/single_fp16"
    S1 = ROOT / "kurosiwo_s1_cache/single_fp16"

    def kept_dates(tile):
        r = rec[tile]
        q = r["scl_clear_fraction"]
        k = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        return [str(r["times"][i])[:10] for i in k]

    def load(path, phen, fold_filter=None):
        out = []
        for l in Path(path).read_text().splitlines():
            if not l.strip():
                continue
            x = json.loads(l)
            if x["type"] != "Q1" or (fold_filter and x["fold"] != fold_filter):
                continue
            if phen == "landslide":
                if not (S2 / f"{x['tile']}.npy").exists() or x["tile"] not in rec:
                    continue
                x.setdefault("kind", "pos" if x["answer"] == "yes" else "neg")
            elif not (S1 / f"{x['tile']}.npy").exists():
                continue
            x["phen"] = phen
            out.append(x)
        return out

    train = load(ROOT / "sentinel_qa_train_v0/items.jsonl", "landslide") + \
        load(ROOT / "flood_qa_v0/items.jsonl", "flood", "train")
    test = load(ROOT / "sentinel_qa_v0_1/items.jsonl", "landslide") + \
        load(ROOT / "flood_qa_v0/items.jsonl", "flood", "test")
    plan = eval_plan(test)
    manifest = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "e0_code_sha256": hashlib.sha256((Path(__file__).parent / "earthtalk_content_controls_v0.py").read_bytes()).hexdigest(),
                "train": {f"{k[0]}|{k[1]}": v for k, v in Counter((x["phen"], x["answer"]) for x in train).items()},
                "test": {f"{k[0]}|{k[1]}": v for k, v in Counter((x["phen"], x["kind"]) for x in test).items()},
                "plan": {k: len(v) for k, v in plan.items()}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)

    llm_dir = str(ROOT / "olmo_llm/Olmo-3-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(llm_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    EMB = llm.get_input_embeddings()
    H = EMB.weight.shape[1]
    emb_rms = float(EMB.weight.detach().float().pow(2).mean().sqrt())

    class Projector(nn.Module):
        def __init__(s):
            super().__init__()
            s.mlp = nn.Sequential(nn.Linear(768, 2048), nn.GELU(), nn.Linear(2048, H))
            s.ttype = nn.Embedding(4, H)
            s.norm = nn.LayerNorm(768)
            s.out = nn.LayerNorm(H)
            s.gain = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(s.ttype.weight, std=0.02)

        def forward(s, tokens, types):
            return s.out(s.mlp(s.norm(tokens))) * (emb_rms * s.gain) + s.ttype(types) * emb_rms

    cache = {}

    def eo_tokens(it):
        key = (it["phen"], it["tile"], tuple(it["dates"]))
        if key in cache:
            return cache[key]
        if it["phen"] == "landslide":
            S = np.load(S2 / f"{it['tile']}.npy")
            kd = kept_dates(it["tile"])
            idx = [kd.index(d) for d in it["dates"]]
            pool = 4
        else:
            S = np.load(S1 / f"{it['tile']}.npy")
            idx = [FLOOD_SLOTS[s] for s in it["slots"]]
            pool = 6
        T = torch.from_numpy(S[idx].astype("float32"))
        T = F.avg_pool2d(T, pool).flatten(2).permute(0, 2, 1)  # n,64,768
        parts = [T[i] for i in range(len(idx))] + [T[-1] - T[0]]
        types = [torch.full((64,), min(i, 2), dtype=torch.long) for i in range(len(idx))] + [torch.full((64,), 3, dtype=torch.long)]
        cache[key] = (torch.cat(parts), torch.cat(types))
        return cache[key]

    def build(proj, src, emb_item, zero, answer=None):
        sensor, word = PHEN[src["phen"]]
        user = (f"These are 2 {sensor} observations of the same area in chronological order, taken on "
                f"{', '.join(src['dates'])}: <EO> Did a {word} occur between the two observations? Answer with yes or no.")
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        pre, post = text.split("<EO>")
        ids_pre = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        ids_post = tok(post, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        t, ty = eo_tokens(emb_item if emb_item is not None else src)
        t, ty = t.to(dev), ty.to(dev)
        if zero:
            t = torch.zeros_like(t)
        e = proj(t, ty).to(torch.bfloat16)
        embs = [EMB(ids_pre), e, EMB(ids_post)]
        labels = [torch.full((len(ids_pre) + len(e) + len(ids_post),), -100, device=dev)]
        if answer is not None:
            ids_ans = tok(answer + tok.eos_token, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
            embs.append(EMB(ids_ans))
            labels.append(ids_ans)
        return torch.cat(embs), torch.cat(labels)

    def train_proj(seed, blind, items, epochs):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        proj = Projector().to(dev)
        opt = torch.optim.AdamW(proj.parameters(), 1e-4, weight_decay=0.01)
        t0, step = time.perf_counter(), 0
        for ep in range(epochs):
            order = rng.permutation(len(items))
            tot = 0.0
            for i in range(0, len(order), 8):
                seqs = [build(proj, items[j], None, blind, items[j]["answer"]) for j in order[i:i + 8]]
                L = max(s[0].shape[0] for s in seqs)
                E = torch.zeros(len(seqs), L, H, dtype=torch.bfloat16, device=dev)
                Y = torch.full((len(seqs), L), -100, device=dev)
                M = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
                for k, (e, y) in enumerate(seqs):
                    E[k, :len(e)] = e
                    Y[k, :len(y)] = y
                    M[k, :len(e)] = 1
                out = llm(inputs_embeds=E, attention_mask=M, labels=Y)
                opt.zero_grad()
                out.loss.backward()
                opt.step()
                tot += float(out.loss)
                step += 1
                if step % 50 == 0:
                    print(f"  seed {seed} blind {blind} ep {ep} step {step} loss {out.loss.item():.4f} {time.perf_counter() - t0:.0f}s", flush=True)
            print(f"  epoch {ep} mean loss {tot / max(1, math.ceil(len(order) / 8)):.4f}", flush=True)
        return proj.eval(), time.perf_counter() - t0

    def evaluate(proj, blind, jobs_by_arm):
        rows_by_arm = {}
        with torch.no_grad():
            for arm, jobs in jobs_by_arm.items():
                rows = []
                for src, emb_item, emb_gold in jobs:
                    zero = blind or arm == "zero_embedding"
                    e, _ = build(proj, src, emb_item, zero)
                    out = llm.generate(inputs_embeds=e[None], attention_mask=torch.ones(1, e.shape[0], dtype=torch.long, device=dev),
                                       max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
                    raw = tok.decode(out[0], skip_special_tokens=True)
                    rows.append({"id": src["id"], "tile": src["tile"], "fold": src["fold"], "phen": src["phen"], "kind": src["kind"],
                                 "emb_item": emb_item["id"] if emb_item else None, "text_gold": src["answer"],
                                 "emb_gold": emb_gold, "answer_raw": raw, "parsed": parse("Q1", raw)})
                rows_by_arm[arm] = rows
        return rows_by_arm

    # probe: catches shape/prompt/device errors before hours of training; outputs discarded
    if not a.no_probe:
        rng = random.Random(0)
        probe_train = rng.sample(train, 16)
        pj, _ = train_proj(1, False, probe_train, 1)
        small = {k: v[:4] for k, v in plan.items()}
        pr = evaluate(pj, False, small)
        print("probe ok:", {k: [r["answer_raw"][:12] for r in v[:2]] for k, v in pr.items()}, flush=True)
        del pj
        torch.cuda.empty_cache()

    scores = {}
    for arm in ("reader", "blind"):
        for seed in SEEDS:
            d = out_root / f"{arm}_seed{seed}"
            if (d / "scores.json").exists():
                scores[(arm, seed)] = json.loads((d / "scores.json").read_text())["per_phenomenon"]
                print(f"skip finished {d.name}", flush=True)
                continue
            d.mkdir(exist_ok=True)
            blind = arm == "blind"
            proj, train_s = train_proj(seed, blind, train, 3)
            torch.save(proj.state_dict(), d / "projector.pt")
            jobs = {"real_all": plan["real_all"]} if blind else plan
            t0 = time.perf_counter()
            rows = evaluate(proj, blind, jobs)
            for k, v in rows.items():
                (d / f"answers_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
            per = {p: phen_scores(rows, p) for p in PHEN}
            (d / "scores.json").write_text(json.dumps({"per_phenomenon": per, "train_s": train_s,
                                                       "eval_s": time.perf_counter() - t0}, indent=1))
            scores[(arm, seed)] = per
            print(f"{d.name}: " + json.dumps({p: {"ba": round(s["balanced_acc"], 3),
                                                   "d_swap": s.get("d_swap", {}).get("value")} for p, s in per.items()}), flush=True)
            del proj
            torch.cuda.empty_cache()
    verdict, passes = final_verdict(scores)
    final = {"verdict": verdict, "seeds_passing_per_phenomenon": passes,
             "table": {f"{k[0]}_seed{k[1]}": v for k, v in scores.items()}, "manifest": manifest,
             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps({"verdict": verdict, "passes": passes}), flush=True)
    print("E2 DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--out", default="e2_multi_reader_v0")
    args = ap.parse_args()
    selftest() if args.selftest else run(args)
