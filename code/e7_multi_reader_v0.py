#!/usr/bin/env python3
"""E7: three phenomena (landslide S2 pair, flood S1 pair, solar-farm S2 single) + cross questions.

Prereg: config/e7_multi_reader_prereg_v0.json. Extends E2 (code/e2_multi_reader_v0.py) with
(1) a second Sentinel-2 phenomenon so sensor no longer identifies the phenomenon,
(2) cross questions (solar question on landslide tiles, pre vs post event), (3) phenomenon-balanced
sampling at the E2 per-epoch budget, (4) a non-LLM head trained on the same batches.
Arms per seed (1,2,3): reader, blind (EO input zero), head. One process holds the GPU.

  python3 -B code/e7_multi_reader_v0.py --selftest
  CUDA_VISIBLE_DEVICES=0 python3 -B code/e7_multi_reader_v0.py --out e7_multi_reader_v0
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import (balanced_acc, bootstrap, cross_tile_donor, d_stat, parse,  # noqa: E402
                                           within_tile_pairs)

ROOT = Path("/home/work/data/olmoearth")
SEEDS = (1, 2, 3)
EPOCH_ITEMS = 4234            # E2 per-epoch budget (1,036 landslide + 3,198 flood)
PAIR_PHEN = ("landslide", "flood")
QUESTION = {"landslide": 0, "flood": 1, "solar": 2, "cross_solar_on_landslide": 2}
FLOOD_SLOTS = {"pre_1": 0, "pre_2": 1, "post": 2}


# ---------------------------------------------------------------- pure logic

def epoch_sample(pools, rng, n=EPOCH_ITEMS):
    """Equal share for landslide / flood / solar-question; the solar share is half solar chips
    (pos/neg balanced) and half cross items. Sampling with replacement inside each pool."""
    third = n // 3
    solar_pos = [x for x in pools["solar"] if x["answer"] == "yes"]
    solar_neg = [x for x in pools["solar"] if x["answer"] == "no"]
    pick = lambda pool, k: [pool[i] for i in rng.integers(0, len(pool), k)]
    out = (pick(pools["landslide"], third) + pick(pools["flood"], third) +
           pick(solar_pos, third // 4) + pick(solar_neg, third // 4) +
           pick(pools["cross_solar_on_landslide"], n - 2 * third - 2 * (third // 4)))
    return [out[i] for i in rng.permutation(len(out))]


def eval_plan(test):
    plan = defaultdict(list)
    for it in test:
        plan["real_all"].append((it, it, it["answer"]))
    for phen in PAIR_PHEN:
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
    solar = [it for it in test if it["phen"] == "solar"]
    for it in solar:
        plan["zero_embedding"].append((it, None, None))
        d = cross_tile_donor(it, solar, lambda s, c: c["answer"] != s["answer"])
        if d:
            plan["swap_cross_tile"].append((it, d, d["answer"]))
    return dict(plan)


def cross_scores(rows, n_boot=2000, seed=20260926):
    cr = [r for r in rows if r["phen"] == "cross_solar_on_landslide"]
    by = defaultdict(dict)
    for r in cr:
        by[r["tile"]][r["kind"]] = r["parsed"] == "yes"
    tiles = [t for t, v in by.items() if "cross_pre" in v and "cross_post" in v]
    if not tiles:
        return None
    diff = lambda ts: sum(by[t]["cross_post"] for t in ts) / len(ts) - sum(by[t]["cross_pre"] for t in ts) / len(ts)
    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(tiles) for _ in tiles]) for _ in range(n_boot))
    return {"n_tiles": len(tiles), "fpr_all": sum(r["parsed"] == "yes" for r in cr) / len(cr),
            "fpr_pre": sum(by[t]["cross_pre"] for t in tiles) / len(tiles),
            "fpr_post": sum(by[t]["cross_post"] for t in tiles) / len(tiles),
            "post_minus_pre": {"value": diff(tiles), "ci95": [boots[int(.025 * n_boot)], boots[int(.975 * n_boot) - 1]]},
            "parse_fail": sum(r["parsed"] is None for r in cr) / len(cr)}


def phen_scores(rows_by_arm, phen):
    R = {a: [r for r in rows if r["phen"] == phen] for a, rows in rows_by_arm.items()}
    real = R.get("real_all", [])
    if not real:
        return None
    out = {"n_real": len(real), "balanced_acc": balanced_acc(real, "text_gold"),
           "parse_fail": {a: sum(r["parsed"] is None for r in v) / len(v) for a, v in R.items() if v}}
    pos = [r for r in real if r["kind"] == "pos"]
    out["recall"] = sum(r["parsed"] == "yes" for r in pos) / max(len(pos), 1)
    for kind in ("neg", "hard_neg"):
        k = [r for r in real if r["kind"] == kind]
        out[f"fpr_{kind}"] = (sum(r["parsed"] == "yes" for r in k) / len(k)) if k else None
    if phen in PAIR_PHEN and R.get("swap_within_tile"):
        paired = [r for r in real if r["kind"] in ("pos", "neg")]
        out["d_real"] = {"value": d_stat(paired, "emb"), "ci95": bootstrap(paired, "emb")}
        out["d_swap"] = {"value": d_stat(R["swap_within_tile"], "emb"), "ci95": bootstrap(R["swap_within_tile"], "emb")}
    if R.get("swap_cross_tile"):
        out["d_cross"] = {"value": d_stat(R["swap_cross_tile"], "emb"), "ci95": bootstrap(R["swap_cross_tile"], "emb")}
    if phen == "landslide":
        out["ba_by_region"] = {g: balanced_acc([r for r in real if r.get("region") == g], "text_gold")
                               for g in sorted({r.get("region") for r in real})}
    return out


def content_control(s, phen):
    return s.get("d_swap") if phen in PAIR_PHEN else s.get("d_cross")


def final_verdicts(scores):
    """scores[(arm, seed)] = {"phen": {...}, "cross": {...}}"""
    for per in scores.values():
        for s in list(per["phen"].values()) + [per.get("cross") or {}]:
            pf = s.get("parse_fail") if s else None
            if isinstance(pf, dict) and pf and max(pf.values()) > 0.10:
                return {"validity": "invalid"}
            if isinstance(pf, float) and pf > 0.10:
                return {"validity": "invalid"}
    reads = {}
    for phen in ("landslide", "flood", "solar"):
        ok = 0
        for seed in SEEDS:
            r, b = scores[("reader", seed)]["phen"][phen], scores[("blind", seed)]["phen"][phen]
            c = content_control(r, phen)
            if c and c["value"] is not None and c["value"] >= 0.10 and c["ci95"][0] > 0 and \
                    r["balanced_acc"] - b["balanced_acc"] >= 0.05:
                ok += 1
        reads[phen] = {"seeds_passing": ok, "reads": ok >= 2}
    anomaly = sum(1 for s in SEEDS if (c := scores[("reader", s)]["cross"]) and
                  c["post_minus_pre"]["value"] >= 0.10 and c["post_minus_pre"]["ci95"][0] > 0)
    specific = sum(1 for s in SEEDS if (c := scores[("reader", s)]["cross"]) and
                   c["fpr_all"] <= 0.10 and abs(c["post_minus_pre"]["value"]) < 0.05)
    cross = "anomaly_driven" if anomaly >= 2 else "question_specific" if specific >= 2 else "unclear"
    return {"validity": "valid", "reads": reads, "cross_question": cross,
            "cross_counts": {"anomaly_seeds": anomaly, "specific_seeds": specific}}


def selftest():
    test, pools = [], defaultdict(list)
    for phen in PAIR_PHEN:
        for t in range(30):
            base = {"tile": f"{phen}{t}", "fold": "test", "type": "Q1", "phen": phen, "dates": ["a", "b"], "region": "r0"}
            test += [{**base, "id": f"{phen}{t}_pos", "answer": "yes", "kind": "pos"},
                     {**base, "id": f"{phen}{t}_neg", "answer": "no", "kind": "neg"}]
    for t in range(40):
        test.append({"tile": f"s{t}", "fold": "test", "type": "P", "phen": "solar", "id": f"s{t}", "answer": "yes" if t < 10 else "no",
                     "kind": "pos" if t < 10 else "neg"})
    for t in range(30):
        for w in ("pre", "post"):
            test.append({"tile": f"landslide{t}", "fold": "test", "type": "P", "phen": "cross_solar_on_landslide",
                         "id": f"c{t}{w}", "answer": "no", "kind": f"cross_{w}", "dates": ["a"]})
    for it in test:
        pools[it["phen"]].append(it)
    rng = __import__("numpy").random.default_rng(0)
    ep = epoch_sample(pools, rng)
    c = Counter(x["phen"] for x in ep)
    assert len(ep) == EPOCH_ITEMS and c["landslide"] == c["flood"] == EPOCH_ITEMS // 3, c
    solar_ans = Counter(x["answer"] for x in ep if x["phen"] == "solar")
    assert solar_ans["yes"] == solar_ans["no"], solar_ans
    plan = eval_plan(test)
    assert all(s["phen"] == e["phen"] for s, e, _ in plan["swap_cross_tile"])
    assert len([1 for s, e, _ in plan["swap_cross_tile"] if s["phen"] == "solar"]) == 40

    def rows(fn, arms):
        return {a: [{"id": s["id"], "tile": s["tile"], "phen": s["phen"], "kind": s["kind"], "region": s.get("region"),
                     "text_gold": s["answer"], "emb_gold": eg, "parsed": fn(s, eg)} for s, e, eg in plan[a]] for a in arms}

    reader = lambda s, eg: ("yes" if s["kind"] == "cross_post" else "no") if s["phen"].startswith("cross") else (eg if eg is not None else "no")
    blind = lambda s, eg: "no"
    sc = {}
    for seed in SEEDS:
        rr, bb = rows(reader, list(plan)), rows(blind, ["real_all"])
        sc[("reader", seed)] = {"phen": {p: phen_scores(rr, p) for p in ("landslide", "flood", "solar")}, "cross": cross_scores(rr["real_all"])}
        sc[("blind", seed)] = {"phen": {p: phen_scores(bb, p) for p in ("landslide", "flood", "solar")}, "cross": cross_scores(bb["real_all"])}
    v = final_verdicts(sc)
    assert all(x["reads"] for x in v["reads"].values()) and v["cross_question"] == "anomaly_driven", v
    reader2 = lambda s, eg: "no" if s["phen"].startswith("cross") else (eg if eg is not None else "no")
    for seed in SEEDS:
        rr = rows(reader2, list(plan))
        sc[("reader", seed)] = {"phen": {p: phen_scores(rr, p) for p in ("landslide", "flood", "solar")}, "cross": cross_scores(rr["real_all"])}
    assert final_verdicts(sc)["cross_question"] == "question_specific"
    print("selftest ok: balanced epoch, plan, per-phenomenon reads, cross verdict branches")


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
    S2, S1, SOL = ROOT / "olmo_streaming_dev/single_fp16", ROOT / "kurosiwo_s1_cache/single_fp16", ROOT / "task2_cache/emb_fp16"

    def kept_dates(tile):
        r = rec[tile]
        q = r["scl_clear_fraction"]
        k = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        return [str(r["times"][i])[:10] for i in k]

    def jl(p):
        return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

    def landslide(path):
        out = []
        for x in jl(path):
            if x["type"] == "Q1" and (S2 / f"{x['tile']}.npy").exists() and x["tile"] in rec:
                x.update(phen="landslide", kind="pos" if x["answer"] == "yes" else "neg", region=x["fold"])
                out.append(x)
        return out

    flood_all = [dict(x, phen="flood", region=str(x["event"])) for x in jl(ROOT / "flood_qa_v0/items.jsonl") if (S1 / f"{x['tile']}.npy").exists()]
    e7 = jl(ROOT / a.items / "items.jsonl")
    train = {"landslide": landslide(ROOT / "sentinel_qa_train_v0/items.jsonl"),
             "flood": [x for x in flood_all if x["fold"] == "train"],
             "solar": [x for x in e7 if x["phen"] == "solar" and x["fold"] == "train"],
             "cross_solar_on_landslide": [x for x in e7 if x["phen"] == "cross_solar_on_landslide" and x["fold"] == "train"]}
    test = (landslide(ROOT / "sentinel_qa_v0_1/items.jsonl") + [x for x in flood_all if x["fold"] == "test"] +
            [x for x in e7 if x["fold"] == "test"])
    plan = eval_plan(test)
    manifest = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "items_sha256": hashlib.sha256((ROOT / a.items / "items.jsonl").read_bytes()).hexdigest(),
                "train_pools": {k: len(v) for k, v in train.items()},
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

    class Head(nn.Module):
        """Non-LLM baseline: mean-pooled slot vectors [first, second, second-first] + question one-hot."""
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.LayerNorm(3 * 768 + 3), nn.Linear(3 * 768 + 3, 256), nn.GELU(), nn.Linear(256, 1))

        def forward(s, x):
            return s.net(x).squeeze(-1)

    cache = {}

    def slots(it):
        """Per-date token blocks (n_dates, 64, 768) for the item's own observations."""
        key = (it["phen"], it["tile"], tuple(it.get("dates") or ()), tuple(it.get("slots") or ()))
        if key in cache:
            return cache[key]
        if it["phen"] in ("landslide", "cross_solar_on_landslide"):
            S = np.load(S2 / f"{it['tile']}.npy")
            kd = kept_dates(it["tile"])
            T = torch.from_numpy(S[[kd.index(d) for d in it["dates"]]].astype("float32"))
            T = F.avg_pool2d(T, 4)
        elif it["phen"] == "flood":
            S = np.load(S1 / f"{it['tile']}.npy")
            T = F.avg_pool2d(torch.from_numpy(S[[FLOOD_SLOTS[s] for s in it["slots"]]].astype("float32")), 6)
        else:
            T = F.avg_pool2d(torch.from_numpy(np.load(SOL / f"{it['tile']}.npy").astype("float32"))[None], 4)
        cache[key] = T.flatten(2).permute(0, 2, 1)
        return cache[key]

    def eo_tokens(it):
        T = slots(it)
        parts = [T[i] for i in range(len(T))]
        types = [torch.full((64,), min(i, 2), dtype=torch.long) for i in range(len(T))]
        if len(T) >= 2:
            parts.append(T[-1] - T[0])
            types.append(torch.full((64,), 3, dtype=torch.long))
        return torch.cat(parts).to(dev), torch.cat(types).to(dev)

    def head_input(src, emb_item, zero):
        T = slots(emb_item if emb_item is not None else src).mean(1)  # n_dates,768
        first, second = T[0], T[-1] if len(T) > 1 else torch.zeros(768)
        diff = (T[-1] - T[0]) if len(T) > 1 else torch.zeros(768)
        v = torch.cat([first, second, diff])
        if zero:
            v = torch.zeros_like(v)
        q = torch.zeros(3)
        q[QUESTION[src["phen"]]] = 1
        return torch.cat([v, q])

    def prompt(src):
        if src["phen"] in PAIR_PHEN:
            sensor, word = ("Sentinel-2", "landslide") if src["phen"] == "landslide" else ("Sentinel-1", "flood")
            return (f"These are 2 {sensor} observations of the same area in chronological order, taken on "
                    f"{', '.join(src['dates'])}: <EO> Did a {word} occur between the two observations? Answer with yes or no.")
        return "This is a Sentinel-2 observation of an area: <EO> Is there a solar farm in this area? Answer with yes or no."

    def build(proj, src, emb_item, zero, answer=None):
        text = tok.apply_chat_template([{"role": "user", "content": prompt(src)}], tokenize=False, add_generation_prompt=True)
        pre, post = text.split("<EO>")
        ids_pre = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        ids_post = tok(post, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        t, ty = eo_tokens(emb_item if emb_item is not None else src)
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

    def schedule(seed):
        rng = np.random.default_rng(seed)
        return [epoch_sample(train, rng) for _ in range(3)]

    def train_proj(seed, blind, epochs_items):
        torch.manual_seed(seed)
        proj = Projector().to(dev)
        opt = torch.optim.AdamW(proj.parameters(), 1e-4, weight_decay=0.01)
        t0, step = time.perf_counter(), 0
        for ep, items in enumerate(epochs_items):
            tot = 0.0
            for i in range(0, len(items), 8):
                seqs = [build(proj, it, None, blind, it["answer"]) for it in items[i:i + 8]]
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
                tot += out.loss.item()
                step += 1
                if step % 100 == 0:
                    print(f"  seed {seed} blind {blind} ep {ep} step {step} loss {out.loss.item():.4f} {time.perf_counter() - t0:.0f}s", flush=True)
            print(f"  epoch {ep} mean loss {tot / max(1, math.ceil(len(items) / 8)):.4f}", flush=True)
        return proj.eval(), time.perf_counter() - t0

    def train_head(seed, epochs_items):
        torch.manual_seed(seed)
        head = Head().to(dev)
        opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=0.01)
        for items in epochs_items:
            for i in range(0, len(items), 8):
                b = items[i:i + 8]
                x = torch.stack([head_input(it, None, False) for it in b]).to(dev)
                y = torch.tensor([1.0 if it["answer"] == "yes" else 0.0 for it in b], device=dev)
                loss = F.binary_cross_entropy_with_logits(head(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return head.eval()

    def row(src, emb_item, emb_gold, raw, parsed):
        return {"id": src["id"], "tile": src["tile"], "fold": src["fold"], "phen": src["phen"], "kind": src["kind"],
                "region": src.get("region"), "emb_item": emb_item["id"] if emb_item else None,
                "text_gold": src["answer"], "emb_gold": emb_gold, "answer_raw": raw, "parsed": parsed}

    def evaluate_llm(proj, blind, jobs_by_arm):
        res = {}
        with torch.no_grad():
            for arm, jobs in jobs_by_arm.items():
                rows = []
                for src, emb_item, emb_gold in jobs:
                    e, _ = build(proj, src, emb_item, blind or arm == "zero_embedding")
                    out = llm.generate(inputs_embeds=e[None], attention_mask=torch.ones(1, e.shape[0], dtype=torch.long, device=dev),
                                       max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
                    raw = tok.decode(out[0], skip_special_tokens=True)
                    rows.append(row(src, emb_item, emb_gold, raw, parse("Q1", raw)))
                res[arm] = rows
        return res

    def evaluate_head(head, jobs_by_arm):
        res = {}
        with torch.no_grad():
            for arm, jobs in jobs_by_arm.items():
                rows = []
                for src, emb_item, emb_gold in jobs:
                    z = float(head(head_input(src, emb_item, arm == "zero_embedding")[None].to(dev)))
                    rows.append(row(src, emb_item, emb_gold, f"{z:.4f}", "yes" if z > 0 else "no"))
                res[arm] = rows
        return res

    def summarize(rows):
        return {"phen": {p: phen_scores(rows, p) for p in ("landslide", "flood", "solar")}, "cross": cross_scores(rows["real_all"])}

    if not a.no_probe:
        pj, _ = train_proj(1, False, [epoch_sample(train, np.random.default_rng(0), 16)])
        pr = evaluate_llm(pj, False, {k: v[:3] for k, v in plan.items()})
        print("probe ok:", {k: [r["answer_raw"][:10] for r in v] for k, v in pr.items()}, flush=True)
        del pj
        torch.cuda.empty_cache()

    scores = {}
    for arm in ("head", "reader", "blind"):
        for seed in SEEDS:
            d = out_root / f"{arm}_seed{seed}"
            if (d / "scores.json").exists():
                scores[(arm, seed)] = json.loads((d / "scores.json").read_text())["summary"]
                print(f"skip finished {d.name}", flush=True)
                continue
            d.mkdir(exist_ok=True)
            sched = schedule(seed)
            t0 = time.perf_counter()
            if arm == "head":
                model = train_head(seed, sched)
                torch.save(model.state_dict(), d / "head.pt")
                rows = evaluate_head(model, plan)
                train_s = time.perf_counter() - t0
            else:
                model, train_s = train_proj(seed, arm == "blind", sched)
                torch.save(model.state_dict(), d / "projector.pt")
                rows = evaluate_llm(model, arm == "blind", {"real_all": plan["real_all"]} if arm == "blind" else plan)
            for k, v in rows.items():
                (d / f"answers_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
            summ = summarize(rows)
            (d / "scores.json").write_text(json.dumps({"summary": summ, "train_s": train_s,
                                                       "total_s": time.perf_counter() - t0}, indent=1))
            scores[(arm, seed)] = summ
            print(f"{d.name}: " + json.dumps({p: round(s["balanced_acc"], 3) for p, s in summ["phen"].items() if s} |
                                             {"cross_post_minus_pre": summ["cross"] and round(summ["cross"]["post_minus_pre"]["value"], 3)}), flush=True)
            del model
            torch.cuda.empty_cache()
    verdicts = final_verdicts(scores)
    head_vs_reader = {p: [scores[("reader", s)]["phen"][p]["balanced_acc"] - scores[("head", s)]["phen"][p]["balanced_acc"]
                          for s in SEEDS] for p in ("landslide", "flood", "solar")}
    final = {"verdicts": verdicts, "reader_minus_head_ba": head_vs_reader,
             "table": {f"{k[0]}_seed{k[1]}": v for k, v in scores.items()}, "manifest": manifest,
             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps({"verdicts": verdicts, "reader_minus_head_ba": head_vs_reader}), flush=True)
    print("E7 DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--items", default="e7_items_v1")
    ap.add_argument("--out", default="e7_multi_reader_v0")
    args = ap.parse_args()
    selftest() if args.selftest else run(args)
