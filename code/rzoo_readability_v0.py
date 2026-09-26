#!/usr/bin/env python3
"""R-ZOO v0: language readability of EO foundation-model embeddings (pilot of an encoder-zoo benchmark).

Prereg: config/rzoo_readability_prereg_v0.json. One task, many frozen encoders, same tiles/labels/splits:
"During the observed period, did a landslide occur in this area?" on Sen12 window embeddings (12 timesteps
encoded into one spatial map per tile). yes = landslide mask >= 200 px, no = empty mask, in-between dropped.
Encoders = existing caches on identical 6,834 tiles (dims 128-1024, grids 8-32; all pooled to 8x8 = 64 tokens).
Per encoder: reader (projector -> frozen Olmo-3-7B) seeds 1-3 and a non-LLM head seeds 1-3; one shared blind
reader (EO input zero, encoder-independent) seeds 1-3. Controls: cross-tile swap (same test region, opposite
answer), zero embedding. Train regions = the E-series train regions; test = hiroshima, indonesia, china (china untouched).

  python3 -B code/rzoo_readability_v0.py --selftest
  CUDA_VISIBLE_DEVICES=0 python -B code/rzoo_readability_v0.py --encoders olmo_nano,olmo_tiny,olmo_base_half --out rzoo_v0_a
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
from earthtalk_content_controls_v0 import balanced_acc, bootstrap, cross_tile_donor, d_stat, parse  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
ENCODERS = ("olmo_nano", "olmo_tiny", "olmo_base_half", "galileo_nano", "galileo_tiny", "galileo_cache",
            "prithvi_cache", "clay_cache", "clay_cache_native16")
TRAIN_REGIONS = ("chimanimani", "hokkaido", "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur", "itogon")
TEST_REGIONS = ("hiroshima", "indonesia", "china")
SEEDS = (1, 2, 3)
EPOCHS = 2
QUESTION = ("This is a Sentinel-2 time series of an area, encoded by an Earth-observation foundation model: <EO> "
            "During the observed period, did a landslide occur in this area? Answer with yes or no.")


def build_items(labels, regions):
    """labels: {tile: mask_px}, regions: {tile: region} -> item list."""
    out = []
    for t in sorted(labels):
        r, px = regions.get(t), labels[t]
        if r not in TRAIN_REGIONS + TEST_REGIONS or (0 < px < 200):
            continue
        ans = "yes" if px >= 200 else "no"
        out.append({"id": f"{t}_win", "tile": t, "region": r, "fold": "test" if r in TEST_REGIONS else "train",
                    "answer": ans, "kind": "pos" if ans == "yes" else "neg"})
    return out


def eval_plan(test):
    plan = {"real": [(it, it, it["answer"]) for it in test], "swap_cross_tile": [], "zero_embedding": []}
    by_region = defaultdict(list)
    for it in test:
        by_region[it["region"]].append(it)
    for it in test:
        pool = [dict(c, fold=c["region"]) for c in by_region[it["region"]]]
        d = cross_tile_donor(dict(it, fold=it["region"]), pool, lambda s, c: c["answer"] != s["answer"])
        if d:
            plan["swap_cross_tile"].append((it, next(x for x in by_region[it["region"]] if x["id"] == d["id"]), d["answer"]))
        plan["zero_embedding"].append((it, None, None))
    return plan


def scores(rows_by_arm):
    real = rows_by_arm["real"]
    out = {"balanced_acc": balanced_acc(real, "text_gold"),
           "ba_by_region": {g: balanced_acc([r for r in real if r["region"] == g], "text_gold") for g in TEST_REGIONS},
           "parse_fail": {a: sum(r["parsed"] is None for r in v) / len(v) for a, v in rows_by_arm.items() if v},
           "p_yes": sum(r["parsed"] == "yes" for r in real) / len(real)}
    if rows_by_arm.get("swap_cross_tile"):
        sw = rows_by_arm["swap_cross_tile"]
        out["d_cross"] = {"value": d_stat(sw, "emb"), "ci95": bootstrap(sw, "emb")}
    if rows_by_arm.get("zero_embedding"):
        z = rows_by_arm["zero_embedding"]
        out["p_yes_zero"] = sum(r["parsed"] == "yes" for r in z) / len(z)
    return out


def verdicts(sc):
    """Per encoder: readable if in >= 2/3 seeds reader BA - blind BA >= .05 and d_cross >= .10 with CI lower > 0."""
    out = {}
    for enc in {k.split("/")[0] for k in sc if not k.startswith("blind")}:
        ok = 0
        for s in SEEDS:
            r, b = sc.get(f"{enc}/reader_seed{s}"), sc.get(f"blind/reader_seed{s}")
            if not r or not b:
                continue
            d = r.get("d_cross")
            if d and d["value"] >= 0.10 and d["ci95"][0] > 0 and r["balanced_acc"] - b["balanced_acc"] >= 0.05:
                ok += 1
        heads = [sc[f"{enc}/head_seed{s}"]["balanced_acc"] for s in SEEDS if f"{enc}/head_seed{s}" in sc]
        readers = [sc[f"{enc}/reader_seed{s}"]["balanced_acc"] for s in SEEDS if f"{enc}/reader_seed{s}" in sc]
        out[enc] = {"readable": ok >= 2, "seeds_passing": ok, "reader_ba": readers, "head_ba": heads}
    return out


def selftest():
    labels = {f"t{i}": (300 if i % 3 == 0 else 0) for i in range(120)}
    regions = {f"t{i}": (TRAIN_REGIONS + TEST_REGIONS)[i % 10] for i in range(120)}
    labels["tmid"], regions["tmid"] = 50, "hiroshima"
    items = build_items(labels, regions)
    assert all(i["tile"] != "tmid" for i in items)
    test = [i for i in items if i["fold"] == "test"]
    plan = eval_plan(test)
    assert all(s["region"] == e["region"] and s["answer"] != e["answer"] for s, e, _ in plan["swap_cross_tile"])
    mk = lambda fn, arms: {a: [{"id": s["id"], "tile": s["tile"], "region": s["region"], "text_gold": s["answer"], "emb_gold": eg,
                                "parsed": fn(s, eg)} for s, e, eg in plan[a]] for a in arms}
    good = scores(mk(lambda s, eg: eg if eg else "no", list(plan)))
    blind = scores(mk(lambda s, eg: "no", ["real"]))
    sc = {f"enc/reader_seed{s}": good for s in SEEDS} | {f"blind/reader_seed{s}": blind for s in SEEDS} | {f"enc/head_seed{s}": good for s in SEEDS}
    assert verdicts(sc)["enc"]["readable"]
    print("selftest ok: items (mid dropped), region-matched cross swap, scores, readability verdict")


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
    ref = ROOT / "olmo_nano"
    tiles = sorted(p.stem for p in (ref / "mask_u8").glob("*.npy"))
    labels = {t: int(np.load(ref / "mask_u8" / f"{t}.npy").sum()) for t in tiles}
    items = build_items(labels, {t: rec[t]["region"] for t in tiles if t in rec})
    train = [i for i in items if i["fold"] == "train"]
    test = [i for i in items if i["fold"] == "test"]
    plan = eval_plan(test)
    encoders = [e for e in a.encoders.split(",") if e]
    manifest = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "encoders": encoders,
                "n_train": dict(Counter(i["answer"] for i in train)), "n_test": {f"{r}|{k}": v for (r, k), v in Counter((i["region"], i["answer"]) for i in test).items()},
                "items_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / f"manifest_{a.out}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)

    llm_dir = str(ROOT / "olmo_llm/Olmo-3-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(llm_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    EMB = llm.get_input_embeddings()
    H = EMB.weight.shape[1]
    emb_rms = float(EMB.weight.detach().float().pow(2).mean().sqrt())
    text = tok.apply_chat_template([{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True)
    pre_s, post_s = text.split("<EO>")
    ids_pre = tok(pre_s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
    ids_post = tok(post_s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)

    class Projector(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.norm = nn.LayerNorm(d)
            s.mlp = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Linear(2048, H))
            s.out = nn.LayerNorm(H)
            s.gain = nn.Parameter(torch.tensor(1.0))
            s.pos = nn.Parameter(torch.zeros(64, H))

        def forward(s, t):
            return s.out(s.mlp(s.norm(t))) * (emb_rms * s.gain) + s.pos * emb_rms

    class Head(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))

        def forward(s, x):
            return s.net(x).squeeze(-1)

    def load_tokens(enc):
        cache = {}
        for it in items:
            A = torch.from_numpy(np.load(ROOT / enc / "emb_fp16" / f"{it['tile']}.npy").astype("float32"))[None]
            cache[it["tile"]] = F.adaptive_avg_pool2d(A, 8)[0].flatten(1).T.contiguous()  # 64,d
        return cache

    def seq(proj, tokens, answer=None):
        e = proj(tokens.to(dev)).to(torch.bfloat16)
        embs = [EMB(ids_pre), e, EMB(ids_post)]
        lab = [torch.full((len(ids_pre) + 64 + len(ids_post),), -100, device=dev)]
        if answer is not None:
            ids_ans = tok(answer + tok.eos_token, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
            embs.append(EMB(ids_ans))
            lab.append(ids_ans)
        return torch.cat(embs), torch.cat(lab)

    def train_reader(cache, d, seed, blind):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        proj = Projector(d).to(dev)
        opt = torch.optim.AdamW(proj.parameters(), 1e-4, weight_decay=0.01)
        for ep in range(EPOCHS):
            order = rng.permutation(len(train))
            for i in range(0, len(order), 8):
                b = [train[j] for j in order[i:i + 8]]
                ss = [seq(proj, torch.zeros(64, d) if blind else cache[it["tile"]], it["answer"]) for it in b]
                L = max(x[0].shape[0] for x in ss)
                E = torch.zeros(len(ss), L, H, dtype=torch.bfloat16, device=dev)
                Y = torch.full((len(ss), L), -100, device=dev)
                M = torch.zeros(len(ss), L, dtype=torch.long, device=dev)
                for k, (e, y) in enumerate(ss):
                    E[k, :len(e)], Y[k, :len(y)], M[k, :len(e)] = e, y, 1
                loss = llm(inputs_embeds=E, attention_mask=M, labels=Y).loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        return proj.eval()

    def train_head(cache, d, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = Head(d).to(dev)
        opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=0.01)
        for ep in range(EPOCHS):
            order = rng.permutation(len(train))
            for i in range(0, len(order), 8):
                b = [train[j] for j in order[i:i + 8]]
                x = torch.stack([cache[it["tile"]].mean(0) for it in b]).to(dev)
                y = torch.tensor([1.0 if it["answer"] == "yes" else 0.0 for it in b], device=dev)
                loss = F.binary_cross_entropy_with_logits(head(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return head.eval()

    def evaluate(model, cache, d, kind, blind=False):
        rows_by_arm = {}
        jobs = {"real": plan["real"]} if blind else plan
        with torch.no_grad():
            for arm, js in jobs.items():
                rows = []
                for src, emb_item, eg in js:
                    zero = blind or arm == "zero_embedding"
                    t = torch.zeros(64, d) if zero else cache[(emb_item or src)["tile"]]
                    if kind == "head":
                        z = float(model(t.mean(0)[None].to(dev)))
                        raw, parsed = f"{z:.3f}", ("yes" if z > 0 else "no")
                    else:
                        e, _ = seq(model, t)
                        out = llm.generate(inputs_embeds=e[None], attention_mask=torch.ones(1, e.shape[0], dtype=torch.long, device=dev),
                                           max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
                        raw = tok.decode(out[0], skip_special_tokens=True)
                        parsed = parse("Q1", raw)
                    rows.append({"id": src["id"], "tile": src["tile"], "region": src["region"], "text_gold": src["answer"],
                                 "emb_gold": eg, "raw": raw, "parsed": parsed})
                rows_by_arm[arm] = rows
        return rows_by_arm

    sc = {}
    runs = [("blind", "reader", s) for s in SEEDS] if a.with_blind else []
    runs += [(enc, kind, s) for enc in encoders for kind in ("head", "reader") for s in SEEDS]
    cache, cache_enc = None, None
    for enc, kind, s in runs:
        name = f"{enc}/{kind}_seed{s}"
        f = out_root / f"{enc}__{kind}_seed{s}.json"
        if f.exists():
            sc[name] = json.loads(f.read_text())["scores"]
            continue
        src_enc = encoders[0] if enc == "blind" else enc
        if cache_enc != src_enc:
            cache, cache_enc = load_tokens(src_enc), src_enc
        d = next(iter(cache.values())).shape[1]
        t0 = time.perf_counter()
        if kind == "head":
            model = train_head(cache, d, s)
            rows = evaluate(model, cache, d, "head")
        else:
            model = train_reader(cache, d, s, enc == "blind")
            rows = evaluate(model, cache, d, "reader", blind=(enc == "blind"))
        sc[name] = scores(rows)
        f.write_text(json.dumps({"scores": sc[name], "seconds": time.perf_counter() - t0, "dim": d,
                                 "rows": {k: v for k, v in rows.items()}}, indent=0))
        print(name, "ba", round(sc[name]["balanced_acc"], 3), {g: round(v, 3) for g, v in sc[name]["ba_by_region"].items() if v is not None},
              "d_cross", sc[name].get("d_cross", {}).get("value"), f"{time.perf_counter() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    (out_root / f"summary_{a.out}.json").write_text(json.dumps({"scores": sc, "manifest": manifest}, indent=1))
    print("RZOO PART DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--with-blind", action="store_true")
    ap.add_argument("--out", default="rzoo_v0")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
