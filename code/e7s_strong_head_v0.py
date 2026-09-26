#!/usr/bin/env python3
"""E7s: strongest non-LLM baseline for E7 — a spatial attention head on the same tokens, same schedule.

Prereg: config/e7s_strong_head_prereg_v0.json (meaningfulness rule config/meaningfulness_rule_20260927.json).
Uses E7's items, balanced epoch sampler (same seeds -> same item schedule) and token construction
(per-date 64 tokens; pairs get [first, second, second-first]). Head: per-slot Linear(768->256) + slot type +
position, 2-layer Transformer over all tokens (+ question one-hot token), mean pool, linear -> yes logit.
Compares with E7 reader answers (e7_multi_reader_v0/reader_seed*/answers_real_all.jsonl), paired by item.

  python3 -B code/e7s_strong_head_v0.py --selftest
  CUDA_VISIBLE_DEVICES=0 python -B code/e7s_strong_head_v0.py --out e7s_strong_head_v0
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import balanced_acc  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
SEEDS = (1, 2, 3)
PHENS = ("landslide", "flood", "solar")


def paired_ba_diff(rows_a, rows_b, n=2000, seed=20260927):
    """BA(a) - BA(b) on the same items, tile bootstrap."""
    b_by = {r["id"]: r for r in rows_b}
    pairs = [(r, b_by[r["id"]]) for r in rows_a if r["id"] in b_by]
    tiles = sorted({r["tile"] for r, _ in pairs})
    by_tile = defaultdict(list)
    for p in pairs:
        by_tile[p[0]["tile"]].append(p)

    def diff(ps):
        A, B = [x for x, _ in ps], [y for _, y in ps]
        ba, bb = balanced_acc(A, "text_gold"), balanced_acc(B, "text_gold")
        return None if ba is None or bb is None else ba - bb
    rng = random.Random(seed)
    boots = sorted(v for v in (diff([p for t in (rng.choice(tiles) for _ in tiles) for p in by_tile[t]]) for _ in range(n)) if v is not None)
    return {"n": len(pairs), "value": diff(pairs), "ci95": [boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]]}


def verdict(diffs):
    """diffs[phen][seed] = paired diff (reader - attn head). meaningful if >= .10 with CI lower > 0 in >= 2 seeds."""
    return {p: {"seeds_meaningful": sum(1 for d in diffs[p].values() if d["value"] >= 0.10 and d["ci95"][0] > 0),
                "meaningful_10pp": sum(1 for d in diffs[p].values() if d["value"] >= 0.10 and d["ci95"][0] > 0) >= 2}
            for p in diffs}


def selftest():
    rows_a, rows_b = [], []
    for t in range(100):
        g = "yes" if t % 2 else "no"
        rows_a.append({"id": f"i{t}", "tile": f"t{t}", "text_gold": g, "parsed": g})
        rows_b.append({"id": f"i{t}", "tile": f"t{t}", "text_gold": g, "parsed": g if t % 4 else ("no" if g == "yes" else "yes")})
    d = paired_ba_diff(rows_a, rows_b)
    assert abs(d["value"] - 0.25) < 1e-9 and d["ci95"][0] > 0.1, d
    v = verdict({"solar": {1: d, 2: d, 3: d}})
    assert v["solar"]["meaningful_10pp"]
    print("selftest ok: paired BA difference with tile bootstrap, 10pp verdict")


def run(a):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import e7_multi_reader_v0 as e7

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}
    S2, S1, SOL = ROOT / "olmo_streaming_dev/single_fp16", ROOT / "kurosiwo_s1_cache/single_fp16", ROOT / "task2_cache/emb_fp16"
    jl = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

    def kept_dates(tile):
        r = rec[tile]
        q = r["scl_clear_fraction"]
        k = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        return [str(r["times"][i])[:10] for i in k]

    def landslide(path):
        res = []
        for x in jl(path):
            if x["type"] == "Q1" and (S2 / f"{x['tile']}.npy").exists() and x["tile"] in rec:
                x.update(phen="landslide", kind="pos" if x["answer"] == "yes" else "neg", region=x["fold"])
                res.append(x)
        return res

    flood_all = [dict(x, phen="flood", region=str(x["event"])) for x in jl(ROOT / "flood_qa_v0/items.jsonl") if (S1 / f"{x['tile']}.npy").exists()]
    it7 = jl(ROOT / "e7_items_v1/items.jsonl")
    train = {"landslide": landslide(ROOT / "sentinel_qa_train_v0/items.jsonl"), "flood": [x for x in flood_all if x["fold"] == "train"],
             "solar": [x for x in it7 if x["phen"] == "solar" and x["fold"] == "train"],
             "cross_solar_on_landslide": [x for x in it7 if x["phen"] == "cross_solar_on_landslide" and x["fold"] == "train"]}
    test = landslide(ROOT / "sentinel_qa_v0_1/items.jsonl") + [x for x in flood_all if x["fold"] == "test"] + [x for x in it7 if x["fold"] == "test"]
    cache = {}

    def slots(it):
        key = (it["phen"], it["tile"], tuple(it.get("dates") or ()), tuple(it.get("slots") or ()))
        if key not in cache:
            if it["phen"] in ("landslide", "cross_solar_on_landslide"):
                S = np.load(S2 / f"{it['tile']}.npy")
                kd = kept_dates(it["tile"])
                T = F.avg_pool2d(torch.from_numpy(S[[kd.index(d) for d in it["dates"]]].astype("float32")), 4)
            elif it["phen"] == "flood":
                S = np.load(S1 / f"{it['tile']}.npy")
                T = F.avg_pool2d(torch.from_numpy(S[[e7.FLOOD_SLOTS[s] for s in it["slots"]]].astype("float32")), 6)
            else:
                T = F.avg_pool2d(torch.from_numpy(np.load(SOL / f"{it['tile']}.npy").astype("float32"))[None], 4)
            cache[key] = T.flatten(2).permute(0, 2, 1)
        return cache[key]

    def tokens(it):
        T = slots(it)
        parts = [T[i] for i in range(len(T))] + ([T[-1] - T[0]] if len(T) >= 2 else [])
        types = [min(i, 2) for i in range(len(T))] + ([3] if len(T) >= 2 else [])
        x = torch.zeros(3, 64, 768)
        m = torch.zeros(3, dtype=torch.bool)
        ty = torch.zeros(3, dtype=torch.long)
        for k, (p, t) in enumerate(zip(parts, types)):
            x[k], m[k], ty[k] = p, True, t
        return x, m, ty

    class AttnHead(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256))
            s.pos = nn.Parameter(torch.zeros(64, 256))
            s.ttype = nn.Embedding(4, 256)
            s.q = nn.Embedding(3, 256)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, 0.0, batch_first=True, norm_first=True), 2)
            s.cls = nn.Linear(256, 1)

        def forward(s, x, m, ty, q):  # x B,3,64,768  m B,3  ty B,3  q B
            B = x.shape[0]
            h = s.inp(x) + s.pos + s.ttype(ty)[:, :, None, :]
            h = torch.cat([s.q(q)[:, None, :], h.reshape(B, 3 * 64, 256)], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), (~m)[:, :, None].expand(B, 3, 64).reshape(B, 192)], 1)
            z = s.enc(h, src_key_padding_mask=pad)
            keep = (~pad).float()[..., None]
            return s.cls((z * keep).sum(1) / keep.sum(1)).squeeze(-1)

    def batch(items):
        xs = [tokens(it) for it in items]
        x = torch.stack([a[0] for a in xs]).to(dev)
        m = torch.stack([a[1] for a in xs]).to(dev)
        ty = torch.stack([a[2] for a in xs]).to(dev)
        q = torch.tensor([e7.QUESTION[it["phen"]] for it in items], device=dev)
        return x, m, ty, q

    manifest = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    diffs, res = defaultdict(dict), {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sched = [e7.epoch_sample(train, rng) for _ in range(3)]   # identical schedule to E7 for this seed
        torch.manual_seed(seed)
        head = AttnHead().to(dev)
        opt = torch.optim.AdamW(head.parameters(), 3e-4, weight_decay=0.01)
        for items in sched:
            for i in range(0, len(items), 8):
                b = items[i:i + 8]
                y = torch.tensor([1.0 if it["answer"] == "yes" else 0.0 for it in b], device=dev)
                loss = F.binary_cross_entropy_with_logits(head(*batch(b)), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        head.eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(test), 64):
                b = test[i:i + 64]
                z = head(*batch(b)).tolist()
                rows += [{"id": it["id"], "tile": it["tile"], "phen": it["phen"], "text_gold": it["answer"], "parsed": "yes" if v > 0 else "no"}
                         for it, v in zip(b, z)]
        (out / f"answers_attn_head_seed{seed}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        reader = jl(ROOT / f"e7_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl")
        for p in PHENS:
            hr = [r for r in rows if r["phen"] == p]
            rr = [r for r in reader if r["phen"] == p]
            res[f"{p}_seed{seed}"] = {"attn_head_ba": balanced_acc(hr, "text_gold"), "reader_ba": balanced_acc(rr, "text_gold")}
            diffs[p][seed] = paired_ba_diff(rr, hr)
        print(f"seed {seed}", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items() if k.endswith(str(seed))}), flush=True)
    final = {"manifest": manifest, "per_seed": res, "reader_minus_attn_head": {p: diffs[p] for p in diffs}, "verdict": verdict(diffs)}
    (out / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final["verdict"]), flush=True)
    print("E7s DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="e7s_strong_head_v0")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
