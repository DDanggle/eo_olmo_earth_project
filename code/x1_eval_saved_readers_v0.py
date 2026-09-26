#!/usr/bin/env python3
"""X1: evaluate already-trained landslide readers on the untouched Sen12 region italy (no retraining).

Prereg: config/x1_italy_external_prereg_v0.json. Models are loaded from their saved state dicts:
  e2_multi_reader_v0/{reader,blind}_seed{1,2,3}/projector.pt   (MS-157)
  e7_multi_reader_v0/{reader,blind}_seed{1,2,3}/projector.pt, head_seed{1,2,3}/head.pt   (E7)
Items: x1_italy_items_v0 (963 tiles x pos/neg). Embeddings: x1_italy_stream/emb_time_fp16 (same extractor
and settings as olmo_streaming_dev/single_fp16). Prompt and token construction are identical to E2/E7 landslide.
Arms: reader/head real + swap_within_tile (+ zero_embedding for seed 1); blind real.

  python3 -B code/x1_eval_saved_readers_v0.py --selftest
  CUDA_VISIBLE_DEVICES=1 python -B code/x1_eval_saved_readers_v0.py --models e2 --out x1_italy_eval_e2
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import balanced_acc, bootstrap, d_stat, parse, within_tile_pairs  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
ITEMS = ROOT / "x1_italy_items_v0/items.jsonl"
EMB = ROOT / "x1_italy_stream/emb_time_fp16"
MODEL_SETS = {
    "e2": [("e2_multi_reader_v0", arm, s) for arm in ("reader", "blind") for s in (1, 2, 3)],
    "e7": [("e7_multi_reader_v0", arm, s) for arm in ("reader", "blind", "head") for s in (1, 2, 3)],
}


def plan(items, with_zero):
    pairs = within_tile_pairs(items)
    paired = [it for t in sorted(pairs) for it in pairs[t]]
    jobs = {"real": [(it, it, it["answer"]) for it in paired], "swap_within_tile": []}
    for pos, neg in (pairs[t] for t in sorted(pairs)):
        jobs["swap_within_tile"] += [(pos, neg, "no"), (neg, pos, "yes")]
    if with_zero:
        jobs["zero_embedding"] = [(it, None, None) for it in paired]
    return jobs


def score(rows_by_arm):
    real = rows_by_arm["real"]
    pos = [r for r in real if r["text_gold"] == "yes"]
    neg = [r for r in real if r["text_gold"] == "no"]
    out = {"n_real": len(real), "balanced_acc": balanced_acc(real, "text_gold"),
           "recall": sum(r["parsed"] == "yes" for r in pos) / len(pos), "fpr": sum(r["parsed"] == "yes" for r in neg) / len(neg),
           "p_yes": sum(r["parsed"] == "yes" for r in real) / len(real),
           "parse_fail": {a: sum(r["parsed"] is None for r in v) / len(v) for a, v in rows_by_arm.items()},
           "d_real": {"value": d_stat(real, "emb"), "ci95": bootstrap(real, "emb")}}
    if "swap_within_tile" in rows_by_arm:
        sw = rows_by_arm["swap_within_tile"]
        out["d_swap"] = {"value": d_stat(sw, "emb"), "ci95": bootstrap(sw, "emb")}
    if "zero_embedding" in rows_by_arm:
        out["p_yes_zero"] = sum(r["parsed"] == "yes" for r in rows_by_arm["zero_embedding"]) / len(rows_by_arm["zero_embedding"])
    return out


def verdict(scores, source):
    """Primary: per seed, reader d_swap >= .10 with CI lower > 0 AND reader BA - blind BA >= .05; transfers if >= 2/3 seeds."""
    for s in scores.values():
        if max(s["parse_fail"].values()) > 0.10:
            return {"verdict": "invalid_run"}
    ok = []
    for seed in (1, 2, 3):
        r, b = scores.get(f"{source}/reader_seed{seed}"), scores.get(f"{source}/blind_seed{seed}")
        if r is None or b is None:
            return {"verdict": "incomplete"}
        ok.append(r["d_swap"]["value"] >= 0.10 and r["d_swap"]["ci95"][0] > 0 and r["balanced_acc"] - b["balanced_acc"] >= 0.05)
    return {"verdict": "transfers" if sum(ok) >= 2 else "does_not_transfer", "seeds_passing": sum(ok)}


def selftest():
    items = []
    for t in range(50):
        base = {"tile": f"t{t}", "fold": "holdout_italy", "type": "Q1", "dates": ["2019-01-01", "2019-03-01"]}
        items += [{**base, "id": f"t{t}_pos", "answer": "yes"}, {**base, "id": f"t{t}_neg", "answer": "no"}]
    jobs = plan(items, True)
    assert len(jobs["real"]) == 100 and len(jobs["swap_within_tile"]) == 100 and len(jobs["zero_embedding"]) == 100

    def rows(fn, arms):
        return {a: [{"id": s["id"], "tile": s["tile"], "text_gold": s["answer"], "emb_gold": eg, "parsed": fn(s, eg)}
                    for s, e, eg in jobs[a]] for a in arms}

    good = score(rows(lambda s, eg: eg if eg else "no", ("real", "swap_within_tile", "zero_embedding")))
    blind = score(rows(lambda s, eg: "no", ("real",)))
    sc = {f"e7/{a}_seed{k}": v for k in (1, 2, 3) for a, v in (("reader", good), ("blind", blind))}
    assert verdict(sc, "e7")["verdict"] == "transfers"
    texter = score(rows(lambda s, eg: s["answer"], ("real", "swap_within_tile")))
    sc = {f"e7/{a}_seed{k}": v for k in (1, 2, 3) for a, v in (("reader", texter), ("blind", blind))}
    assert verdict(sc, "e7")["verdict"] == "does_not_transfer"
    print("selftest ok: plan, score, transfer verdict branches")


def run(a):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_root = ROOT / a.out
    out_root.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    rec = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ROOT / "sen12_gp_contract/sample_contract.jsonl") if l.strip()}
    items = [x for x in (json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()) if (EMB / f"{x['tile']}.npy").exists()]

    def kept_dates(tile):
        r = rec[tile]
        q = r["scl_clear_fraction"]
        k = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        return [str(r["times"][i])[:10] for i in k]

    llm_dir = str(ROOT / "olmo_llm/Olmo-3-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(llm_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_dir, dtype=torch.bfloat16).to(dev).eval()
    E = llm.get_input_embeddings()
    H = E.weight.shape[1]
    emb_rms = float(E.weight.detach().float().pow(2).mean().sqrt())

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

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.LayerNorm(3 * 768 + 3), nn.Linear(3 * 768 + 3, 256), nn.GELU(), nn.Linear(256, 1))

        def forward(s, x):
            return s.net(x).squeeze(-1)

    cache = {}

    def slots(it):
        key = (it["tile"], tuple(it["dates"]))
        if key not in cache:
            S = np.load(EMB / f"{it['tile']}.npy")
            kd = kept_dates(it["tile"])
            T = F.avg_pool2d(torch.from_numpy(S[[kd.index(d) for d in it["dates"]]].astype("float32")), 4)
            cache[key] = T.flatten(2).permute(0, 2, 1)
        return cache[key]

    def tokens(it, zero):
        T = slots(it)
        t = torch.cat([T[0], T[1], T[1] - T[0]]).to(dev)
        ty = torch.cat([torch.full((64,), 0), torch.full((64,), 1), torch.full((64,), 3)]).long().to(dev)
        return (torch.zeros_like(t) if zero else t), ty

    def gen(proj, src, emb_item, zero):
        user = (f"These are 2 Sentinel-2 observations of the same area in chronological order, taken on "
                f"{', '.join(src['dates'])}: <EO> Did a landslide occur between the two observations? Answer with yes or no.")
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        pre, post = text.split("<EO>")
        ids_pre = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        ids_post = tok(post, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        t, ty = tokens(emb_item if emb_item is not None else src, zero)
        e = torch.cat([E(ids_pre), proj(t, ty).to(torch.bfloat16), E(ids_post)])
        out = llm.generate(inputs_embeds=e[None], attention_mask=torch.ones(1, e.shape[0], dtype=torch.long, device=dev),
                           max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    def head_logit(head, src, emb_item, zero):
        T = slots(emb_item if emb_item is not None else src).mean(1)
        v = torch.cat([T[0], T[1], T[1] - T[0]])
        if zero:
            v = torch.zeros_like(v)
        q = torch.tensor([1.0, 0.0, 0.0])            # landslide question one-hot (E7 QUESTION["landslide"] = 0)
        return float(head(torch.cat([v, q])[None].to(dev)))

    scores, manifest = {}, {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "items_sha256": hashlib.sha256(ITEMS.read_bytes()).hexdigest(), "n_items": len(items),
                            "n_tiles": len({x['tile'] for x in items}), "models": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with torch.no_grad():
        for src_dir, arm, seed in MODEL_SETS[a.models]:
            d = ROOT / src_dir / f"{arm}_seed{seed}"
            wfile = d / ("head.pt" if arm == "head" else "projector.pt")
            manifest["models"].append({"name": f"{a.models}/{arm}_seed{seed}", "weights_sha256": hashlib.sha256(wfile.read_bytes()).hexdigest()})
            jobs = plan(items, with_zero=(seed == 1 and arm != "blind"))
            if arm == "blind":
                jobs = {"real": jobs["real"]}
            rows_by_arm = {}
            if arm == "head":
                model = Head().to(dev)
                model.load_state_dict(torch.load(wfile, map_location=dev))
                model.eval()
                for k, js in jobs.items():
                    rows_by_arm[k] = [{"id": s["id"], "tile": s["tile"], "text_gold": s["answer"], "emb_gold": eg,
                                       "parsed": "yes" if head_logit(model, s, e, k == "zero_embedding") > 0 else "no"} for s, e, eg in js]
            else:
                model = Projector().to(dev)
                model.load_state_dict(torch.load(wfile, map_location=dev))
                model.eval()
                for k, js in jobs.items():
                    rows = []
                    for s, e, eg in js:
                        raw = gen(model, s, e, arm == "blind" or k == "zero_embedding")
                        rows.append({"id": s["id"], "tile": s["tile"], "text_gold": s["answer"], "emb_gold": eg, "raw": raw, "parsed": parse("Q1", raw)})
                    rows_by_arm[k] = rows
            name = f"{a.models}/{arm}_seed{seed}"
            for k, v in rows_by_arm.items():
                (out_root / f"answers_{arm}_seed{seed}_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
            scores[name] = score(rows_by_arm)
            print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in scores[name].items() if k in ("balanced_acc", "recall", "fpr")}),
                  "d_swap", scores[name].get("d_swap"), flush=True)
            del model
    final = {"manifest": manifest, "scores": scores, "verdict": verdict(scores, a.models) if a.models in ("e2", "e7") else None,
             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final["verdict"]), flush=True)
    print("X1 DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", choices=sorted(MODEL_SETS), default="e7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        a.out = a.out or f"x1_italy_eval_{a.models}"
        run(a)
