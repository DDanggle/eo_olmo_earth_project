#!/usr/bin/env python3
"""A2b: few-shot landslide mapping in a new region - OlmoEarth per-acquisition embeddings vs AlphaEarth vs raw S2, same head.

Prereg: config/a2b_fewshot_landslide_prereg_v0.json. Tiles/dates from A2 (a2/tiles.json), restricted to tiles with AlphaEarth
Y and Y-1. For each target region: pre-train the head on the 7 other regions, then adapt on K positive + K negative target
tiles (10 draws, same draws for every feature set), score the remaining target tiles.
  T1 tile AUC (tile score = 95th percentile of token probabilities), T2 token AUC inside positive query tiles.

  python3 -B code/a2b_fewshot_landslide_v0.py --selftest
  CUDA_VISIBLE_DEVICES=1 .venv-master/bin/python -B code/a2b_fewshot_landslide_v0.py
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_aef_vs_olmoearth_floods_v0 import auc, cos_change, dequant, pool4  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
OUT = ROOT / "a2b"
AEF_TILES = ROOT / "a1_aef/tiles"
SRC = {"italy": (ROOT / "x1_italy_stream/emb_time_fp16", ROOT / "sen12_pilot/holdout_italy")}
DEFAULT_SRC = (ROOT / "olmo_streaming_dev/single_fp16", ROOT / "olmo_streaming_dev")
PRIMARY = ("hiroshima", "hokkaido", "thrissur", "chimanimani")
SECONDARY = ("itogon",)
FEATS = ("olmo", "aef", "s2raw", "olmo_post", "olmo_aef")
BASELINES = ("aef", "s2raw")
KS, DRAWS, SEEDS = (0, 5, 20), 10, (0, 1, 2)
B04, B08 = 2, 6


def p95(prob):
    return float(np.quantile(prob, .95))


def draw_support(ids_pos, ids_neg, k, region, d):
    """Deterministic support: same for every feature set and seed."""
    h = lambda s: hashlib.sha256(f"a2b|{region}|{k}|{d}|{s}".encode()).hexdigest()
    return sorted(ids_pos, key=h)[:k], sorted(ids_neg, key=h)[:k]


def token_labels(mask_frac, kind):
    y = np.full(mask_frac.shape, -1, np.int8)
    y[mask_frac == 0] = 0
    if kind == "pos":
        y[mask_frac >= .5] = 1
    return y


def selftest():
    a, b = draw_support(list("abcdefg"), list("hijklmn"), 3, "r", 0)
    assert len(a) == 3 and len(b) == 3 and draw_support(list("abcdefg"), list("hijklmn"), 3, "r", 0) == (a, b)
    assert draw_support(list("abcdefg"), list("hijklmn"), 3, "r", 1) != (a, b)
    y = token_labels(np.array([[0, .3], [.5, 1]]), "pos")
    assert y.tolist() == [[0, -1], [1, 1]]
    assert token_labels(np.array([[0, .3]]), "neg").tolist() == [[0, -1]]
    print("selftest ok: deterministic paired support draws, token labels")


def load():
    tiles = json.loads((ROOT / "a2/tiles.json").read_text())
    X = defaultdict(list)
    meta, Y, LF = [], [], []
    for t in tiles:
        fY, fB = (AEF_TILES / f"{t['id']}_{t['year'] + d}.npy" for d in (0, -1))
        if not (fY.exists() and fB.exists()):
            continue
        emb_dir, src = SRC.get(t["region"], DEFAULT_SRC)
        p1, p0, q = t["idx"]
        S = np.load(emb_dir / f"{t['id']}.npy").astype(np.float32)
        pre, post = S[p0], S[q]
        eY, eB = (np.nan_to_num(pool4c(dequant(np.load(f)))) for f in (fY, fB))
        raw = np.load(src / "raw_u16" / f"{t['id']}.npy").astype(np.float32) / 10000.0
        rp, rq = raw[:, p0], raw[:, q]
        nd = lambda r: (r[B08] - r[B04]) / np.where(r[B08] + r[B04] > 0, r[B08] + r[B04], np.nan)
        s2 = np.concatenate([rp, rq, rq - rp, np.stack([nd(rp), nd(rq), nd(rq) - nd(rp)])])
        s2 = np.nan_to_num(pool4c(s2))
        m = np.load(src / "mask_u8" / f"{t['id']}.npy") > 0
        X["olmo"].append(np.concatenate([pre, post, post - pre]).astype(np.float16))
        X["olmo_post"].append(post.astype(np.float16))
        X["aef"].append(np.concatenate([eB, eY, eY - eB]).astype(np.float16))
        X["s2raw"].append(s2.astype(np.float16))
        Y.append(token_labels(pool4(m.astype(np.float32)), t["kind"]))
        LF.append({"olmo_raw": cos_change(post, pre), "aef_raw": np.nan_to_num(cos_change(eY, eB)),
                   "dndvi_raw": np.nan_to_num(pool4(nd(rp) - nd(rq)))})
        meta.append({"id": t["id"], "region": t["region"], "kind": t["kind"]})
    X = {k: np.stack(v) for k, v in X.items()}
    X["olmo_aef"] = np.concatenate([X["olmo"], X["aef"]], 1)
    return X, np.stack(Y), meta, LF


def pool4c(x):
    c, h, w = x.shape
    return np.nanmean(x.reshape(c, h // 4, 4, w // 4, 4), axis=(2, 4))


def run(a):
    import torch
    nn = torch.nn
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    X, Y, meta, LF = load()
    print("loaded", {k: v.shape for k, v in X.items()}, f"{time.perf_counter() - t0:.0f}s", flush=True)
    reg = np.array([m["region"] for m in meta])
    kind = np.array([m["kind"] for m in meta])
    Yt = torch.from_numpy(Y.astype(np.int64)).to(dev)

    def head(c):
        return nn.Sequential(nn.Conv2d(c, 128, 1), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 1, 1)).to(dev)

    def loss_fn(logit, y, pw):
        m = y >= 0
        return nn.functional.binary_cross_entropy_with_logits(logit[m], y[m].float(), pos_weight=torch.tensor(pw, device=dev))

    def fit(model, Xg, idx, steps, lr, seed, pw):
        g = torch.Generator(device="cpu").manual_seed(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        idx = torch.as_tensor(idx)
        for _ in range(steps):
            b = idx[torch.randint(len(idx), (min(32, len(idx)),), generator=g)].to(dev)
            loss = loss_fn(model(Xg[b].float())[:, 0], Yt[b], pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def predict(model, Xg, idx):
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(Xg[torch.as_tensor(idx[i:i + 64]).to(dev)].float())[:, 0]) for i in range(0, len(idx), 64)]).cpu().numpy()

    def metrics(prob, idx):
        k = kind[idx]
        t1 = auc([p95(p) for p, kk in zip(prob, k) if kk == "pos"], [p95(p) for p, kk in zip(prob, k) if kk == "neg"])
        pos, neg = [], []
        for p, i in zip(prob, idx):
            if kind[i] == "pos":
                pos += list(p[Y[i] == 1])
                neg += list(p[Y[i] == 0])
        return t1, auc(pos, neg)

    rows = []
    targets = PRIMARY + SECONDARY
    for f in FEATS:
        Xg = torch.from_numpy(X[f]).to(dev)
        for tg in targets:
            src_idx = np.flatnonzero(reg != tg)
            mu = Xg[torch.as_tensor(src_idx).to(dev)].float().mean((0, 2, 3), keepdim=True)
            sd = Xg[torch.as_tensor(src_idx).to(dev)].float().std((0, 2, 3), keepdim=True).clamp(min=1e-4)
            Xn = ((Xg.float() - mu) / sd).half()
            ys = Y[src_idx]
            pw = float(min((ys == 0).sum() / max((ys == 1).sum(), 1), 10))
            tpos = [i for i in np.flatnonzero(reg == tg) if kind[i] == "pos"]
            tneg = [i for i in np.flatnonzero(reg == tg) if kind[i] == "neg"]
            for seed in SEEDS:
                torch.manual_seed(seed)
                base = fit(head(Xn.shape[1]), Xn, src_idx, 1500, 1e-3, seed, pw)
                state = {k: v.clone() for k, v in base.state_dict().items()}
                for K in KS:
                    if K and (len(tpos) < K + 3 or len(tneg) < K + 3) or K == 20 and len(tpos) < 30:
                        continue
                    for d in (range(DRAWS) if K else [0]):
                        sp, sn = draw_support(tpos, tneg, K, tg, d)
                        sup = sp + sn
                        ss = set(sup)
                        q = np.array([i for i in np.flatnonzero(reg == tg) if i not in ss])
                        m = head(Xn.shape[1])
                        m.load_state_dict(state)
                        if K:
                            m = fit(m, Xn, np.array(sup), 100, 3e-4, 1000 * seed + d, pw)
                        t1, t2 = metrics(predict(m.eval(), Xn, q), q)
                        rows.append({"feat": f, "region": tg, "seed": seed, "K": K, "draw": d, "T1": t1, "T2": t2, "n_query": len(q)})
            print(f, tg, f"{time.perf_counter() - t0:.0f}s", flush=True)
            del Xn
        del Xg
        torch.cuda.empty_cache()

    # label-free reference on all target tiles (context only)
    lf = {}
    for tg in targets:
        idx = np.flatnonzero(reg == tg)
        lf[tg] = {k: metrics([LF[i][k] for i in idx], idx) for k in ("olmo_raw", "aef_raw", "dndvi_raw")}

    agg = defaultdict(list)
    for r in rows:
        agg[(r["feat"], r["region"], r["K"])].append(r)
    mean = lambda f, tg, K, mtr: (lambda v: float(np.mean(v)) if v else None)([r[mtr] for r in agg[(f, tg, K)] if r[mtr] is not None])
    res = {"n_tiles": len(meta), "per_region_tiles": {tg: {"pos": int(((reg == tg) & (kind == "pos")).sum()), "neg": int(((reg == tg) & (kind == "neg")).sum())} for tg in targets},
           "label_free_reference": lf, "table": {}, "verdicts": {}}
    for K in KS:
        tab = {mtr: {f: {tg: mean(f, tg, K, mtr) for tg in targets} for f in FEATS} for mtr in ("T1", "T2")}
        res["table"][f"K{K}"] = tab
        ok = {}
        for mtr in ("T1", "T2"):
            macro = {f: (lambda v: float(np.mean(v)) if len(v) == len(PRIMARY) else None)([tab[mtr][f][tg] for tg in PRIMARY if tab[mtr][f][tg] is not None]) for f in FEATS}
            best = max(BASELINES, key=lambda b: macro[b] if macro[b] is not None else -1)
            if macro["olmo"] is None or macro[best] is None:
                ok[mtr] = {"macro": macro, "valid": False}
                continue
            wins = sum(tab[mtr]["olmo"][tg] > tab[mtr][best][tg] for tg in PRIMARY)
            draw_win = []
            for tg in PRIMARY:
                o = {(r["seed"], r["draw"]): r[mtr] for r in agg[("olmo", tg, K)]}
                b = {(r["seed"], r["draw"]): r[mtr] for r in agg[(best, tg, K)]}
                draw_win += [o[k] > b[k] for k in o if k in b and o[k] is not None and b[k] is not None]
            ok[mtr] = {"macro": macro, "strongest": best, "diff": macro["olmo"] - macro[best], "region_wins": wins,
                       "draw_win_fraction": float(np.mean(draw_win)) if draw_win else None,
                       "pass": macro["olmo"] - macro[best] >= .05 and wins >= 3}
        v = ("meaningful" if all(ok[m].get("pass") for m in ok) else "t1_only" if ok["T1"].get("pass") else
             "t2_only" if ok["T2"].get("pass") else "not_meaningful")
        res["verdicts"][f"K{K}"] = {"verdict": v, **ok}
    res["rows"] = rows
    res["seconds"] = round(time.perf_counter() - t0)
    res["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"verdicts": {k: v["verdict"] for k, v in res["verdicts"].items()}, "seconds": res["seconds"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
