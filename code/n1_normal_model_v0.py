#!/usr/bin/env python3
"""N1: learn each location's 'normal' from ~3 years of monthly OlmoEarth S1 embeddings, score the post acquisition by surprise.

Prereg: config/n1_learned_normal_prereg_v0.json (registered before download). Inputs: n1_s1/emb/<id>.npz from
code/n1_embed_s1_v0.py (emb24: (T,768,24,24) at 80 m, roles hist/pre/post).

Model (normal forecaster, per token location): the previous K=12 unit-normalised embeddings (+ day-of-year sin/cos and
gap-to-target in years) -> a query token carrying the target's day-of-year -> predicted mean (768) and log-variance (768).
Loss = Gaussian NLL. Targets and inputs are HISTORY acquisitions only (all before event-5 d). Events are split into 3 folds;
the model that scores an event was trained on the other two folds' tiles only. 3 seeds.

Scores at 80 m tokens (24x24), all on the same PC RTC chips:
  n1_surprise  mean_d (post - mu)^2 / var given history ending with pre (full method)
  raw          1-cos(post, pre)
  hand_norm    raw / (1-cos(pre, previous acquisition) + eps), eps = median over tiles (A1b rule)
  clim         1-cos(post, mean of history acquisitions in the post's calendar month)
External (A1b, KuroSiwo GRD / AlphaEarth, pooled to 80 m): aef_raw, aef_norm, sar_raw, sar_norm; reference (not a
competitor in the prereg, reported): olmo4_norm (A1b full T1 method on the KuroSiwo cache).
Tile score = mean over valid tokens; T1 = tile AUC flood vs dry+perm (primary) and vs perm; T2 = token AUC at 80 m in
flood tiles (flood fraction >= .5 positive, 0 negative). Event-clustered bootstrap 2000.
Interpretation fixed before any score: verdict rules are applied per seed and need >= 2 of 3 seeds; the 3-seed mean
surprise ("n1_surprise_ens") is reported alongside.

  python3 -B code/n1_normal_model_v0.py --selftest
  CUDA_VISIBLE_DEVICES=0 .venv-master/bin/python -B code/n1_normal_model_v0.py
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/home/work/data/olmoearth")
SRC = ROOT / "n1_s1"
KURO = ROOT / "kurosiwo_s1_cache"
AEF_TILES = ROOT / "a1_aef/tiles"
OUT = ROOT / "n1_out"
K, KMIN, NTOK_TRAIN, D = 12, 6, 64, 768
SEEDS = (0, 1, 2)
BASE = ("raw", "hand_norm", "clim", "aef_raw", "aef_norm", "sar_raw", "sar_norm")


def unit(x, axis):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def pool(x, f):
    h, w = x.shape
    return np.nanmean(x.reshape(h // f, f, w // f, f), axis=(1, 3))


def day(s):
    return date.fromisoformat(s).toordinal()


def doy_feats(ords):
    """ordinal days -> (..., 2) sin/cos of day-of-year."""
    doy = np.array([date.fromordinal(int(o)).timetuple().tm_yday for o in np.ravel(ords)], float).reshape(np.shape(ords))
    a = 2 * math.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a)], -1)


def build_model(torch, d=192, layers=3):
    nn = torch.nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(D + 3, d)
            self.q = nn.Linear(2, d)
            self.pos = nn.Parameter(torch.zeros(K + 1, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
            self.mu = nn.Linear(d, D)
            self.lv = nn.Linear(d, D)

        def forward(self, x, feats, qfeat, pad):
            """x (B,K,D) unit embeddings, feats (B,K,3) [doy sin, cos, gap years], qfeat (B,2), pad (B,K) True = missing."""
            h = torch.cat([self.inp(torch.cat([x, feats], -1)), self.q(qfeat)[:, None]], 1) + self.pos
            m = torch.cat([pad, torch.zeros_like(pad[:, :1])], 1)
            z = self.enc(h, src_key_padding_mask=m)[:, -1]
            return self.mu(z), self.lv(z).clamp(-10, 5)

    return Forecaster()


def nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / lv.exp()).mean()


def window(ords, t):
    """Indices of the K inputs before target index t (negative = padding)."""
    return np.arange(t - K, t)


def selftest():
    assert list(window(None, 3)) == list(range(-9, 3))
    x = np.ones((4, 4))
    x[0, 0] = np.nan
    assert pool(x, 2)[0, 0] == 1
    f = doy_feats(np.array([day("2020-01-01"), day("2020-07-01")]))
    assert f.shape == (2, 2) and abs(f[0, 1] - 1) < 1e-3 and f[1, 1] < -0.99
    try:
        import torch
        m = build_model(torch)
        B = 5
        mu, lv = m(torch.randn(B, K, D), torch.randn(B, K, 3), torch.randn(B, 2), torch.zeros(B, K, dtype=torch.bool))
        assert mu.shape == (B, D) and lv.shape == (B, D)
        print("params", sum(p.numel() for p in m.parameters()))
    except ImportError:
        print("(torch not available locally; model shape check skipped)")
    print("selftest ok")


def load_tiles():
    from a1_aef_vs_olmoearth_floods_v0 import cos_change, dequant
    tiles = {t["id"]: t for t in json.loads((ROOT / "a1b/tiles.json").read_text())}
    per = []
    for f in sorted((SRC / "emb").glob("*.npz")):
        t = tiles.get(f.stem)
        z = np.load(f)
        roles, dates = list(z["roles"]), list(z["dates"])
        if t is None or "pre" not in roles or "post" not in roles:
            continue
        ipre, ipost = roles.index("pre"), roles.index("post")
        hist = [i for i, r in enumerate(roles) if r == "hist" and dates[i] != dates[ipre]]
        if len(hist) < KMIN:
            continue
        E = unit(z["emb24"].astype(np.float32), 1)                    # (T,768,24,24)
        seq = hist + [ipre]
        ords = np.array([day(dates[i]) for i in seq])
        post_ord = day(dates[ipost])
        # validity: chips not no-data (-30 dB fill) at pre and post, KuroSiwo valid mask
        ok = np.ones((24, 24), bool)
        for i in (ipre, ipost):
            c = np.load(SRC / "chips" / f"{f.stem}_{dates[i]}.npy").astype(np.float32)
            ok &= pool((c[0] > -29.9).astype(np.float32), 8) > 0.5
        m = np.load(KURO / "mask_u8" / f"{f.stem}.npy")
        v = (np.load(KURO / "valid_u8" / f"{f.stem}.npy") == 1) & (m > 0)
        vv = (pool(v.astype(np.float32), 8) > 0.5) & ok
        ff = pool(((m == 3) & v).astype(np.float32), 8)
        pre, post, prev = E[ipre], E[ipost], E[hist[-1]]
        pm = date.fromordinal(post_ord).month
        same = [i for i in hist if int(dates[i][5:7]) == pm]
        ext = {}
        a = [AEF_TILES / f"{f.stem}_{t['year'] + d}.npy" for d in (-2, -1, 0)]
        if all(x.exists() for x in a):
            e2, e1, e0 = (dequant(np.load(x)) for x in a)
            ext["aef_post"], ext["aef_pre"] = pool(cos_change(e0, e1), 8), pool(cos_change(e1, e2), 8)
        raw = np.load(ROOT / "kurosiwo_npy/raw_f32" / f"{f.stem}.npy")[:, :, 16:208, 16:208].astype(np.float64)
        db = 10 * np.log10(np.clip(raw, 1e-6, None))
        db[raw == 0] = np.nan
        ext["sar_post"] = pool(np.nanmean(db[:, 1] - db[:, 2], 0), 8)
        ext["sar_pre"] = pool(np.nanmean(np.abs(db[:, 0] - db[:, 1]), 0), 8)
        S = np.load(KURO / "single_fp16" / f"{f.stem}.npy").astype(np.float32)
        ext["o4_post"], ext["o4_pre"] = pool(cos_change(S[2], S[1]), 2), pool(cos_change(S[1], S[0]), 2)
        per.append({**t, "E": E[seq].astype(np.float16), "ords": ords, "post": post, "post_ord": post_ord,
                    "raw": 1 - (post * pre).sum(0), "d_pre": 1 - (pre * prev).sum(0),
                    "clim": (1 - (post * unit(E[same].mean(0), 0)).sum(0)) if same else np.full((24, 24), np.nan),
                    "vv": vv, "ff": ff, **ext})
    return per


def train_fold(torch, train, seed, dev, steps, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # training tensor: per tile NTOK_TRAIN random token locations, history only (drop the trailing 'pre')
    Tm = max(len(p["ords"]) - 1 for p in train)
    X = np.zeros((len(train), Tm, NTOK_TRAIN, D), np.float16)
    O = np.zeros((len(train), Tm))
    L = np.zeros(len(train), int)
    for j, p in enumerate(train):
        h = len(p["ords"]) - 1
        loc = rng.choice(576, NTOK_TRAIN, replace=False)
        X[j, :h] = p["E"][:h].reshape(h, D, 576)[:, :, loc].transpose(0, 2, 1)
        O[j, :h], L[j] = p["ords"][:h], h
    X = torch.from_numpy(X).to(dev)
    pairs = np.array([(j, t) for j in range(len(train)) for t in range(KMIN, L[j])])
    DOY = doy_feats(np.where(O > 0, O, day("2000-01-01")))
    model = build_model(torch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    B = 1024
    for s in range(steps):
        pj = pairs[rng.integers(len(pairs), size=B)]
        j, t = pj[:, 0], pj[:, 1]
        idx = t[:, None] + np.arange(-K, 0)[None]
        pad = idx < 0
        idc = np.clip(idx, 0, None)
        tok = rng.integers(NTOK_TRAIN, size=B)
        tj = torch.from_numpy(j).to(dev)
        x = X[tj[:, None], torch.from_numpy(idc).to(dev), torch.from_numpy(tok).to(dev)[:, None]].float()
        y = X[tj, torch.from_numpy(t).to(dev), torch.from_numpy(tok).to(dev)].float()
        gap = (O[j[:, None], idc] - O[j, t][:, None]) / 365.25
        feats = np.concatenate([DOY[j[:, None], idc], gap[..., None]], -1)
        feats[pad] = 0
        f = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
        padt = torch.from_numpy(pad).to(dev)
        x = x.masked_fill(padt[..., None], 0)
        mu, lv = model(x, f(feats), f(DOY[j, t]), padt)
        loss = nll(mu, lv, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 500 == 0 or s == steps - 1:
            log.append((s, float(loss)))
    return model.eval()


def surprise(torch, model, p, dev):
    ords, E = p["ords"][-K:], p["E"][-K:].astype(np.float32)
    n = len(ords)
    x = E.reshape(n, D, 576).transpose(2, 0, 1)                           # (576, n, D)
    pad = np.zeros((576, K), bool)
    if n < K:
        x = np.concatenate([np.zeros((576, K - n, D), np.float32), x], 1)
        pad[:, :K - n] = True
        ords = np.concatenate([np.full(K - n, ords[0]), ords])
    feats = np.concatenate([doy_feats(ords), ((ords - p["post_ord"]) / 365.25)[:, None]], -1)
    feats = np.broadcast_to(feats, (576, K, 3)).copy()
    feats[pad] = 0
    q = np.broadcast_to(doy_feats(np.array([p["post_ord"]]))[0], (576, 2))
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)
    with torch.no_grad():
        mu, lv = model(f(x), f(feats), f(q), torch.from_numpy(pad).to(dev))
    y = f(p["post"].reshape(D, 576).T)
    return (((y - mu) ** 2 / lv.exp()).mean(1)).cpu().numpy().reshape(24, 24)


def run(a):
    import torch
    from a1_aef_vs_olmoearth_floods_v0 import auc, event_bootstrap
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    per = load_tiles()
    evs = sorted({p["event"] for p in per})
    rs = random.Random(20260927)
    rs.shuffle(evs)
    fold = {e: i % 3 for i, e in enumerate(evs)}
    logs = {}
    for seed in SEEDS:
        for k in range(3):
            model = train_fold(torch, [p for p in per if fold[p["event"]] != k], seed, dev, a.steps, logs.setdefault(f"s{seed}_f{k}", []))
            for p in per:
                if fold[p["event"]] == k:
                    p[f"n1_s{seed}"] = surprise(torch, model, p, dev)
            print(f"seed {seed} fold {k} done {time.perf_counter() - t0:.0f}s loss {logs[f's{seed}_f{k}'][-1]}", flush=True)
    eps = {k: float(np.nanmedian(np.concatenate([p[pre].ravel() for p in per if pre in p])))
           for k, pre in (("hand", "d_pre"), ("aef", "aef_pre"), ("sar", "sar_pre"), ("o4", "o4_pre"))}
    for p in per:
        p["hand_norm"] = p["raw"] / (p["d_pre"] + eps["hand"])
        p["n1_surprise_ens"] = np.mean([p[f"n1_s{s}"] for s in SEEDS], 0)
        if "aef_post" in p:
            p["aef_raw"], p["aef_norm"] = p["aef_post"], p["aef_post"] / (p["aef_pre"] + eps["aef"])
        p["sar_raw"], p["sar_norm"] = p["sar_post"], p["sar_post"] / (p["sar_pre"] + eps["sar"])
        p["olmo4_norm"] = p["o4_post"] / (p["o4_pre"] + eps["o4"])
    # common tile set: every method defined (AEF needs Y-2..Y)
    per = [p for p in per if "aef_raw" in p and p["vv"].any()]
    by_ev = defaultdict(list)
    for p in per:
        by_ev[p["event"]].append(p)
    METHODS = [f"n1_s{s}" for s in SEEDS] + ["n1_surprise_ens"] + list(BASE) + ["olmo4_norm"]

    def ts(p, k):
        x = p[k]
        sel = np.isfinite(x) & p["vv"]
        return float(np.mean(x[sel])) if sel.any() else None

    def T1(ps, k, negs):
        pos = [ts(p, k) for p in ps if p["kind"] == "flood"]
        neg = [ts(p, k) for p in ps if p["kind"] in negs]
        return auc([x for x in pos if x is not None], [x for x in neg if x is not None])

    def T2(ps, k):
        pos, neg = [], []
        for p in ps:
            if p["kind"] == "flood":
                x = p[k]
                sel = np.isfinite(x) & p["vv"]
                pos += list(x[sel & (p["ff"] >= .5)])
                neg += list(x[sel & (p["ff"] == 0)])
        return auc(pos, neg)

    flat = lambda gs: [p for g in gs for p in g]
    metrics = {"T1_flood_vs_dry_perm": lambda ps, k: T1(ps, k, ("dry", "perm")), "T1_flood_vs_perm": lambda ps, k: T1(ps, k, ("perm",)),
               "T2_token_80m": T2}
    res = {"n_tiles": len(per), "kinds": dict(Counter(p["kind"] for p in per)), "n_events": len(by_ev), "eps": eps,
           "median_history": float(np.median([len(p["ords"]) - 1 for p in per])), "train_logs": logs, "steps": a.steps}
    for name, fn in metrics.items():
        r = {k: fn(per, k) for k in METHODS}
        r["strongest_hand_or_clim"] = max(("hand_norm", "clim"), key=lambda k: r[k] if r[k] is not None else -1)
        r["strongest_all"] = max(BASE, key=lambda k: r[k] if r[k] is not None else -1)
        for s in [f"n1_s{s}" for s in SEEDS] + ["n1_surprise_ens"]:
            for tag, b in (("vs_hand_or_clim", r["strongest_hand_or_clim"]), ("vs_strongest", r["strongest_all"])):
                diff = lambda gs, f=s, b=b: (lambda x, y: None if x is None or y is None else x - y)(fn(flat(gs), f), fn(flat(gs), b))
                r[f"{s}_{tag}"] = {"diff": r[s] - r[b], "ci95": event_bootstrap(by_ev, diff)}
        r["per_event"] = {str(e): {k: fn(ps, k) for k in ("n1_surprise_ens", "raw", "hand_norm", "clim", "sar_raw", "aef_raw")} for e, ps in by_ev.items()}
        res[name] = r
    ok = lambda d: d["diff"] >= .05 and d["ci95"] is not None and d["ci95"][0] > 0
    T1n, T2n = "T1_flood_vs_dry_perm", "T2_token_80m"
    helps = sum(ok(res[T1n][f"n1_s{s}_vs_hand_or_clim"]) for s in SEEDS)
    strong = sum(ok(res[T1n][f"n1_s{s}_vs_strongest"]) and ok(res[T2n][f"n1_s{s}_vs_strongest"]) for s in SEEDS)
    res["verdict"] = {"learned_normal_helps_seeds": helps, "meaningful_vs_strongest_seeds": strong,
                      "verdict": "meaningful_vs_strongest" if strong >= 2 else "learned_normal_helps" if helps >= 2 else "no_gain",
                      "invalid": len(by_ev) < 8}
    res["seconds"] = round(time.perf_counter() - t0)
    (out / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n_tiles", "kinds", "n_events", "verdict")}, default=float))
    for name in metrics:
        print(name, {k: round(res[name][k], 3) for k in METHODS if res[name][k] is not None})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=str(OUT), help="smoke tests write elsewhere and are not looked at")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
