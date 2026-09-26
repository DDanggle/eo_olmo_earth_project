#!/usr/bin/env python3
"""Feasibility count for the problem-driven SAR flood task: how many KuroSiwo tiles have much PERMANENT water
but no flood (post-image water that is NOT new flooding), vs tiles with flood, per split and event.

Mask (code/extract_kurosiwo_s1_cache.py): 0 no-data, 1 no water, 2 permanent water, 3 flood; valid_u8 1 = valid.
CPU only, reads masks.   python3 -B code/audit_kuro_perm_water_v0.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

C = Path("/home/work/data/olmoearth/kurosiwo_s1_cache")
meta = [json.loads(l) for l in (C / "meta.jsonl").read_text().splitlines() if l.strip()]
cnt, events = Counter(), defaultdict(set)
for r in meta:
    m = np.load(C / "mask_u8" / f"{r['id']}.npy")
    v = (np.load(C / "valid_u8" / f"{r['id']}.npy") == 1) & (m > 0)
    n = max(int(v.sum()), 1)
    perm, flood = float(((m == 2) & v).sum()) / n, float(((m == 3) & v).sum()) / n
    if flood >= 0.02 and perm >= 0.05:
        k = "flood+perm"
    elif flood >= 0.02:
        k = "flood_only"
    elif flood == 0 and perm >= 0.05:
        k = "perm_only(water,no flood)"
    elif flood == 0 and perm == 0:
        k = "dry"
    else:
        k = "other"
    cnt[(r["split"], k)] += 1
    events[(r["split"], k)].add(r["actid"])
out = {f"{s}|{k}": {"tiles": c, "events": len(events[(s, k)])} for (s, k), c in sorted(cnt.items())}
print(json.dumps(out, indent=1))
