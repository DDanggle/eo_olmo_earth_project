#!/usr/bin/env python3
"""Decision for trust_aware_reader_prereg_v0: arm vs C_lora (MS-143) per unseen event, tile-paired bootstrap CI on Q_when accuracy and cloud_stability.
signal_helps = arm beats C_lora by >= .03 on BOTH metrics with 95% CI excluding 0 on >= 2 of 3 unseen events."""
import json, sys, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); REF=ROOT/"vlm_memory_stage2_v0"; NEW=ROOT/"vlm_trust_reader_v0"; ARMS=sys.argv[1].split(",") if len(sys.argv)>1 else ["E_trust_z","E_z_only","E_trust_only"]
EV=["thrissur","itogon","hokkaido"]; rng=np.random.default_rng(20260920); NB=10000
def load(p): return [json.loads(l) for l in p.read_text().splitlines() if l]
def per_tile(rows):
    st={(r["tile"],r["step"]):r["pred"] for r in rows if r["q"]=="Q_state"}; W={}; C={}
    for r in rows:
        if r["q"]=="Q_when": W.setdefault(r["tile"],[]).append(r["pred"]==r["gold"])
        if r["q"]=="Q_update" and r["gold"] in ("keep","abstain") and r["step"]>0: C.setdefault(r["tile"],[]).append(st.get((r["tile"],r["step"]))==st.get((r["tile"],r["step"]-1)))
    return {t:np.mean(v) for t,v in W.items()},{t:np.mean(v) for t,v in C.items()}
def paired(a,b):
    ts=sorted(set(a)&set(b)); x=np.array([a[t] for t in ts]); y=np.array([b[t] for t in ts]); d=x-y
    bs=d[rng.integers(0,len(d),(NB,len(d)))].mean(1); return {"n_tiles":len(ts),"arm":float(x.mean()),"ref":float(y.mean()),"diff":float(d.mean()),"ci95":[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]}
out={"schema":"trust-reader-decision-v0","ref":"C_lora (vlm_memory_stage2_v0)","rule":"diff>=.03 on Q_when AND cloud_stability, CI excl 0, >=2/3 unseen events","arms":{}}
for arm in ARMS:
    res={"events":{},"wins":0,"available":0}
    for ev in EV:
        pa=NEW/f"answers_{arm}_{ev}.jsonl"; pr=REF/f"answers_C_lora_{ev}.jsonl"
        if not (pa.exists() and pr.exists()): res["events"][ev]="pending"; continue
        Wa,Ca=per_tile(load(pa)); Wr,Cr=per_tile(load(pr)); w=paired(Wa,Wr); c=paired(Ca,Cr); win=w["diff"]>=.03 and w["ci95"][0]>0 and c["diff"]>=.03 and c["ci95"][0]>0
        res["events"][ev]={"Q_when":w,"cloud_stability":c,"win":bool(win)}; res["available"]+=1; res["wins"]+=int(win)
    res["signal_helps"]=bool(res["available"]==3 and res["wins"]>=2); res["decided"]=res["available"]==3 or (3-res["available"]+res["wins"])<2
    out["arms"][arm]=res; print(arm,json.dumps(res,default=float)[:1500],flush=True)
(NEW/"decision_v0.json").write_text(json.dumps(out,indent=1,default=float)); print("DECISION DONE")
