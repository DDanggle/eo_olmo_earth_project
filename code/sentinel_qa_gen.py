#!/usr/bin/env python3
"""Sentinel temporal QA v0 generator (config/sentinel_temporal_qa_prereg_v0.json). Sen12 only in v0. Writes items.jsonl + RGB PNGs."""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image
ROOT=Path("/home/work/data/olmoearth"); import sys as _s; OUT=ROOT/(_s.argv[1] if len(_s.argv)>1 else "sentinel_qa_v0"); (OUT/"img").mkdir(parents=True,exist_ok=True)
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
rng=np.random.default_rng(0); PER=int(_s.argv[3]) if len(_s.argv)>3 else 400
REGIONS=_s.argv[2].split(",") if len(_s.argv)>2 else None; NOPNG=("--nopng" in _s.argv)
def kept12(r): q=r["scl_clear_fraction"]; return sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12])
def rgb_png(raw,t,path):
    if NOPNG: return str(path)
    if not path.exists():
        x=np.stack([raw[2,t],raw[1,t],raw[0,t]],-1); x=np.clip(x/3000.0,0,1); Image.fromarray((x*255).astype("uint8")).resize((512,512),Image.BICUBIC).save(path)
    return str(path)
def quadrants(m):
    h,w=m.shape; tot=m.sum(); out=[]
    for name,sl in (("NW",(slice(0,h//2),slice(0,w//2))),("NE",(slice(0,h//2),slice(w//2,w))),("SW",(slice(h//2,h),slice(0,w//2))),("SE",(slice(h//2,h),slice(w//2,w)))):
        if tot and m[sl].sum()/tot>=0.25: out.append(name)
    return out
items=[]
FOLDS=[f"holdout_{r}" for r in REGIONS] if REGIONS else ["holdout_hiroshima","holdout_indonesia"]
for fold in FOLDS:
    SRC=ROOT/"sen12_pilot"/fold; region=fold.replace("holdout_","")
    if not (SRC/"mask_u8").exists(): print("skip missing fold",fold); continue
    ids=[s for s in sorted(p.stem for p in (SRC/"mask_u8").glob("*.npy")) if REC[s]["region"]==region and REC[s].get("post_index") is not None]
    pos=[s for s in ids if np.load(SRC/"mask_u8"/f"{s}.npy").sum()>=200]; neg=[s for s in ids if np.load(SRC/"mask_u8"/f"{s}.npy").sum()==0]
    rng.shuffle(pos); rng.shuffle(neg)
    def slots(s):
        r=REC[s]; k=kept12(r); pre=[j for j,i in enumerate(k) if i<r["post_index"]]; post=[j for j,i in enumerate(k) if i>=r["post_index"]]; return r,k,pre,post
    # Q1: positives pre/post; negatives = positive tiles' pre/pre or post/post pairs + zero-mask tiles pre/post (both count as "no event between")
    n1=0
    for s in pos:
        r,k,pre,post=slots(s)
        if len(pre)<2 or len(post)<2: continue
        raw=np.load(SRC/"raw_u16"/f"{s}.npy").astype("float32"); d=lambda j:str(r["times"][k[j]])[:10]
        from datetime import date as _date
        dd=lambda j:_date.fromisoformat(d(j)); a,b=pre[-1],post[0]; pos_gap=(dd(b)-dd(a)).days
        # gap-matched negative: a pre/pre or post/post pair whose day gap is closest to pos_gap and within +-50%
        cands=[(x,y) for grp in (pre,post) for x in grp for y in grp if y>x]; cands=[(x,y) for x,y in cands if abs((dd(y)-dd(x)).days-pos_gap)<=0.5*pos_gap]
        if not cands: continue
        x,y=min(cands,key=lambda p:abs((dd(p[1])-dd(p[0])).days-pos_gap))
        items.append({"id":f"{s}_q1_pos","fold":fold,"tile":s,"type":"Q1","dates":[d(a),d(b)],"gap_days":pos_gap,"images":[rgb_png(raw,a,OUT/"img"/f"{s}_{a}.png"),rgb_png(raw,b,OUT/"img"/f"{s}_{b}.png")],"answer":"yes"})
        items.append({"id":f"{s}_q1_neg","fold":fold,"tile":s,"type":"Q1","dates":[d(x),d(y)],"gap_days":(dd(y)-dd(x)).days,"images":[rgb_png(raw,x,OUT/"img"/f"{s}_{x}.png"),rgb_png(raw,y,OUT/"img"/f"{s}_{y}.png")],"answer":"no"})
        # Q2: 4 dates, event between exactly one adjacent pair
        if len(pre)>=2 and len(post)>=2:
            c=rng.integers(0,3); seq={0:[pre[-1],post[0],post[1],post[-1]] if len(post)>=3 else None,1:[pre[0],pre[-1],post[0],post[-1]],2:[pre[0],pre[-2],pre[-1],post[0]] if len(pre)>=3 else None}[int(c)]
            if seq and len(set(seq))==4:
                gaps=[(dd(seq[i+1])-dd(seq[i])).days for i in range(3)]
                if max(gaps)>2*max(min(gaps),1): seq=None
            if seq and len(set(seq))==4:
                items.append({"id":f"{s}_q2","fold":fold,"tile":s,"type":"Q2","dates":[d(j) for j in seq],"images":[rgb_png(raw,j,OUT/"img"/f"{s}_{j}.png") for j in seq],"answer":str(int(c)+1)})
        # Q3: quadrants
        qd=quadrants(np.load(SRC/"mask_u8"/f"{s}.npy")>0)
        if qd: items.append({"id":f"{s}_q3","fold":fold,"tile":s,"type":"Q3","dates":[d(pre[-1]),d(post[0])],"images":[rgb_png(raw,pre[-1],OUT/"img"/f"{s}_{pre[-1]}.png"),rgb_png(raw,post[0],OUT/"img"/f"{s}_{post[0]}.png")],"answer":qd})
        # Q4: control
        nimg=int(rng.integers(2,5)); seq=sorted(rng.choice(12,nimg,replace=False).tolist()); items.append({"id":f"{s}_q4","fold":fold,"tile":s,"type":"Q4","dates":[d(j) for j in seq],"images":[rgb_png(raw,j,OUT/"img"/f"{s}_{j}.png") for j in seq],"answer":str(nimg)})
        n1+=1
        if n1>=PER: break
    for s in neg[:PER//2]:
        r,k,pre,post=slots(s)
        if not pre or not post: continue
        raw=np.load(SRC/"raw_u16"/f"{s}.npy").astype("float32"); d=lambda j:str(r["times"][k[j]])[:10]; a,b=pre[-1],post[0]
        from datetime import date as _date
        items.append({"id":f"{s}_q1_zero","gap_days":(_date.fromisoformat(d(b))-_date.fromisoformat(d(a))).days,"fold":fold,"tile":s,"type":"Q1","dates":[d(a),d(b)],"images":[rgb_png(raw,a,OUT/"img"/f"{s}_{a}.png"),rgb_png(raw,b,OUT/"img"/f"{s}_{b}.png")],"answer":"no"})
(OUT/"items.jsonl").write_text("\n".join(json.dumps(x) for x in items)+"\n")
from collections import Counter; print(Counter((x["fold"],x["type"]) for x in items)); print("Q1 balance",Counter(x["answer"] for x in items if x["type"]=="Q1")); import statistics; print("Q1 gap median yes/no",statistics.median([x["gap_days"] for x in items if x["type"]=="Q1" and x["answer"]=="yes"]),statistics.median([x["gap_days"] for x in items if x["type"]=="Q1" and x["answer"]=="no"])); print("SENTINEL QA GEN DONE")
