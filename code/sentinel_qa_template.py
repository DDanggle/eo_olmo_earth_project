#!/usr/bin/env python3
"""No-LLM baseline for sentinel_qa_v0: OlmoEarth single-observation embeddings + change score + template answers.
Change score between dates = mean over tokens of (1 - cos) between single embeddings. Q1: yes if score > median of all Q1 items (label-free, balanced by construction).
Q2: adjacent pair with the largest score. Q3: quadrants whose share of the change map >= 25%. Q4: number of images."""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); import sys as _s; OUT=ROOT/(_s.argv[1] if len(_s.argv)>1 and not _s.argv[1].isdigit() else "sentinel_qa_v0"); SRC=ROOT/"olmo_streaming_dev/single_fp16"
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
items=[json.loads(l) for l in (OUT/"items.jsonl").read_text().splitlines() if l]
def kept12(r): q=r["scl_clear_fraction"]; return sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12])
cache={}
def single(tile):
    if tile not in cache:
        p=SRC/f"{tile}.npy"; cache[tile]=torch.from_numpy(np.load(p).astype("float32")) if p.exists() else None
    return cache[tile]
def slot_of(tile,date):
    r=REC[tile]; k=kept12(r); ds=[str(r["times"][i])[:10] for i in k]; return ds.index(date)
def change_map(S,a,b): return 1-torch.nn.functional.cosine_similarity(S[a],S[b],dim=0)  # 32x32
rows=[]; missing=0
for it in items:
    S=single(it["tile"])
    if S is None: missing+=1; rows.append({"id":it["id"],"type":it["type"],"parsed":None,"gold":it["answer"],"score":None}); continue
    sl=[slot_of(it["tile"],d) for d in it["dates"]]
    if it["type"]=="Q1": rows.append({"id":it["id"],"type":"Q1","score":float(change_map(S,sl[0],sl[1]).mean()),"gold":it["answer"]})
    elif it["type"]=="Q2": sc=[float(change_map(S,sl[i],sl[i+1]).mean()) for i in range(3)]; rows.append({"id":it["id"],"type":"Q2","parsed":str(int(np.argmax(sc))+1),"gold":it["answer"],"score":sc})
    elif it["type"]=="Q3":
        m=change_map(S,sl[0],sl[1]).numpy(); h,w=m.shape; tot=m.sum(); q=[]
        for name,s in (("NW",(slice(0,h//2),slice(0,w//2))),("NE",(slice(0,h//2),slice(w//2,w))),("SW",(slice(h//2,h),slice(0,w//2))),("SE",(slice(h//2,h),slice(w//2,w)))):
            if m[s].sum()/tot>=0.25: q.append(name)
        rows.append({"id":it["id"],"type":"Q3","parsed":q,"gold":it["answer"]})
    else: rows.append({"id":it["id"],"type":"Q4","parsed":str(len(it["images"])),"gold":it["answer"]})
q1=[r for r in rows if r["type"]=="Q1" and r.get("score") is not None]; med=float(np.median([r["score"] for r in q1]))
for r in q1: r["parsed"]="yes" if r["score"]>med else "no"
(OUT/"answers_template.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
sc={"missing_single_embeddings":missing,"q1_threshold_median":med}
for t in ("Q1","Q2","Q3","Q4"):
    R=[r for r in rows if r["type"]==t and r.get("parsed") is not None]
    if not R: continue
    if t=="Q3": sc[t]={"n":len(R),"exact":sum(1 for r in R if r["parsed"]==r["gold"])/len(R),"jaccard":sum(len(set(r["parsed"])&set(r["gold"]))/len(set(r["parsed"])|set(r["gold"])) for r in R)/len(R)}
    else:
        sc[t]={"n":len(R),"acc":sum(1 for r in R if r["parsed"]==r["gold"])/len(R)}
        if t=="Q1": neg=[r for r in R if r["gold"]=="no"]; pos=[r for r in R if r["gold"]=="yes"]; sc[t]["fpr_on_negatives"]=sum(1 for r in neg if r["parsed"]=="yes")/max(len(neg),1); sc[t]["recall_on_positives"]=sum(1 for r in pos if r["parsed"]=="yes")/max(len(pos),1); sc[t]["balanced_acc"]=0.5*(sc[t]["recall_on_positives"]+1-sc[t]["fpr_on_negatives"])
(OUT/"scores_template.json").write_text(json.dumps(sc,indent=1)); print(json.dumps(sc,indent=1)); print("SENTINEL QA TEMPLATE DONE")
