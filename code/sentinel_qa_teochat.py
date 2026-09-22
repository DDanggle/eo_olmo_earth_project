#!/usr/bin/env python3
"""TEOChat zero-shot on sentinel_qa_v0 items (fp16). Writes answers.jsonl and scores.json with registered regex parsing."""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0,"/home/work/data/olmoearth/teochat/code")
ROOT=Path("/home/work/data/olmoearth"); import sys as _s; OUT=ROOT/(_s.argv[1] if len(_s.argv)>1 and not _s.argv[1].isdigit() else "sentinel_qa_v0"); items=[json.loads(l) for l in (OUT/"items.jsonl").read_text().splitlines() if l]
limit=int(sys.argv[2]) if len(sys.argv)>2 else 0
if limit: items=items[:limit]
from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single
tok, model, proc = load_model(model_path=str(ROOT/"teochat/TEOChat"), model_base=None, load_8bit=False, device="cuda")
def prompt(it):
    n=len(it["images"]); head=f"These are {n} Sentinel-2 satellite images of the same area in chronological order, taken on {', '.join(it['dates'])}: <video> "
    return head+{"Q1":"Did a landslide occur between the two images? Answer with yes or no.",
                 "Q2":"A landslide occurred between exactly one adjacent pair of these images. Between which pair did it occur? Answer with 1 (between image 1 and 2), 2 (between 2 and 3), or 3 (between 3 and 4).",
                 "Q3":"A landslide occurred between the two images. In which quadrant(s) of the image is the landslide located? Answer with one or more of NW, NE, SW, SE.",
                 "Q4":"How many images are shown? Answer with a number."}[it["type"]]
def parse(t,a):
    a=a.strip().lower()
    if t=="Q1": m=re.search(r"\b(yes|no)\b",a); return m.group(1) if m else None
    if t=="Q2": m=re.search(r"\b([123])\b",a); return m.group(1) if m else None
    if t=="Q3": q=sorted(set(re.findall(r"\b(nw|ne|sw|se)\b",a))); return [x.upper() for x in q] or None
    if t=="Q4": m=re.search(r"\b([2-9]|two|three|four)\b",a); return {"two":"2","three":"3","four":"4"}.get(m.group(1),m.group(1)) if m else None
done={json.loads(l)["id"] for l in (OUT/"answers_teochat.jsonl").read_text().splitlines()} if (OUT/"answers_teochat.jsonl").exists() else set()
f=open(OUT/"answers_teochat.jsonl","a"); t0=time.perf_counter()
for i,it in enumerate(items):
    if it["id"] in done: continue
    try: ans=run_inference_single(model, proc, tok, prompt(it), it["images"], timestamps=it["dates"], max_new_tokens=64)
    except Exception as e: ans=f"ERROR {e}"
    f.write(json.dumps({"id":it["id"],"type":it["type"],"answer_raw":ans,"parsed":parse(it["type"],ans),"gold":it["answer"]})+"\n"); f.flush()
    if i%50==0: print(i,f"{time.perf_counter()-t0:.0f}s",flush=True)
f.close()
rows=[json.loads(l) for l in (OUT/"answers_teochat.jsonl").read_text().splitlines() if l]
sc={}
for t in ("Q1","Q2","Q3","Q4"):
    R=[r for r in rows if r["type"]==t]; 
    if not R: continue
    pf=sum(1 for r in R if r["parsed"] is None)/len(R)
    if t=="Q3": acc=sum(1 for r in R if r["parsed"]==r["gold"])/len(R); jac=sum((len(set(r["parsed"] or [])&set(r["gold"]))/len(set(r["parsed"] or [])|set(r["gold"]))) for r in R)/len(R); sc[t]={"n":len(R),"exact":acc,"jaccard":jac,"parse_fail":pf}
    else:
        acc=sum(1 for r in R if r["parsed"]==r["gold"])/len(R); sc[t]={"n":len(R),"acc":acc,"parse_fail":pf}
        if t=="Q1":
            neg=[r for r in R if r["gold"]=="no"]; pos=[r for r in R if r["gold"]=="yes"]; sc[t]["fpr_on_negatives"]=sum(1 for r in neg if r["parsed"]=="yes")/max(len(neg),1); sc[t]["recall_on_positives"]=sum(1 for r in pos if r["parsed"]=="yes")/max(len(pos),1); sc[t]["balanced_acc"]=0.5*(sc[t]["recall_on_positives"]+1-sc[t]["fpr_on_negatives"])
(OUT/"scores_teochat.json").write_text(json.dumps(sc,indent=1)); print(json.dumps(sc,indent=1)); print("SENTINEL QA TEOCHAT DONE")
