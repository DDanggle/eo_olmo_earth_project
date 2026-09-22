#!/usr/bin/env python3
"""G1-lite v0.3 (config/bottleneck_diag_lite_prereg_v0_3.json): as v0.2 but privileged frames for Q1 span the question window (start-1, first appearance, one clear mid month, cutoff); Q2 privileged unchanged: quadrant crops at NATIVE resolution for Q1/Q2, MIN_NEW=8, Q3 dropped, reader Qwen3-VL-8B (run) — otherwise identical to v0.
  build : episodes (12 dev AOIs x 2 cutoffs), quadrant regions, silver gold from labels_match, four evidence conditions, PNG frames.
  run   : frozen TEOChat answers for every (episode, question, condition).
  score : per-condition accuracy, paired over episodes, bootstrap CI over AOIs.
Silver gold = building-id first appearance month in labels_match (omniscient labels; NOT prefix-visible-verified)."""
import json, re, sys, time, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); SN7=ROOT/"spacenet7/train"; OUT=ROOT/"spacenet7/diag_lite_v0_3"; OUT.mkdir(parents=True,exist_ok=True)
CUTOFFS=[12,20]; K=4; QUADS=["NW","NE","SW","SE"]; WIN=6; MIN_NEW=8
mon=lambda name: "-".join(re.search(r"global_monthly_(\d{4})_(\d{2})",name).groups())
def dev_aois():
    a=sorted(p.name for p in SN7.iterdir() if p.is_dir()); return a[::5][:12]
def quad_of(x,y,W,H): return ("N" if y<H/2 else "S")+("W" if x<W/2 else "E")
def build():
    import rasterio
    from shapely.geometry import shape
    from PIL import Image
    items=[]
    for aoi in dev_aois():
        imgs=sorted((SN7/aoi/"images_masked").glob("*.tif")) or sorted((SN7/aoi/"images").glob("*.tif"))
        months=[mon(p.name) for p in imgs]; udm={mon(p.name):p for p in (SN7/aoi/"UDM_masks").glob("*.tif")}
        fr_dir=OUT/"frames"/aoi; fr_dir.mkdir(parents=True,exist_ok=True); png={}; arrs={}
        for p,m in zip(imgs,months):
            with rasterio.open(p) as ds: a=ds.read([1,2,3]).transpose(1,2,0); W,H=ds.width,ds.height; tr=ds.transform
            png.setdefault(m,{})
            for q in QUADS:  # native-resolution quadrant crops (~512x512 px at 4 m)
                r0,r1=(0,H//2) if q[0]=="N" else (H//2,H); c0,c1=(0,W//2) if q[1]=="W" else (W//2,W); o=fr_dir/f"{m}_{q}.png"
                if not o.exists(): Image.fromarray(a[r0:r1,c0:c1].astype("uint8")).save(o)
                png[m][q]=str(o)
            arrs[m]=a[::4,::4].astype("float32").mean(2)
        unus={}
        for m,p in udm.items():
            with rasterio.open(p) as ds: unus[m]=float((ds.read(1)!=0).mean())
        # first appearance per building id, with quadrant by centroid (pixel coords via transform)
        first={}
        for p in sorted((SN7/aoi/"labels_match").glob("*.geojson")):
            m=mon(p.name)
            try: fc=json.loads(p.read_text())
            except Exception: continue
            for f in fc.get("features",[]):
                pr=f.get("properties") or {}; i=pr.get("Id",pr.get("id"))
                if i is None or not f.get("geometry"): continue
                if i not in first or m<first[i][0]:
                    c=shape(f["geometry"]).centroid; col,row=~tr*(c.x,c.y); first[i]=(m,quad_of(col,row,W,H))
        base=months[0]; new_by_mq={}
        for i,(m,q) in first.items():
            if m!=base: new_by_mq.setdefault((m,q),0); new_by_mq[(m,q)]+=1
        for cut in CUTOFFS:
            if cut>=len(months): continue
            pre=months[:cut+1]; cm=months[cut]
            # Q1: for each quadrant, did >=MIN_NEW new buildings appear in the last WIN months? balanced pick: one yes, one no if available
            cnt={q:sum(new_by_mq.get((m,q),0) for m in pre[-WIN:]) for q in QUADS}
            yes=[q for q in QUADS if cnt[q]>=MIN_NEW]; no=[q for q in QUADS if cnt[q]==0]
            # Q2: quadrant with events: first month (index>0) where cumulative new >= MIN_NEW
            q2=None
            for q in sorted(QUADS,key=lambda q:-sum(new_by_mq.get((m,q),0) for m in pre)):
                cum=0
                for m in pre[1:]:
                    cum+=new_by_mq.get((m,q),0)
                    if cum>=MIN_NEW: q2=(q,m); break
                if q2: break
            # Q3: quadrant with most new buildings in last 3 months
            last3={q:sum(new_by_mq.get((m,q),0) for m in pre[-3:]) for q in QUADS}; q3=max(last3,key=last3.get) if max(last3.values())>0 else None
            def conds(gold_month,q1_window=None):
                c={"full_prefix":pre,"latest_k":pre[-K:]}
                diffs=[(float(np.abs(arrs[pre[j]]-arrs[pre[j-1]]).mean()),pre[j]) for j in range(1,len(pre))]
                c["change_topk"]=sorted([m for _,m in sorted(diffs,reverse=True)[:K]])
                if gold_month and gold_month in pre:
                    gi=pre.index(gold_month)
                    if q1_window:  # v0.3: Q1 evidence must span the question window: frame before window, first appearance, one clear mid month, cutoff
                        ws=pre.index(q1_window[0]); mid=[m for m in pre[gi+1:-1] if m>q1_window[0]]; mid=sorted(mid,key=lambda m:unus.get(m,0.0))[:1]
                        c["privileged_silver"]=sorted(set([pre[max(ws-1,0)],gold_month]+mid+[pre[-1]]))
                    else:
                        later=sorted([m for m in pre[gi+1:]],key=lambda m:unus.get(m,0.0))[:2]
                        c["privileged_silver"]=sorted(set([pre[max(gi-1,0)],gold_month]+later))
                else: c["privileged_silver"]=None
                return c
            for q in (yes[:1]+no[:1]):
                items.append({"id":f"{aoi}|{cm}|Q1|{q}","aoi":aoi,"cutoff":cm,"q":"Q1","region":q,"window":[pre[-WIN],cm],"gold":"yes" if q in yes else "no","conds":conds(None if q not in yes else next((m for m in pre[-WIN:] if new_by_mq.get((m,q),0)>0),None),q1_window=[pre[-WIN],cm])})
            if q2: items.append({"id":f"{aoi}|{cm}|Q2|{q2[0]}","aoi":aoi,"cutoff":cm,"q":"Q2","region":q2[0],"gold":q2[1],"conds":conds(q2[1])})
        for it in items:
            if it["aoi"]==aoi: it["png"]=png
        print(aoi,len(months),"items so far",len(items),flush=True)
    (OUT/"items.jsonl").write_text("\n".join(json.dumps(i) for i in items)+"\n"); print("BUILD DONE",len(items))
def prompt(it,frames):
    head=f"These are {len(frames)} monthly satellite images of the same area (one quadrant of a larger tile) in chronological order, taken in {', '.join(frames)}: <video> "
    if it["q"]=="Q1": return head+f"Did new buildings appear in this area between {it['window'][0]} and {it['window'][1]}? Answer with yes or no."
    if it["q"]=="Q2": return head+f"In which month did new construction first become clearly visible in this area? Answer with the month in YYYY-MM format from the list above."
    return head+"Which quadrant of the image contains the most recent new construction? Answer with one of NW, NE, SW, SE."
def parse(q,a):
    a=a.strip().lower()
    if q=="Q1": m=re.search(r"\b(yes|no)\b",a); return m.group(1) if m else None
    if q=="Q2": m=re.search(r"(20\d\d)[-_/ ](\d\d)",a); return f"{m.group(1)}-{m.group(2)}" if m else None
    m=re.search(r"\b(nw|ne|sw|se)\b",a); return m.group(1).upper() if m else None
def run():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    MODEL=ROOT/"models/Qwen3-VL-8B-Instruct"; proc=AutoProcessor.from_pretrained(MODEL); model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    items=[json.loads(l) for l in (OUT/"items.jsonl").read_text().splitlines() if l]; ap=OUT/"answers.jsonl"
    done={(r["id"],r["cond"]) for r in map(json.loads,ap.read_text().splitlines())} if ap.exists() else set(); f=open(ap,"a"); t0=time.perf_counter(); n=0
    for it in items:
        for cond,frames in it["conds"].items():
            if frames is None or (it["id"],cond) in done: continue
            paths=[it["png"][m][it["region"]] for m in frames]; text=prompt(it,frames).replace("<video> ","")
            content=[]
            for m,pth in zip(frames,paths): content+=[{"type":"text","text":f"[{m}]"},{"type":"image","image":pth}]
            content.append({"type":"text","text":text})
            try:
                imgs=[Image.open(pth).convert("RGB") for pth in paths]
                chat=proc.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
                inp=proc(text=[chat],images=imgs,return_tensors="pt").to("cuda")
                with torch.no_grad(): out=model.generate(**inp,max_new_tokens=32,do_sample=False)
                ans=proc.batch_decode(out[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0]
            except Exception as e: ans=f"ERROR {str(e)[:200]}"
            f.write(json.dumps({"id":it["id"],"aoi":it["aoi"],"q":it["q"],"cond":cond,"n_frames":len(frames),"raw":ans[:200],"pred":parse(it["q"],ans),"gold":it["gold"]})+"\n"); f.flush(); n+=1
            if n%20==0: print(n,f"{time.perf_counter()-t0:.0f}s",flush=True)
    f.close(); print("RUN DONE")
def score():
    rows=[json.loads(l) for l in (OUT/"answers.jsonl").read_text().splitlines() if l]; rng=np.random.default_rng(0)
    conds=sorted({r["cond"] for r in rows}); out={"schema":"sn7-diag-lite-v0","per_condition":{},"paired_vs":{}}
    def acc_rows(R):
        if not R: return None
        return {"n":len(R),"acc":float(np.mean([r["pred"]==r["gold"] for r in R])),"parse_fail":float(np.mean([r["pred"] is None for r in R])),"error":float(np.mean([r["raw"].startswith("ERROR") for r in R]))}
    for c in conds:
        out["per_condition"][c]={q:acc_rows([r for r in rows if r["cond"]==c and r["q"]==q]) for q in ("Q1","Q2")}
    aois=sorted({r["aoi"] for r in rows}); byk={(r["id"],r["cond"]):r for r in rows}
    for q in ("Q1","Q2"):
        for c in conds:
            if c=="privileged_silver": continue
            ids=[i for i in {r["id"] for r in rows if r["q"]==q} if (i,c) in byk and (i,"privileged_silver") in byk]
            if not ids: continue
            d={i:float(byk[(i,"privileged_silver")]["pred"]==byk[(i,"privileged_silver")]["gold"])-float(byk[(i,c)]["pred"]==byk[(i,c)]["gold"]) for i in ids}
            per_aoi={a:np.mean([d[i] for i in ids if i.startswith(a+"|")]) for a in aois if any(i.startswith(a+"|") for i in ids)}; v=np.array(list(per_aoi.values()))
            bs=[rng.choice(v,len(v)).mean() for _ in range(5000)]
            out["paired_vs"][f"{q}:privileged-{c}"]={"n_items":len(ids),"n_aois":len(v),"diff":float(v.mean()),"ci95":[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]}
    (OUT/"scores.json").write_text(json.dumps(out,indent=1)); print(json.dumps(out,indent=1)); print("SCORE DONE")
{"build":build,"run":run,"score":score}[sys.argv[1]]()
