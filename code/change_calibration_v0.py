#!/usr/bin/env python3
"""Change-score calibration v0 (config/change_calibration_prereg_v0.json). Consecutive single-observation states -> 1-cos; normal-change tables by gap/season/region; z-scores; LOEO transfer; event/cloud validation."""
import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"change_calibration_v0"; OUT.mkdir(exist_ok=True)
SIZE=sys.argv[1] if len(sys.argv)>1 else "base"; REGIONS=sys.argv[2].split(",") if len(sys.argv)>2 else ["hiroshima","thrissur","itogon","hokkaido"]
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
def gapbin(g): return next(l for l,a,b in GAPS if a<=g<=b)
def season(m): return "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
rows=[]
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg if SIZE=="base" else ROOT/f"arrival_v0_{SIZE}"/reg
    for p in sorted((D/"single_fp16").glob("*.npy")):
        s=p.stem; m=json.loads((ROOT/"arrival_v0"/reg/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
        mask=np.load(ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"/f"{s}.npy")>0 if (ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"/f"{s}.npy").exists() else None
        S=torch.from_numpy(np.load(p).astype("float32")).to(dev); T=S.shape[0]; F=S.flatten(2)
        cos=torch.nn.functional.cosine_similarity(F[:-1],F[1:],dim=1).mean(1).cpu().numpy()
        for i in range(T-1):
            crosses=(post is not None and i<post<=i+1); positive=bool(mask.any()) if mask is not None else False
            kind="cloudy" if (clear[i]<0.5 or clear[i+1]<0.5) else ("event" if (crosses and positive) else ("normal" if not crosses else "cross_negative"))
            rows.append({"region":reg,"tile":s,"i":i,"gap":(d[i+1]-d[i]).days,"season":season(d[i+1].month),"clear_min":min(clear[i],clear[i+1]),"change":float(1-cos[i]),"kind":kind})
# v0.1: clear-to-clear pairs (last clear pre-event <-> first clear post-event = EVENT; any clear-to-clear non-crossing span = NORMAL_SPAN)
rows2=[]
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg if SIZE=="base" else ROOT/f"arrival_v0_{SIZE}"/reg
    for p in sorted((D/"single_fp16").glob("*.npy")):
        s=p.stem; m=json.loads((ROOT/"arrival_v0"/reg/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
        mp=ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"/f"{s}.npy"; positive=bool((np.load(mp)>0).any()) if mp.exists() else False
        S=torch.from_numpy(np.load(p).astype("float32")).to(dev); F=S.flatten(2); cl=[i for i in range(S.shape[0]) if clear[i]>=0.5]
        def ch(i,j): return float(1-torch.nn.functional.cosine_similarity(F[i],F[j],dim=0).mean())
        if post is not None and positive:
            pre=[i for i in cl if i<post]; pst=[i for i in cl if i>=post]
            if pre and pst: i,j=pre[-1],pst[0]; rows2.append({"region":reg,"tile":s,"gap":(d[j]-d[i]).days,"season":season(d[j].month),"change":ch(i,j),"kind":"event"})
        for a_,b_ in zip(cl[:-1],cl[1:]):
            if post is None or (a_<post and b_<post) or (a_>=post and b_>=post):
                rows2.append({"region":reg,"tile":s,"gap":(d[b_]-d[a_]).days,"season":season(d[b_].month),"change":ch(a_,b_),"kind":"normal_span","pre":bool(post is None or b_<post)})
(OUT/f"pairs_{SIZE}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
def table(sel):
    t={}
    for l,_,_ in GAPS:
        for se in ("DJF","MAM","JJA","SON"):
            v=np.array([r["change"] for r in sel if gapbin(r["gap"])==l and r["season"]==se])
            if len(v)>=20: t[f"{l}|{se}"]={"n":len(v),"mu":float(v.mean()),"sigma":float(v.std()+1e-6),"p50":float(np.median(v)),"p95":float(np.quantile(v,.95))}
    return t
def z(r,t):
    k=f"{gapbin(r['gap'])}|{r['season']}"
    if k in t: return (r["change"]-t[k]["mu"])/t[k]["sigma"]
    v=[x for x in t.values()]; mu=np.mean([x["mu"] for x in v]); sg=np.mean([x["sigma"] for x in v]); return (r["change"]-mu)/sg
normal=[r for r in rows if r["kind"]=="normal"]; T_all=table(normal)
res={"schema":"change-calibration-v0","size":SIZE,"n_pairs":len(rows),"kinds":{k:sum(1 for r in rows if r["kind"]==k) for k in ("normal","event","cloudy","cross_negative")},
     "raw_change_by_kind":{k:{"median":float(np.median([r["change"] for r in rows if r["kind"]==k])),"p90":float(np.quantile([r["change"] for r in rows if r["kind"]==k],.9))} for k in ("normal","event","cloudy") if any(r["kind"]==k for r in rows)},
     "normal_table_pooled":T_all,"by_gap_normal":{l:float(np.median([r["change"] for r in normal if gapbin(r["gap"])==l])) for l,_,_ in GAPS if any(gapbin(r["gap"])==l for r in normal)},
     "by_region_normal_median":{reg:float(np.median([r["change"] for r in normal if r["region"]==reg])) for reg in REGIONS}}
for k in ("event","cloudy","normal"):
    zs=[z(r,T_all) for r in rows if r["kind"]==k]; res[f"z_{k}"]={"n":len(zs),"frac_z_ge_2":float(np.mean([v>=2 for v in zs])) if zs else None,"median_z":float(np.median(zs)) if zs else None}
if len(REGIONS)>1:
    lo={}
    for held in REGIONS:
        Tt=table([r for r in normal if r["region"]!=held]); ev=[z(r,Tt) for r in rows if r["region"]==held and r["kind"]=="event"]; no=[z(r,Tt) for r in rows if r["region"]==held and r["kind"]=="normal"]; cl=[z(r,Tt) for r in rows if r["region"]==held and r["kind"]=="cloudy"]
        lo[held]={"event_frac_z_ge_2":float(np.mean([v>=2 for v in ev])) if ev else None,"normal_frac_z_ge_2":float(np.mean([v>=2 for v in no])) if no else None,"cloudy_frac_z_ge_2":float(np.mean([v>=2 for v in cl])) if cl else None,"n_event":len(ev)}
    res["loeo_transfer"]=lo
# v0.1 tables on clear-to-clear spans, region variance factor from PRE-EVENT spans only, LOEO with/without the factor
ns=[r for r in rows2 if r["kind"]=="normal_span"]; ev=[r for r in rows2 if r["kind"]=="event"]; Tn=table(ns)
res["v0_1"]={"n_normal_span":len(ns),"n_event":len(ev),"event_change_median":float(np.median([r["change"] for r in ev])) if ev else None,"normal_span_median":float(np.median([r["change"] for r in ns])),
    "z_event_frac_ge_2":float(np.mean([z(r,Tn)>=2 for r in ev])) if ev else None,"z_normal_frac_ge_2":float(np.mean([z(r,Tn)>=2 for r in ns])),"loeo":{}}
def spread(sel): v=np.array([r["change"] for r in sel]); return float(v.std()) if len(v)>=30 else None
sp_pool=spread([r for r in ns if r.get("pre")])
for held in REGIONS:
    Tt=table([r for r in ns if r["region"]!=held]); sr=spread([r for r in ns if r["region"]==held and r.get("pre")]); f=(sr/sp_pool) if (sr and sp_pool) else 1.0
    def zf(r): return z(r,Tt)/f
    E=[r for r in ev if r["region"]==held]; N=[r for r in ns if r["region"]==held and not r.get("pre")]
    res["v0_1"]["loeo"][held]={"n_event":len(E),"region_factor":round(f,3),"event_z_ge_2":float(np.mean([z(r,Tt)>=2 for r in E])) if E else None,"event_z_ge_2_factor":float(np.mean([zf(r)>=2 for r in E])) if E else None,"normal_post_z_ge_2":float(np.mean([z(r,Tt)>=2 for r in N])) if N else None,"normal_post_z_ge_2_factor":float(np.mean([zf(r)>=2 for r in N])) if N else None}
(OUT/f"summary_{SIZE}.json").write_text(json.dumps(res,indent=1)); print(json.dumps({k:res[k] for k in ("size","n_pairs","kinds","raw_change_by_kind","by_gap_normal","by_region_normal_median","z_event","z_cloudy","z_normal")},indent=1)); print(json.dumps(res.get("loeo_transfer",{}),indent=1)); print("V01",json.dumps(res["v0_1"])); print("CHANGE CALIBRATION DONE")
