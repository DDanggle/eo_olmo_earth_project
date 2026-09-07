"""Summarise field-adaptation dev runs and ceilings: per region x K, mean over seeds of FP-matched IoU and AP per arm; ceiling A1_pool/A4w_pool; gap closure."""
import json, glob, numpy as np, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/artifacts")
def load(p):
    r=json.load(open(p)); agg=defaultdict(list); ap=defaultdict(list)
    for x in r["runs"]: agg[(x["region"],x["K"],x["arm"])].append(x["eval"]["iou_fp_matched"]); ap[(x["region"],x["K"],x["arm"])].append(x["eval"]["tie_ap"])
    return agg,ap
ceil={}
for task,files in {"sen12":["fewshot_ceiling/sen12_a1/report_fixed_exposure.json","fewshot_ceiling/sen12_a4w/report_fixed_exposure.json"],"solar":["fewshot_ceiling/solar_a1/report_fixed_exposure.json","fewshot_ceiling/solar_a4w/report_fixed_exposure.json"]}.items():
    for f in files:
        if (ROOT/f).exists():
            a,_=load(ROOT/f)
            for (g,K,arm),v in a.items(): ceil[(task,g,arm if arm!="A0" else "A0")]=float(np.mean(v))
print("== ceilings (in-region full pool)")
for task in ("sen12","solar"):
    regs=sorted({k[1] for k in ceil if k[0]==task}); 
    for g in regs: print(task,g,"A0 %.3f | A1_pool %.3f | A4w_pool %.3f"%(ceil.get((task,g,"A0"),float("nan")),ceil.get((task,g,"A1"),float("nan")),ceil.get((task,g,"A4w"),float("nan"))))
print("\n== dev arms (mean over seeds; FP-matched IoU / AP)")
for name in sorted(glob.glob(str(ROOT/"field_adapt/dev_*/report_fixed_update.json"))):
    a,ap=load(name); regs=sorted({k[0] for k in a}); arms=sorted({k[2] for k in a},key=lambda s:(len(s),s)); print("--",name.split("/")[-2])
    for g in regs:
        for K in (None,5,20):
            row=[f"{arm}:{np.mean(a[(g,K,arm)]):.3f}/{np.mean(ap[(g,K,arm)]):.3f}" for arm in arms if (g,K,arm) in a]
            if row: print(g,K,"  ".join(row))
print("\n== aggregates per arm (Solar dev 8 folds; Sen12 dev 2 regions): mean FP-IoU, wins vs A1, worst-fold, mean AP")
for name in sorted(glob.glob(str(ROOT/"field_adapt/dev_*/report_fixed_update.json"))):
    a,ap=load(name); regs=sorted({k[0] for k in a}); arms=sorted({k[2] for k in a},key=lambda s:(len(s),s)); print("--",name.split("/")[-2])
    for K in (5,20):
        for arm in arms:
            if arm=="A0" or not all((g,K,arm) in a for g in regs): continue
            v=[np.mean(a[(g,K,arm)]) for g in regs]; b=[np.mean(a[(g,K,"A1")]) for g in regs]; w=sum(x>y+1e-9 for x,y in zip(v,b)); apv=[np.mean(ap[(g,K,arm)]) for g in regs]
            print(f"K={K} {arm:6s} mean {np.mean(v):.3f} (A1 {np.mean(b):.3f}) wins {w}/{len(regs)} worst {min(v):.3f} (A1 worst {min(b):.3f}) AP {np.mean(apv):.3f}")
