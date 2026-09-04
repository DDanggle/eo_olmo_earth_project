"""Per-region bars: zero-target (reuse P4 vs raw P2) and few-shot K=5 stratified (A0 reuse / A1 cache+head adapt / A4w raw adapt), landslide 8 regions + solar 8 folds."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
ROOT=Path("/home/work/data/olmoearth/artifacts")
def fewshot(path,K=5):
    r=json.load(open(path)); agg=defaultdict(list)
    for x in r["runs"]:
        if x["arm"]=="A0" or x["K"]==K: agg[(x["region"],x["arm"])].append(x["eval"]["iou_fp_matched"])
    regs=sorted({k[0] for k in agg}); return regs,{arm:[np.mean(agg[(g,arm)]) for g in regs] for arm in ("A0","A1","A4w")}
s=json.load(open(ROOT/"confirmatory_8region_summary.json")); zs=[(x["fold"].replace("holdout_",""),x["primary_mean"]["reuse"],x["primary_mean"]["raw_strong"]) for x in s["regions"]]
t=json.load(open(ROOT/"task2_fewshot/verdict.json")); zt=[(x["fold"].replace("task2_",""),x["P4"],x["P2"]) for x in t["source"]["rows"]]
fs_l=fewshot(ROOT/"fewshot_confirmatory/fu/report_fixed_update.json"); fs_t=fewshot(ROOT/"task2_fewshot/fu/report_fixed_update.json")
fig,ax=plt.subplots(2,2,figsize=(14,8.5))
def zero(a,rows,title):
    x=np.arange(len(rows)); w=0.38; a.bar(x-w/2,[r[1] for r in rows],w,label="reuse: frozen cache + decoder (P4)",color="#1f77b4"); a.bar(x+w/2,[r[2] for r in rows],w,label="retrain: raw UNet3D (P2)",color="#b0b0b0")
    a.set_xticks(x); a.set_xticklabels([r[0] for r in rows],rotation=30,ha="right",fontsize=8); a.set_ylabel("positive-tile macro IoU"); a.set_title(title,fontsize=10); a.legend(fontsize=8)
    wins=sum(r[1]>r[2] for r in rows); a.text(0.02,0.95,f"reuse wins {wins}/{len(rows)}   macro {np.mean([r[1] for r in rows]):.3f} vs {np.mean([r[2] for r in rows]):.3f}",transform=a.transAxes,va="top",fontsize=9)
def few(a,fs,title):
    regs,d=fs; x=np.arange(len(regs)); w=0.27
    for i,(arm,lab,c) in enumerate((("A0","A0 reuse, 0 labels","#1f77b4"),("A1","A1 cache + head adapt, K=5","#ff7f0e"),("A4w","A4w raw adapt, K=5","#b0b0b0"))): a.bar(x+(i-1)*w,d[arm],w,label=lab,color=c)
    a.set_xticks(x); a.set_xticklabels([g.replace("task2_","") for g in regs],rotation=30,ha="right",fontsize=8); a.set_ylabel("FP-matched macro IoU"); a.set_title(title,fontsize=10); a.legend(fontsize=8)
    a.text(0.02,0.95,"A1>A4w %d/%d   A1>A0 %d/%d   macro A0 %.3f A1 %.3f A4w %.3f"%(sum(np.array(d["A1"])>np.array(d["A4w"])),len(regs),sum(np.array(d["A1"])>np.array(d["A0"])),len(regs),np.mean(d["A0"]),np.mean(d["A1"]),np.mean(d["A4w"])),transform=a.transAxes,va="top",fontsize=9)
zero(ax[0,0],zs,"Landslide (Sen12, 8 held-out regions) — zero target labels (M65)"); zero(ax[0,1],zt,"Solar farm (Task-2, 8 UTM folds) — zero target labels (MS-98)")
few(ax[1,0],fs_l,"Landslide — 5 target labels, stratified support (MS-96/97)"); few(ax[1,1],fs_t,"Solar farm — 5 target labels, stratified support (MS-99)")
fig.suptitle("Before (raw retrain) vs after (frozen-cache reuse / cache-side adaptation): per-region, 3 seeds averaged",fontsize=11); plt.tight_layout(rect=(0,0,1,0.96))
out=ROOT/"figures/region_bars_landslide_solar.png"; fig.savefig(out,dpi=120); print("saved",out)
