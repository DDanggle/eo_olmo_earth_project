#!/usr/bin/env python3
"""Task-2 판정: (1) 소스 재현(M65식: P4 > 최고 raw ≥5/8 폴드) (2) few-shot 헤드라인(addendum_v1: A1−A4w ≥+.01 ≥6/8 at K5 & K20 같은 방향; A1−A4h ≥6/8) + random + fe."""
import json,glob,collections,statistics as st
from pathlib import Path
R=Path("/home/work/data/olmoearth"); out={"schema":"task2-verdict-v1"}
# 소스
src=collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(str(R/"task2_source_v1/holdout_*/*_pilot.json")):
    d=json.load(open(f)); fold=d.get("fold") or f.split("holdout_")[1].split("_seed")[0]
    for arm,v in d["arms"].items(): src[fold.replace("holdout_","")][arm].append(v["test"]["positive_patch_macro_iou"])
folds=sorted(src); rows=[]
for fo in folds:
    p4=st.mean(src[fo]["P4"]) if src[fo]["P4"] else None; p2=st.mean(src[fo]["P2"]) if src[fo]["P2"] else None
    rows.append({"fold":fo,"P4":p4,"P2":p2,"n_seeds":(len(src[fo]["P4"]),len(src[fo]["P2"]))})
wins=sum(1 for r in rows if r["P4"] is not None and r["P2"] is not None and r["P4"]>r["P2"])
out["source"]={"rows":rows,"P4_macro":st.mean([r["P4"] for r in rows if r["P4"] is not None]) if rows else None,"P2_macro":st.mean([r["P2"] for r in rows if r["P2"] is not None]) if rows else None,"P4_gt_P2_folds":wins,"n_folds":len(rows),"reproduces_rule_ge5of8":wins>=5}
print("SOURCE:",{k:(round(v,4) if isinstance(v,float) else v) for k,v in out["source"].items() if k!="rows"})
for r in rows: print("  %-12s P4=%s P2=%s seeds=%s"%(r["fold"],None if r["P4"] is None else round(r["P4"],3),None if r["P2"] is None else round(r["P2"],3),r["n_seeds"]))
# few-shot
def load(tag):
    fs=glob.glob(str(R/f"artifacts/task2_fewshot/{tag}/report_*.json"))
    if not fs: return None
    d=json.load(open(fs[0])); agg=collections.defaultdict(list)
    for r in d["runs"]: agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"])
    return {k:sum(v)/len(v) for k,v in agg.items()}
out["fewshot"]={}
for tag in ("fu","fu_random","fe"):
    m=load(tag)
    if not m: continue
    regs=sorted({k[0] for k in m}); cnt=lambda f: sum(1 for r in regs if f(r)); res={"n_regions":len(regs)}
    def g(r,K,a): return m.get((r,K,a))
    for K in (5,20):
        if all(g(r,K,"A1") is not None and g(r,K,"A4w") is not None for r in regs):
            res[f"A1_gt_A4w_K{K}"]=cnt(lambda r: g(r,K,"A1")-g(r,K,"A4w")>=0.01); res[f"A4w_gt_A1_K{K}"]=cnt(lambda r: g(r,K,"A4w")-g(r,K,"A1")>=0.01)
            res[f"macro_A1_K{K}"]=st.mean(g(r,K,"A1") for r in regs); res[f"macro_A4w_K{K}"]=st.mean(g(r,K,"A4w") for r in regs)
        if all(g(r,K,"A1") is not None and g(r,K,"A4h") is not None for r in regs):
            res[f"A1_gt_A4h_K{K}"]=cnt(lambda r: g(r,K,"A1")-g(r,K,"A4h")>=0.01); res[f"macro_A4h_K{K}"]=st.mean(g(r,K,"A4h") for r in regs)
        if all(g(r,K,"A1") is not None and g(r,None,"A0") is not None for r in regs):
            res[f"A1_gt_A0_K{K}"]=cnt(lambda r: g(r,K,"A1")>g(r,None,"A0"))
    if all(g(r,None,"A0") is not None for r in regs): res["macro_A0"]=st.mean(g(r,None,"A0") for r in regs)
    n=len(regs); need=6 if n>=8 else max(1,round(n*0.75))
    if "A1_gt_A4w_K5" in res and "A1_gt_A4w_K20" in res:
        res["verdict_replicates"]=bool(res["A1_gt_A4w_K5"]>=need and res["macro_A1_K20"]>res["macro_A4w_K20"])
    if "A1_gt_A4h_K5" in res: res["verdict_representation"]=bool(res["A1_gt_A4h_K5"]>=need)
    out["fewshot"][tag]=res; print(tag,{k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items()})
    for r in regs: print("  %-12s"%r," ".join("%s%s=%.3f"%(a,"" if K is None else K,m[(r,K,a)]) for K in (None,5,20) for a in ("A0","A1","A4h","A4w") if (r,K,a) in m))
(R/"artifacts/task2_fewshot/verdict.json").parent.mkdir(parents=True,exist_ok=True); (R/"artifacts/task2_fewshot/verdict.json").write_text(json.dumps(out,indent=1))
