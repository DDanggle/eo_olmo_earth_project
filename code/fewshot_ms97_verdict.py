#!/usr/bin/env python3
"""MS-97 판정: rawctl(A4w0,A4h,A4p) + fu(A0,A1,A4w) + random 결합. 규칙: fewshot prereg addendum_v1."""
import json,collections
from pathlib import Path
R=Path("/home/work/data/olmoearth/artifacts/fewshot_confirmatory")
def load(tag):
    d=json.load(open(next((R/tag).glob("report_*.json")))); agg=collections.defaultdict(list); par={}
    for r in d["runs"]:
        agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"]); par[r["arm"]]=r["train"].get("trainable_params")
    m={k:sum(v)/len(v) for k,v in agg.items()}; return m,par
fu,_=load("fu"); rc,par=load("fu_rawctl"); rnd,_=load("fu_random"); fe,_=load("fe")
regs=sorted({k[0] for k in fu})
def cnt(f): return sum(1 for r in regs if f(r))
out={"trainable":{"A1":237537,"A4w":par.get("A4w",2693121),"A4h":par.get("A4h"),"A4p":par.get("A4p")}}
print("trainable params:",out["trainable"])
print("\n%-12s %6s %6s %6s %6s %6s | %6s %6s %6s"%("region","A0","A4w0","A1_5","A4h_5","A4w_5","A1_20","A4h_20","A4w_20"))
for r in regs:
    print("%-12s %.3f  %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f"%(r,fu[(r,None,"A0")],rc[(r,5,"A4w0")],fu[(r,5,"A1")],rc[(r,5,"A4h")],fu[(r,5,"A4w")],fu[(r,20,"A1")],rc[(r,20,"A4h")],fu[(r,20,"A4w")]))
mac=lambda f: sum(f(r) for r in regs)/len(regs)
print("\nmacro A4w0 %.4f | A4h K5 %.4f K20 %.4f | A4p K5 %.4f K20 %.4f"%(mac(lambda r:rc[(r,5,"A4w0")]),mac(lambda r:rc[(r,5,"A4h")]),mac(lambda r:rc[(r,20,"A4h")]),mac(lambda r:rc[(r,5,"A4p")]),mac(lambda r:rc[(r,20,"A4p")])))
v={}
v["raw_adaptation_hurts_raw_K5"]=cnt(lambda r: rc[(r,5,"A4w0")]-fu[(r,5,"A4w")]>=0.01); v["raw_adaptation_hurts_raw_K20"]=cnt(lambda r: rc[(r,5,"A4w0")]-fu[(r,20,"A4w")]>=0.01)
v["A4h_beats_A4w_K5"]=cnt(lambda r: rc[(r,5,"A4h")]-fu[(r,5,"A4w")]>=0.01)
v["A1_beats_A4h_K5"]=cnt(lambda r: fu[(r,5,"A1")]-rc[(r,5,"A4h")]>=0.01); v["A1_beats_A4h_K20"]=cnt(lambda r: fu[(r,20,"A1")]-rc[(r,20,"A4h")]>=0.01)
v["A4h_within_001_of_A1_K5"]=cnt(lambda r: abs(fu[(r,5,"A1")]-rc[(r,5,"A4h")])<0.01)
v["A1_beats_A4w0_K5"]=cnt(lambda r: fu[(r,5,"A1")]-rc[(r,5,"A4w0")]>=0.01); v["A0_beats_A4w0"]=cnt(lambda r: fu[(r,None,"A0")]-rc[(r,5,"A4w0")]>=0.01)
v["random_A1_beats_A4w_K5"]=cnt(lambda r: rnd[(r,5,"A1")]-rnd[(r,5,"A4w")]>=0.01); v["random_A1_beats_A4h_K5"]=cnt(lambda r: rnd[(r,5,"A1")]-rnd[(r,5,"A4h")]>=0.01)
v["random_A1_gt_A0_K5"]=cnt(lambda r: rnd[(r,5,"A1")]>rnd[(r,None,"A0")]); v["strat_A1_gt_A0_K5"]=cnt(lambda r: fu[(r,5,"A1")]>fu[(r,None,"A0")])
v["fe_A1_beats_A4w_K5"]=cnt(lambda r: fe[(r,5,"A1")]-fe[(r,5,"A4w")]>=0.01); v["fe_A1_beats_A4w_K20"]=cnt(lambda r: fe[(r,20,"A1")]-fe[(r,20,"A4w")]>=0.01)
for k,val in v.items(): print(f"{k:34s} {val}/8")
out["verdicts"]=v; out["macro"]={"A0":mac(lambda r:fu[(r,None,"A0")]),"A4w0":mac(lambda r:rc[(r,5,"A4w0")]),"A1_5":mac(lambda r:fu[(r,5,"A1")]),"A4h_5":mac(lambda r:rc[(r,5,"A4h")]),"A4w_5":mac(lambda r:fu[(r,5,"A4w")]),"A1_20":mac(lambda r:fu[(r,20,"A1")]),"A4h_20":mac(lambda r:rc[(r,20,"A4h")]),"A4w_20":mac(lambda r:fu[(r,20,"A4w")]),"rand_A1_5":mac(lambda r:rnd[(r,5,"A1")]),"rand_A4w_5":mac(lambda r:rnd[(r,5,"A4w")]),"rand_A4h_5":mac(lambda r:rnd[(r,5,"A4h")])}
json.dump(out,open(R/"ms97_verdict.json","w"),indent=1)
