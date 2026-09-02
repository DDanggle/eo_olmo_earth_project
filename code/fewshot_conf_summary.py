#!/usr/bin/env python3
"""확증 few-shot 결과 요약 + 등록 판정(config/fewshot_a1_vs_a4_prereg_v0.json confirmatory_protocol_registered_now)."""
import json,glob,collections,sys
from pathlib import Path
root=Path("/home/work/data/olmoearth/artifacts/fewshot_confirmatory")
out={}
for f in sorted(root.glob("*/report_*.json")):
    d=json.load(open(f)); agg=collections.defaultdict(list)
    for r in d["runs"]: agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"])
    tag=f.parent.name; out[tag]={}
    print("==",tag,len(d["runs"]),"runs")
    regs=sorted({k[0] for k in agg})
    for reg in regs:
        m={}
        for (rg,K,arm),v in agg.items():
            if rg==reg: m[f"{arm}{'' if K is None else K}"]=(sum(v)/len(v),len(v))
        out[tag][reg]={k:v[0] for k,v in m.items()}
        print("  %-12s"%reg," ".join(f"{k}={v[0]:.3f}(n{v[1]})" for k,v in sorted(m.items())))
    if all(f"A1{K}" in out[tag].get(r,{}) and f"A4w{K}" in out[tag].get(r,{}) for r in regs for K in (5,20)):
        w5=sum(1 for r in regs if out[tag][r]["A15"]-out[tag][r]["A4w5"]>=0.01); w20=sum(1 for r in regs if out[tag][r]["A120"]-out[tag][r]["A4w20"]>=0.01)
        l5=sum(1 for r in regs if out[tag][r]["A4w5"]-out[tag][r]["A15"]>=0.01); l20=sum(1 for r in regs if out[tag][r]["A4w20"]-out[tag][r]["A120"]>=0.01)
        tie20=sum(1 for r in regs if out[tag][r]["A4w20"]-out[tag][r]["A120"]>=-0.01)
        macro={k:sum(out[tag][r][k] for r in regs)/len(regs) for k in ("A0","A15","A4w5","A120","A4w20") if all(k in out[tag][r] for r in regs)}
        print("   macro:",{k:round(v,4) for k,v in macro.items()})
        print(f"   A1>A4w(+.01): K5 {w5}/{len(regs)}  K20 {w20}/{len(regs)} | A4w>A1: K5 {l5} K20 {l20} | A4w>=A1-.01 at K20: {tie20}")
        print("   verdict reuse_wins(>=6/8 at K5):", w5>=6, "| crossover(K5>=6 & K20 tie/win>=5):", w5>=6 and tie20>=5, "| retrain_wins:", l5>=6 and l20>=6)
        ind=out[tag].get("indonesia",{}); print("   indonesia:",{k:round(v,3) for k,v in ind.items()})
json.dump(out,open(root/"summary.json","w"),indent=1)
