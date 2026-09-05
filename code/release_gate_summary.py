"""Apply the registered A gates to a release-migration screen report (8 folds): per-fold mean AP/IoU per arm, compatibility gate (AP within .02 AND IoU within 5% of R0, >=6/8), beats-identity (>= .01, >=6/8)."""
import json, sys, numpy as np
from collections import defaultdict
from pathlib import Path
for path in sys.argv[1:]:
    r=json.load(open(path)); agg=defaultdict(list)
    for x in r["runs"]: agg[(x["region"],x["arm"])].append((x["eval"]["tie_ap"],x["eval"]["iou_fp_matched"] or 0))
    regs=sorted({k[0] for k in agg}); arms=[a for a in ["R0_old_reference","R1_identity","R2_mean_shift","R3_procrustes","R4_affine_ridge","R5_spatial_stitch","R6_new_native_head"] if any(k[1]==a for k in agg)]
    M={k:float(np.mean([t[0] for t in v])) for k,v in agg.items()}; I={k:float(np.mean([t[1] for t in v])) for k,v in agg.items()}
    print("==",path,"new_cache=",r.get("new_cache","task2_cache_v12"),"folds",len(regs))
    print("fold | "+" | ".join(a.split("_")[0] for a in arms)+"  ||  IoU: "+" ".join(a.split("_")[0] for a in arms))
    f=lambda D,g,a: ("%.3f"%D[(g,a)]) if (g,a) in D else "  -  "
    for g in regs: print(g.replace("task2_",""),"| "+" | ".join(f(M,g,a) for a in arms),"|| "+" ".join(f(I,g,a) for a in arms))
    print("macro | "+" | ".join("%.3f"%np.mean([M[(g,a)] for g in regs if (g,a) in M]) for a in arms),"|| "+" ".join("%.3f"%np.mean([I[(g,a)] for g in regs if (g,a) in I]) for a in arms))
    out={}
    for b in arms[2:]:
        if b=="R6_new_native_head": continue
        comp=sum(abs(M[(g,b)]-M[(g,"R0_old_reference")])<=0.02 and I[(g,b)]>=0.95*I[(g,"R0_old_reference")] for g in regs); gain=sum(M[(g,b)]-M[(g,"R1_identity")]>=0.01 for g in regs)
        out[b]={"compat_both":comp,"beats_identity":gain,"macro_ap":float(np.mean([M[(g,b)] for g in regs])),"macro_iou":float(np.mean([I[(g,b)] for g in regs])),"ap_retention":float(np.mean([M[(g,b)] for g in regs])/np.mean([M[(g,"R0_old_reference")] for g in regs]))}
        print(b,"compat(AP±.02 & IoU 95%%) %d/%d | beats identity %d/%d | AP retention %.3f"%(comp,len(regs),gain,len(regs),out[b]["ap_retention"]))
    Path(path).with_name("gate_summary.json").write_text(json.dumps({"folds":regs,"mean_ap":{f"{g}|{a}":M[(g,a)] for (g,a) in M},"mean_iou":{f"{g}|{a}":I[(g,a)] for (g,a) in I},"gates":out},indent=1))
