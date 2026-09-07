#!/usr/bin/env python3
"""Aggregate T1 streaming-update runs: per fold x module, seed-mean downstream AP / macro IoU, recovery = (student-frozen)/(teacher-frozen), agreement."""
import json, glob, sys, statistics as S, collections
rows=collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else "artifacts/streaming_t1/*.json")):
    d=json.load(open(f)); ds=d["downstream_c12"]; t,fz,st=ds["teacher_full_reencode"]["auprc_exact"],ds["frozen_m4"]["auprc_exact"],ds["student"]["auprc_exact"]
    import os; arm=os.path.basename(f).replace(d["fold"]+"_","").rsplit("_seed",1)[0]; rows[(d["fold"],arm)].append({"seed":d["seed"],"teacher_ap":t,"frozen_ap":fz,"student_ap":st,"singles_ap":ds["singles_mean"]["auprc_exact"],"recovery":(st-fz)/max(t-fz,1e-9),"cos":d["agreement_c12"]["student"]["cosine"],"student_macro":ds["student"]["positive_patch_macro_iou"],"teacher_macro":ds["teacher_full_reencode"]["positive_patch_macro_iou"]})
out={}
for k,v in sorted(rows.items()):
    m={kk:round(S.mean(x[kk] for x in v),3) for kk in v[0] if kk!="seed"}; m["n_seeds"]=len(v); out[f"{k[0]}|{k[1]}"]=m
    print(k, "n",len(v), "teacher AP",m["teacher_ap"],"frozen",m["frozen_ap"],"student",m["student_ap"],"singles",m["singles_ap"],"recovery",m["recovery"],"cos",m["cos"])
json.dump(out,open("artifacts/streaming_t1_summary.json","w"),indent=1) if rows else None
