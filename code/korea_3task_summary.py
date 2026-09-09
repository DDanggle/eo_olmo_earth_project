#!/usr/bin/env python3
"""Summarise artifacts/korea_3task/<run>/report.json: per task, per K/draw -> mean±sd over seeds of the primary metric for CACHE_K and RAW_K, FULL refs, cost."""
import json,sys,numpy as np
from pathlib import Path
rp=Path(sys.argv[1] if len(sys.argv)>1 else "/home/work/data/olmoearth/artifacts/korea_3task/run_v1/report.json"); R=json.loads(rp.read_text())
PM={"land_cover":"cluster_macro_miou","logged":"all_cluster_ap","landslide":"all_cluster_ap"}
out={"n":R["n"],"tasks":{}}
for task,T in R["tasks"].items():
    pm=PM[task]; o={"primary_metric":pm}
    for ref in ("FULL_CACHE","FULL_RAW"):
        if ref in T: o[ref]={"test":T[ref]["test"][pm],"pos_cluster_iou":T[ref]["test"].get("positive_cluster_macro_iou"),"gpu_s":T[ref]["train"]["gpu_s"],"train_bytes":T[ref]["train"]["train_bytes_read"],"test_bytes":T[ref]["test_bytes_read"],"clusters":{k:(v.get("miou") if pm=="cluster_macro_miou" else v.get("ap")) for k,v in T[ref]["test"]["clusters"].items()}}
    for K in (5,20):
        for draw in ("random","posaware"):
            row={}
            for arm in ("CACHE_K","RAW_K"):
                vals=[T[k]["test"][pm] for k in T if k.startswith(f"{arm}_K{K}_{draw}_") and T[k]["test"][pm] is not None]
                if vals: row[arm]={"mean":float(np.mean(vals)),"sd":float(np.std(vals)),"n":len(vals),"vals":[round(v,4) for v in vals],"gpu_s":float(np.mean([T[k]["train"]["gpu_s"] for k in T if k.startswith(f"{arm}_K{K}_{draw}_")])),"support_pos":[T[k]["support"]["support_pos_chips"] for k in T if k.startswith(f"{arm}_K{K}_{draw}_")]}
            if row:
                if "CACHE_K" in row and "RAW_K" in row: row["cache_minus_raw"]=row["CACHE_K"]["mean"]-row["RAW_K"]["mean"]; row["cache_wins_seeds"]=int(sum(c>r for c,r in zip(row["CACHE_K"]["vals"],row["RAW_K"]["vals"])))
                o[f"K{K}_{draw}"]=row
    out["tasks"][task]=o
print(json.dumps(out,indent=1))
