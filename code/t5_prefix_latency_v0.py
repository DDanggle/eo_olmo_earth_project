#!/usr/bin/env python3
"""T5 v0: detection latency in days from prefix-window states (config/t5_prefix_latency_prereg_v0.json)."""
import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
fold="holdout_hiroshima"; SEALED=ROOT/"sen12_pilot"/fold; T=ROOT/"olmo_streaming_dev/teacher_fp16"; OUT=ROOT/"t5_prefix_latency_v0"; OUT.mkdir(exist_ok=True); dev=torch.device("cuda"); W=slice(8,120); CUT=[4,6,8,10,12]
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); model=EmbDecoder(ck["cin"]).to(dev); model.load_state_dict(ck["model_state"]); model.eval(); mean,sd=emb_stats_from_cache(SEALED,fold)
def kept(r): q=r["scl_clear_fraction"]; return sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12])
rows=[]
ids=[s for s in sorted(p.stem for p in T.glob("hiroshima_*.npy")) if (SEALED/"mask_u8"/f"{s}.npy").exists()]
with torch.no_grad():
    for s in ids:
        r=REC[s]; Y=np.load(SEALED/"mask_u8"/f"{s}.npy")[W,W]>0
        if not Y.any() or r.get("post_index") is None: continue
        k=kept(r); d=[datetime.fromisoformat(str(r["times"][i])[:19]) for i in k]; fp=next((j for j,i in enumerate(k) if i>=r["post_index"]),None)
        if fp is None or fp<2 or fp>10: continue
        S=torch.from_numpy(np.load(T/f"{s}.npy").astype("float32")); X=(S-mean)/sd; P=torch.sigmoid(model(X.to(dev)).float()).cpu().squeeze(1).numpy()[:,W,W]>0.5
        for ci,c in enumerate(CUT):
            p=P[ci]; u=(p|Y).sum(); iou=float((p&Y).sum()/u) if u else 1.0; n_post=sum(1 for j in range(c) if j>=fp); days=(d[c-1]-d[fp]).days
            rows.append({"id":s,"c":c,"n_post":n_post,"days_since_event":days,"iou":iou,"detected":bool(p.any()),"first_post_slot":fp})
(OUT/"rows.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n")
def agg(sel): return {"n":len(sel),"det_rate":float(np.mean([x["detected"] for x in sel])) if sel else None,"mean_iou":float(np.mean([x["iou"] for x in sel])) if sel else None}
bins=[("<0",-10**6,-1),("0-30",0,30),("31-60",31,60),("61-120",61,120),(">120",121,10**6)]
tiles=sorted({x["id"] for x in rows}); first={}
for t in tiles:
    xs=sorted([x for x in rows if x["id"]==t],key=lambda x:x["c"]); det=[x for x in xs if x["detected"] and x["n_post"]>0]; first[t]=det[0]["days_since_event"] if det else None
fd=[v for v in first.values() if v is not None]
res={"schema":"t5-prefix-latency-v0","n_tiles":len(tiles),"by_cutoff":{str(c):agg([x for x in rows if x["c"]==c]) for c in CUT},"by_n_post":{str(n):agg([x for x in rows if x["n_post"]==n]) for n in sorted({x["n_post"] for x in rows})},
     "by_days":{lab:agg([x for x in rows if lo<=x["days_since_event"]<=hi]) for lab,lo,hi in bins},"days_to_first_detection":{"n_detected":len(fd),"n_never":sum(1 for v in first.values() if v is None),"median":float(np.median(fd)) if fd else None,"p25":float(np.quantile(fd,.25)) if fd else None,"p75":float(np.quantile(fd,.75)) if fd else None},
     "pre_event_false_alarm_rate":agg([x for x in rows if x["n_post"]==0])}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("T5 PREFIX DONE")
