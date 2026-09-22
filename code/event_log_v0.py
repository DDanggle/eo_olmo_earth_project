#!/usr/bin/env python3
"""Rule-based event-log baseline input (design doc §9.4 fourth arm): per tile and arrival step, the v0.3 latest-observation head's predicted area and quadrants. Thresholds are NOT applied here (applied at read time from the v0.3 validation thresholds). Inference only."""
import json, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import emb_stats_from_cache, ROOT
dev=torch.device("cuda"); W=slice(8,120); REGIONS=sys.argv[1].split(",") if len(sys.argv)>1 else ["hiroshima","thrissur","itogon","hokkaido"]; OUT=ROOT/"event_log_v0"; OUT.mkdir(exist_ok=True)
mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_hiroshima","holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return F.interpolate(s.f(x),size=(128,128),mode="bilinear",align_corners=False)
h=Head(); h.load_state_dict(torch.load(ROOT/"p1_causal_v0_3/head_last_seed1.pt",map_location="cpu")); h=h.to(dev).eval()
def quadrants(p):
    hh,ww=p.shape; tot=p.sum(); out=[]
    for name,sl in (("NW",(slice(0,hh//2),slice(0,ww//2))),("NE",(slice(0,hh//2),slice(ww//2,ww))),("SW",(slice(hh//2,hh),slice(0,ww//2))),("SE",(slice(hh//2,hh),slice(ww//2,ww)))):
        if tot and p[sl].sum()/tot>=0.25: out.append(name)
    return out
with torch.no_grad():
    for reg in REGIONS:
        D=ROOT/"arrival_v0"/reg; ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (D/"single_fp16"/f"{p.stem}.npy").exists()); rows=[]
        for s in ids:
            U=torch.from_numpy(np.load(D/"single_fp16"/f"{s}.npy").astype("float32")).to(dev); m=json.loads((D/"meta"/f"{s}.json").read_text())
            P=torch.sigmoid(h((U-mean)/sd).float()).squeeze(1).cpu().numpy()[:,W,W]>0.5
            rows.append({"tile":s,"dates":[d[:10] for d in m["dates"]],"clear":[float(v) for v in m["clear"]],"area":[int(p.sum()) for p in P],"quadrants":[quadrants(p) for p in P]})
        (OUT/f"{reg}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n"); print(reg,len(rows),"tiles",flush=True)
print("EVENT LOG V0 DONE")
