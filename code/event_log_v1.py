#!/usr/bin/env python3
"""Event-log v1 (config/event_log_prereg_v1.json): sealed cross-region readout on causal prefix states, per region's own LOCO fold. Inference only."""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
dev=torch.device("cuda"); W=slice(8,120); REGIONS=sys.argv[1].split(",") if len(sys.argv)>1 else ["hiroshima","thrissur","itogon","hokkaido"]; OUT=ROOT/"event_log_v1"; OUT.mkdir(exist_ok=True)
def quadrants(p):
    hh,ww=p.shape; tot=p.sum(); out=[]
    for name,sl in (("NW",(slice(0,hh//2),slice(0,ww//2))),("NE",(slice(0,hh//2),slice(ww//2,ww))),("SW",(slice(hh//2,hh),slice(0,ww//2))),("SE",(slice(hh//2,hh),slice(ww//2,ww)))):
        if tot and p[sl].sum()/tot>=0.25: out.append(name)
    return out
with torch.no_grad():
    for reg in REGIONS:
        fold=f"holdout_{reg}"; mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot"/fold,fold); mean,sd=mean.to(dev),sd.to(dev)
        ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
        D=ROOT/"arrival_v0"/reg; ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (D/"prefix_fp16"/f"{p.stem}.npy").exists()); rows=[]
        for s in ids:
            T=torch.from_numpy(np.load(D/"prefix_fp16"/f"{s}.npy").astype("float32")).to(dev); m=json.loads((D/"meta"/f"{s}.json").read_text())
            Pr=torch.sigmoid(dec((T-mean)/sd).float()).squeeze(1).cpu().numpy()[:,W,W]; P=Pr>0.5
            rows.append({"tile":s,"dates":[d[:10] for d in m["dates"]],"clear":[float(v) for v in m["clear"]],"post_index":m["post_index"],"area":[int(p.sum()) for p in P],"prob_sum":[float(p.sum()) for p in Pr],"quadrants":[quadrants(p) for p in P]})
        (OUT/f"{reg}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n"); print(reg,len(rows),"tiles",flush=True)
print("EVENT LOG V1 DONE")
