#!/usr/bin/env python3
"""Re-evaluate saved T1 student checkpoints under several frozen decoders (seed 1 = sealed control, seeds 2/3 = artifacts/control_seeds).
Usage: t1_reeval_decoders.py <fold> <ckpt.pt> [<ckpt.pt> ...]  -> artifacts/streaming_t1_reeval/<fold>_<ckptname>.json"""
import json,sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,"/home/work/data/olmoearth/code"); from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, metrics
sys.argv_module=None
ROOT=Path("/home/work/data/olmoearth"); D=ROOT/"olmo_streaming_dev"; dev=torch.device("cuda"); fold=sys.argv[1]; CUT=(4,6,8,10,12)
exec(open(ROOT/"code/streaming_update_train.py").read().split("model={")[0].split("class EMA")[1].join(["class EMA",""]) if False else "")  # (module classes are re-declared below to stay independent)
import torch.nn as nn
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
ids=[s for s in json.loads((ROOT/"sen12_gp_contract/t1_manifest.json").read_text())[fold]["test"] if (D/"teacher_fp16"/f"{s}.npy").exists()]
T=torch.from_numpy(np.stack([np.load(D/"teacher_fp16"/f"{s}.npy") for s in ids]).astype("float32")); S=torch.from_numpy(np.stack([np.load(D/"single_fp16"/f"{s}.npy") for s in ids]).astype("float32"))
U=torch.stack([S[:,c-2:c].mean(1) for c in CUT[1:]],1); Y=torch.from_numpy(np.stack([np.load(D/"mask_u8"/f"{s}.npy") for s in ids]).astype("float32")); del S
mu,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_chimanimani",fold)
decs=[]
for seed,path in ((1,ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt"),(2,ROOT/"artifacts/control_seeds"/f"{fold}_seed2_best.pt"),(3,ROOT/"artifacts/control_seeds"/f"{fold}_seed3_best.pt")):
    if path.exists(): ck=torch.load(path,map_location="cpu"); d=EmbDecoder(ck["cin"]).to(dev); d.load_state_dict(ck["model_state"]); d.eval(); decs.append((seed,d))
def ap(dec,X):
    out=[]
    with torch.no_grad():
        for i in range(0,len(X),32):
            with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(dec(((X[i:i+32]-mu)/sd).to(dev)).float()).cpu())
    return metrics(torch.cat(out).squeeze(1),Y)["auprc_exact"]
OUT=ROOT/"artifacts/streaming_t1_reeval"; OUT.mkdir(exist_ok=True)
for ck in sys.argv[2:]:
    st=torch.load(ck,map_location="cpu"); assert st["module"]=="gru", ck; g=GRU().to(dev); g.load_state_dict(st["model_state"]); g.eval()
    with torch.no_grad():
        outs=[]
        for i in range(0,len(T),16):
            m=T[i:i+16,0].to(dev)
            for k in range(4):
                with torch.autocast("cuda",dtype=torch.bfloat16): m=g(m,U[i:i+16,k].to(dev)).float()
            outs.append(m.cpu())
        student=torch.cat(outs)
    rep={"fold":fold,"ckpt":ck,"decoders":{}}
    for seed,d in decs:
        t,f,s_=ap(d,T[:,-1]),ap(d,T[:,0]),ap(d,student); rep["decoders"][f"decoder_seed{seed}"]={"teacher":t,"frozen":f,"student":s_,"recovery":(s_-f)/max(t-f,1e-9)}
    (OUT/f"{fold}_{Path(ck).stem}.json").write_text(json.dumps(rep,indent=1)); print(json.dumps(rep["decoders"]))
