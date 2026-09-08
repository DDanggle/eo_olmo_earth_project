#!/usr/bin/env python3
"""Material for the explanation page: for the hiroshima fold, pick test tiles with landslide pixels and export per tile:
S2 RGB (last timestep), label, decoder probability on frozen e4 / GRU-updated state / teacher e12; plus per-timestep single-image RGB thumbnails (t=0..11)."""
import json,sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,"/home/work/data/olmoearth/code"); from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache
import torch.nn as nn
ROOT=Path("/home/work/data/olmoearth"); D=ROOT/"olmo_streaming_dev"; SRC=ROOT/"sen12_pilot/holdout_chimanimani"; dev=torch.device("cuda"); fold="holdout_hiroshima"
ids=[s for s in json.loads((ROOT/"sen12_gp_contract/t1_manifest.json").read_text())[fold]["test"] if (D/"teacher_fp16"/f"{s}.npy").exists()]
Y=np.stack([np.load(SRC/"mask_u8"/f"{s}.npy") for s in ids]); pos=Y.reshape(len(ids),-1).mean(1); order=np.argsort(-pos)
pick=[ids[i] for i in order[:40:5]]+[ids[i] for i in order[60:90:10]]   # mix of large and small positives
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
g=GRU().to(dev); g.load_state_dict(torch.load(ROOT/"artifacts/streaming_t1v"/f"{fold}_gru_seed1_best.pt",map_location="cpu")["model_state"]); g.eval()
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval(); mu,sd=emb_stats_from_cache(SRC,fold)
def prob(X):
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16): return torch.sigmoid(dec(((X-mu)/sd).to(dev)).float()).cpu().squeeze(1).numpy()
out={}
for s in pick:
    T=torch.from_numpy(np.load(D/"teacher_fp16"/f"{s}.npy").astype("float32")); S=torch.from_numpy(np.load(D/"single_fp16"/f"{s}.npy").astype("float32"))
    m=T[0:1].to(dev)
    with torch.no_grad():
        for c in (6,8,10,12): m=g(m,S[c-2:c].mean(0,keepdim=True).to(dev))
    raw=np.load(SRC/"raw_u16"/f"{s}.npy").astype("float32")   # (10,T,128,128) bands B02,B03,B04,...
    def rgb(t): x=raw[[2,1,0],t]; return np.clip(x/3000.0,0,1)
    out[s]={"rgb_last":rgb(raw.shape[1]-1),"rgb_t":np.stack([rgb(t) for t in range(raw.shape[1])])[:, :, ::4, ::4],"label":np.load(SRC/"mask_u8"/f"{s}.npy"),
            "p_frozen":prob(T[0:1])[0],"p_gru":prob(m.cpu())[0],"p_teacher":prob(T[4:5])[0],"p_c8":prob(T[2:3])[0]}
np.savez_compressed(ROOT/"artifacts/fig_streaming_material_hiroshima.npz",ids=np.array(pick),**{f"{k}__{s}":v for s,d in out.items() for k,v in d.items()}); print("saved",len(pick),"tiles")
