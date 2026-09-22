import numpy as np, json, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
C=Path("/home/work/data/olmoearth/kurosiwo_s1_cache"); ids=[json.loads(l)["id"] for l in open(C/"meta.jsonl") if l.strip()]
tr=[json.loads(l) for l in open(C/"meta.jsonl") if l.strip()]; tr=[m["id"] for m in tr if m["split"]=="train"]
bad=[s for s in tr[:200] if not np.isfinite(np.load(C/"stale2_fp16"/f"{s}.npy").astype("float32")).all()]; print("stale2 nonfinite in first 200 train:",len(bad))
M=torch.from_numpy(np.stack([np.load(C/"stale2_fp16"/f"{s}.npy") for s in tr[:16]]).astype("float32")); U=torch.from_numpy(np.stack([np.load(C/"single_fp16"/f"{s}.npy")[2] for s in tr[:16]]).astype("float32")); T=torch.from_numpy(np.stack([np.load(C/"teacher3_fp16"/f"{s}.npy") for s in tr[:16]]).astype("float32"))
print("finite",torch.isfinite(M).all().item(),torch.isfinite(U).all().item(),torch.isfinite(T).all().item(),"absmax",M.abs().max().item(),U.abs().max().item(),T.abs().max().item(),"std",T.std().item())
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
dev=torch.device("cuda"); g=GRU().to(dev); sc=T.std().item()
with torch.autocast("cuda",dtype=torch.bfloat16): out=g(M.to(dev),U.to(dev)).float()
print("fwd finite",torch.isfinite(out).all().item(),"loss",float(F.mse_loss(out,T.to(dev))/sc**2))
# scan all train tiles for nonfinite in any of the three arrays
nb={"stale2":0,"single":0,"teacher3":0}
for s in tr:
    for k in nb:
        if not np.isfinite(np.load(C/f"{k}_fp16"/f"{s}.npy").astype("float32")).all(): nb[k]+=1
print("nonfinite tiles over all train",nb,"of",len(tr))
