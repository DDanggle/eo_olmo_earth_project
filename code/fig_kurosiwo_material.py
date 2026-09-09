#!/usr/bin/env python3
"""KuroSiwo before/after material: test tiles with flood pixels -> S1 VV (pre_2, post) dB images, flood label, decoder probs on stale2 / GRU-updated / teacher3."""
import json,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); C=ROOT/"kurosiwo_s1_cache"; NPY=ROOT/"kurosiwo_npy"; OUT=ROOT/"artifacts/kurosiwo"; dev=torch.device("cuda")
src=open(ROOT/"code/kurosiwo_pipeline.py").read(); i=src.index("def conv_bn"); j=src.index("MODULES="); exec(src[i:j])
META=[json.loads(l) for l in (C/"meta.jsonl").read_text().splitlines() if l]; test=[m for m in META if m["split"]=="test"]
def L(kind,s): return np.load(C/f"{kind}_fp16"/f"{s}.npy").astype("float32")
ck=torch.load(OUT/"decoder_seed1.pt",map_location="cpu"); dec=EmbDecoder(768).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval(); mu,sd=ck["mu"],ck["sd"]
g=GRU().to(dev); g.load_state_dict(torch.load(OUT/"updater_gru_seed1.pt",map_location="cpu")["model_state"]); g.eval()
def prob(X):
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16): return torch.sigmoid(dec(((X-mu)/sd).to(dev)).float()).cpu().squeeze(1).numpy()
cands=sorted([m for m in test if m["pflood"]>3 and np.isfinite(L("teacher3",m["id"])).all()],key=lambda m:-m["pflood"])
pick=[cands[k] for k in (0,3,8,15,25,40) if k<len(cands)]
out={}
for m in pick:
    s=m["id"]; T=torch.from_numpy(L("teacher3",s))[None]; M=torch.from_numpy(L("stale2",s))[None]; P=torch.from_numpy(L("single",s)[2])[None]
    with torch.no_grad(): upd=g(M.to(dev),P.to(dev)).float().cpu()
    raw=np.load(NPY/"raw_f32"/f"{s}.npy")[:,:,16:208,16:208]; db=lambda x: np.clip((10*np.log10(np.clip(x,1e-6,None))+25)/25,0,1)   # -25..0 dB -> 0..1
    mk=np.load(C/"mask_u8"/f"{s}.npy")
    out[s]={"vv_pre":db(raw[0,1]),"vv_post":db(raw[0,2]),"label":(mk==3).astype(np.float32),"p_stale":prob(M)[0],"p_gru":prob(upd)[0],"p_teacher":prob(T)[0],"meta":{"actid":m["actid"],"flood_date":m["flood_date"],"pflood":m["pflood"]}}
np.savez_compressed(ROOT/"artifacts/fig_kurosiwo_material.npz",ids=np.array([m["id"] for m in pick]),meta=json.dumps({m["id"]:out[m["id"]]["meta"] for m in pick}),**{f"{k}__{s}":v for s,d in out.items() for k,v in d.items() if k!="meta"}); print("saved",len(pick))
