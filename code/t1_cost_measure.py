#!/usr/bin/env python3
"""Measured (not counted) cost of keeping a cache alive. On a fixed set of N tiles, times on ONE GPU with nothing else of ours running:
  A. full re-encode at every cutoff  (windows 0:6, 0:8, 0:10, 0:12)      -> encoder GPU-s, raw bytes read (all 12 timesteps each time)
  B. streaming: single-acquisition encode of the 8 new images + GRU steps -> encoder GPU-s + updater GPU-s, raw bytes read (new images only)
  C. final map only: one window 0:12                                       -> the 'you only need the last map' reference
Initial encode (0:4) is reported separately and added to every path. Reports wall-clock with cuda synchronisation, per tile."""
import argparse, json, time, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0,"/home/work/data/olmoearth/code")
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=64); ap.add_argument("--gru-ckpt",required=True); ap.add_argument("--out",default="artifacts/cost/t1_cost_measured.json"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
import torch.nn as nn
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; dev=torch.device("cuda"); PATCH=4
from datetime import datetime
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
def real_ts(sid,T):
    r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:T]); return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
g=GRU().to(dev); g.load_state_dict(torch.load(a.gru_ckpt,map_location="cpu")["model_state"]); g.eval()
ids=sorted(p.stem for p in (SRC/"raw_u16").glob("*.npy"))[:a.n]
def pooled(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,_,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        return ((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float()
def window(cube,ts,t0,t1):
    feat=torch.empty((768,32,32),device=dev)
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)): feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=pooled(np.ascontiguousarray(cube[:,t0:t1,y0:y0+64,x0:x0+64]),ts[t0:t1])
    return feat
def singles_batched(cube,ts,t0,t1):
    """All (t1-t0) single-timestep windows x 4 crops in ONE forward call (batch = 4*(t1-t0))."""
    inputs=[]
    for t in range(t0,t1):
        for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
            image=torch.from_numpy(np.ascontiguousarray(cube[:,t:t+1,y0:y0+64,x0:x0+64])).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(ts[t],ts[t])])}; w.normalizer(inp,{}); inputs.append(inp)
    sample,_,_=w._prepare_modality_inputs(ModelContext(inputs=inputs,metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1)).permute(0,3,1,2).float()   # (B,768,16,16)
    out=[]
    for i in range(t1-t0):
        feat=torch.empty((768,32,32),device=dev)
        for j,(y0,x0) in enumerate(((0,0),(0,64),(64,0),(64,64))): feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=pooled[i*4+j]
        out.append(feat)
    return out
def window_batched(cube,ts,t0,t1):
    """One window [t0:t1], 4 crops in ONE forward call (fair batched baseline for re-encode / final-only paths)."""
    inputs=[]
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        image=torch.from_numpy(np.ascontiguousarray(cube[:,t0:t1,y0:y0+64,x0:x0+64])).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts[t0:t1]])}; w.normalizer(inp,{}); inputs.append(inp)
    sample,_,_=w._prepare_modality_inputs(ModelContext(inputs=inputs,metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1)).permute(0,3,1,2).float()
    feat=torch.empty((768,32,32),device=dev)
    for j,(y0,x0) in enumerate(((0,0),(0,64),(64,0),(64,64))): feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=pooled[j]
    return feat
def timed(fn):
    torch.cuda.synchronize(); t=time.perf_counter(); r=fn(); torch.cuda.synchronize(); return time.perf_counter()-t, r
res={"n_tiles":len(ids),"per_tile_s":{},"raw_bytes_per_tile":{}}
acc={"init_0_4":0,"init_0_4_batched":0,"A_reencode_6_8_10_12":0,"A_reencode_batched":0,"B_singles_8":0,"B_singles_8_batched":0,"B_gru_4steps":0,"C_final_0_12":0,"C_final_batched":0}; bytes_={}
with torch.no_grad():
    for i,sid in enumerate(ids):
        raw=np.load(SRC/"raw_u16"/f"{sid}.npy"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw.astype("float32"); ts=real_ts(sid,T)
        per_t_bytes=raw.nbytes//T
        if i==0: window(cube,ts,0,12)  # warm-up
        dt,m4=timed(lambda: window(cube,ts,0,4)); acc["init_0_4"]+=dt
        dt,_=timed(lambda: [window(cube,ts,0,c) for c in (6,8,10,12)]); acc["A_reencode_6_8_10_12"]+=dt
        dt,ab=timed(lambda: [window_batched(cube,ts,0,c) for c in (6,8,10,12)]); acc["A_reencode_batched"]+=dt
        dt,_=timed(lambda: window_batched(cube,ts,0,4)); acc["init_0_4_batched"]+=dt
        dt,cb=timed(lambda: window_batched(cube,ts,0,12)); acc["C_final_batched"]+=dt
        if i==1: print("window batched-vs-loop max|diff| c12:",float((cb-window(cube,ts,0,12)).abs().max()),flush=True)
        dt,singles=timed(lambda: [window(cube,ts,t,t+1) for t in range(4,12)]); acc["B_singles_8"]+=dt
        dt,sb=timed(lambda: singles_batched(cube,ts,4,12)); acc["B_singles_8_batched"]+=dt
        if i==1: print("batched-vs-loop max|diff|",float(max((a-b).abs().max() for a,b in zip(sb,singles))),flush=True)
        def gru_steps():
            m=m4.unsqueeze(0)
            for k in range(4): m=g(m,torch.stack(singles[2*k:2*k+2]).mean(0,keepdim=True))
            return m
        dt,_=timed(gru_steps); acc["B_gru_4steps"]+=dt
        dt,_=timed(lambda: window(cube,ts,0,12)); acc["C_final_0_12"]+=dt
        bytes_={"init_0_4":4*per_t_bytes,"A_reencode_6_8_10_12":(6+8+10+12)*per_t_bytes,"B_singles_8":8*per_t_bytes,"C_final_0_12":12*per_t_bytes}
res["per_tile_s"]={k:v/len(ids) for k,v in acc.items()}; res["raw_bytes_per_tile"]=bytes_
p=res["per_tile_s"]; res["paths_per_tile_s"]={"A_all_maps_reencode":p["init_0_4"]+p["A_reencode_6_8_10_12"],"B_all_maps_streaming":p["init_0_4"]+p["B_singles_8"]+p["B_gru_4steps"],"B_all_maps_streaming_batched":p["init_0_4"]+p["B_singles_8_batched"]+p["B_gru_4steps"],"C_final_map_only":p["C_final_0_12"],"A_all_maps_reencode_batched":p["init_0_4_batched"]+p["A_reencode_batched"],"B_all_maps_streaming_batched_fair":p["init_0_4_batched"]+p["B_singles_8_batched"]+p["B_gru_4steps"],"C_final_map_only_batched":p["C_final_batched"]}
res["note"]="wall-clock incl. crop batching overhead on one H200; measured with other jobs possibly present (see gpu_procs)"; res["gpu_procs_at_start"]=int(torch.cuda.device_count())
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(res,indent=1)); print(json.dumps(res["paths_per_tile_s"]), json.dumps(res["raw_bytes_per_tile"]))
