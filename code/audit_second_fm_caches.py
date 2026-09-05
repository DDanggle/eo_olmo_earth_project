"""Label-free sanity audit of second-FM caches vs OlmoEarth: (1) input normalisation scale actually fed to each encoder, (2) Clay latlon/time values, (3) token statistics: per-channel std, dead channels, spatial autocorrelation (adjacent-token cosine vs random-token cosine), cross-tile variance; (4) decoder training curves (best val IoU) per cache."""
import json, sys, glob, math, numpy as np, torch
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"
ids=sorted(p.stem for p in (SRC/"emb_fp16").glob("*.npy"))[::400][:12]
def tokstats(cache):
    X=np.stack([np.load(ROOT/cache/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids])   # N,C,G,G
    N,C,G,_=X.shape; chstd=X.reshape(N,C,-1).transpose(1,0,2).reshape(C,-1).std(1); dead=int((chstd<1e-4).sum())
    T=torch.from_numpy(X); Tn=torch.nn.functional.normalize(T,dim=1)
    adj=float((Tn[:,:,:,1:]*Tn[:,:,:,:-1]).sum(1).mean()); rnd=float((Tn[:,:,0,0].unsqueeze(-1).unsqueeze(-1)*Tn[torch.randperm(N)]).sum(1).mean())
    tilemean=T.mean(dim=(2,3)); between=float(tilemean.std(0).mean()); within=float(T.std(dim=(2,3)).mean())
    return {"shape":[C,G,G],"ch_std_median":float(np.median(chstd)),"dead_channels":dead,"adjacent_token_cos":adj,"random_token_cos":rnd,"between_tile_std":between,"within_tile_std":within}
out={}
for c in ["sen12_pilot/holdout_chimanimani","olmo_cache_pool16","clay_cache_in256","clay_cache_native16","galileo_cache","prithvi_cache"]:
    try: out[c]=tokstats(c)
    except Exception as e: out[c]={"err":str(e)[:100]}
# decoder curves
for c in ["olmo_cache_pool16","clay_cache_in256","clay_cache_native16","galileo_cache","prithvi_cache"]:
    fs=glob.glob(str(ROOT/"bv1_runs"/c/"*.json")); bv=[json.load(open(f))["best_val_iou"] for f in fs]; be=[json.load(open(f))["best_val_epoch"] for f in fs]
    out[c].update({"best_val_iou_mean":float(np.mean(bv)) if bv else None,"best_epoch_mean":float(np.mean(be)) if be else None})
# input scale checks
sid=ids[0]; raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); out["raw_DN_stats"]={"mean":float(raw.mean()),"p99":float(np.percentile(raw,99))}
sys.path.insert(0,str(ROOT/"third_party/pydeps")); sys.path.insert(0,str(ROOT/"third_party/galileo"))
from src.data.utils import construct_galileo_input
s2=torch.from_numpy(raw[[0,1,2,4,5,6,3,7,8,9]]).permute(2,3,1,0).contiguous(); m=construct_galileo_input(s2=s2,months=torch.arange(12),normalize=True)
stx=m.space_time_x; out["galileo_input_after_normalize"]={"mean":float(stx[...,2:12].mean()),"std":float(stx[...,2:12].std()),"s1_cols_mean":float(stx[...,:2].mean()),"ndvi_mean":float(stx[...,12].mean()),"mask_seen_groups":torch.unique(m.space_time_mask).tolist(),"mask_zero_frac":float((m.space_time_mask==0).float().mean())}
import yaml
meta=yaml.safe_load(open(ROOT/"third_party/clay/configs/metadata.yaml")); out["clay_metadata_s2"]=str(meta["sentinel-2-l2a"])[:300]
(ROOT/"artifacts/second_fm_cache_audit.json").write_text(json.dumps(out,indent=1)); print(json.dumps(out,indent=1))
