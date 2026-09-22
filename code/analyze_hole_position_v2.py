#!/usr/bin/env python3
"""v2 analysis: per-tile IoU delta (arm - ctrl) for position-controlled 3-hole arms, restricted to positive tiles where the hole was placed
without fallback (first_post_slot in [3,9]). Sealed readout, window 8..120. Applies decision_rules_v2 descriptively."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
fold=sys.argv[1]; BASE=ROOT/"frozen_sensitivity_v0"/fold; SEALED=ROOT/"sen12_pilot"/fold; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); W=slice(8,120)
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); model=EmbDecoder(ck["cin"]).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
mean,sd=emb_stats_from_cache(SEALED,fold); ids=sorted(p.stem for p in (BASE/"ctrl/emb_fp16").glob("*.npy"))
Y={s:(np.load(SEALED/"mask_u8"/f"{s}.npy")[W,W]>0) for s in ids}; pos=[s for s in ids if Y[s].any()]
@torch.no_grad()
def tile_iou(arm):
    out={}
    for i in range(0,len(pos),32):
        b=pos[i:i+32]; E=torch.from_numpy(np.stack([np.load(BASE/arm/"emb_fp16"/f"{s}.npy").astype("float32") for s in b])); X=(E-mean)/sd
        P=(torch.sigmoid(model(X.to(dev)).float()).cpu().squeeze(1).numpy()[:,W,W]>0.5)
        for s,p in zip(b,P): y=Y[s]; u=(p|y).sum(); out[s]=float((p&y).sum()/u) if u else 1.0
    return out
ctrl=tile_iou("ctrl"); res={"fold":fold,"n_positive":len(pos),"arms":{}}
for arm in ("hole_pre3","hole_post3","hole_span3","hole_late3"):
    if not (BASE/arm/"extract_audit.json").exists(): res["arms"][arm]="missing"; continue
    flags={json.loads(l)["id"]:json.loads(l) for l in (BASE/arm/"drop_positions.jsonl").read_text().splitlines() if l}
    a=tile_iou(arm); ok=[s for s in pos if s in flags and not flags[s]["fallback"]]; d=np.array([a[s]-ctrl[s] for s in ok]); dall=np.array([a[s]-ctrl[s] for s in pos])
    res["arms"][arm]={"n_analysis":len(ok),"n_fallback_positive":len(pos)-len(ok),"mean_delta_iou":float(d.mean()) if len(d) else None,"median_delta_iou":float(np.median(d)) if len(d) else None,"ci95_boot":None,"mean_delta_all_positive":float(dall.mean())}
    if len(d)>10:
        rng=np.random.default_rng(0); bs=[rng.choice(d,len(d)).mean() for _ in range(2000)]; res["arms"][arm]["ci95_boot"]=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]
pre=res["arms"].get("hole_pre3",{}); post=res["arms"].get("hole_post3",{})
if isinstance(pre,dict) and isinstance(post,dict) and pre.get("mean_delta_iou") is not None:
    p,q=pre["mean_delta_iou"],post["mean_delta_iou"]
    res["rules_v2"]={"information_explanation":bool(abs(p)<=0.02 and q<=-3*abs(p) and q<p),"structure_explanation":bool(p<=-0.05),"pre":p,"post":q}
(BASE/"hole_position_v2.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("HOLE V2 DONE")
