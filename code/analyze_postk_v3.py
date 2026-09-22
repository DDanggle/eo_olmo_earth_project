#!/usr/bin/env python3
"""v3: observations-to-detection curve. Positive tiles with >=4 post-event observations; per-tile IoU for ctrl and each post-k arm."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
fold=sys.argv[1]; BASE=ROOT/"frozen_sensitivity_v0"/fold; SEALED=ROOT/"sen12_pilot"/fold; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); W=slice(8,120)
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); model=EmbDecoder(ck["cin"]).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
mean,sd=emb_stats_from_cache(SEALED,fold); ids=sorted(p.stem for p in (BASE/"ctrl/emb_fp16").glob("*.npy"))
Y={s:(np.load(SEALED/"mask_u8"/f"{s}.npy")[W,W]>0) for s in ids}; pos=[s for s in ids if Y[s].any()]
flags={json.loads(l)["id"]:json.loads(l) for l in (BASE/"postkeep3/drop_positions.jsonl").read_text().splitlines() if l}
sel=[s for s in pos if s in flags and not flags[s]["fallback"] and flags[s]["n_post"]>=4]
@torch.no_grad()
def tile_iou(arm):
    out={}
    for i in range(0,len(sel),32):
        b=sel[i:i+32]; E=torch.from_numpy(np.stack([np.load(BASE/arm/"emb_fp16"/f"{s}.npy").astype("float32") for s in b])); X=(E-mean)/sd
        P=(torch.sigmoid(model(X.to(dev)).float()).cpu().squeeze(1).numpy()[:,W,W]>0.5)
        for s,p in zip(b,P): y=Y[s]; u=(p|y).sum(); out[s]=(float((p&y).sum()/u) if u else 1.0, bool(p.any()))
    return out
arms=["postnone","postkeep1","postkeep2","postkeep3","postlast1","ctrl"]; R={a:tile_iou(a) for a in arms}
rng=np.random.default_rng(0); res={"fold":fold,"n_analysis":len(sel),"n_positive_total":len(pos),"n_post_dist":{str(v):sum(1 for s in sel if flags[s]["n_post"]==v) for v in sorted({flags[s]["n_post"] for s in sel})},"arms":{}}
for a in arms:
    v=np.array([R[a][s][0] for s in sel]); det=np.mean([R[a][s][1] for s in sel]); bs=[rng.choice(v,len(v)).mean() for _ in range(2000)]
    res["arms"][a]={"mean_iou":float(v.mean()),"ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],"tile_detection_rate":float(det),"frac_of_ctrl":float(v.mean()/max(np.mean([R["ctrl"][s][0] for s in sel]),1e-9))}
res["postlast1_minus_postkeep1"]=res["arms"]["postlast1"]["mean_iou"]-res["arms"]["postkeep1"]["mean_iou"]
(BASE/"postk_v3.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("POSTK V3 DONE")
