#!/usr/bin/env python3
"""Eval-only frozen-sensitivity metrics (prereg config/frozen_sensitivity_prereg_v0.json).
Applies the sealed p4_native_control readout (frozen, seed 1) to every arm cache of a fold, shifts spatial-arm predictions back to the
original pixel frame, and compares decisions to the ctrl arm on the common window rows/cols 8..120. Label-free flip rate is primary."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, metrics, ROOT
ap=argparse.ArgumentParser(); ap.add_argument("--fold",required=True); ap.add_argument("--outroot",default="frozen_sensitivity_v0"); ap.add_argument("--prereg",default=str(ROOT/"config/frozen_sensitivity_prereg_v0.json")); a=ap.parse_args()
P=json.loads(Path(a.prereg).read_text()); ARMS=P["arms"]; BASE=ROOT/a.outroot/a.fold; SEALED=ROOT/"sen12_pilot"/a.fold; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{a.fold}_seed1_best.pt",map_location="cpu"); model=EmbDecoder(ck["cin"]).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
mean,sd=emb_stats_from_cache(SEALED,a.fold)
ids=sorted(p.stem for p in (BASE/"ctrl/emb_fp16").glob("*.npy"))
W=slice(8,120)
def load_arm(arm): return np.stack([np.load(BASE/arm/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids])
@torch.no_grad()
def predict(E,bs=32):
    X=(torch.from_numpy(E)-mean)/sd; out=[]
    for i in range(0,len(X),bs): out.append(torch.sigmoid(model(X[i:i+bs].to(dev)).float()).cpu())
    return torch.cat(out).squeeze(1).numpy()
def unshift(pm,dy,dx):
    o=np.full_like(pm,np.nan); o[:,dy:,dx:]=pm[:,:128-dy,:128-dx]; return o
Y=np.stack([np.load(SEALED/"mask_u8"/f"{s}.npy") for s in ids]).astype("float32")
E_ctrl=load_arm("ctrl"); P_ctrl=predict(E_ctrl); B_ctrl=P_ctrl>0.5
E_sealed=np.stack([np.load(SEALED/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids])
def tokcos(A,B): 
    a=A.reshape(len(A),768,-1); b=B.reshape(len(B),768,-1); return float(np.mean(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)))
res={"schema":"frozen-sensitivity-eval-v0","fold":a.fold,"n_tiles":len(ids),"window":"8:120","ctrl_vs_sealed_token_cos":tokcos(E_ctrl,E_sealed),"arms":{}}
Yw=torch.from_numpy(Y[:,W,W]); pos=Y[:,W,W].sum((1,2))>0
for arm,spec in ARMS.items():
    if not (BASE/arm/"extract_audit.json").exists(): res["arms"][arm]={"status":"missing"}; continue
    E=load_arm(arm); dy,dx=spec["offset_px"]; Pm=unshift(predict(E),dy,dx); Pw=Pm[:,W,W]; assert not np.isnan(Pw).any()
    Bw=Pw>0.5; Cw=B_ctrl[:,W,W]; flip=(Bw!=Cw); either=Bw|Cw
    r={"offset_px":[dy,dx],"time":spec["time"],"decision_flip_rate":float(flip.mean()),"flip_rate_among_predicted_positive":float(flip[either].mean()) if either.any() else 0.0,
       "prob_mae":float(np.abs(Pw-P_ctrl[:,W,W]).mean()),"per_tile_flip_p50":float(np.median(flip.mean((1,2)))),"per_tile_flip_p90":float(np.quantile(flip.mean((1,2)),.9))}
    if (dy,dx)==(0,0): r["token_cos_vs_ctrl"]=tokcos(E,E_ctrl)
    elif dy%4==0 and dx%4==0: r["token_cos_vs_ctrl_realigned"]=tokcos(E[:,:,:32-dy//4,:32-dx//4],E_ctrl[:,:,dy//4:,dx//4:])
    m=metrics(torch.from_numpy(Pw),Yw); r["window_macro_iou"]=m["positive_patch_macro_iou"]; r["window_micro_iou"]=m["iou"]; r["window_auprc"]=m["auprc_exact"]
    res["arms"][arm]=r
c=res["arms"]["ctrl"]; ys=res["arms"].get("s40_xy",{}).get("decision_flip_rate")
if ys is not None:
    res["per_arm_rule"]={k:{"ratio_to_s40":v["decision_flip_rate"]/max(ys,1e-9),"present":bool(v["decision_flip_rate"]>=2*ys and v["decision_flip_rate"]>=0.02),"macro_iou_delta_vs_ctrl":v["window_macro_iou"]-res["arms"]["ctrl"]["window_macro_iou"]} for k,v in res["arms"].items() if "decision_flip_rate" in v}
    s20=[res["arms"][k]["decision_flip_rate"] for k in ("s20_y","s20_x","s20_xy") if k in res["arms"] and "decision_flip_rate" in res["arms"][k]]
    tm=[res["arms"][k]["decision_flip_rate"] for k in ("t_plus1","t_shuffle") if "decision_flip_rate" in res["arms"].get(k,{})]
    res["rules"]={"yardstick_s40_flip":ys,"spatial_axis_present":bool(s20) and np.mean(s20)>=2*ys and np.mean(s20)>=0.02,"time_axis_present":bool(tm) and max(tm)>=2*ys and max(tm)>=0.02,
                  "time_axis_inert":all(res["arms"][k]["decision_flip_rate"]<0.005 and res["arms"][k].get("token_cos_vs_ctrl",0)>.995 for k in ("t_synth","t_plus1","t_shuffle") if "decision_flip_rate" in res["arms"].get(k,{})),
                  "control_ok":res["ctrl_vs_sealed_token_cos"]>=.99}
def _py(o):
    import numpy as _np
    return bool(o) if isinstance(o,(_np.bool_,)) else float(o) if isinstance(o,_np.floating) else int(o) if isinstance(o,_np.integer) else str(o)
(BASE/"eval_summary.json").write_text(json.dumps(res,indent=1,default=_py)); print(json.dumps(res,indent=1,default=_py)); print("FROZEN SENSITIVITY EVAL DONE")
