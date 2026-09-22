#!/usr/bin/env python3
"""MS-123 follow-up: is the gap-arm IoU loss explained by whether the dropped timesteps removed the post-event observations?
Recomputes the seeded drop per tile (same rng as extract_olmo_perturb.py), maps kept-12 slots to original 15 indices, and
splits per-tile IoU delta (arm - ctrl) by post-event coverage. Eval-only, sealed readout, window 8..120."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
fold=sys.argv[1]; BASE=ROOT/"frozen_sensitivity_v0"/fold; SEALED=ROOT/"sen12_pilot"/fold; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); W=slice(8,120)
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); model=EmbDecoder(ck["cin"]).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
mean,sd=emb_stats_from_cache(SEALED,fold); ids=sorted(p.stem for p in (BASE/"ctrl/emb_fp16").glob("*.npy"))
def kept12(sid):
    q=REC[sid]["scl_clear_fraction"]; return sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12])
def dropped_slots(sid,mode):
    T=12; rng=np.random.default_rng(int.from_bytes(sid.encode()[-8:],"little")%(2**32)); k=int(mode[-1]); s0=int(rng.integers(0,T-k+1))
    return set(rng.choice(T,k,replace=False).tolist()) if mode.startswith("rand") else set(range(s0,s0+k))
@torch.no_grad()
def tile_iou(arm):
    out=[]
    for i in range(0,len(ids),32):
        E=torch.from_numpy(np.stack([np.load(BASE/arm/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids[i:i+32]])); X=(E-mean)/sd
        P=(torch.sigmoid(model(X.to(dev)).float()).cpu().squeeze(1).numpy()[:,W,W]>0.5)
        Y=np.stack([np.load(SEALED/"mask_u8"/f"{s}.npy") for s in ids[i:i+32]])[:,W,W]>0
        for p,y in zip(P,Y):
            inter=(p&y).sum(); uni=(p|y).sum(); out.append((inter/uni if uni else 1.0, y.sum()>0))
    return out
ctrl=tile_iou("ctrl"); res={"fold":fold,"n":len(ids),"arms":{}}; fine={}
for arm in ("gap_r3","gap_r6","gap_c3","gap_c6"):
    a=tile_iou(arm); groups={"post_removed_all":[],"post_removed_some":[],"post_intact":[]}; nopost=0
    for sid,(ci,pos),(ai,_) in zip(ids,ctrl,a):
        if not pos: continue
        r=REC[sid]; post=r.get("post_index")
        if post is None: nopost+=1; continue
        k=kept12(sid); dslots=dropped_slots(sid,arm[4:]); dropped_orig={k[s] for s in dslots}; post_kept={i for i in k if i>=post}
        if not post_kept: nopost+=1; continue
        rem=len(post_kept&dropped_orig); g="post_removed_all" if rem==len(post_kept) else "post_removed_some" if rem else "post_intact"
        groups[g].append(ai-ci)
        remaining=len(post_kept)-rem; kept_sorted=sorted(k); first_post_slot=next((j for j,i in enumerate(kept_sorted) if i>=post),None)
        boundary=bool(first_post_slot is not None and (first_post_slot in dslots or (first_post_slot-1) in dslots))
        fine.setdefault(arm,[]).append({"delta":float(ai-ci),"remaining_post":remaining,"post_total":len(post_kept),"boundary_covered":boundary,"frac_post_removed":rem/len(post_kept)})
    res["arms"][arm]={g:{"n":len(v),"mean_delta_iou":float(np.mean(v)) if v else None} for g,v in groups.items()}; res["arms"][arm]["no_post_info"]=nopost
    allp=[ai-ci for (ci,pos),(ai,_) in zip(ctrl,a) if pos]; res["arms"][arm]["all_positive_mean_delta"]=float(np.mean(allp))
for arm,rows in fine.items():
    d=np.array([r["delta"] for r in rows]); fr=np.array([r["frac_post_removed"] for r in rows]); b=np.array([r["boundary_covered"] for r in rows]); rp=np.array([r["remaining_post"] for r in rows])
    res["arms"][arm]["fine"]={"corr_delta_vs_frac_post_removed":float(np.corrcoef(d,fr)[0,1]) if fr.std()>0 else None,
        "by_boundary":{"covered":{"n":int(b.sum()),"mean":float(d[b].mean()) if b.any() else None},"not_covered":{"n":int((~b).sum()),"mean":float(d[~b].mean()) if (~b).any() else None}},
        "by_remaining_post":{str(v):{"n":int((rp==v).sum()),"mean":float(d[rp==v].mean())} for v in sorted(set(rp.tolist()))},
        "by_frac_removed_bin":{lab:{"n":int(m.sum()),"mean":float(d[m].mean()) if m.any() else None} for lab,m in (("0",fr==0),("(0,.34]",(fr>0)&(fr<=.34)),("(.34,.67]",(fr>.34)&(fr<=.67)),("(.67,1]",fr>.67))}}
(BASE/"gap_position_analysis.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("GAP POSITION DONE")
