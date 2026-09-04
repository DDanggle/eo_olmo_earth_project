"""Qualitative before/after figure: landslide (hiroshima) + solar (fold0). columns: pre RGB, post RGB, GT, cache P4 prob, raw P2 prob. 4 largest-positive test tiles (deterministic)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth")
def load(task):
    if task=="landslide":
        cache=ROOT/"sen12_pilot/holdout_chimanimani"; base=ROOT/"confirmatory/holdout_hiroshima"; p4=base/"P4_seed1/prob_maps/holdout_hiroshima"; p2=base/"P2_seed1/prob_maps/holdout_hiroshima"; tag="hiroshima"
    else:
        cache=ROOT/"task2_cache"; p4=ROOT/"task2_source_v1/holdout_task2_fold0_seed1_P4/prob_maps/holdout_task2_fold0"; p2=ROOT/"task2_source_v1/holdout_task2_fold0_seed1_P2/prob_maps/holdout_task2_fold0"; tag="task2_fold0"
    idx=json.loads((p4/"P4_test_probs_index.json").read_text()); ids=idx if isinstance(idx,list) else idx.get("sample_ids") or idx.get("ids") or list(idx.values())[0]
    idx2=json.loads((p2/"P2_test_probs_index.json").read_text()); ids2=idx2 if isinstance(idx2,list) else idx2.get("sample_ids") or idx2.get("ids") or list(idx2.values())[0]
    P4=np.load(p4/"P4_test_probs_u8.npy").astype("float32")/255; P2=np.load(p2/"P2_test_probs_u8.npy").astype("float32")/255; pos2={s:i for i,s in enumerate(ids2)}
    return cache,ids,P4,P2,pos2,tag
def rgb(cube,t):  # cube (10,T,H,W) uint16 bands B02,B03,B04,...
    x=cube[[2,1,0],t].astype("float32")/10000; return np.clip(x/0.3,0,1).transpose(1,2,0)
def iou(p,y,thr=0.5): b=p>=thr; u=(b|y).sum(); return (b&y).sum()/u if u else 0
fig,axes=plt.subplots(8,5,figsize=(12.5,20)); r=0
for task in ("landslide","solar"):
    cache,ids,P4,P2,pos2,tag=load(task)
    areas=[]
    for i,s in enumerate(ids):
        m=np.load(cache/"mask_u8"/f"{s}.npy"); areas.append((m.sum(),i))
    pick=[i for _,i in sorted(areas,reverse=True)[:4]]
    for i in pick:
        s=ids[i]; m=np.load(cache/"mask_u8"/f"{s}.npy").astype(bool); cube=np.load(cache/"raw_u16"/f"{s}.npy"); T=cube.shape[1]
        a=axes[r]; a[0].imshow(rgb(cube,0)); a[1].imshow(rgb(cube,T-1)); a[2].imshow(m,cmap="gray"); a[3].imshow(P4[i],vmin=0,vmax=1,cmap="magma"); a[3].contour(m,levels=[0.5],colors="cyan",linewidths=0.6)
        q=P2[pos2[s]] if s in pos2 else np.zeros_like(P4[i]); a[4].imshow(q,vmin=0,vmax=1,cmap="magma"); a[4].contour(m,levels=[0.5],colors="cyan",linewidths=0.6)
        a[0].set_ylabel(f"{tag}\n{str(s)[:14]}",fontsize=8)
        for j,t in enumerate(["S2 before (t0)","S2 after (t-1)" if task=="landslide" else f"S2 last (t{T-1})","label","cache P4  IoU %.2f"%iou(P4[i],m),"raw UNet3D P2  IoU %.2f"%iou(q,m)]): a[j].set_title(t if r in (0,4) or j>2 else "",fontsize=9)
        for ax in a: ax.set_xticks([]); ax.set_yticks([])
        r+=1
fig.suptitle("Zero-target reuse: frozen-cache decoder (P4) vs raw UNet3D (P2), held-out region, seed 1, 4 largest-positive test tiles\nlandslide = Sen12 hiroshima (P4 .27 vs P2 .20 macro over 8 regions)   |   solar = Task-2 fold0 (P4 .59 vs P2 .33, 8/8)",fontsize=10)
plt.tight_layout(rect=(0,0,1,0.97)); out=ROOT/"artifacts/figures/before_after_landslide_solar.png"; out.parent.mkdir(exist_ok=True,parents=True); fig.savefig(out,dpi=110); print("saved",out)
