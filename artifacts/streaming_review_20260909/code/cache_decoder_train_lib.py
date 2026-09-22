"""Re-exported pieces of cache_decoder_train.py (identical definitions) so other scripts can apply a saved decoder."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth")
def conv_bn(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):
    def __init__(s,cin,base=128):
        super().__init__(); s.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); s.u1,s.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); s.head=nn.Conv2d(base//4,1,1)
    def forward(s,x):
        x=s.proj(x); x=s.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=s.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(s.head(x),size=(128,128),mode="bilinear",align_corners=False)
def emb_stats_from_cache(cache,fold_name,sample=400):
    folds=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text()); fold=next(f for f in folds["folds"] if f["fold"]==fold_name)
    recs=[json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l]
    tr=sorted(r["sample_id"] for r in recs if r["region"] in fold["train_regions"] and not r.get("error") and r.get("s15_eligible",True) and (cache/"mask_u8"/f"{r['sample_id']}.npy").exists())
    idx=np.linspace(0,len(tr)-1,min(sample,len(tr))).astype(int); acc=acc2=None; n=0
    for j in idx:
        x=np.load(cache/"emb_fp16"/f"{tr[j]}.npy").astype("float32"); m=x.mean(axis=(1,2)); m2=(x**2).mean(axis=(1,2)); acc=m if acc is None else acc+m; acc2=m2 if acc2 is None else acc2+m2; n+=1
    mean=acc/n; sdv=np.sqrt(np.maximum(acc2/n-mean**2,1e-6)); return torch.tensor(mean).view(-1,1,1).float(), torch.tensor(sdv).view(-1,1,1).float()
def exact_ap(scores,labels):
    o=np.argsort(-scores,kind="mergesort"); s=scores[o]; l=labels[o]; P=l.sum()
    if P==0: return None
    b=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]; tp=np.cumsum(l)[b]; fp=(b+1)-tp; prec=tp/np.maximum(tp+fp,1); rec=tp/P; return float(np.sum(np.diff(np.r_[0,rec])*prec))
def metrics(P,Y):
    pred=(P>0.5).float(); tp=(pred*Y).flatten(1).sum(1); fp=(pred*(1-Y)).flatten(1).sum(1); fn=((1-pred)*Y).flatten(1).sum(1); mp=Y.flatten(1).sum(1)
    den=tp+fp+fn; piou=torch.where(den>0,tp/den.clamp(min=1e-9),torch.ones_like(den)); posm=float(piou[mp>0].mean()) if (mp>0).any() else None
    return {"iou":float(tp.sum()/max(float((tp+fp+fn).sum()),1e-9)),"positive_patch_macro_iou":posm,"positive_patch_n":int((mp>0).sum()),"auprc_exact":exact_ap(P.numpy().ravel().astype("float64"),Y.numpy().ravel().astype("uint8"))}
