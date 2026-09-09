#!/usr/bin/env python3
"""Italy: size-stratified evaluation of saved decoder probability maps is not available (cache_decoder_train saves no probs), so re-apply saved
checkpoints to the test tiles and compute (a) overall metrics, (b) IoU restricted to tiles whose largest landslide component >= 16 px (one 40 m token),
(c) recall of components by size bin (a component counts as detected if any predicted pixel overlaps it). Arms: decoder(40m), p4_upsample2, p2_native, p2_avgpool2."""
import json,sys,numpy as np,torch
from pathlib import Path
from scipy import ndimage
sys.path.insert(0,"/home/work/data/olmoearth/code"); from cache_grid_controls import transform_embedding
from cache_decoder_train_lib import EmbDecoder
ROOT=Path("/home/work/data/olmoearth"); dev=torch.device("cuda")
folds=json.loads((ROOT/"sen12_gp_contract/loco_folds_italy.json").read_text())["folds"][0]; recs=[json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l]
ARMS={"decoder":("sen12_pilot/holdout_italy","native"),"p4_upsample2":("sen12_pilot/holdout_italy","upsample2"),"p2_native":("olmo_italy_p2_rt","native"),"p2_avgpool2":("olmo_italy_p2_rt","avgpool2")}
def members(cache,regions): return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (cache/"mask_u8"/f"{r['sample_id']}.npy").exists())
out={}
for arm,(cdir,tr) in ARMS.items():
    C=ROOT/cdir; test=members(C,[folds["test_region"]]); train=members(C,folds["train_regions"])
    idx=np.linspace(0,len(train)-1,400).astype(int); acc=acc2=None
    for j in idx:
        x=transform_embedding(np.load(C/"emb_fp16"/f"{train[j]}.npy").astype("float32"),tr); m=x.mean(axis=(1,2)); m2=(x**2).mean(axis=(1,2)); acc=m if acc is None else acc+m; acc2=m2 if acc2 is None else acc2+m2
    mu=torch.tensor(acc/400).view(-1,1,1).float(); sd=torch.tensor(np.sqrt(np.maximum(acc2/400-(acc/400)**2,1e-6))).view(-1,1,1).float()
    res=[]
    for s in (1,2,3):
        ck=torch.load(ROOT/"artifacts/italy_sealed"/arm/f"holdout_italy_seed{s}_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
        big_iou=[]; small_only_iou=[]; det={"<16":[0,0],"16-64":[0,0],">=64":[0,0]}; tp=fp=fn=0
        for t in test:
            y=np.load(C/"mask_u8"/f"{t}.npy")>0
            if not y.any(): continue
            x=torch.from_numpy(transform_embedding(np.load(C/"emb_fp16"/f"{t}.npy").astype("float32"),tr))[None]
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16): p=torch.sigmoid(dec(((x-mu)/sd).to(dev)).float()).cpu().numpy()[0,0]>0.5
            inter=(p&y).sum(); union=(p|y).sum(); iou=inter/max(union,1)
            lab,n=ndimage.label(y); sizes=np.bincount(lab.ravel())[1:]
            (big_iou if sizes.max()>=16 else small_only_iou).append(iou)
            for k in range(1,n+1):
                sz=sizes[k-1]; b="<16" if sz<16 else "16-64" if sz<64 else ">=64"; det[b][1]+=1; det[b][0]+=int((p&(lab==k)).any())
        res.append({"seed":s,"macro_iou_tiles_with_component_ge16px":float(np.mean(big_iou)),"n_tiles_ge16":len(big_iou),"macro_iou_tiles_all_small":float(np.mean(small_only_iou)) if small_only_iou else None,"n_tiles_small_only":len(small_only_iou),"component_recall":{k:(round(v[0]/max(v[1],1),3),v[1]) for k,v in det.items()}})
        print(arm,res[-1],flush=True)
    out[arm]=res
json.dump(out,open(ROOT/"artifacts/italy_sealed/size_stratified.json","w"),indent=1); print("DONE")
