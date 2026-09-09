#!/usr/bin/env python3
"""Landslide label geometry per region: connected-component sizes (pixels, 10 m) and positive fraction, from sealed mask_u8. Explains why positive-tile macro IoU differs across regions."""
import json, numpy as np, sys
from pathlib import Path
from scipy import ndimage
ROOT=Path("/home/work/data/olmoearth"); recs=[json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l]
cache={"italy":ROOT/"sen12_pilot/holdout_italy/mask_u8"}; default=ROOT/"sen12_pilot/holdout_chimanimani/mask_u8"
out={}
for region in ["italy","hiroshima","thrissur","chimanimani","newzealand","kyrgyzstan1","hokkaido","indonesia","itogon","kyrgyzstan2"]:
    ids=[r["sample_id"] for r in recs if r["region"]==region and not r.get("error") and r.get("s15_eligible",True)]
    d=cache.get(region,default); sizes=[]; posfrac=[]; ncomp=[]
    for s in ids:
        p=d/f"{s}.npy"
        if not p.exists(): continue
        m=np.load(p)>0
        if not m.any(): posfrac.append(0.0); continue
        lab,n=ndimage.label(m); cs=np.bincount(lab.ravel())[1:]; sizes+=cs.tolist(); ncomp.append(n); posfrac.append(float(m.mean()))
    sizes=np.array(sizes); out[region]={"tiles":len(ids),"pos_tiles":int(sum(1 for f in posfrac if f>0)),"pos_frac_mean_pos_tiles":round(float(np.mean([f for f in posfrac if f>0])) if any(posfrac) else 0,4),"components":int(sizes.size),"comp_px_median":float(np.median(sizes)) if sizes.size else None,"comp_px_p25":float(np.percentile(sizes,25)) if sizes.size else None,"comp_px_p75":float(np.percentile(sizes,75)) if sizes.size else None,"frac_components_below_16px":round(float((sizes<16).mean()),3) if sizes.size else None,"comps_per_pos_tile":round(float(np.mean(ncomp)),2) if ncomp else None}
    print(region,out[region],flush=True)
json.dump(out,open(ROOT/"artifacts/label_size_audit.json","w"),indent=1)
