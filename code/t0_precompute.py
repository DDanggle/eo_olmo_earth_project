#!/usr/bin/env python3
"""One-pass precompute for the T0 screen so each of the 24 runs does not re-read 123 GB of per-timestep tokens.
Writes, under the temporal cache dir:
  emb_diff_fp16/<sid>.npy        (768,32,32)  late-half mean minus early-half mean  (fold-independent, training-free)
  proj_<fold>/pca.npz            PCA-64 fit on the fold's 100 evenly spaced source-train tiles (subsampled pixels), fold-specific
  proj_<fold>/<sid>.npy          (T,64,32,32) fp16 per-timestep projection for every tile of the fold (train/val/test)
The trainer uses these when present; numerics identical to its in-line computation."""
import json, sys, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); CACHE=ROOT/sys.argv[1]; folds=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text())["folds"]
recs=[json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l]
def members(fold,split):
    regions=fold["train_regions"] if split=="train" else [fold["val_region"]] if split=="val" else [fold["test_region"]]
    return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (CACHE/"emb_time_fp16"/f"{r['sample_id']}.npy").exists())
def pca_fit(vecs,k):
    mu=vecs.mean(0); X=vecs-mu; U,S,Vt=np.linalg.svd(X,full_matrices=False); comp=Vt[:k]; proj=X@comp.T; return mu.astype("float32"),comp.astype("float32"),proj.std(0).astype("float32")+1e-6
(CACHE/"emb_diff_fp16").mkdir(exist_ok=True)
for fn in sys.argv[2:]:
    fold=next(f for f in folds if f["fold"]==fn); ids={s:members(fold,s) for s in ("train","val","test")}; out=CACHE/f"proj_{fn}"; out.mkdir(exist_ok=True)
    tr=ids["train"]; fit_idx=np.linspace(0,len(tr)-1,min(400,len(tr))).astype(int); fit_ids=[tr[j] for j in fit_idx][:100]
    if not (out/"pca.npz").exists():
        samp=np.concatenate([np.load(CACHE/"emb_time_fp16"/f"{s}.npy").astype("float32").transpose(0,2,3,1).reshape(-1,768)[::16] for s in fit_ids]); mu,comp,sd=pca_fit(samp,64); np.savez(out/"pca.npz",mu=mu,comp=comp,sd=sd); del samp
    z=np.load(out/"pca.npz"); mu,comp,sd=z["mu"],z["comp"],z["sd"]; n=0
    for split,lst in ids.items():
        for s in lst:
            pp=out/f"{s}.npy"; dp=CACHE/"emb_diff_fp16"/f"{s}.npy"
            if pp.exists() and dp.exists(): continue
            t=np.load(CACHE/"emb_time_fp16"/f"{s}.npy").astype("float32"); h=t.shape[0]//2
            if not dp.exists(): np.save(dp,(t[h:].mean(0)-t[:h].mean(0)).astype("float16"))
            if not pp.exists(): np.save(pp,(((t.transpose(0,2,3,1)-mu)@comp.T)/sd).transpose(0,3,1,2).astype("float16"))
            n+=1
            if n%500==0: print(fn,n,flush=True)
    print(fn,"done",{k:len(v) for k,v in ids.items()},flush=True)
print("T0 PRECOMPUTE DONE")
