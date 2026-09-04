#!/usr/bin/env python3
"""A — release migration screen (config/release_migration_prereg_draft_v0.json, registered efce8b8).
Scenario: v1 cache + v1-trained P4 head kept; new v1.2 embeddings must be mapped into the v1 contract. Bridges fit on SOURCE-TRAIN paired tokens only (no labels), λ selected on source-val pairs by reconstruction MSE; threshold frozen from R0 on source val; target-fold labels read only at the end.
Arms: R0 old head/old emb · R1 identity · R2 channel mean shift · R3 Procrustes(+translation) · R4 affine ridge · R6 new v1.2 head (task2_source_v12, if present) · R7 = R6 + extraction cost reference.
Metrics: tie-correct AP (primary), positive-tile macro IoU at frozen thr, IoU@0.5, empty-tile FP at frozen thr; secondary same-token cosine, linear CKA, token R@1; cost: bridge fit s, bridge infer s/tile, bytes."""
import json, os, sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth"); CACHE=ROOT/"task2_cache"; C12=ROOT/"task2_cache_v12"; CONTRACT=ROOT/"task2_contract/sample_contract.jsonl"; FOLDS=json.loads((ROOT/"task2_contract/loco_folds.json").read_text())
ap=argparse.ArgumentParser(); ap.add_argument("--folds",default="task2_fold0,task2_fold1"); ap.add_argument("--seeds",default="1,2,3"); ap.add_argument("--out",default=str(ROOT/"artifacts/release_migration/screen")); ap.add_argument("--fit-tiles",type=int,default=200); ap.add_argument("--val-tiles",type=int,default=60)
a=ap.parse_args(); OUT=Path(a.out); OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda")
sys.argv=[sys.argv[0],"--task2"]  # reuse helpers from the few-shot runner without triggering its main loop
src=(ROOT/"code/fewshot_a1_a4.py").read_text().split("rep={\"schema\"")[0]; ns={}; exec(compile(src,"fewshot_helpers","exec"),ns)
EmbDecoder,members,load_masks,probs,pos_macro_iou,empty_fp,tie_ap,evaluate=[ns[k] for k in ("EmbDecoder","members","load_masks","probs","pos_macro_iou","empty_fp","tie_ap","evaluate")]
def stats(cache,ids,sample=400):
    idx=np.linspace(0,len(ids)-1,min(sample,len(ids))).astype(int); acc=acc2=None; n=0
    for j in idx:
        x=np.load(cache/"emb_fp16"/f"{ids[j]}.npy").astype("float32"); m=x.mean(axis=(1,2)); m2=(x**2).mean(axis=(1,2))
        acc=m if acc is None else acc+m; acc2=m2 if acc2 is None else acc2+m2; n+=1
    mean=acc/n; sd=np.sqrt(np.maximum(acc2/n-mean**2,1e-6)); return torch.tensor(mean).view(-1,1,1).float(), torch.tensor(sd).view(-1,1,1).float()
def load_raw_emb(cache,ids): return torch.from_numpy(np.stack([np.load(cache/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids]))
def tokens(X): return X.permute(0,2,3,1).reshape(-1,X.shape[1]).double()  # (N*1024, C)
def fit_bridges(X2,X1,V2,V1):
    """X2->X1 (v1.2 -> v1). returns dict name -> (fn, fit_seconds, meta). All in raw (un-normalized) embedding space."""
    out={}; t0=time.perf_counter(); m2=X2.mean(0); m1=X1.mean(0)
    out["R2_mean_shift"]=((lambda Z:Z-m2+m1),time.perf_counter()-t0,{})
    t0=time.perf_counter(); A=X2-m2; B=X1-m1; U,_,Vt=torch.linalg.svd(A.T@B,full_matrices=False); W=U@Vt
    out["R3_procrustes"]=((lambda Z:(Z-m2)@W+m1),time.perf_counter()-t0,{})
    t0=time.perf_counter(); G=A.T@A; tr=float(torch.trace(G))/G.shape[0]; best=None
    for lam in (1e-3,1e-2,1e-1,1.0,10.0):
        Wr=torch.linalg.solve(G+lam*tr*torch.eye(G.shape[0],dtype=G.dtype),A.T@B); mse=float((((V2-m2)@Wr+m1-V1)**2).mean())
        if best is None or mse<best[0]: best=(mse,lam,Wr)
    mse,lam,Wr=best; out["R4_affine_ridge"]=((lambda Z:(Z-m2)@Wr+m1),time.perf_counter()-t0,{"lambda_rel":lam,"val_mse":mse})
    return out
def apply_bridge(fn,X):  # X (N,C,32,32) raw -> bridged raw
    N,C,H,W=X.shape; Z=tokens(X); t0=time.perf_counter(); Y=fn(Z).float(); dt=time.perf_counter()-t0
    return Y.reshape(N,H,W,C).permute(0,3,1,2).contiguous(), dt/N
def cka(X,Y):
    X=X-X.mean(0); Y=Y-Y.mean(0); return float((torch.norm(X.T@Y)**2)/(torch.norm(X.T@X)*torch.norm(Y.T@Y)))
def r_at_1(Q,G):  # fraction of query tokens whose nearest gallery token is its own index
    Qn=F.normalize(Q.float(),dim=1); Gn=F.normalize(G.float(),dim=1); nn_=(Qn@Gn.T).argmax(1); return float((nn_==torch.arange(len(Q))).float().mean())
def best_thr(P,Y):
    ths=np.linspace(0.05,0.95,19); v=[pos_macro_iou(P,Y,t) or 0 for t in ths]; return float(ths[int(np.argmax(v))])
rep={"schema":"release-migration-screen-v1","preregistration":"config/release_migration_prereg_draft_v0.json","folds":a.folds,"seeds":a.seeds,"bridge_fit":"source-train paired tokens, no labels; lambda on source-val pairs","runs":[]}
for region in a.folds.split(","):
    fold=next(f for f in FOLDS["folds"] if f["fold"]==f"holdout_{region}"); tr_ids=members(fold,"train"); va_ids=members(fold,"val"); te_ids=members(fold,"test")
    st1=stats(CACHE,tr_ids); st12=stats(C12,tr_ids)
    rng=np.random.RandomState(0); fit_ids=[tr_ids[i] for i in sorted(rng.choice(len(tr_ids),min(a.fit_tiles,len(tr_ids)),replace=False))]; vfit_ids=[va_ids[i] for i in sorted(rng.choice(len(va_ids),min(a.val_tiles,len(va_ids)),replace=False))]
    X1f=tokens(load_raw_emb(CACHE,fit_ids)); X2f=tokens(load_raw_emb(C12,fit_ids)); V1=tokens(load_raw_emb(CACHE,vfit_ids)); V2=tokens(load_raw_emb(C12,vfit_ids))
    bridges=fit_bridges(X2f,X1f,V2,V1); bytes_pairs=int(len(fit_ids)*2*768*1024*2)
    Yv=load_masks(va_ids).squeeze(1).numpy().astype(bool); Yt=load_masks(te_ids).squeeze(1).numpy().astype(bool)
    E1v=load_raw_emb(CACHE,va_ids); E1t=load_raw_emb(CACHE,te_ids); E12t=load_raw_emb(C12,te_ids); E12v=load_raw_emb(C12,va_ids)
    # secondary geometry on a token sample from the test fold (label-free)
    sidx=torch.from_numpy(np.random.RandomState(1).choice(E1t.shape[0]*1024,4000,replace=False)); T1=tokens(E1t)[sidx]
    for seed in [int(s) for s in a.seeds.split(",")]:
        ck=torch.load(ROOT/"task2_source_v1"/f"holdout_{region}_seed{seed}_P4/checkpoints/holdout_{region}/P4_best.pt",map_location="cpu"); ck=ck.get("model_state",ck.get("state_dict",ck))
        head=EmbDecoder(cin=ck["proj.0.weight"].shape[1]).to(dev); head.load_state_dict(ck,strict=True); head.eval()
        norm1=lambda X:(X-st1[0])/st1[1]
        Pv0=probs(head,norm1(E1v)); thr=best_thr(Pv0,Yv)  # frozen on source val with R0
        Pt0=probs(head,norm1(E1t)); budget=empty_fp(Pt0,Yt,thr)
        def rec(arm,P,extra):
            ev=evaluate(P,Yt,budget); ev.update({"iou_frozen_thr":pos_macro_iou(P,Yt,thr),"empty_fp_frozen_thr":empty_fp(P,Yt,thr)})
            r={"region":region,"seed":seed,"arm":arm,"frozen_thr":thr,"fp_budget":budget,"eval":ev}; r.update(extra); rep["runs"].append(r); print(region,seed,arm,"AP %.4f IoU@thr %s"%(ev["tie_ap"],None if ev["iou_frozen_thr"] is None else round(ev["iou_frozen_thr"],4)),flush=True)
        rec("R0_old_reference",Pt0,{"geom":{"cos":1.0,"cka":1.0,"r1":1.0},"cost":{"fit_s":0,"infer_s_per_tile":0,"bytes":0}})
        Tb=tokens(E12t)[sidx]; rec("R1_identity",probs(head,norm1(E12t)),{"geom":{"cos":float(F.cosine_similarity(Tb.float(),T1.float(),dim=1).mean()),"cka":cka(Tb,T1),"r1":r_at_1(Tb,T1)},"cost":{"fit_s":0,"infer_s_per_tile":0,"bytes":0}})
        for name,(fn,fs,meta) in bridges.items():
            Eb,ipt=apply_bridge(fn,E12t); Tb=tokens(Eb)[sidx]
            rec(name,probs(head,norm1(Eb)),{"geom":{"cos":float(F.cosine_similarity(Tb.float(),T1.float(),dim=1).mean()),"cka":cka(Tb,T1),"r1":r_at_1(Tb,T1)},"cost":{"fit_s":fs,"infer_s_per_tile":ipt,"bytes":bytes_pairs},"meta":meta})
        ck6=ROOT/"task2_source_v12"/f"holdout_{region}_seed{seed}_P4/checkpoints/holdout_{region}/P4_best.pt"
        if ck6.exists():
            c6=torch.load(ck6,map_location="cpu"); c6=c6.get("model_state",c6.get("state_dict",c6)); h6=EmbDecoder(cin=c6["proj.0.weight"].shape[1]).to(dev); h6.load_state_dict(c6,strict=True); h6.eval()
            norm12=lambda X:(X-st12[0])/st12[1]; thr6=best_thr(probs(h6,norm12(E12v)),Yv); P6=probs(h6,norm12(E12t)); ev=evaluate(P6,Yt,budget); ev.update({"iou_frozen_thr":pos_macro_iou(P6,Yt,thr6),"empty_fp_frozen_thr":empty_fp(P6,Yt,thr6)})
            rep["runs"].append({"region":region,"seed":seed,"arm":"R6_new_native_head","frozen_thr":thr6,"fp_budget":budget,"eval":ev,"cost":{"note":"new head training under source recipe + full v1.2 re-extraction of the archive (R7 reference)"}}); print(region,seed,"R6 AP %.4f"%ev["tie_ap"],flush=True)
    (OUT/"report.json").write_text(json.dumps(rep,indent=1))
# screen verdict per registered rule
from collections import defaultdict
agg=defaultdict(list)
for r in rep["runs"]: agg[(r["region"],r["arm"])].append(r["eval"]["tie_ap"])
M={k:float(np.mean(v)) for k,v in agg.items()}; regs=sorted({k[0] for k in M})
ok_ret=all(M[(g,"R4_affine_ridge")]>=0.9*M[(g,"R0_old_reference")] for g in regs); ok_gain=all(M[(g,"R4_affine_ridge")]-M[(g,"R1_identity")]>=0.05 for g in regs)
rep["summary"]={"mean_ap":{f"{g}|{arm}":round(M[(g,arm)],4) for (g,arm) in M},"screen_pass_R4":bool(ok_ret and ok_gain),"retention_ok":ok_ret,"gain_over_identity_ok":ok_gain,"rule":"R4 >= 0.9*R0 AP and R4-R1 >= 0.05 AP in both folds"}
(OUT/"report.json").write_text(json.dumps(rep,indent=1)); print(json.dumps(rep["summary"],indent=1)); print("SCREEN DONE")
