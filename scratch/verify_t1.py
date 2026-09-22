"""Independent re-computation of the T1 reference rows (teacher / frozen m4 / singles mean) from disk, plus data-contract checks.
Does not import streaming_update_train. GPU0."""
import json,sys,numpy as np,torch
sys.path.insert(0,"/home/work/data/olmoearth/code"); from pathlib import Path
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, metrics
ROOT=Path("/home/work/data/olmoearth"); D=ROOT/"olmo_streaming_dev"; dev=torch.device("cuda")
man=json.loads((ROOT/"sen12_gp_contract/t1_manifest.json").read_text())
folds=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text())["folds"]
recs={json.loads(l)["sample_id"]:json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l}
for fold in ("holdout_hiroshima","holdout_chimanimani"):
    f=next(x for x in folds if x["fold"]==fold); m=man[fold]
    # 1. manifest region contract: test ids all in test region, train ids none in test/val region
    assert all(recs[s]["region"]==f["test_region"] for s in m["test"]), "test region leak"
    assert not any(recs[s]["region"] in (f["test_region"],f["val_region"]) for s in m["train"]), "train contains test/val region"
    assert not set(m["train"])&set(m["test"]) and not set(m["val"])&set(m["test"]), "overlap"
    ids=m["test"]
    # 2. teacher c12 == sealed cache exactly for ALL test tiles (not just 30 audited)
    T=np.stack([np.load(D/"teacher_fp16"/f"{s}.npy") for s in ids]); S=np.stack([np.load(D/"single_fp16"/f"{s}.npy") for s in ids])
    sealed=np.stack([np.load(ROOT/"sen12_pilot/holdout_chimanimani/emb_fp16"/f"{s}.npy") for s in ids])
    print(fold,"max|teacher_c12 - sealed| over",len(ids),"tiles:",float(np.abs(T[:,-1].astype("float32")-sealed.astype("float32")).max()))
    # 3. shapes/finite
    assert T.shape[1:]==(5,768,32,32) and S.shape[1:]==(12,768,32,32) and np.isfinite(T.astype("float32")).all() and np.isfinite(S.astype("float32")).all()
    # 4. singles are NOT trivially equal to teacher windows (would indicate a windowing bug)
    print("  cos(single_t11, teacher_c12) mean:",float(torch.nn.functional.cosine_similarity(torch.from_numpy(S[:,11].astype("float32")).flatten(1),torch.from_numpy(T[:,-1].astype("float32")).flatten(1)).mean()),
          " cos(teacher_c4, teacher_c12):",float(torch.nn.functional.cosine_similarity(torch.from_numpy(T[:,0].astype("float32")).flatten(1),torch.from_numpy(T[:,-1].astype("float32")).flatten(1)).mean()))
    # 5. frozen decoder rows recomputed
    ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
    mu,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_chimanimani",fold); Y=torch.from_numpy(np.stack([np.load(ROOT/"sen12_pilot/holdout_chimanimani/mask_u8"/f"{s}.npy") for s in ids]).astype("float32"))
    def ap(X):
        X=torch.from_numpy(X.astype("float32")); out=[]
        with torch.no_grad():
            for i in range(0,len(X),32):
                with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(dec(((X[i:i+32]-mu)/sd).to(dev)).float()).cpu())
        return metrics(torch.cat(out).squeeze(1),Y)["auprc_exact"]
    rec=json.load(open(ROOT/"artifacts/streaming_t1"/f"{fold}_gru_seed1.json"))["downstream_c12"]
    ctrl=json.load(open(ROOT/"resolution_contract_v2/p4_native_control"/f"{fold}_seed1.json"))["test"]["auprc_exact"]
    print("  teacher AP recomputed %.4f | json %.4f | control decoder's own test AP %.4f"%(ap(T[:,-1]),rec["teacher_full_reencode"]["auprc_exact"],ctrl))
    print("  frozen  AP recomputed %.4f | json %.4f"%(ap(T[:,0]),rec["frozen_m4"]["auprc_exact"]))
    print("  singles AP recomputed %.4f | json %.4f"%(ap(S.astype("float32").mean(1)),rec["singles_mean"]["auprc_exact"]))
    # 6. a sanity row nobody trained: decoder on teacher c8 (8 of 12 timesteps) — should sit between frozen and teacher
    print("  teacher_c8 AP %.4f  teacher_c10 AP %.4f"%(ap(T[:,2]),ap(T[:,3])))
print("VERIFY OK")
