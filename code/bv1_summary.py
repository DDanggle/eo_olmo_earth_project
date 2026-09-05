"""B-v1 diagnostics summary: per-fold positive-patch macro IoU for each cache (bv1_runs/<cache>/holdout_<fold>_seed1.json) vs OlmoEarth P4 (sealed pilot) and raw P2 (raw_strong). Applies addendum_v1a readout rules."""
import json, glob, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); ref={x["fold"]:x["primary_mean"] for x in json.load(open(ROOT/"artifacts/confirmatory_8region_summary.json"))["regions"]}
folds=["holdout_hiroshima","holdout_hokkaido","holdout_indonesia","holdout_itogon","holdout_kyrgyzstan1","holdout_kyrgyzstan2","holdout_newzealand","holdout_thrissur"]
caches=["olmo_cache_pool16","clay_cache_native16","clay_cache_native16_last","galileo_cache","prithvi_cache","clay_cache_in256","galileo_cache_groupcat"]
out={"schema":"bv1-diagnostics-summary-v1","status":"single seed diagnostics (addendum_v1a); not headline","rows":{},"verdict":{}}
for c in caches:
    r={}
    for f in folds:
        p=ROOT/"bv1_runs"/c/f"{f}_seed1.json"
        if p.exists(): d=json.load(open(p)); r[f]={"iou":d["test"]["positive_patch_macro_iou"],"micro":d["test"]["iou"],"ap":d["test"]["auprc_exact"],"best_epoch":d["best_val_epoch"],"shape":d["emb_shape"]}
    out["rows"][c]=r
    if r:
        done=list(r); beats_raw=sum(r[f]["iou"]>ref[f]["raw_strong"] for f in done); beats_olmo=sum(r[f]["iou"]>ref[f]["reuse"] for f in done)
        out["verdict"][c]={"n":len(done),"macro":float(np.mean([r[f]["iou"] for f in done])),"raw_macro_same_folds":float(np.mean([ref[f]["raw_strong"] for f in done])),"olmo_macro_same_folds":float(np.mean([ref[f]["reuse"] for f in done])),"beats_raw":beats_raw,"beats_olmo":beats_olmo,"rule_beats_raw_ge6of8":(beats_raw>=6 and len(done)==8)}
print("cache | n | macro | raw | olmo | >raw | >olmo")
for c,v in out["verdict"].items(): print(f"{c} | {v['n']} | {v['macro']:.3f} | {v['raw_macro_same_folds']:.3f} | {v['olmo_macro_same_folds']:.3f} | {v['beats_raw']} | {v['beats_olmo']}")
print("\nfold | " + " | ".join(caches) + " | olmo P4 | raw P2")
for f in folds: print(f.replace("holdout_",""), "| " + " | ".join(("%.3f"%out["rows"][c][f]["iou"]) if f in out["rows"].get(c,{}) else "  -  " for c in caches), "| %.3f | %.3f"%(ref[f]["reuse"],ref[f]["raw_strong"]))
(ROOT/"artifacts/bv1_diagnostics_summary.json").write_text(json.dumps(out,indent=1))
