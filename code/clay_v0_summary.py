"""Clay v0 (16->32 bilinear, exploratory) summary: zero-target P4(Clay) vs OlmoEarth P4 vs raw P2 per region; few-shot A0/A1/A4w/A4h on the Clay cache (shared FP budget)."""
import json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
ROOT=Path("/home/work/data/olmoearth"); out={"schema":"clay-v0-summary-v1","status":"EXPLORATORY (16x16 native tokens bilinear-upsampled to 32x32; not the registered B-v1 protocol)"}
# zero-target: clay_source_v1/<fold>_seed<s>/<fold>_pilot.json
z=defaultdict(list)
for p in glob.glob(str(ROOT/"clay_source_v1/holdout_*_seed*/holdout_*_pilot.json")):
    d=json.load(open(p)); fold=Path(p).stem.replace("_pilot",""); 
    try: v=d["arms"]["P4"]["test"]["positive_patch_macro_iou"]
    except Exception: v=None
    if v is None:
        # fallback: search any key containing test macro iou
        s=json.dumps(d); import re; m=re.search(r'"pos_macro_iou":\s*([0-9.]+)',s); v=float(m.group(1)) if m else None
    z[fold].append(v)
olmo=json.load(open(ROOT/"artifacts/confirmatory_8region_summary.json"))["regions"]; ref={x["fold"]:x["primary_mean"] for x in olmo}
rows=[]
for fold in sorted(z):
    rows.append({"fold":fold,"clay_P4":float(np.mean([v for v in z[fold] if v is not None])),"n":len(z[fold]),"olmo_P4":ref[fold]["reuse"],"raw_P2":ref[fold]["raw_strong"]})
out["zero_target"]={"rows":rows,"clay_beats_raw":sum(r["clay_P4"]>r["raw_P2"] for r in rows),"clay_beats_olmo":sum(r["clay_P4"]>r["olmo_P4"] for r in rows),"macro":{k:float(np.mean([r[k] for r in rows])) for k in ("clay_P4","olmo_P4","raw_P2")}}
# few-shot
fs=json.load(open(ROOT/"artifacts/clay_fewshot/fu/report_fixed_update.json")); agg=defaultdict(list)
for r in fs["runs"]: agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"])
regs=sorted({k[0] for k in agg}); few={}
for K in (5,20):
    M={arm:{g:float(np.mean(agg[(g,K,arm)])) for g in regs} for arm in ("A1","A4w","A4h")}; A0={g:float(np.mean(agg[(g,None,"A0")])) for g in regs}
    few[f"K{K}"]={"macro":{"A0":float(np.mean(list(A0.values()))),**{a:float(np.mean(list(M[a].values()))) for a in M}},"A1_gt_A4w":sum(M["A1"][g]>M["A4w"][g] for g in regs),"A1_gt_A4h":sum(M["A1"][g]>M["A4h"][g] for g in regs),"A1_gt_A0":sum(M["A1"][g]>A0[g] for g in regs),"n":len(regs),"per_region":{g:{"A0":round(A0[g],4),**{a:round(M[a][g],4) for a in M}} for g in regs}}
out["fewshot_stratified"]=few
Path(ROOT/"artifacts/clay_fewshot/clay_v0_summary.json").write_text(json.dumps(out,indent=1)); print(json.dumps({k:out[k] for k in ("zero_target","fewshot_stratified")},indent=1)[:3000])
