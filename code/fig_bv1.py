import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/artifacts"); d=json.load(open(ROOT/"bv1_diagnostics_summary.json")); ref={x["fold"]:x["primary_mean"] for x in json.load(open(ROOT/"confirmatory_8region_summary.json"))["regions"]}
folds=["holdout_hiroshima","holdout_hokkaido","holdout_indonesia","holdout_itogon","holdout_kyrgyzstan1","holdout_kyrgyzstan2","holdout_newzealand","holdout_thrissur"]
series=[("OlmoEarth v1 32x32 (sealed, 3 seeds)",[ref[f]["reuse"] for f in folds],"#1f77b4"),("OlmoEarth pooled 16x16",[d["rows"]["olmo_cache_pool16"][f]["iou"] for f in folds],"#7fb3d5"),
        ("Clay v1.5 in256 -> 32x32",[d["rows"]["clay_cache_in256"][f]["iou"] for f in folds],"#e07b39"),("Clay v1.5 native 16x16",[d["rows"]["clay_cache_native16"][f]["iou"] for f in folds],"#f2b586"),
        ("Galileo base 32x32 (temporal-native)",[d["rows"]["galileo_cache"][f]["iou"] for f in folds],"#5b9e5b"),("Prithvi-EO-2.0 8x8 (contract shift)",[d["rows"]["prithvi_cache"][f]["iou"] for f in folds],"#b0b0b0"),
        ("raw UNet3D P2 (3 seeds)",[ref[f]["raw_strong"] for f in folds],"#333333")]
fig,ax=plt.subplots(figsize=(14,5.5)); x=np.arange(len(folds)); w=0.11
for i,(lab,v,c) in enumerate(series): ax.bar(x+(i-3)*w,v,w,label=f"{lab}  [macro {np.mean(v):.3f}]",color=c)
ax.set_xticks(x); ax.set_xticklabels([f.replace("holdout_","") for f in folds]); ax.set_ylabel("positive-tile macro IoU (zero target labels)"); ax.legend(fontsize=8,ncol=2)
ax.set_title("Which frozen cache is worth keeping? Sen12 landslides, 8 held-out regions, same decoder & recipe (single seed except sealed references)",fontsize=10)
plt.tight_layout(); fig.savefig(ROOT/"figures/bv1_diagnostics.png",dpi=120); print("saved")
