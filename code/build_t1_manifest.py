#!/usr/bin/env python3
"""T1 development manifest: for folds holdout_chimanimani and holdout_hiroshima, take test region (all), val region (all), and an
evenly spaced 600-tile subsample of the train regions (same filter as the sealed folds). Writes a newline id list + json summary."""
import json, sys
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); recs=[json.loads(l) for l in (ROOT/"sen12_gp_contract/sample_contract.jsonl").read_text().splitlines() if l]
folds=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text())["folds"]; cache=ROOT/"sen12_pilot/holdout_chimanimani"
def ids(regions): return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (cache/"emb_fp16"/f"{r['sample_id']}.npy").exists())
FOLDS=sys.argv[1:] or ["holdout_chimanimani","holdout_hiroshima"]
mp=ROOT/"sen12_gp_contract/t1_manifest.json"; out=json.loads(mp.read_text()) if mp.exists() else {}; allids=set()
for fn in FOLDS:
    f=next(x for x in folds if x["fold"]==fn); tr=ids(f["train_regions"]); step=max(1,len(tr)//600); trs=tr[::step][:600]
    out[fn]={"train":trs,"val":ids([f["val_region"]]),"test":ids([f["test_region"]])}; [allids.update(v) for v in out[fn].values()]
mp.write_text(json.dumps(out)); (ROOT/"sen12_gp_contract/t1_ids_new.txt").write_text("\n".join(sorted(allids)))
print(json.dumps({k:{s:len(v) for s,v in d.items()} for k,d in out.items()}),"total_unique",len(allids))
