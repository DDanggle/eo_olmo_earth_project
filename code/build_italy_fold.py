#!/usr/bin/env python3
"""Sealed-region fold for Italy (Sen12Landslides, Emilia-Romagna 2022-10..2023-09): test=italy, val=china, train=the 8 headline regions.
Italy was excluded from HEADLINE_REGIONS because its NetCDF annotation attributes/event dates are absent (annotated_attr False), not because of data errors.
Membership filter and hashing are identical to build_sen12_gp_contract.build_loco_folds (region, not error, s15_eligible; sha256 of sorted ids)."""
import json, hashlib, sys
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/sen12_gp_contract"); recs=[json.loads(l) for l in (ROOT/"sample_contract.jsonl").read_text().splitlines() if l]
TRAIN=["chimanimani","hiroshima","hokkaido","indonesia","itogon","kyrgyzstan1","kyrgyzstan2","newzealand","thrissur"]; VAL="china"; TEST="italy"
def ids(regions): return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True))
def h(v): return hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()
roles={"train":ids(TRAIN),"val":ids([VAL]),"test":ids([TEST])}
it=[r for r in recs if r["region"]==TEST and not r.get("error") and r.get("s15_eligible",True)]
fold={"fold":"holdout_italy","test_region":TEST,"val_region":VAL,"train_regions":TRAIN,"sample_counts":{k:len(v) for k,v in roles.items()},"sample_sha256":{k:h(v) for k,v in roles.items()},
      "caveats":{"annotated_attr_false":sum(1 for r in it if not r.get("annotated_attr")),"event_date_null":sum(1 for r in it if r.get("event_date") is None),"positive_tiles":sum(1 for r in it if r.get("mask_positive_pixels",0)>0),"note":"Italy masks exist and are time-invariant; annotation provenance attributes are absent, so this region is a development-sealed replication, not a confirmatory fold"}}
base=json.loads((ROOT/"loco_folds.json").read_text()); out={"schema":"sen12-loco-folds-italy-v0","sample_contract_sha256":base.get("sample_contract_sha256"),"folds":[fold]}
(ROOT/"loco_folds_italy.json").write_text(json.dumps(out,indent=1)); print(json.dumps({k:fold[k] for k in ("sample_counts","caveats")}))
