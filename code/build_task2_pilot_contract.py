#!/usr/bin/env python3
"""Task-2 → pilot 입력 계약 변환: task2_geo_folds.json 을 pilot 이 읽는 loco_folds 형식(fold/test_region/val_region/train_regions)과
sample_contract.jsonl(sample_id, region=fold 이름) 로 변환. 캐시에 실제 존재하는 칩만 포함."""
import json, hashlib
from pathlib import Path
A=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm"); C=Path("/home/work/data/olmoearth/task2_cache"); OUT=Path("/home/work/data/olmoearth/task2_contract"); OUT.mkdir(exist_ok=True)
folds=json.loads((A/"task2_geo_folds.json").read_text())["folds"]; have={p.stem for p in (C/"emb_fp16").glob("*.npy")}
rows=[]; counts={}
for f in folds:
    ids=[i for i in f["test_ids"] if i in have]; counts[f["fold"]]=len(ids)
    for i in ids: rows.append({"sample_id":i,"region":f["fold"],"error":None,"s15_eligible":True})
(OUT/"sample_contract.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
names=[f["fold"] for f in folds]; loco={"schema":"task2-loco-folds-v1","folds":[]}
for k,f in enumerate(folds):
    val=names[(k+1)%8]; train=[n for n in names if n not in (f["fold"],val)]
    loco["folds"].append({"fold":f"holdout_{f['fold']}","test_region":f["fold"],"val_region":val,"train_regions":train,
                          "sample_counts":{"test":counts[f["fold"]],"val":counts[val],"train":sum(counts[n] for n in train)},"zones":f["zones"]})
(OUT/"loco_folds.json").write_text(json.dumps(loco,indent=1))
print("chips in cache",len(have),"contract rows",len(rows),{k:v for k,v in counts.items()}); print("sha", hashlib.sha256((OUT/"sample_contract.jsonl").read_bytes()).hexdigest()[:16])
