#!/usr/bin/env python3
"""random-support 재현성 감사: fewshot_a1_a4.py --support random 과 동일한 deterministic draw 를 재구성해 support_ids·양성 타일 수·픽셀 prevalence·manifest SHA 를 봉인. 재학습 없음."""
import json,random,hashlib
from pathlib import Path
import numpy as np
ROOT=Path("/home/work/data/olmoearth"); MAN=ROOT/"artifacts/fewshot_confirmatory_manifests"; CACHE=ROOT/"sen12_pilot/holdout_chimanimani/mask_u8"
out={"schema":"random-support-audit-v1","draw_rule":"random.Random(100+seed+K*7).sample(sorted pool ids as stored in manifest['support_pool']['ids'], K); sorted","regions":{}}
for mf in sorted(MAN.glob("*_manifest.json")):
    man=json.loads(mf.read_text()); region=man["region"]; pool=man["support_pool"]["ids"]
    pos_pool=sum(1 for s in pool if np.load(CACHE/f"{s}.npy").any())
    reg={"manifest_sha256":hashlib.sha256(mf.read_bytes()).hexdigest(),"pool_n":len(pool),"pool_positive_tiles":pos_pool,"pool_positive_rate":pos_pool/len(pool),"draws":{}}
    for K in (5,20):
        for seed in (1,2,3):
            rng=random.Random(100+seed+K*7); sids=sorted(rng.sample(pool,K))
            masks=[np.load(CACHE/f"{s}.npy") for s in sids]
            reg["draws"][f"K{K}_s{seed}"]={"support_ids":sids,"positive_tiles":int(sum(m.any() for m in masks)),"pixel_prevalence":float(np.mean([m.mean() for m in masks]))}
    out["regions"][region]=reg
    print(region,"pool pos rate %.3f"%reg["pool_positive_rate"],{k:v["positive_tiles"] for k,v in reg["draws"].items()})
(ROOT/"artifacts/fewshot_confirmatory/random_support_audit.json").write_text(json.dumps(out,indent=1)); print("AUDIT DONE")
