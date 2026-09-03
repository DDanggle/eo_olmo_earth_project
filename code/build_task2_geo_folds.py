#!/usr/bin/env python3
"""Task-2 지리 폴드 봉인 — 등록 addendum_v2. 입력: artifacts/task2_solar_farm/window_index.jsonl(전량 감사 산출물).
그룹 = UTM 존; 같은 CRS 안 bounds 겹침은 union-find 로 병합. 그룹을 존 이름 정렬 순으로 8폴드에 탐욕 균형 배정. val = 다음 폴드. SHA 봉인."""
import json, hashlib, collections
from pathlib import Path
OUT=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm"); rows=[json.loads(l) for l in open(OUT/"window_index.jsonl")]
elig=[r for r in rows if r["s2_timesteps"]==4 and r["s2_bands_complete"] and r["label_pos_px"]>=0]
print("windows",len(rows),"eligible",len(elig))
parent={r["id"]:r["id"] for r in elig}
def find(a):
    while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
    return a
bycrs=collections.defaultdict(list)
for r in elig: bycrs[r["crs"]].append(r)
for crs,lst in bycrs.items():
    lst.sort(key=lambda r:r["bounds"][0])
    for i,a in enumerate(lst):
        for b in lst[i+1:i+60]:
            if b["bounds"][0]>=a["bounds"][2]: break
            if a["bounds"][0]<b["bounds"][2] and b["bounds"][0]<a["bounds"][2] and a["bounds"][1]<b["bounds"][3] and b["bounds"][1]<a["bounds"][3]:
                parent[find(a["id"])]=find(b["id"])
groups=collections.defaultdict(list)
for r in elig: groups[(r["crs"],find(r["id"]))].append(r["id"])
# 존 단위 묶음: 같은 CRS 의 그룹들을 하나의 존 그룹으로 (겹침 병합은 존 안에서 이미 보장)
zone_groups=collections.defaultdict(list)
for (crs,_),ids in groups.items(): zone_groups[crs]+=ids
zones=sorted(zone_groups.items(), key=lambda kv:(-len(kv[1]),kv[0]))  # 큰 존부터 배정하면 균형이 좋음
folds=[[] for _ in range(8)]; fz=[[] for _ in range(8)]
for crs,ids in zones:
    k=min(range(8), key=lambda i:len(folds[i])); folds[k]+=ids; fz[k].append(crs)
out={"schema":"task2-geo-folds-v1","rule":"UTM zone groups (overlap-merged) packed greedily into 8 folds; val fold = (k+1)%8; test fold = k","n_eligible":len(elig),"folds":[]}
for k in range(8):
    ids=sorted(folds[k]); pos=sum(1 for r in elig if r["id"] in set(ids) and r["label_pos_px"]>0)
    out["folds"].append({"fold":f"task2_fold{k}","test_ids":ids,"n":len(ids),"positive_windows":pos,"zones":sorted(fz[k]),"val_fold":f"task2_fold{(k+1)%8}","sha256_ids":hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    print(f"fold{k}: n={len(ids)} pos={pos} zones={len(fz[k])}")
(OUT/"task2_geo_folds.json").write_text(json.dumps(out,indent=1)); print("FOLDS SEALED")
