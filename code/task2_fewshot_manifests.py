#!/usr/bin/env python3
"""Task-2 few-shot support/query manifest (규칙 = 확증 v2: 10 km block, 3 km buffer, pool 양성 상한 45%, K=5/20 × 3 draw, 양성 ≥ ceil(0.4K)).
Task-2 는 ann_id 가 없으므로 그룹 = 같은 CRS 안 bounds 겹침(union-find) 만 사용. 지표 열람 전 실행."""
import json, hashlib, math, random, collections
from pathlib import Path
import numpy as np
A=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm"); C=Path("/home/work/data/olmoearth/task2_cache"); OUT=Path("/home/work/data/olmoearth/artifacts/task2_fewshot_manifests"); OUT.mkdir(parents=True,exist_ok=True)
idx={r["id"]:r for r in (json.loads(l) for l in open(A/"window_index.jsonl"))}; folds=json.loads((A/"task2_geo_folds.json").read_text())["folds"]
KS=[5,20]; BLOCK=10_000; BUF=3_000; POSF=0.40
def sha(o): return hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
summ={"schema":"task2-fewshot-manifest-v1","rule":"v2 (pool positive cap 45%), groups = same-CRS overlapping bounds","regions":{}}
for f in folds:
    reg=f["fold"]; tiles=[]
    for sid in f["test_ids"]:
        if not (C/"mask_u8"/f"{sid}.npy").exists(): continue
        r=idx[sid]; b=r["bounds"]; m=np.load(C/"mask_u8"/f"{sid}.npy"); tiles.append({"sid":sid,"crs":r["crs"],"x":(b[0]+b[2])/2,"y":(b[1]+b[3])/2,"b":b,"positive":bool(m.any())})
    parent={t["sid"]:t["sid"] for t in tiles}
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    bycrs=collections.defaultdict(list)
    for t in tiles: bycrs[t["crs"]].append(t)
    for lst in bycrs.values():
        lst.sort(key=lambda t:t["b"][0])
        for i,a in enumerate(lst):
            for c in lst[i+1:i+60]:
                if c["b"][0]>=a["b"][2]: break
                if a["b"][0]<c["b"][2] and c["b"][0]<a["b"][2] and a["b"][1]<c["b"][3] and c["b"][1]<a["b"][3]: parent[find(a["sid"])]=find(c["sid"])
    groups=collections.defaultdict(list)
    for t in tiles: groups[find(t["sid"])].append(t)
    blocks=collections.defaultdict(list)
    for g,ts in groups.items():
        gx=np.mean([t["x"] for t in ts]); gy=np.mean([t["y"] for t in ts]); blocks[(ts[0]["crs"],int(gx//BLOCK),int(gy//BLOCK))].extend(ts)
    order=sorted(blocks.keys()); total_pos=sum(t["positive"] for t in tiles); pool=[]; query=[]; acc=0
    for k in order:
        bp=sum(t["positive"] for t in blocks[k])
        if acc+bp<=0.45*total_pos and acc<POSF*total_pos: pool+=blocks[k]; acc+=bp
        else: query+=blocks[k]
    def near_pool(t):
        cand=[p for p in pool if p["crs"]==t["crs"]]
        return any(math.hypot(p["x"]-t["x"],p["y"]-t["y"])<BUF for p in cand)
    dropped=[t["sid"] for t in query if near_pool(t)]; query=[t for t in query if t["sid"] not in set(dropped)]
    pool_pos=[t for t in pool if t["positive"]]; q_pos=sum(t["positive"] for t in query); draws={}
    for K in KS:
        need=math.ceil(K*POSF); feas=len(pool_pos)>=need and len(pool)>=K and q_pos>=10; draws[str(K)]={"feasible":feas,"need_pos":need,"pool_pos":len(pool_pos),"query_pos_tiles":q_pos,"draws":[]}
        if not feas: continue
        for seed in (1,2,3):
            rng=random.Random(20260903*100+seed+K); pp=rng.sample([t["sid"] for t in pool_pos],need); rest=[t["sid"] for t in pool if t["sid"] not in pp]; pick=pp+rng.sample(rest,K-need)
            draws[str(K)]["draws"].append({"seed":seed,"support_ids":sorted(pick),"support_pos_tiles":sum(1 for s in pick if s in {t["sid"] for t in pool_pos})})
    man={"region":reg,"n_tiles":len(tiles),"n_positive_tiles":total_pos,"n_groups":len(groups),"n_blocks":len(blocks),"support_pool":{"n":len(pool),"positive":len(pool_pos),"ids":sorted(t["sid"] for t in pool)},
         "query":{"n":len(query),"positive":q_pos,"ids":sorted(t["sid"] for t in query)},"buffer_dropped":sorted(dropped),"draws":draws}
    assert not set(man["support_pool"]["ids"])&set(man["query"]["ids"]); man["sha256"]=sha({k:v for k,v in man.items() if k!="sha256"})
    (OUT/f"{reg}_manifest.json").write_text(json.dumps(man,indent=1))
    summ["regions"][reg]={"pool":(len(pool),len(pool_pos)),"query":(len(query),q_pos),"dropped":len(dropped),"feasible":{K:draws[str(K)]["feasible"] for K in KS}}
    print(reg,"tiles",len(tiles),"pos",total_pos,"| pool",len(pool),"(pos",len(pool_pos),") query",len(query),"(pos",q_pos,") dropped",len(dropped),"| feasible",{K:draws[str(K)]["feasible"] for K in KS},flush=True)
(OUT/"summary.json").write_text(json.dumps(summ,indent=1,default=str)); print("MANIFESTS DONE")
