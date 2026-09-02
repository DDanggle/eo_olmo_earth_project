#!/usr/bin/env python3
"""PT-0 — CacheTune target support/query 공간분리 manifest (CPU, 모델 지표 미열람).

사전 등록: config/cachetune_pt0_preregistration_v0.json. 개발 지역 후보 china·chimanimani.
규칙(결과 보기 전 고정):
  - 타일 footprint 1.28 km. 같은 ann_id(이벤트)를 공유하는 타일은 한 그룹. 그룹 중심을 10 km 격자 셀에 묶어 spatial block.
  - block 을 x 좌표 순으로 정렬해 앞쪽부터 support pool 에 배정하되 support pool 양성 타일이 전체 양성의 40% 에 도달하면 나머지는 query.
  - buffer: query 타일 중 support pool 타일과 중심 거리 < 3 km 인 것은 제거(양쪽 경계 오염 방지).
  - support draw: K∈{5,20}, 3회(seed 1,2,3). 양성 타일 최소 ceil(K*0.4) 포함, 나머지는 pool 전체에서 무작위. 같은 ID 를 모든 arm 이 공유.
  - K 실현 불가(pool 양성 < ceil(K*0.4) 또는 query 양성 < 10) 이면 그 K 는 infeasible 로 봉인하고 모델 결과 없이 대체 사유 기록.
"""
import json, hashlib, math, glob, random
from pathlib import Path
import numpy as np, xarray as xr
NC=Path("/home/work/data/sen12landslides/extracted"); CACHE=Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani")
OUT=Path("/home/work/data/olmoearth/artifacts/cachetune_pt0"); OUT.mkdir(parents=True,exist_ok=True)
REGIONS=["china","chimanimani"]; KS=[5,20]; BLOCK_M=10_000; BUFFER_M=3_000; POS_FRAC_SUPPORT=0.40
def sha(o): return hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
rep={"schema":"cachetune-pt0-manifest-v1","rules":__doc__,"regions":{}}
for region in REGIONS:
    tiles=[]
    for f in sorted(glob.glob(str(NC/f"{region}_s2_*.nc"))):
        sid=Path(f).stem
        with xr.open_dataset(f, cache=False) as ds:
            at=ds.attrs; cx=float(at.get("center_lon") or 0); cy=float(at.get("center_lat") or 0); crs=str(at.get("crs")); ann=str(at.get("ann_id") or "")
        m=np.load(CACHE/"mask_u8"/f"{sid}.npy"); pos=int(m.sum())
        tiles.append({"sid":sid,"x":cx,"y":cy,"crs":crs,"ann":ann,"pos_px":pos,"positive":pos>0})
    crss={t["crs"] for t in tiles}
    # union-find: ann_id 공유 → 같은 그룹
    parent={t["sid"]:t["sid"] for t in tiles}
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b): parent[find(a)]=find(b)
    by_ann={}
    for t in tiles:
        for a in [s for s in t["ann"].split(",") if s]:
            by_ann.setdefault(a,[]).append(t["sid"])
    for sids in by_ann.values():
        for s in sids[1:]: union(sids[0],s)
    groups={}
    for t in tiles: groups.setdefault(find(t["sid"]),[]).append(t)
    # 그룹 → block (10 km 셀, 그룹 중심 기준)
    blocks={}
    for g,ts in groups.items():
        gx=np.mean([t["x"] for t in ts]); gy=np.mean([t["y"] for t in ts])
        key=(int(gx//BLOCK_M), int(gy//BLOCK_M)); blocks.setdefault(key,[]).extend(ts)
    order=sorted(blocks.keys())  # x 셀 우선 정렬
    total_pos=sum(t["positive"] for t in tiles); pool=[]; query=[]; acc=0
    for k in order:
        if acc < POS_FRAC_SUPPORT*total_pos: pool+=blocks[k]; acc+=sum(t["positive"] for t in blocks[k])
        else: query+=blocks[k]
    # buffer
    px=np.array([[t["x"],t["y"]] for t in pool]) if pool else np.zeros((0,2))
    def near_pool(t): return len(px)>0 and float(np.min(np.hypot(px[:,0]-t["x"],px[:,1]-t["y"])))<BUFFER_M
    dropped=[t["sid"] for t in query if near_pool(t)]; query=[t for t in query if not near_pool(t)]
    pool_pos=[t for t in pool if t["positive"]]; pool_neg=[t for t in pool if not t["positive"]]
    q_pos=sum(t["positive"] for t in query)
    draws={}
    for K in KS:
        need_pos=math.ceil(K*0.4); feasible = len(pool_pos)>=need_pos and len(pool)>=K and q_pos>=10
        draws[str(K)]={"feasible":feasible,"need_pos":need_pos,"pool_pos":len(pool_pos),"query_pos_tiles":q_pos,"draws":[]}
        if not feasible: continue
        for seed in (1,2,3):
            rng=random.Random(20260902*100+seed+K)
            pos_pick=rng.sample([t["sid"] for t in pool_pos], need_pos)
            rest=[t["sid"] for t in pool if t["sid"] not in pos_pick]
            pick=pos_pick+rng.sample(rest, K-need_pos)
            draws[str(K)]["draws"].append({"seed":seed,"support_ids":sorted(pick),"support_pos_tiles":sum(1 for s in pick if s in {t["sid"] for t in pool_pos})})
    man={"region":region,"crs":sorted(crss),"n_tiles":len(tiles),"n_positive_tiles":total_pos,"n_groups":len(groups),"n_blocks":len(blocks),
         "support_pool":{"n":len(pool),"positive":len(pool_pos),"ids":sorted(t["sid"] for t in pool)},
         "query":{"n":len(query),"positive":q_pos,"ids":sorted(t["sid"] for t in query)},"buffer_dropped":sorted(dropped),
         "block_m":BLOCK_M,"buffer_m":BUFFER_M,"draws":draws}
    assert not set(man["support_pool"]["ids"]) & set(man["query"]["ids"])
    man["sha256"]=sha({k:v for k,v in man.items() if k!="sha256"})
    (OUT/f"{region}_manifest.json").write_text(json.dumps(man,indent=1)); rep["regions"][region]={k:v for k,v in man.items() if k not in ("support_pool","query","buffer_dropped","draws")}
    rep["regions"][region]["support_pool_n_pos"]=(len(pool),len(pool_pos)); rep["regions"][region]["query_n_pos"]=(len(query),q_pos); rep["regions"][region]["buffer_dropped_n"]=len(dropped)
    rep["regions"][region]["K_feasible"]={K:draws[str(K)]["feasible"] for K in KS}
    print(region, "tiles",len(tiles),"pos",total_pos,"groups",len(groups),"blocks",len(blocks),"| pool",len(pool),"(pos",len(pool_pos),") query",len(query),"(pos",q_pos,") dropped",len(dropped),"| feasible",{K:draws[str(K)]["feasible"] for K in KS}, flush=True)
(OUT/"pt0_summary.json").write_text(json.dumps(rep,indent=1,default=str)); print("PT0 DONE")
