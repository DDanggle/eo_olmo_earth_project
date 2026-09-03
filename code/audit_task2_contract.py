#!/usr/bin/env python3
"""Task-2 Solar Farm — 계약 감사 (CPU, 모델 지표 없음). 등록: config/task2_extension_prereg_v0.json addendum.
윈도우마다: CRS(UTM 존)·bounds·크기·split·S2 시점 수(sentinel2, .1, .2, .3 의 completed)·밴드 그룹 완전성·label_raster 값 분포·양성 픽셀 수.
출력: contract_audit.json(요약) + window_index.jsonl(윈도우별). 지리 폴드 구성 가능성(UTM 존 분포·양성 분포)까지만 판단."""
import json, collections, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path("/home/work/data/task2_solar_farm/extracted"); OUT=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm"); OUT.mkdir(parents=True,exist_ok=True)
S2_GROUPS=["B02_B03_B04_B08","B05_B06_B07_B8A_B11_B12","B01_B09_B10"]
idx=open(OUT/"window_index.jsonl","w"); crs_cnt=collections.Counter(); split_cnt=collections.Counter(); ts_cnt=collections.Counter(); sizes=collections.Counter(); labvals=collections.Counter()
n=0; n_full=0; pos_tiles=0; pos_px_total=0; overlap_check=collections.defaultdict(list); missing_label=0
wins=sorted((ROOT/"windows/default").iterdir())
for w in wins:
    md=json.loads((w/"metadata.json").read_text()); crs=md["projection"]["crs"]; b=md["bounds"]; sz=(b[2]-b[0], b[3]-b[1]); split=md.get("options",{}).get("split")
    ts=[t for t in ["sentinel2","sentinel2.1","sentinel2.2","sentinel2.3"] if (w/"layers"/t/"completed").exists()]
    full=all((w/"layers"/t/g/"geotiff.tif").exists() for t in ts for g in S2_GROUPS)
    lp=w/"layers/label_raster/label/image.png"
    if lp.exists():
        a=np.array(Image.open(lp)); u,c=np.unique(a,return_counts=True); labvals.update(dict(zip(u.tolist(),c.tolist()))); pos=int((a>0).sum()); shape=a.shape
    else: missing_label+=1; pos=-1; shape=None
    rec={"id":w.name,"crs":crs,"bounds":b,"size_px":sz,"split":split,"s2_timesteps":len(ts),"s2_bands_complete":full,"label_pos_px":pos,"label_shape":shape,"time_range":md.get("time_range")}
    idx.write(json.dumps(rec)+"\n"); n+=1; crs_cnt[crs]+=1; split_cnt[split]+=1; ts_cnt[len(ts)]+=1; sizes[tuple(sz)]+=1
    if full and len(ts)==4: n_full+=1
    if pos>0: pos_tiles+=1; pos_px_total+=pos
    overlap_check[crs].append((b,w.name))
    if n%2000==0: print(n,"windows…",flush=True)
idx.close()
# 같은 CRS 안 bounds 겹침(공간 누수) — O(n^2) 회피: x 정렬 sweep
overlaps=0
for crs,lst in overlap_check.items():
    lst.sort(key=lambda t:t[0][0])
    for i,(b1,_) in enumerate(lst):
        for b2,_ in lst[i+1:i+50]:
            if b2[0]>=b1[2]: break
            if b1[0]<b2[2] and b2[0]<b1[2] and b1[1]<b2[3] and b2[1]<b1[3]: overlaps+=1
rep={"schema":"task2-solar-farm-contract-audit-v1","n_windows":n,"n_full_4x12band":n_full,"s2_timesteps_dist":dict(ts_cnt),"size_px_dist":{str(k):v for k,v in sizes.most_common(5)},
     "crs_utm_zones":len(crs_cnt),"crs_top":crs_cnt.most_common(15),"split_dist":dict(split_cnt),"label_values":{str(k):v for k,v in sorted(labvals.items())},"label_missing":missing_label,
     "positive_tiles":pos_tiles,"positive_tile_rate":pos_tiles/max(n,1),"positive_px_total":pos_px_total,"bounds_overlaps_same_crs":overlaps,
     "notes":["label positive = value>0 (semantics to be confirmed from config.json label class map)","time_range 6 months; S2 4 timesteps if .0-.3 completed","UTM zone = geographic grouping candidate"]}
(OUT/"contract_audit.json").write_text(json.dumps(rep,indent=1)); print(json.dumps(rep,indent=1)); print("AUDIT DONE")
