#!/usr/bin/env python3
"""KuroSiwo (GEO-Bench-2) data audit before freezing the streaming prereg: units, nodata, invalid fraction, class balance, activations per split, date spread."""
import json, numpy as np, tacoreader, rasterio, collections, sys
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla")
t=tacoreader.load(str(ROOT)); n=len(t); print("samples",n)
splits=collections.Counter(t["tortilla:data_split"]); print("splits",dict(splits))
acts=collections.defaultdict(set)
for sp,a in zip(t["tortilla:data_split"],t["actid"]): acts[sp].add(int(a))
print("activations per split",{k:len(v) for k,v in acts.items()},"train∩test",len(acts["train"]&acts["test"]))
fd=t["flood_date"]; print("flood_date range",str(min(fd))[:10],str(max(fd))[:10])
print("pflood stats",float(np.mean(t["pflood"])),float(np.median(t["pflood"])),"tiles with flood>0:",int((t["pflood"]>0).sum()))
rng=np.random.default_rng(0); idx=rng.choice(n,size=min(60,n),replace=False); vv=[];vh=[];inv=[];cls=collections.Counter();neg=0;zero=0
for i in idx:
    s=t.read(int(i))
    with rasterio.open(s.read(0)) as r: a=r.read().astype("float64"); vv.append(a[0].ravel()); vh.append(a[1].ravel()); neg+=int((a<0).sum()); zero+=int((a==0).sum())
    with rasterio.open(s.read(4)) as r: m=r.read(1); cls.update(collections.Counter(m.ravel().tolist()))
    with rasterio.open(s.read(5)) as r: inv.append(float((r.read(1)>0).mean()))
vv=np.concatenate(vv); vh=np.concatenate(vh)
print("VV linear p1/p50/p99",np.percentile(vv,[1,50,99]).round(4).tolist(),"-> dB",(10*np.log10(np.clip(np.percentile(vv,[1,50,99]),1e-6,None))).round(1).tolist())
print("VH linear p1/p50/p99",np.percentile(vh,[1,50,99]).round(4).tolist(),"-> dB",(10*np.log10(np.clip(np.percentile(vh,[1,50,99]),1e-6,None))).round(1).tolist())
print("negative values",neg,"zeros",zero,"of",vv.size*2,"| invalid-mask fraction mean",round(float(np.mean(inv)),4))
print("mask class pixel counts (0 nodata,1 no water,2 perm water,3 flood)",dict(cls))
json.dump({"n":n,"splits":dict(splits),"activations":{k:len(v) for k,v in acts.items()},"vv_db_p":(10*np.log10(np.clip(np.percentile(vv,[1,50,99]),1e-6,None))).round(1).tolist(),"vh_db_p":(10*np.log10(np.clip(np.percentile(vh,[1,50,99]),1e-6,None))).round(1).tolist(),"neg":neg,"zeros":zero,"invalid_frac":float(np.mean(inv)),"classes":{str(k):v for k,v in cls.items()}},open("/home/work/data/olmoearth/artifacts/kurosiwo_audit.json","w"),indent=1); print("AUDIT DONE")
