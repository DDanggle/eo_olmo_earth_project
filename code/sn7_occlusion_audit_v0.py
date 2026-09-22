#!/usr/bin/env python3
"""SpaceNet 7 occlusion audit v0 (config/sn7_occlusion_audit_prereg_v0.json). CPU. Per AOI x month UDM unusable fraction; change events (building id first appearance) vs footprint occlusion at t-1,t,t+1."""
import json, sys, re, numpy as np
from pathlib import Path
import rasterio
from shapely.geometry import shape
from rasterio import features
ROOT=Path("/home/work/data/olmoearth/spacenet7/train"); OUT=Path("/home/work/data/olmoearth/spacenet7/audit_v0"); OUT.mkdir(parents=True,exist_ok=True)
PROBE="--probe" in sys.argv; aois=sorted(p.name for p in ROOT.iterdir() if p.is_dir()); aois=aois[:1] if PROBE else aois
mon=lambda name: re.search(r"global_monthly_(\d{4})_(\d{2})",name).groups()
res={"schema":"sn7-occlusion-audit-v0","aois":{},"events":[]}
for aoi in aois:
    # SN7 ships a UDM file only for months that have unusable pixels; months without a UDM file are treated as fully usable (frac 0).
    udm=sorted((ROOT/aoi/"UDM_masks").glob("*.tif")); lab=sorted((ROOT/aoi/"labels_match").glob("*.geojson")); imgs=sorted((ROOT/aoi/"images").glob("*.tif"))
    months=sorted({"-".join(mon(p.name)) for p in imgs}); frac={k:0.0 for k in months}
    with rasterio.open(imgs[0]) as ds: tr=ds.transform; shp=(ds.height,ds.width)
    udm_by={}
    for p in udm:
        y,m=mon(p.name); key=f"{y}-{m}"; udm_by[key]=p
        with rasterio.open(p) as ds: a=ds.read(1); frac[key]=float((a!=0).mean())
    occ2=[k for k in months if frac[k]>.2]; occ5=[k for k in months if frac[k]>.5]
    run=0;best=0
    for k in months: run=run+1 if frac[k]>.2 else 0; best=max(best,run)
    # change events: first appearance month per building id
    first={}; geoms={}
    for p in lab:
        y,m=mon(p.name); key=f"{y}-{m}"
        try: fc=json.loads(p.read_text())
        except Exception: continue
        for f in fc.get("features",[]):
            pr=f.get("properties") or {}; i=pr.get("Id",pr.get("id"))
            if i is None or f.get("geometry") is None: continue
            if i not in first or key<first[i]: first[i]=key; geoms[i]=shape(f["geometry"])
    idx={k:i for i,k in enumerate(months)}; ev_occ=0; ev_n=0; udm_arr={}
    def udm_of(key):
        if key not in udm_arr:
            if key in udm_by:
                with rasterio.open(udm_by[key]) as ds: udm_arr[key]=ds.read(1)!=0
            else: udm_arr[key]=np.zeros(shp,dtype=bool)
        return udm_arr[key]
    for i,k in first.items():
        if k not in idx or idx[k]==0: continue
        t=idx[k]; nb=[months[j] for j in (t-1,t,t+1) if 0<=j<len(months)]
        try: mask=features.rasterize([(geoms[i].buffer(2*abs(tr.a)),1)],out_shape=shp,transform=tr,fill=0,dtype="uint8").astype(bool)
        except Exception: continue
        if mask.sum()==0: continue
        fr={m_:float(udm_of(m_)[mask].mean()) for m_ in nb}; occ=any(v>.5 for v in fr.values()); ev_n+=1; ev_occ+=int(occ)
        if len(res["events"])<5000: res["events"].append({"aoi":aoi,"id":str(i),"first":k,"nb_unusable":fr,"occluded_neighbour":occ})
    res["aois"][aoi]={"n_months":len(months),"n_udm_files":len(udm),"months":months,"udm_unusable_frac":frac,"n_partially_occluded(>.2)":len(occ2),"n_mostly_occluded(>.5)":len(occ5),"longest_occluded_run":best,"n_change_events":ev_n,"n_events_occluded_neighbour":ev_occ}
    print(aoi,len(months),"occ>.2:",len(occ2),"occ>.5:",len(occ5),"run:",best,"events:",ev_n,"occluded_nb:",ev_occ,flush=True)
A=res["aois"]; tot_ev=sum(v["n_change_events"] for v in A.values()); tot_occ=sum(v["n_events_occluded_neighbour"] for v in A.values())
res["aggregate"]={"n_aois":len(A),"aois_with_>=2_partially_occluded_months":sum(1 for v in A.values() if v["n_partially_occluded(>.2)"]>=2),"change_events":tot_ev,"frac_events_with_occluded_neighbour":(tot_occ/tot_ev if tot_ev else None)}
g=res["aggregate"]; res["gate"]={"phenomenon_present":bool(g["frac_events_with_occluded_neighbour"] is not None and g["frac_events_with_occluded_neighbour"]>=.15 and g["aois_with_>=2_partially_occluded_months"]>=20)}
(OUT/("summary_probe.json" if PROBE else "summary.json")).write_text(json.dumps(res,indent=1)); print(json.dumps({**g,**res["gate"]})); print("SN7 AUDIT DONE")
