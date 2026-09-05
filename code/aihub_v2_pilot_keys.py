"""AIHUB_CUBE_V2_CONTRACT gate 1: stratified 40-key pilot = 20 v1-severe (all-band-zero >=10%) + 20 v1-clean (<0.1%), spread over dates/platforms deterministically (seed 0)."""
import json, random, numpy as np
from pathlib import Path
V1=Path("/home/work/data/olmoearth/aihub/s2_12band/arrays"); OUT=Path("/home/work/data/olmoearth/aihub/s2_12band_v2"); OUT.mkdir(exist_ok=True)
inv={json.loads(l)["key"]:json.loads(l) for l in open("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl") if l.strip()}
rows=[]
for p in sorted(V1.glob("*.npy")):
    a=np.load(p,mmap_mode="r"); z=float((np.asarray(a)==0).all(axis=0).mean()) if a.ndim==3 else float((np.asarray(a)==0).all(axis=(0,1)).mean()); rows.append((p.stem,z))
sev=[k for k,z in rows if z>=0.10]; cln=[k for k,z in rows if z<0.001]
def spread(keys,n):
    r=random.Random(0); by={}
    for k in keys: by.setdefault((inv.get(k,{}).get("date",k[-8:]),inv.get(k,{}).get("platform","?")),[]).append(k)
    groups=sorted(by); r.shuffle(groups); out=[]
    while len(out)<n and groups:
        for g in list(groups):
            if by[g]: out.append(by[g].pop(0))
            else: groups.remove(g)
            if len(out)>=n: break
    return sorted(out)
keys=spread(sev,20)+spread(cln,20); (OUT/"pilot40_keys.txt").write_text("\n".join(keys)+"\n")
json.dump({"n_v1":len(rows),"n_severe":len(sev),"n_clean":len(cln),"pilot_severe":spread(sev,20),"pilot_clean":spread(cln,20),"rule":"seed 0, spread over (date,platform)"},open(OUT/"pilot40_selection.json","w"),indent=1); print("keys",len(keys),"severe",len(sev),"clean",len(cln))
