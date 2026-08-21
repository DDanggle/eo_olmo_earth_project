import glob, numpy as np, rasterio, json
from pyproj import Transformer
from scipy.ndimage import uniform_filter

BASE="/home/work/data/olmoearth/embed_search/dataset/windows/default"
YEARS={"2023":"jeju23_","2024":"jeju_","2025":"jeju25_","2026":"jeju26r_"}
data={}
for y,pref in YEARS.items():
    d={}
    for t in glob.glob(f"{BASE}/{pref}*/layers/embeddings/*/geotiff.tif"):
        name=t.split("/windows/default/")[1].split("/")[0]
        key=name.replace(pref,"")
        with rasterio.open(t) as s:
            d[key]={"arr":s.read().astype(np.float32),"tr":s.transform,"crs":s.crs}
    data[y]=d
    print(y,len(d),"windows",flush=True)

for y,d in data.items():
    mu=np.mean([w["arr"].mean(axis=(1,2)) for w in d.values()],axis=0)[:,None,None]
    for w in d.values():
        a=w["arr"]-mu; n=np.linalg.norm(a,axis=0,keepdims=True); n[n==0]=1; w["arr"]=a/n
print("centered",flush=True)

pairs=[("2023","2024"),("2024","2025"),("2025","2026")]
keys=sorted(set.intersection(*[set(d) for d in data.values()]))
print("matched:",len(keys),flush=True)

def stitch(vals):
    ws=[data["2024"][k] for k in keys]
    xs=[w["tr"].c for w in ws]; ys=[w["tr"].f for w in ws]
    x0,y0=min(xs),max(ys); px=40.0
    W=int((max(xs)-x0)/px)+256; H=int((y0-min(ys))/px)+256
    cv=np.full((H,W),np.nan,np.float32)
    for k in keys:
        w=data["2024"][k]
        cx=int((w["tr"].c-x0)/px); cy=int((y0-w["tr"].f)/px)
        cv[cy:cy+256,cx:cx+256]=vals[k]
    return cv,(x0,y0,px)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(15,17))
tr_back=Transformer.from_crs(data["2024"][keys[0]]["crs"],"EPSG:4326",always_xy=True)
report={}
for ax,(y1,y2) in zip(axes,pairs):
    vals={k: 1.0-np.einsum("chw,chw->hw",data[y1][k]["arr"],data[y2][k]["arr"]) for k in keys}
    cv,geo=stitch(vals)
    im=ax.imshow(cv,cmap="inferno",vmin=0,vmax=0.6)
    ax.set_title(f"change {y1} -> {y2} (1 - cosine)"); ax.axis("off")
    plt.colorbar(im,ax=ax,fraction=0.02)
    sm=uniform_filter(np.nan_to_num(cv,nan=0),size=3)
    order=np.argsort(sm.ravel())[::-1][:4000]
    seen=[]; tops=[]
    for f in order:
        r,c=divmod(int(f),cv.shape[1])
        if any(abs(r-rr)<12 and abs(c-cc)<12 for rr,cc in seen): continue
        seen.append((r,c)); x=geo[0]+c*geo[2]; y=geo[1]-r*geo[2]
        lon,lat=tr_back.transform(x,y)
        tops.append([round(float(sm[r,c]),3),round(lat,4),round(lon,4)])
        ax.plot(c,r,"o",ms=9,mfc="none",mec="cyan",mew=1.4)
        if len(tops)>=20: break
    report[f"{y1}->{y2}"]=tops
plt.tight_layout(); plt.savefig("/home/work/data/olmoearth/embed_search/jeju_change_4yr.png",dpi=95,bbox_inches="tight")
json.dump(report,open("/home/work/data/olmoearth/embed_search/jeju_change_top.json","w"),indent=1)
print("DONE",flush=True)
