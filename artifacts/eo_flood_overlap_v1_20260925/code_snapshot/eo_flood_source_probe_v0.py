#!/usr/bin/env python3
"""Bounded, CPU-only recovery of original KuroSiwo geometry and timestamps."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import rasterio
import tacoreader

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True);p.add_argument('--id',default='ks_00276');a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 t=tacoreader.load(str(a.source)); target=int(a.id.removeprefix('ks_')); inds=np.flatnonzero(t['tortilla:id'].astype(int).to_numpy()==target)
 if len(inds)!=1:raise ValueError('id not unique')
 i=int(inds[0]);r=t.iloc[i];s=t.read(i)
 if str(r['tortilla:data_split'])!='train':raise ValueError('Engineering pilot restricted to train')
 a.out.mkdir(parents=True)
 report={'id':a.id,'source':str(a.source),'source_size_bytes':a.source.stat().st_size,'selection':'single known positive train development case; not representative evaluation','row':{str(k):str(v) for k,v in r.items()},'sample_metadata':str(s),'sample_rows':[{str(k):str(v) for k,v in row.items()} for _,row in s.iterrows()],'assets':[],'code_sha256':sha(__file__)}
 for k in range(6):
  loc=s.read(k)
  with rasterio.open(loc) as src:
   data=src.read();out=a.out/f'asset_{k}.tif'
   profile=src.profile.copy();profile.update(driver='GTiff',compress='deflate');profile.pop('blockxsize',None);profile.pop('blockysize',None)
   with rasterio.open(out,'w',**profile) as dst:
    dst.write(data);dst.update_tags(**src.tags())
   un,co=np.unique(data,return_counts=True) if k>=4 else ([],[])
   report['assets'].append({'index':k,'path':out.name,'source_locator':str(loc),'sha256':sha(out),'crs':str(src.crs),'transform':list(src.transform),'bounds':list(src.bounds),'shape':list(data.shape),'dtype':str(data.dtype),'nodata':src.nodata,'tags':src.tags(),'band_tags':[src.tags(b) for b in src.indexes],'descriptions':list(src.descriptions),'counts':{str(int(v)):int(n) for v,n in zip(un,co)}})
 (a.out/'source_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
