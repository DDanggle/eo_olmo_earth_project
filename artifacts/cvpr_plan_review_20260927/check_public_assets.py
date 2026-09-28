"""Public metadata requests only; never downloads model weights or dataset tarballs."""
import json,urllib.request
from pathlib import Path
from datetime import datetime,timezone

URLS={
 'official_1k_h5':'https://storage.googleapis.com/ai2-olmoearth-projects-public-data/pretraining_data/subset_1k/subset_1k_h5py_data.tar',
 'official_1k_geo':'https://storage.googleapis.com/ai2-olmoearth-projects-public-data/pretraining_data/subset_1k/subset_1k.tar',
 'current_base_api':'https://huggingface.co/api/models/allenai/OlmoEarth-v1_2-Base',
}
out=[]
for name,url in URLS.items():
 method='GET' if name.endswith('_api') else 'HEAD'
 row={'name':name,'url':url,'method':method}
 try:
  req=urllib.request.Request(url,method=method,headers={'User-Agent':'EO-research-readiness/1.0'})
  with urllib.request.urlopen(req,timeout=30) as r:
   row.update(status=r.status,final_url=r.url,content_length=r.headers.get('Content-Length'),etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'))
   if method=='GET':
    raw=r.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('metadata response too large')
    x=json.loads(raw)
    row.update(model_id=x.get('id'),revision=x.get('sha'),last_modified_api=x.get('lastModified'),gated=x.get('gated'),files=[a.get('rfilename') for a in x.get('siblings',[])])
 except Exception as e:row['error']=repr(e)
 out.append(row)
p=Path('/private/tmp/cvpr_plan_20260927/public_asset_metadata.json')
value={'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'metadata_only_no_corpus_or_weights_downloaded','assets':out}
p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
