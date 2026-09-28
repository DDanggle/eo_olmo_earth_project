"""HTTP integration checks of the local evidence browser; no model accuracy claim."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url',default='http://127.0.0.1:8774')
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
checks=[]
def get(path):
    with urlopen(args.url+path,timeout=10) as r:
        return r.status,r.read(),r.headers.get_content_type()
def query(q):
    return json.loads(get('/api/search?'+urlencode({'q':json.dumps(q)}))[1])
def check(name,condition,detail=None):
    if not condition: raise AssertionError(name)
    checks.append({'name':name,'passed':True,'detail':detail})

meta=json.loads(get('/api/meta')[1])
check('catalog counts',meta['kuro_records']==7000 and meta['kuro_events']==43 and meta['preview_count']==6)
check('A summary',meta['h']['answer_counts']=={'change_supported':3,'no_visible_change':4,'insufficient_evidence':5})
q={'dataset':'kurosiwo','bbox':[84,24,87,27],'start':'2019-01-01','end':'2019-12-31','min_flood_pct':10}
r=query(q)
check('spatial date query',r['matched_count']==72,{'query':q,'matches':r['matched_count']})
first=r['records'][0]
check('evidence association',first['id']=='ks_05229' and len(first['evidence'])==1)
check('event date not acquisition',first['time_basis']=='event_date' and first['evidence'][0]['acquisition_dates'] is None)
_,png,mime=get(first['evidence'][0]['image_url'])
check('preview image integrity',mime=='image/png' and hashlib.sha256(png).hexdigest()==first['evidence'][0]['source']['preview_sha256'])
for status,count in [('change_supported',3),('no_visible_change',4),('insufficient_evidence',5)]:
    r=query({'dataset':'sn7','status':status})
    check('A '+status,r['matched_count']==count)
r=query({'dataset':'sn7','status':'change_supported','start':'2018-02-15','end':'2018-02-28'})
check('month precision is interval',r['matched_count']==1 and r['records'][0]['first_change_date']=='2018-02' and r['records'][0]['time_interval']==['2018-02-01','2018-02-28'])
check('validation split',query({'dataset':'kurosiwo','split':'validation'})['matched_count']==1000)
for land in ['cropland','forest']:
    r=query({'dataset':'kurosiwo','land_cover':land})
    check('missing '+land+' is not negative',r['status']=='needs_data' and bool(r.get('blockers')))
for path,expected in [('/api/search?'+urlencode({'q':json.dumps({'min_flood_pct':-1})}),400),('/asset_routes.private.json',404),('/previews/%2e%2e/asset_routes.private.json',404)]:
    try:
        code=get(path)[0]
    except HTTPError as e:
        code=e.code
    check('reject '+path,code==expected)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps({'scope':'HTTP integration checks, not scientific/model performance', 'base_url':args.url,'checks':checks,'passed':len(checks)},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'report':str(args.out)},ensure_ascii=False))
