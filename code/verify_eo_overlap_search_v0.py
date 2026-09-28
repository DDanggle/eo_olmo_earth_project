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
check('catalog counts',meta['kuro_records']==7000 and meta['kuro_events']==43 and meta['preview_count']==7 and meta['land_cover_assessed_records']==1)
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
for land,expected in [('cropland',1),('forest',0)]:
    r=query({'dataset':'kurosiwo','land_cover':land})
    c=r['coverage']
    check('partial '+land+' scope',r['status']=='partial_coverage' and c['scope_count']==7000 and c['assessed_count']==1 and c['unassessed_count']==6999 and r['matched_count']==expected,c)
    if land=='cropland':
        tile=r['records'][0];m=tile['land_cover_overlap']
        check('selected train case',tile['id']=='ks_00276' and tile['split']=='train' and tile['location_name']=='Larkana, Pakistan')
        check('geodesic measured overlap',abs(m['classes']['cropland']['flood_overlap_m2']-3665141.6010337174)<1e-5 and m['naive_projected_footprint_m2']>m['footprint_area_m2'])
        check('acquisition remains unknown',all(v is None for v in m['acquisition_dates'].values()))
        check('baseline and availability limits',m['baseline']['reference_end']=='2020-12-31' and m['baseline']['historical_byte_identity_verified'] is False)
        e=tile['evidence'][0];_,png,mime=get(e['image_url'])
        check('spatial preview integrity',mime=='image/png' and hashlib.sha256(png).hexdigest()==e['source']['preview_sha256'])
    else:check('observed tree zero not general absence',c['observed_zero_count']==1 and r['records']==[])
r=query({'dataset':'kurosiwo','land_cover':'cropland','aoi_id':'1111009','min_flood_pct':10})
check('filtered coverage before display limit',r['coverage']['scope_count']==423 and r['coverage']['assessed_count']==1 and r['coverage']['unassessed_count']==422,r['coverage'])
r=query({'dataset':'kurosiwo','land_cover':'cropland','aoi_id':'1111007'})
check('different event unavailable',r['status']=='needs_data' and r['coverage']['assessed_count']==0)
for path,expected in [('/api/search?'+urlencode({'q':json.dumps({'min_flood_pct':-1})}),400),('/asset_routes.private.json',404),('/previews/%2e%2e/asset_routes.private.json',404)]:
    try:
        code=get(path)[0]
    except HTTPError as e:
        code=e.code
    check('reject '+path,code==expected)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps({'scope':'HTTP integration checks, not scientific/model performance', 'base_url':args.url,'checks':checks,'passed':len(checks)},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'report':str(args.out)},ensure_ascii=False))
