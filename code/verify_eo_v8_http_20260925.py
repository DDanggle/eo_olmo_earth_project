"""Compare local v8 HTTP results with pinned disk catalog and prior reader data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8774')
    a=p.parse_args();base=a.url.rstrip('/')
    require(not a.report.exists(),'New report required')
    u=urlsplit(base);require(u.scheme=='http' and u.hostname in ('127.0.0.1','localhost'),'Local HTTP only')
    def read(name):return json.loads((a.snapshot/name).read_text())
    def get(route):
        with urlopen(base+route,timeout=10) as r:
            require(r.status==200,'HTTP failure');return r.read(),r.headers.get_content_type()
    def api(route):
        data,mime=get(route);require(mime=='application/json','JSON MIME differs');return json.loads(data)
    manifest=read('v8_build_manifest.json')
    for name,digest in manifest['output_files_sha256'].items():
        require(hashlib.sha256((a.snapshot/name).read_bytes()).hexdigest()==digest,'Snapshot changed: '+name)
    meta=api('/api/meta');require(meta==read('meta.json') and meta['preview_count']==9,'Metadata differs')
    research=api('/api/research');require(research['available'] and research['runs']==read('research_runs.json')['runs'],'Research differs')
    runs={r['id']:r for r in research['runs']}
    require('T0-DATE' in runs and '결과 없음' in runs['E5-EB']['status'] and 'metrics' not in runs['E5-EB'],'T0 absent or E5 outcomes invented')
    catalog={r['id']:r for r in read('catalog.json')['records']}
    require(sum('source_date_provenance' in r for r in catalog.values())==7000,'Date provenance coverage differs')
    readers=read('reader_cases.json')['cases']
    cases={}
    for tile in ('ks_06770','ks_05265','ks_05229','ks_00276'):
        result=api('/api/reader-case?'+urlencode({'tile':tile}))
        require(result['available']==(tile in readers),'Reader availability differs')
        if tile in readers:require(result['case']==readers[tile],'Existing reader case changed')
        else:require(result.get('case') is None and result['reason']=='no_saved_case','Invented train answer')
        cases[tile]={'reader_available':result['available']}
        if tile in ('ks_06770','ks_05265'):
            e=catalog[tile]['evidence'][0];body,mime=get(e['image_url'])
            require(mime=='image/png' and hashlib.sha256(body).hexdigest()==e['source']['preview_sha256'],'Source PNG changed')
            cases[tile]['preview_bytes_verified']=True
    q={'dataset':'kurosiwo','land_cover':'all','aoi_id':'562','min_flood_pct':0,'limit':40}
    result=api('/api/search?'+urlencode({'q':json.dumps(q)}))
    require(len(result['records'])==8,'Event support differs')
    for row in result['records']:
        expected=catalog[row['id']]
        d=row['source_date_provenance']
        require(d==expected['source_date_provenance'],'HTTP date provenance differs')
        require(d['published_slot_dates']=={'pre_1':'2021-08-07','pre_2':'2021-08-19','post':'2022-02-03'},'Fixed event documented dates differ')
        require(d['model_assumed_slot_dates']=={'pre_1':'2022-01-05','pre_2':'2022-01-17','post':'2022-01-29'},'Old imputed dates changed')
        require(d['actual_acquisition_verified'] is False and all(v is None for v in d['sensor_utc_acquisition_timestamps'].values()),'Unverified UTC promoted')
        require(d['intervals_days']['pre2_to_post_days']==168 and row['event_date']=='2022-01-29','Event/observation dates conflated')
    body,mime=get('/');require(mime=='text/html' and body==(a.snapshot/'index.html').read_bytes(),'Page bytes differ')
    report={'schema':'eo-v8-http-readback-v0','valid':True,'checked_at':datetime.now(timezone.utc).isoformat(),
        'url':base,'snapshot':str(a.snapshot.resolve()),'manifest_sha256':hashlib.sha256((a.snapshot/'v8_build_manifest.json').read_bytes()).hexdigest(),
        'catalog_provenance_count':7000,'event_562_dates_verified_over_http':8,'cases':cases,
        'previews_unchanged':True,'research_exact':True,'e5_results_included':False,'index_bytes_exact':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with a.report.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
