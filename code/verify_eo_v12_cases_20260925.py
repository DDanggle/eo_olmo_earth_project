"""Independently trace every attached E5 answer, plus selected live HTTP cases."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
import hashlib,json

ROOT=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
OUT=ROOT/'artifacts/eo_evidence_search_v12_20260925'
ARMS=('full/native','pair/native','later/native','delta/native','full/full_no_delta')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def get(path):
    with urlopen('http://127.0.0.1:8774'+path,timeout=15) as r:
        assert r.status==200
        return r.read()
def main():
    report_path=OUT/'http_readback_checks_v12.json';assert not report_path.exists()
    manifest=read(OUT/'v11_build_manifest.json');base=Path(manifest['base'])
    for name,digest in manifest['output_files_sha256'].items():assert sha(OUT/name)==digest,name
    for name,digest in manifest['base_files_sha256'].items():assert sha(base/name)==digest,name
    package=OUT/'e5_cases_v11';source_audit=read(package/'independent_audit.json')
    prov=read(OUT/'reader_cases.json')['e5_control_provenance']
    for name,digest in prov['source_files_sha256'].items():
        assert sha(package/name)==digest==source_audit['hashes_verified'][source_audit['artifact']+'/'+name],name
    assert sha(package/'independent_audit.json')==prov['independent_audit_sha256']
    predictions=lines(package/'predictions.jsonl');items=lines(package/'items.jsonl');prompts=lines(package/'prompts.jsonl')
    indexed={(str(x['seed']),x['model_arm']+'/'+x['eval_arm'],x['id']):x for x in predictions}
    item_ids={x['id']:x for x in items};prompt_ids={x['id']:x for x in prompts}
    primary=set(read(package/'eval_sets.json')['primary_same_prompt'])
    join={x['tile']:x for x in read(package/'case_join_index.json')['records']}
    cases=read(OUT/'reader_cases.json')['cases'];prior=read(base/'reader_cases.json')['cases']
    expected_tiles={x['tile'] for x in items if x['partition']=='test' and x['phen']=='flood' and x['kind'] in ('pos','hard_neg')}
    assert set(cases)==set(prior)==set(join)==expected_tiles and len(cases)==914
    n=0
    for tile,c in cases.items():
        assert {k:v for k,v in c.items() if k!='e5_control'}==prior[tile]
        e=c['e5_control'];key=c['question_id'];item=item_ids[key];j=join[tile]
        assert e['source_gold']==c['reference_label']==item['answer']
        assert e['dates']==c['dates']==item['dates'] and e['slots']==c['slots']==item['slots']
        assert e['primary_membership']==(key in primary)==c['c1_membership']['supported_stratum']
        assert prompt_ids[key]['user_text']==c['prompt']
        assert prompts[j['prompt_record_number_1_based']-1]['id']==key
        for arm in ARMS:
            for seed in ('1','2','3'):
                row=indexed[(seed,arm,key)]
                assert e['predictions'][arm][seed]==row['parsed']
                assert predictions[j['prediction_record_numbers_1_based'][arm][seed]-1]==row
                n+=1
    assert n==13710 and sum(c['e5_control']['primary_membership'] for c in cases.values())==902
    live={}
    for tile in ('ks_06770','ks_05265','ks_05229','ks_00276'):
        answer=json.loads(get('/api/reader-case?tile='+tile))
        assert answer['available']==(tile in cases)
        if tile in cases:assert answer['case']==cases[tile]
        else:assert answer['case'] is None and answer['reason']=='no_saved_case'
        live[tile]={'available':answer['available'],'exact_disk_match':True}
    html=(OUT/'index.html').read_text()
    first=html.index('function renderE5Control(box,c){')
    last=html.index('function renderReaderCase(box,snapshot){',first)
    reverted=(html[:first]+html[last:]).replace('renderE5Control(box,c);\n','')
    assert reverted==(base/'index.html').read_text(), 'Inherited v10 display changed beyond E5 renderer'
    assert get('/')==(OUT/'index.html').read_bytes()
    assert json.loads(get('/api/research'))['runs']==read(OUT/'research_runs.json')['runs']
    assert json.loads(get('/api/meta'))==read(OUT/'meta.json')
    for tile in ('ks_06770','ks_05265'):
        p='previews/'+tile+'_source_grounding_v7.png'
        assert hashlib.sha256(get('/'+p)).hexdigest()==manifest['output_files_sha256'][p]
    result={'schema':'eo-v12-e5-case-readback-v0','valid':True,'checked_at':datetime.now(timezone.utc).isoformat(),
        'snapshot':str(OUT),'manifest_sha256':sha(OUT/'v11_build_manifest.json'),
        'all_cases_verified':914,'all_saved_answers_traced_to_original_rows':13710,'primary_membership_count':902,
        'all_actual_saved_prompts_matched':True,'existing_reader_fields_unchanged':True,
        'live_http_cases':live,'live_research_meta_html_exact':True,'two_preview_http_hashes_exact':True,
        'no_new_inference':True,'script_sha256':sha(Path(__file__)),
        'limits':['Full source-to-case file join verified; live HTTP sampled four cases, not all914.',
                  'Stored E5 outputs are not independent physical-change or damage labels.']}
    with report_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
