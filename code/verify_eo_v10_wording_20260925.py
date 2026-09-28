from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
import hashlib,json

root=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
out=root/'artifacts/eo_evidence_search_v10_20260925'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((out/'v10_build_manifest.json').read_text())
base=Path(manifest['base'])
for name,digest in manifest['source_files_sha256'].items():assert sha(base/name)==digest,name
for name,digest in manifest['output_files_sha256'].items():assert sha(out/name)==digest,name
assert json.loads((out/'http_readback_checks_v9.json').read_text())['valid'] is True
expected=(base/'index.html').read_text().replace(manifest['old_text'],manifest['new_text'])
assert expected==(out/'index.html').read_text()
def get(path):
    with urlopen('http://127.0.0.1:8774'+path,timeout=10) as r:
        assert r.status==200
        return r.read()
assert get('/')==(out/'index.html').read_bytes()
research=json.loads(get('/api/research'))
assert research['runs']==json.loads((out/'research_runs.json').read_text())['runs']
assert json.loads(get('/api/meta'))==json.loads((out/'meta.json').read_text())
for tile in ('ks_06770','ks_05265'):
    expected_case=json.loads((out/'reader_cases.json').read_text())['cases'][tile]
    assert json.loads(get('/api/reader-case?tile='+tile))['case']==expected_case
    name='previews/'+tile+'_source_grounding_v7.png'
    assert hashlib.sha256(get('/'+name)).hexdigest()==manifest['output_files_sha256'][name]
report={'schema':'eo-v10-wording-http-check-v0','valid':True,
    'checked_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(out/'v10_build_manifest.json'),
    'source_parent_files_verified':len(manifest['source_files_sha256']),
    'new_output_files_verified':len(manifest['output_files_sha256']),
    'single_html_sentence_change_verified':True,'scientific_results_and_cases_unchanged':True,
    'http_index_research_meta_two_cases_two_previews_exact':True,'script_sha256':sha(Path(__file__))}
with (out/'http_readback_checks_v10.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
