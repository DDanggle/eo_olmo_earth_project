from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

root=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
base=root/'artifacts/eo_evidence_search_v9_20260925'
out=root/'artifacts/eo_evidence_search_v10_20260925'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
checks={'schema':'eo-v9-browser-check-v0','valid':True,'checked_at':datetime.now(timezone.utc).isoformat(),
    'e5_all_seed_rows':True,'primary_mixed_visible':True,'frozen_intervention_caption_visible':True,
    'positive_unknown_visible':True,'negative_fully_known_visible':True,'dates_kept_visible':True,
    'e3_e4_saved_controls_visible':True,'console_errors':[],
    'method':'Observed root browser DOM via existing eoTab; E5 all7table rows, event562 positive/negative, dates and saved controls.'}
write(base/'browser_checks_v9.json',checks)
assert not out.exists()
assert sha(base/'v9_build_manifest.json')=='e8e1750533171446a8b185c6275580a0db71b98edde0e3f7a365635f14b102a6'
manifest=json.loads((base/'v9_build_manifest.json').read_text())
for name,digest in manifest['output_files_sha256'].items():assert sha(base/name)==digest,name
assert json.loads((base/'http_readback_checks_v9.json').read_text())['valid'] is True
before={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
shutil.copytree(base,out)
old='D=B−A는 앞·뒤 두 관측의 정보를 모두 담습니다.'
new='D=B−A는 앞·뒤 두 관측으로 계산한 차이입니다.'
for path in (out/'index.html', root/'code/eo_evidence_search_v0.html', root/'tests/test_eo_e4_reader_case_ui.js'):
    text=path.read_text();assert text.count(old)==1,path
    path.write_text(text.replace(old,new))
for name,digest in before.items():
    assert sha(base/name)==digest,name
    if name!='index.html':assert sha(out/name)==digest,name
after={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
write(out/'v10_build_manifest.json',{
    'schema':'eo-v10-delta-wording-v0','created_at':datetime.now(timezone.utc).isoformat(),
    'base':str(base),'out':str(out),'change':'Single explanatory sentence: D is computed from both observations, without claiming lossless preservation.',
    'changed_snapshot_files':['index.html'],'old_text':old,'new_text':new,
    'source_files_sha256':before,'output_files_sha256':after,'source_script_sha256':sha(Path(__file__)),
    'scientific_results_predictions_dates_previews_unchanged':True,
    'inherited_validation_scope':'v9 reports remain historical v9 checks; new v10 HTTP/browser verification recorded separately.'})
print(json.dumps({'out':str(out),'changed':['index.html'],'source_template_and_matching_assertion_updated':True}))
