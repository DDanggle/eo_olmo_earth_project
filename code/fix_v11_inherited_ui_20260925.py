from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
r=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
old=r/'artifacts/eo_evidence_search_v11_20260925'
record={'schema':'eo-v11-browser-regression-v0','checked_at':datetime.now(timezone.utc).isoformat(),
    'publication_ready':False,'scientific_source_data_unchanged':True,
    'e5_five_by_three_answers_correct':True,'regression':'--html used the older code template instead of inheriting the full v10 index; T0 dates and source-quality caption absent in browser.',
    'positive_unknown43_38_visible':False,'both_cases_documented_dates_visible':False,
    'fix':'Create v12 with byte-preserved v10 HTML except the E5 renderer and single call; preserve v11 artifact as first attempt.'}
with (old/'browser_regression_v11.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
base=(r/'artifacts/eo_evidence_search_v10_20260925/index.html').read_text()
assert 'function renderE5Control' not in base
assert base.count('function renderReaderCase(box,snapshot){')==1
assert base.count('renderE4Control(box,c);\n}')==1
block=Path('/private/tmp/e5_ui_block_20260925.txt').read_text()
fixed=base.replace('function renderReaderCase(box,snapshot){',block+'function renderReaderCase(box,snapshot){')
fixed=fixed.replace('renderE4Control(box,c);\n}','renderE4Control(box,c);\nrenderE5Control(box,c);\n}')
assert fixed.replace(block,'').replace('renderE5Control(box,c);\n','')==base
(r/'code/eo_evidence_search_v0.html').write_text(fixed)
subprocess.run(['/Users/dongdong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3','-B',
    str(r/'code/build_eo_v11_e5_cases_20260925.py'),
    '--base',str(r/'artifacts/eo_evidence_search_v10_20260925'),
    '--e5',str(r/'artifacts/e5_equal_budget_v0_review_20260925/e5_equal_budget_v0'),
    '--audit',str(r/'artifacts/e5_equal_budget_v0_review_20260925/e5_independent_audit_20260925.json'),
    '--out',str(r/'artifacts/eo_evidence_search_v12_20260925'),
    '--html',str(r/'code/eo_evidence_search_v0.html')],check=True)
manifest=json.loads((r/'artifacts/eo_evidence_search_v12_20260925/v11_build_manifest.json').read_text())
assert manifest['n_saved_answers_attached']==13710
print('v12 inherits full v10 HTML plus only the E5 renderer and call.')
