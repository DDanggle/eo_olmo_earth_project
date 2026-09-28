from pathlib import Path
import json
p=Path('/private/tmp/oe4_execute_20260927')
cfg=json.loads((p/'oe4_execution_scope_v0.json').read_text())
cfg['id']='oe4_native_v12_engineering_v1'
cfg['amendment']={'kind':'Execution bug repair before any GPU launch; no scientific threshold changes', 'preserves':'oe4_execution_scope_v0.json', 'runtime_sha256':'b737a7254c2d66518e3acc8b8314918c8529d21d33e04347b49be6029cac0b2c','changes':['Unknown GPU utilization treated as busy','Only missing official loss registry keys restored after exact pinned recipe match','CPU native contract v0 failed before optimizer; v1 actual1step passed','Own idle controller v1 PID1145593 stopped before GPU launch; controller v2 requires actual CPU PASS','VLM sequence waits for native controller v2'], 'gpu_completed_at_amendment':False}
cfg['cpu_compatibility_run']['name']='native_cpu_contract_v1'
cfg['native_runs'][0]['name']='native_smoke_v1'
cfg['native_runs'][1]['name']='native_development_v1'
cfg['data']['stored_coordinate_groups']=999
cfg['data']['independent_region_count']=None
(p/'oe4_execution_scope_v1.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Preserved original scope; wrote execution-repair amendment')
