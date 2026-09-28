"""Capture only this experiment's status, receipts, last log, latest score, and own PIDs."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
status=json.loads((a.root/'training_v0/status.json').read_text());workers=[]
for w in status['workers']:
 d=a.root/'training_v0'/w['name'];r=json.loads((d/'receipt.json').read_text())
 logs=(d/'train_log.jsonl').read_text().splitlines();last=json.loads(logs[-1]) if logs else {}
 scores=sorted(d.glob('score_step_*.json'));s=json.loads(scores[-1].read_text()) if scores else None
 workers.append({'name':w['name'],'controller_status':w['status'],'receipt_status':r['status'],'completed_updates_receipt':r.get('completed_updates'), 'last_logged_step':last.get('step'),'pid':r['pid'],'elapsed_seconds':r.get('elapsed_seconds'),'latest_score':None if s is None else {'file':scores[-1].name,'target_iou_auc':s['target_iou_auc'],'sha256':hashlib.sha256(scores[-1].read_bytes()).hexdigest()},'receipt_sha256':hashlib.sha256((d/'receipt.json').read_bytes()).hexdigest()})
pids=[status['controller_pid']]+[w['pid'] for w in workers if w['receipt_status']!='training_completed']
assert all(type(x)==int and x>0 for x in pids)
ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,etime,args'],capture_output=True,text=True)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':status['status'],'controller_pid':status['controller_pid'],'current_worker':status.get('current_worker'),'started_utc':status['started_utc'],'gpu_uuid':status['gpu_uuid'],'workers':workers,'own_processes':ps.stdout,'protocol_sha256':status['protocol_sha256'],'status_sha256':hashlib.sha256((a.root/'training_v0/status.json').read_bytes()).hexdigest(),'scope':'live monitoring only; completed results still require frozen full collector'}
with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))

