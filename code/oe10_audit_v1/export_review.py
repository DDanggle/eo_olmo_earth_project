"""Copy small, finalized experiment receipts without model weights or raw imagery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--include-final-predictions',action='store_true');a=p.parse_args()
 root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=False)
 files=[]
 for f in sorted(root.rglob('*')):
  if not f.is_file() or f.is_symlink() or f.name.startswith('._'):continue
  rel=f.relative_to(root)
  if any(x in ('code_snapshot','audit_snapshot','__pycache__') or x.startswith('review_export') for x in rel.parts):continue
  if f.suffix not in {'.json','.jsonl','.log'}:continue
  if f.stat().st_size>64<<20:continue
  if any(x.startswith('predictions_step_') for x in rel.parts):continue
  target=a.out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
  files.append({'path':str(rel),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 if a.include_final_predictions:
  for receipt in root.glob('*/B*_*_train/receipt.json'):
   r=json.loads(receipt.read_text())
   if r['status']!='training_completed':continue
   folder=receipt.parent/f'predictions_step_{r["completed_updates"]:06d}'
   if not folder.is_dir():raise RuntimeError('missing final prediction folder')
   target=a.out/folder.relative_to(root);shutil.copytree(folder,target)
   for f in target.iterdir():
    files.append({'path':str(f.relative_to(a.out)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 (a.out/'export_manifest.json').write_text(json.dumps({'source_root':str(root),'files':files,'weights_included':False,
   'live_receipts_may_be_intermediate':True},indent=2)+'\n')
 print(json.dumps({'out':str(a.out),'files':len(files),'bytes':sum(x['bytes'] for x in files)}))

if __name__=='__main__':main()
