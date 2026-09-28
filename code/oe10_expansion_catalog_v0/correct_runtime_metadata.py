"""Correct stale reference-stage metadata in a new runtime bundle; no data/score changes."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(a):
 old=read(a.source/'export_manifest.json')
 for item in old['files']:
  p=(a.source/item['path']).resolve()
  assert p.is_relative_to(a.source.resolve()) and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 run=read(a.source/'runtime_receipt.json')
 assert run['query_packets_validated']==48 and run['training_target_rows']==384 and run['new_gpu_seconds']==0
 records=[json.loads(x) for x in (a.source/'scoring/scoring_train.jsonl').read_text().splitlines()]
 labels={r['query_label_npz'] for r in records}
 assert len(labels)==48 and all(r['query_label_npz']==f"labels/{r['query_patch_id']}.npz" for r in records)
 assert not a.out.exists()
 shutil.copytree(a.source,a.out)
 (a.out/'export_manifest.json').unlink()
 prep=read(a.out/'preparation_receipt.json')
 prep.update(query_gold_files_opened=48,query_gold_access_purpose='training supervision only after public catalog freeze',raw_packets_opened=48,raw_packets_opened_unit='unique patch packets, not physical reads',scope='Materialized training targets and real CPU arrays verified. No new training, development evaluation, or model performance.',materialization_script_sha256=run['script_sha256'],metadata_correction_script_sha256=sha(Path(__file__)),support_mask_reference_root=str(a.out.resolve()))
 write(a.out/'preparation_receipt.json',prep)
 contract=read(a.out/'episode_contract.json')
 contract.update(query_gold_opened=True,query_gold_access_purpose='training supervision only after public catalog freeze',query_gold_supplied_to_model=False)
 write(a.out/'episode_contract.json',contract)
 correction={'created_utc':datetime.now(timezone.utc).isoformat(),'source_runtime':str(a.source),'source_export_manifest_sha256':sha(a.source/'export_manifest.json'),'source_runtime_receipt_sha256':sha(a.source/'runtime_receipt.json'),'script_sha256':sha(Path(__file__)),'reason':'Copied reference-stage receipt falsely retained zero input/gold access after CPU materialization. Correct only metadata; keep v0 history.','query_label_files_unique':48,'raw_input_packets_unique':48,'model_training_performed':False,'data_files_changed':False,'new_gpu_seconds':0}
 write(a.out/'metadata_correction_receipt.json',correction)
 changed={'episode_contract.json','preparation_receipt.json'}
 for item in old['files']:
  if item['path'] not in changed:assert sha(a.out/item['path'])==item['sha256']
 write(a.out/'export_manifest.json',{'files':[{'path':str(p.relative_to(a.out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.rglob('*')) if p.is_file()]})
 print(json.dumps(correction))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())

