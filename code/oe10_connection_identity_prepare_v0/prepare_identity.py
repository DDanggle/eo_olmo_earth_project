"""Read/hash-only preparation: no model imports, arrays, GPU, launch or reservation."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
from datetime import datetime,timezone

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def require(v,msg):
 if not v:raise ValueError(msg)
def protect(root):
 s=read(root/'oe10_p2_v0/training_v0/status.json');got={}
 for name,ref in s['protected_before'].items():
  p=root/'code'/name;got[name]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size};require(got[name]==ref,'protected source mismatch')
 for name,h in s['source_hashes'].items():require(sha(root/'oe10_p2_v0/code_snapshot/oe10_p2_v3'/name)==h,'P2 source changed')
 return {'protected':got,'snapshot':s['source_hashes'],'protocol':s['protocol_sha256']}
def hash_expected(root,expected):
 for name,h in expected.items():
  p=root/name;require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'asset path boundary')
  require(sha(p)==h,'expected file changed: '+str(p))

def main(a):
 start=time.monotonic();before=protect(a.root)
 require(sha(a.expected)==a.expected_sha256,'expected identity changed')
 expected=read(a.expected);a.out.mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('identity_gate',a.model_code/'input_identity.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 source_gate=gate.verify_source_tree(a.model_code,a.source_manifest_sha256)
 qwen=a.root/'models/Qwen3-VL-8B-Instruct';eo=a.root/'oe4_native_v12_v0/models/OlmoEarth-v1_2-Base';source=a.root/'oe4_native_v12_v0/source'
 reader_names={str(p.relative_to(qwen)) for p in qwen.rglob('*') if p.is_file() and p.suffix in {'.json','.jinja','.safetensors','.txt'}}
 require(reader_names==set(expected['reader_files_sha256']),'unexpected reader assets')
 hash_expected(qwen,expected['reader_files_sha256']);hash_expected(eo,expected['eo_files_sha256']);hash_expected(source,expected['eo_source_files_sha256'])
 identity={k:expected[k] for k in ('reader_files_sha256','eo_files_sha256','eo_source_files_sha256','prepared_manifest_sha256','contexts_sha256','episodes_export_manifest_sha256')}
 prospective={name:sha(source/name) for name in ('olmoearth_pretrain/data/normalize.py','olmoearth_pretrain/data/constants.py')}
 identity['eo_source_files_sha256']={**identity['eo_source_files_sha256'],**prospective}
 prepared=a.root/'oe8_pastis_prepare_v0/prepared_v0';contexts=a.root/'oe10_text_mask_prepare_v0/contexts_v0/contexts.jsonl';episodes=a.root/'oe10_expansion_catalog_v0/runtime_v2'
 require(sha(prepared/'manifest.jsonl')==identity['prepared_manifest_sha256'],'prepared manifest mismatch')
 require(sha(contexts)==identity['contexts_sha256'],'context mismatch')
 identity['schema_version']='oe10_connection_identity_v1';identity['purpose']='future train-only connection check; not a performance experiment or launch authorization'
 dump(a.out/'identity_manifest.json',identity)
 _,episode_gate=gate.verify_episode_binding(a.out/'identity_manifest.json',sha(a.out/'identity_manifest.json'),episodes)
 # Select by a fixed public rule, not presence, quality, label or model score.
 rows=[json.loads(x) for x in contexts.read_text().splitlines() if x.strip()]
 selected=min((r for r in rows if r['audit_only']['k_pairs']==1 and r['audit_only']['support_class_ids_audit_only']['positive']==8),key=lambda r:r['episode_id'])
 after=protect(a.root);require(before==after,'P2 source mutated')
 report={'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'status':'cpu_asset_identity_verified_pending_prereg_and_p2_end',
 'expected_inputs_sha256':sha(a.expected),'source_manifest_sha256':a.source_manifest_sha256,'script_sha256':sha(Path(__file__)),
 'identity_manifest_sha256':sha(a.out/'identity_manifest.json'),'source_gate':source_gate,'episode_gate':episode_gate,
 'prospectively_observed_normalization_sources':prospective,'reader_assets_verified':len(reader_names),'reader_bytes_hashed':sum((qwen/n).stat().st_size for n in reader_names),
 'candidate_episode_id':selected['episode_id'],'candidate_rule':'lexicographically first train K1 episode whose positive support class is8; no query-gold or score selection',
 'protected_before':before,'protected_after':after,'query_gold_arrays_opened':0,'dev_arrays_opened':0,'actual_model_loaded':False,'new_gpu_seconds':0,
 'budget_reserved_seconds':0,'requires':['P2 ended and audited','prospective connection protocol/stop criteria/source/input identities','atomic GPU reservation and idle check','external timeout and durable actual usage receipt'],
 'limits':['No actual OlmoEarth/Qwen path or semantic result.','Expected weights/tokenizer/runtime identities came from already verified outputs; new normalization source hashes are prospective observations.','Hashing scoring/mask files is identity verification, not image/label analysis.']}
 dump(a.out/'preparation_receipt.json',report)
 dump(a.out/'export_manifest.json',{'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.iterdir()) if p.is_file()]})
 print(json.dumps({'status':report['status'],'reader_assets_verified':len(reader_names),'candidate_episode_id':report['candidate_episode_id'],'identity_manifest_sha256':report['identity_manifest_sha256'],'new_gpu_seconds':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('root','expected','model-code','out'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--expected-sha256',required=True);p.add_argument('--source-manifest-sha256',required=True);main(p.parse_args())
