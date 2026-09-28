"""Prepared follow-up check, never launched automatically.

Default --import-only uses CPU imports and externally pinned source validation.
--identity-only validates the source and complete episode export without torch.
--execute requires a separately frozen file-identity manifest and a GPU ledger
reservation after P2 termination/audit. This script performs no optimizer step,
no development scoring and no generation. A launcher must enforce wall-time.
"""
from __future__ import annotations
import argparse,json,os,sys,time,signal
from pathlib import Path
from input_identity import verify_source_tree,verify_episode_binding,sha256 as sha

def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def verify_files(root,expected):
 require(type(expected) is dict and bool(expected),'Empty expected identity')
 actual={}
 for name,value in expected.items():
  p=(root/name).resolve();require(p.is_relative_to(root.resolve()),'Identity path escape')
  actual[name]=sha(p)
 require(actual==expected,'Actual file identity differs from preregistered hashes')
 return actual

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
 p.add_argument('--import-only',action='store_true')
 p.add_argument('--identity-only',action='store_true',help='Standard-library source/episode hash checks only; no model imports')
 p.add_argument('--source-manifest-sha256',required=True)
 p.add_argument('--identity-manifest-sha256')
 for name in ('identity-manifest','reservation','source','deps','eo','qwen','prepared','episodes','contexts','out'):
  p.add_argument('--'+name,type=Path)
 p.add_argument('--episode-id');p.add_argument('--arm',choices=['B0','B2'],default='B2')
 a=p.parse_args()
 source_gate=verify_source_tree(Path(__file__).absolute().parent,a.source_manifest_sha256)
 require(not (a.identity_only and (a.execute or a.import_only)),'Choose one execution mode')
 expected=None;episode_gate=None
 if a.execute or a.identity_only:
  require(a.identity_manifest is not None and a.episodes is not None and a.identity_manifest_sha256 is not None,'External identity manifest/hash and episode root required')
  expected,episode_gate=verify_episode_binding(a.identity_manifest,a.identity_manifest_sha256,a.episodes)
 if a.identity_only:
  print(json.dumps({'status':'cpu_identity_only','source_identity_gate':source_gate,
    'episode_identity_gate':episode_gate,'actual_qwen_loaded':False,'actual_eo_loaded':False,'gpu_started':False}));return
 from contracts import sha,verify_base,load_base
 verify_base()
 import torch
 from text_mask_model import TextConditionedEpisodeModel
 if not a.execute:
  print(json.dumps({'status':'cpu_import_only','immutable_base_verified':True,'torch_version':torch.__version__,'source_identity_gate':source_gate,
                    'actual_qwen_loaded':False,'actual_eo_loaded':False,'gpu_started':False}));return
 require(not a.import_only,'Choose import-only or execute')
 require(all(getattr(a,k) is not None for k in ('identity_manifest','reservation','source','deps','eo','qwen','prepared','episodes','contexts','out','episode_id')),'Execution arguments missing')
 reservation=read(a.reservation)
 require(reservation.get('status')=='reserved' and reservation.get('p2_finished_and_audited') is True,'P2 audit and persisted reservation required')
 require(reservation.get('purpose')=='text_mask_connection_check','Reservation purpose mismatch')
 seconds=reservation.get('reserved_seconds',0);used=reservation.get('prior_gpu_used_seconds',-1)
 require(0<seconds<=1800 and 0<=used and used+seconds<=7200,'Pilot cumulative budget exceeded')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Only prechecked physical GPU1')
 require(not a.out.exists(),'Never overwrite check output');a.out.mkdir(parents=True)
 started=time.monotonic();receipt={'status':'initializing','pid':os.getpid(),'new_training_updates':0,'development_scoring':False,
   'source_identity_gate':source_gate,'episode_identity_gate':episode_gate,'identity_manifest_sha256':a.identity_manifest_sha256}
 def save():
  receipt['elapsed_seconds']=time.monotonic()-started
  (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 def timeout(signum,frame):raise TimeoutError('Reservation wall-clock limit; do not retry outside cumulative budget')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(seconds)
 module=None
 try:
  discovered={str(p.relative_to(a.qwen)) for p in a.qwen.rglob('*') if p.is_file() and p.suffix in {'.json','.jinja','.safetensors','.txt'}}
  require(discovered==set(expected['reader_files_sha256']),'Reader identity must cover all local config/tokenizer/template/weight files')
  require({'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}<=discovered,'Reader files incomplete')
  reader_identity={'files':verify_files(a.qwen,expected['reader_files_sha256'])}
  source_identity=verify_files(a.source,expected['eo_source_files_sha256'])
  require({'olmoearth_pretrain/model_loader.py','olmoearth_pretrain/data/normalize.py','olmoearth_pretrain/data/constants.py'}<=set(source_identity),'EO source/normalization identity missing')
  native=load_base('native_replay')
  require(expected['eo_files_sha256']=={'config.json':native.PINNED_PUBLIC_CONFIG_SHA256,'weights.pth':native.PINNED_WEIGHTS_SHA256},'Expected original pinned OlmoEarth')
  eo_identity={'files':verify_files(a.eo,expected['eo_files_sha256']),'source':source_identity,
               'prepared_manifest_sha256':sha(a.prepared/'manifest.jsonl'),'prepared_normalization':'immutable precomputed arrays; hashes checked by EpisodeLoader'}
  require(eo_identity['prepared_manifest_sha256']==expected['prepared_manifest_sha256'],'Prepared identity changed')
  require(sha(a.contexts)==expected['contexts_sha256'],'Context identity changed')
  record=next(r for r in (json.loads(line) for line in a.contexts.read_text().splitlines()) if r['episode_id']==a.episode_id)
  require(record['audit_only']['split']=='train' and record['audit_only']['k_pairs']==1,'Connection check only train K1')
  contexts=record['model_context_by_condition']
  require(set(contexts)=={'generic','names_only','matched_knowledge','removed_knowledge_diagnostic','swapped_knowledge_diagnostic'},'Five conditions expected')
  loader=load_base('episode_loader').EpisodeLoader(a.prepared,a.episodes,'train')
  item=loader.load(a.episode_id,acquired_positions=[2,5]);target=loader.training_target(a.episode_id,training=True)
  sys.path[:0]=[str(a.source),str(a.deps)]
  from olmoearth_pretrain.model_loader import load_model_from_path
  torch.set_num_threads(2);torch.manual_seed(280928);torch.cuda.manual_seed_all(280928)
  official=load_model_from_path(str(a.eo));encoder=official.encoder;del official
  module=TextConditionedEpisodeModel(encoder,a.arm,'cuda:0',a.qwen,True,
            reader_identity=reader_identity,eo_identity=eo_identity,max_text_tokens=1024)
  module.eval();predictions={};text_meta={}
  with torch.no_grad():
   for condition in contexts:
    out=module(item['model_input'],contexts[condition]);predictions[condition]=out['logits'].cpu();text_meta[condition]=out['metrics']['text']
  difference={name:float((predictions[name]-predictions['names_only']).abs().max()) for name in contexts}
  require(difference['removed_knowledge_diagnostic']==0,'Removed facts must equal names-only')
  require(difference['matched_knowledge']>0,'Actual request text did not affect logits')
  module.train();module.zero_grad(set_to_none=True)
  out=module(item['model_input'],contexts['matched_knowledge'],target);out['loss'].backward()
  gradients={name:sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in obj.parameters())
    for name,obj in [('encoder',module.encoder),('head',module.head),('qwen',module.qwen),('unused_connector',module.connector)]}
  require(gradients['head']>0 and gradients['qwen']==0 and gradients['unused_connector']==0,'Gradient boundaries')
  require((gradients['encoder']>0)==(a.arm=='B2'),'EO arm gradient contract')
  torch.save(module.trainable_state_dict(),a.out/'connection_state.pt')
  module.load_trainable_state_dict(torch.load(a.out/'connection_state.pt',map_location='cpu',weights_only=True))
  module.eval()
  with torch.no_grad():restored=module(item['model_input'],contexts['matched_knowledge'])['logits'].cpu()
  restore_delta=float((restored-predictions['matched_knowledge']).abs().max())
  require(restore_delta<=1e-6,'Same-process save/load mismatch')
  receipt.update(status='actual_connection_check_passed',gradient_nonzero_tensors=gradients,
    text_condition_logit_max_difference_vs_names=difference,text_token_audit=text_meta,
    same_process_restore_max_difference=restore_delta,cold_resume_tested=False,
    reader_identity=reader_identity,eo_identity=eo_identity,actual_human_responses=0,
    claim='Wiring only; no semantic grounding, improvement, generalization or generation claim')
 except Exception as exc:
  receipt.update(status='failed',error=repr(exc));raise
 finally:
  if module is not None:module.close()
  save();signal.alarm(0)
 print(json.dumps({'status':receipt['status'],'output':str(a.out)}))
if __name__=='__main__':main()
