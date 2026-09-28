"""CPU-only tokenizer/contract tests; never instantiate 8B weights or access EO data."""
import argparse,hashlib,inspect,json,os,re,subprocess,sys,time
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def protection(root):
 s=read(root/'oe10_p2_v0/training_v0/status.json');result={}
 for name,ref in s['protected_before'].items():
  p=root/'code'/name;st=p.stat();result[name]={'sha256':sha(p),'mtime_ns':st.st_mtime_ns,'bytes':st.st_size}
  if result[name]!=ref:raise ValueError('protected source modified: '+name)
 for name,h in s['source_hashes'].items():
  if sha(root/'oe10_p2_v0/code_snapshot/oe10_p2_v3'/name)!=h:raise ValueError('running snapshot modified: '+name)
 return {'protected':result,'snapshot':s['source_hashes'],'protocol':s['protocol_sha256']}

def main(a):
 start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
 before=protection(a.root)
 for folder in (a.code,a.model_code):
  for name,h in read(folder/'source_manifest.json')['files'].items():
   if sha(folder/name)!=h:raise ValueError('source changed: '+name)
 for f in read(a.contexts/'export_manifest.json')['files']:
  if sha(a.contexts/f['path'])!=f['sha256']:raise ValueError('context changed')
 records=[json.loads(x) for x in (a.contexts/'contexts.jsonl').read_text().splitlines()]
 sys.path.insert(0,str(a.model_code));import contracts
 contracts.verify_base()
 from transformers import AutoProcessor
 processor=AutoProcessor.from_pretrained(str(a.qwen),local_files_only=True)
 counts=defaultdict(list);tokens_by_context={};name_cases={}
 for r in records:
  for condition,context in r['model_context_by_condition'].items():
   text=contracts.render_context(context)
   rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=False)
   tokens=processor.tokenizer(rendered,add_special_tokens=False,padding=False,truncation=False)['input_ids']
   if not tokens or len(tokens)>1024:raise ValueError('text budget failed')
   if '<|im_start|>assistant' in rendered:raise ValueError('assistant prefix forbidden')
   counts[condition].append(len(tokens));tokens_by_context[contracts.context_sha(context)]=tokens
   name_cases[condition]=len(tokens)
  if r['model_context_by_condition']['removed_knowledge_diagnostic']!=r['model_context_by_condition']['names_only']:raise ValueError('removal changes names')
 # Hash only tokenizer/config inputs. Do not claim this checks the 8B weight shards.
 tokenizer_files={p.name:sha(p) for p in sorted(a.qwen.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.jinja') and p.stat().st_size < 64*1024*1024}
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 api_path=Path(inspect.getfile(Qwen3VLModel))
 api={'source_file':str(api_path),'source_sha256':sha(api_path),'forward_signature':str(inspect.signature(Qwen3VLModel.forward)),
      'forward_source':inspect.getsource(Qwen3VLModel.forward),'scope':'Installed API inspected only; no real Qwen weight forward executed'}
 write(a.out/'qwen_installed_api.json',api)
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.model_code),'-p','test_*.py','-v'],capture_output=True,text=True,timeout=180,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 (a.out/'cpu_unit_tests.log').write_text(result.stdout+result.stderr)
 count_match=re.search(r'Ran (\d+) tests?',result.stdout+result.stderr)
 test_count=int(count_match.group(1)) if count_match else 0
 if test_count<1:raise ValueError('No unit tests actually executed')
 write(a.out/'cpu_unit_tests_receipt.json',{'exit_code':result.returncode,'scope':'toy encoder and toy language model; gradient and dependency wiring only'})
 after=protection(a.root)
 if before!=after:raise ValueError('protection changed')
 report={'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
 'status':'passed' if result.returncode==0 else 'unit_tests_failed','context_records':len(records),'unique_context_texts':len(tokens_by_context),
 'real_qwen_tokenizer_verified':True,'tokenizer_class':type(processor.tokenizer).__name__,'tokenizer_files_sha256':tokenizer_files,
 'per_condition_tokens':{k:{'min':min(v),'max':max(v),'records':len(v)} for k,v in counts.items()},'max_token_cap':1024,'truncation_used':False,
 'equal_actual_text_compute_claimed':False,'text_cache_in_primary_default':False,'unit_tests_exit_code':result.returncode,'unit_tests_count':test_count,
 'new_gpu_seconds':0,'real_qwen_weights_loaded':False,'real_olmoearth_forward':False,'query_gold_opened':False,'dev_opened':False,
 'protected_before':before,'protected_after':after,'actual_expert_responses':0,
 'limitations':['CPU toy tests do not establish real EO/Qwen gradient or semantic understanding.','Only tokenization and installed Qwen API inspected on actual pinned reader assets; weight forwards await a separately frozen pilot.','Descriptions add text tokens; actual counts differ and cannot be called equal compute.']}
 write(a.out/'cpu_validation_receipt.json',report)
 write(a.out/'export_manifest.json',{'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.iterdir()) if p.is_file()]})
 print(json.dumps({k:report[k] for k in ('status','context_records','unique_context_texts','per_condition_tokens','unit_tests_exit_code','new_gpu_seconds')}))
 if result.returncode:sys.exit(result.returncode)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('root','code','model-code','contexts','qwen','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
