#!/usr/bin/env python3
"""Actual local Qwen tokenizer/processor audit; no model weights or GPU."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('qwen','contexts','contracts','adapter','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists()
    spec=importlib.util.spec_from_file_location('_oe11_contracts_token_audit',a.contracts)
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    neutral=None
    for node in ast.parse(a.adapter.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NEUTRAL_COUNTER' for t in node.targets):neutral=ast.literal_eval(node.value)
    assert isinstance(neutral,str)
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    processor=AutoProcessor.from_pretrained(str(a.qwen),local_files_only=True)
    rows=[]
    for key,context in sorted(json.loads(a.contexts.read_text()).items()):
        for role in ('positive','counterexample'):
            isolated=dict(instruction=context['instruction'],positive=context[role],counterexample=neutral)
            request=contract.render_context(isolated)
            rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':request}]}],tokenize=False,add_generation_prompt=False)
            batch=processor.tokenizer(rendered,return_tensors='pt',padding=False,truncation=False,add_special_tokens=False)
            count=int(batch['attention_mask'].sum())
            assert 0<count==batch['input_ids'].shape[1]<=1024
            rows.append({'context':key,'role':role,'tokens':count,'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
    assert len(rows)==48 and not torch.cuda.is_initialized()
    result={'status':'PASS','role_prefills_checked':48,'unique_serialized_role_inputs':len({r['rendered_sha256'] for r in rows}),
            'minimum_tokens':min(r['tokens'] for r in rows),'maximum_tokens':max(r['tokens'] for r in rows),
            'max_per_role_tokens':1024,'truncation':False,'model_weights_loaded':False,'GPU_used':False,
            'cases':rows,'source_sha256':{str(x):sha(x) for x in (a.contexts,a.contracts,a.adapter,Path(__file__))}}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256')}))

if __name__=='__main__':main()
