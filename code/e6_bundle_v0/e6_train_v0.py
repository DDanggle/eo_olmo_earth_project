#!/usr/bin/env python3
"""Frozen E6 head training. No LLM, E5 code import, resume, or CPU fallback.

All three initial states precede training; all three trainings precede test
inference. Lower-level tensor helpers support CPU synthetic tests only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path('/home/work/data/olmoearth')
PLAN_SHA = '66f820db8bc6a3d03c07d66676adcbe6170c012649a824e83b02a94d03194840'
PARENT_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
LOCKS = ['.eo_e3_pair_dependence.lock', '.eo_reader_gpu0.lock', '.eo_reader_gpu1.lock',
         '.eo_e5_equal_budget.lock', '.eo_e6_no_llm.lock']
REQUIRED = {'items.jsonl', 'pairs.npy', 'ordered_ids.json', 'batches.json', 'eval_sets.json',
            'reference_rows.jsonl', 'reference_audit.json', 'prereg.json'}
SEEDS = (1, 2, 3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key: '+key)
        out[key] = value
    return out


def decode(text):
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(path):
    return decode(Path(path).read_text())


def lines(path):
    return [decode(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def atomic_status(out, value):
    temp = out/'status.tmp'
    with temp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temp.replace(out/'status.json')


def relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts), 'Unsafe relative manifest path')
    result = (root/path).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(), 'Missing/escaping frozen file')
    return result


def verify_files(out, manifest):
    require(manifest['schema'] == 'e6-no-llm-prepared-v0', 'Prepared schema mismatch')
    require(manifest['parent_e5_prepared_manifest_sha256'] == PARENT_SHA, 'E5 parent pin differs')
    require((manifest['n_items'], manifest['n_train'], manifest['n_test']) == (5989,4234,1755), 'Prepared counts differ')
    require(REQUIRED <= set(manifest['files_sha256']), 'Missing required file pins')
    for name, expected in manifest['files_sha256'].items():
        require(sha(relative(out, name)) == expected, 'Frozen input changed: '+name)
    require(sha(out/'prereg.json') == PLAN_SHA, 'Frozen E6 plan differs')
    cfg = read(out/'prereg.json')
    for name, expected in cfg['reference']['files_sha256'].items():
        require(manifest['files_sha256'].get(name) == expected, 'Inherited E5 input pin differs: '+name)
    snapshot = out/'code_snapshot'
    require(Path(__file__).resolve().parent == snapshot.resolve(), 'Run only actual frozen code snapshot')
    require(set(manifest['code_snapshot_sha256']) == set(cfg['source_files'])
            and len(cfg['source_files']) == 5, 'Exact code snapshot coverage differs')
    for name, expected in manifest['code_snapshot_sha256'].items():
        require(sha(relative(snapshot, name)) == expected, 'Frozen code changed: '+name)


def validate_config(cfg):
    require(cfg['schema'] == 'e6-no-llm-prereg-v0' and cfg['id'] == 'E6-HEAD-v0', 'Unknown E6 plan')
    expected = {'seeds':[1,2,3], 'epochs':3, 'batch_size':8, 'lr':1e-4, 'weight_decay':.01,
                'betas':[.9,.999], 'eps':1e-8, 'optimizer_foreach':False, 'optimizer_fused':False,
                'expected_updates_per_model':1590, 'expected_exposures_per_model':12702}
    require(all(cfg['training'].get(k) == v for k,v in expected.items()), 'Training settings differ')
    require(cfg['evaluation']['n_rows'] == 5265 and cfg['evaluation']['batch_size'] == 8
            and cfg['evaluation']['model_arm'] == 'full_head' and cfg['evaluation']['eval_arm'] == 'native', 'Evaluation settings differ')
    require(cfg['architecture']['trainable_parameters'] == 367361, 'Head parameter contract differs')
    require(cfg['compute']['gpu_index'] == 0 and cfg['compute']['max_runtime_minutes'] == 60
            and cfg['compute']['max_model_minutes'] == 20 and cfg['compute']['lock_names'] == LOCKS, 'Resource contract differs')


def expected_batches(train_ids, seed):
    rng = np.random.default_rng(seed)
    epochs = []
    for _ in range(3):
        order = rng.permutation(len(train_ids))
        epochs.append([[train_ids[j] for j in order[k:k+8]] for k in range(0,len(order),8)])
    return epochs


def validate_population(items, ordered, batches, pairs, eval_sets):
    require(len(items) == len({i['id'] for i in items}) == 5989, 'Item coverage/duplicates')
    require(set(ordered) == {'train','test'}, 'Ordered ID partitions differ')
    lookup = {item['id']:index for index,item in enumerate(items)}
    for partition, n in [('train',4234),('test',1755)]:
        require(ordered[partition] == [i['id'] for i in items if i['partition'] == partition]
                and len(ordered[partition]) == n, 'Ordered ID list differs: '+partition)
    counts = Counter((i['partition'],i['phen'],i['kind']) for i in items)
    require(counts == {('train','flood','pos'):1066,('train','flood','neg'):1066,('train','flood','hard_neg'):1066,
                      ('train','landslide','pos'):518,('train','landslide','neg'):518,
                      ('test','flood','pos'):457,('test','flood','neg'):457,('test','flood','hard_neg'):457,
                      ('test','landslide','pos'):192,('test','landslide','neg'):192}, 'Population category counts differ')
    for item in items:
        require(item['answer'] == ('yes' if item['kind']=='pos' else 'no'), 'Source gold/kind mismatch')
    for phen, support in [('flood',(27,10)),('landslide',(7,2))]:
        a,b = [[i for i in items if i['phen']==phen and i['partition']==p] for p in ('train','test')]
        require({i['tile'] for i in a}.isdisjoint(i['tile'] for i in b), 'Train/test tile overlap')
        clusters = [{str(i['cluster']) for i in group} for group in (a,b)]
        require(clusters[0].isdisjoint(clusters[1]) and tuple(map(len,clusters))==support, 'Train/test cluster overlap/support')
    require(set(batches) == {'1','2','3'}, 'Batch seed coverage differs')
    for seed in SEEDS:
        require(batches[str(seed)] == expected_batches(ordered['train'],seed), 'Batch order differs')
        require(sum(len(ep) for ep in batches[str(seed)])==1590
                and sum(len(batch) for ep in batches[str(seed)] for batch in ep)==12702, 'Batch budget differs')
    require(pairs.shape==(5989,2,64,768) and pairs.dtype==np.float32, 'Pair shape/dtype differs')
    for index in range(5989):
        require(np.isfinite(pairs[index]).all(), 'Nonfinite source pair')
        with np.errstate(over='ignore',invalid='ignore'):
            difference = pairs[index,1]-pairs[index,0]
        require(np.isfinite(difference).all(), 'Nonfinite full delta')
    expected_sets = {'all_test':1755,'primary_same_prompt':902,'paired_flood':914,
                     'hard_negative_flood':457,'landslide':384,'e3_subset':209}
    require(set(eval_sets)==set(expected_sets), 'Evaluation set keys differ')
    for key,n in expected_sets.items():
        ids = eval_sets[key]
        require(len(ids)==len(set(ids))==n and set(ids)<=set(ordered['test']), 'Evaluation support/train leakage: '+key)
    require(set(eval_sets['all_test'])==set(ordered['test']), 'Full test support differs')
    return lookup


def check_locks(root):
    require(os.environ.get('E6_GPU_INDEX')=='0' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'GPU0 mapping required')
    fds = decode(os.environ['E6_LOCK_FDS'])
    require(set(fds)==set(LOCKS), 'Inherited lock coverage differs')
    for name, fd in fds.items():
        require(type(fd) is int and fd>=3, 'Invalid inherited descriptor')
        current, expected = os.fstat(fd), (root/name).stat()
        require((current.st_dev,current.st_ino)==(expected.st_dev,expected.st_ino), 'Lock descriptor identity differs')
        # A fresh descriptor must be blocked by the inherited existing lease.
        # Never acquire on the inherited FD here: that could silently repair an
        # unlocked launcher descriptor instead of proving ownership.
        probe = os.open(root/name, os.O_RDWR)
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('Inherited lock is not already held: '+name)
        finally:
            os.close(probe)
    return fds


def check_idle():
    uuid = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).strip()
    require(uuid and '\n' not in uuid and uuid==os.environ.get('E6_GPU_UUID'), 'GPU0 UUID differs')
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=20).splitlines()
    require(uuid not in [s.strip() for s in busy], 'GPU0 became busy before model load')
    return uuid


class Budget:
    def __init__(self):
        self.started = time.monotonic(); self.spent = {seed:0.0 for seed in SEEDS}
        self.seed = None; self.phase_started = None

    def check(self):
        total = time.monotonic()-self.started
        require(total < 3600, 'E6 total 60-minute limit exceeded')
        if self.seed is not None:
            require(self.spent[self.seed]+time.monotonic()-self.phase_started < 1200, 'E6 seed 20-minute limit exceeded')

    def arm(self, seed=None):
        self.check(); self.seed=seed; self.phase_started=time.monotonic() if seed is not None else None
        remaining = 3600-(time.monotonic()-self.started)
        if seed is not None: remaining=min(remaining,1200-self.spent[seed])
        signal.setitimer(signal.ITIMER_REAL, max(.001,remaining))

    def finish(self):
        self.check()
        if self.seed is not None: self.spent[self.seed] += time.monotonic()-self.phase_started
        self.seed=None; self.phase_started=None; self.arm()


def finite(torch, tensors, label):
    tensors = list(tensors)
    require(tensors and all(x is not None for x in tensors), 'Missing '+label)
    require(bool(torch.stack([torch.isfinite(x).all() for x in tensors]).all()), 'Nonfinite '+label)


def inputs(torch, pairs, metadata, indices, device):
    # No item dict or answer enters this feature-only helper.
    values = torch.from_numpy(np.array(pairs[indices],dtype=np.float32,copy=True,order='C')).to(device)
    meta = metadata[indices].to(device)
    require(values.shape==(len(indices),2,64,768) and meta.shape==(len(indices),8), 'Batch shape differs')
    finite(torch,[values,meta], 'batch features')
    return values,meta


def targets(torch, items, indices, device):
    answers = [items[i]['answer'] for i in indices]
    require(all(a in ('yes','no') for a in answers), 'Unknown binary source target')
    require(all(items[i]['partition']=='train' for i in indices), 'Test target requested by training')
    return torch.tensor([float(a=='yes') for a in answers],device=device,dtype=torch.float32)


def train_model(torch, model, pairs, metadata, items, lookup, schedule, device, path, budget, progress=None, seed=1):
    require(seed in SEEDS and type(seed) is int, 'Unknown training seed')
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8,
                                 foreach=False,fused=False,amsgrad=False,maximize=False)
    groups = [{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]
    model.train(); steps=exposures=0; losses=[]; started=time.monotonic()
    with path.open('x') as log:
        for epoch,batches in enumerate(schedule):
            weighted=0.0; count=0
            for batch_index,ids in enumerate(batches):
                budget.check(); indices=[lookup[item_id] for item_id in ids]
                x,m=inputs(torch,pairs,metadata,indices,device); y=targets(torch,items,indices,device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x,m); finite(torch,[logits],'training logits')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='mean')
                finite(torch,[loss],'loss'); loss.backward()
                finite(torch,(p.grad for p in model.parameters()),'trainable gradient')
                optimizer.step(); finite(torch,model.parameters(),'updated parameter')
                scalar=float(loss.detach()); steps+=1; exposures+=len(ids); weighted+=scalar*len(ids); count+=len(ids)
                row={'seed':seed,'step':steps,'epoch':epoch,'batch_index':batch_index,'ids':ids,'pair_indices':indices,
                     'batch_size':len(ids),'exposures':exposures,'loss':scalar,'loss_finite':True,
                     'gradients_finite':True,'parameters_finite':True,'elapsed_s':time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                if progress is not None and steps%25==0: progress(steps,exposures)
            require(count>0,'Empty training epoch'); losses.append(weighted/count)
        os.fsync(log.fileno())
    optimizer_steps=[int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
    require(len(optimizer_steps)==len(list(model.parameters())) and set(optimizer_steps)=={steps},'Optimizer step audit differs')
    return {'updates':steps,'exposures':exposures,'epoch_example_mean_bce':losses,
            'train_s':time.monotonic()-started,'optimizer_groups':groups,'steps_sha256':sha(path)}


def prediction_row(seed,item,index,logit,probability):
    require(type(seed) is int and seed in SEEDS and math.isfinite(logit) and math.isfinite(probability)
            and 0<=probability<=1,'Invalid binary output')
    expected = 1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
    require(abs(expected-probability)<=1e-6,'Probability/logit disagreement')
    return {'seed':seed,'model_arm':'full_head','eval_arm':'native','id':item['id'],'tile':item['tile'],
            'cluster':item['cluster'],'phen':item['phen'],'kind':item['kind'],'pair_index':index,
            'source_gold':item['answer'],'transformed_gold':None,'logit':logit,'probability':probability,
            'prediction':'yes' if logit>=0 else 'no'}


def run(out):
    out=Path(out).resolve(); state={'phase':'verify'}; budget=Budget(); previous={}; owned_run=False
    def set_status(status,**extra):
        atomic_status(out,{'schema':'e6-status-v0','status':status,'at':now(),'pid':os.getpid(),**state,**extra})
    def interrupted(signum,frame):
        raise RuntimeError('E6 interrupted or deadline exceeded: '+str(signum))
    try:
        require(out.parent==ROOT and out.name=='e6_no_llm_v0','Dedicated E6 output required')
        check_locks(ROOT)
        require(read(out/'status.json')['status']=='prepared','Run is not prepared; no resume')
        require(not any((out/name).exists() for name in ('failure.json','runtime_environment.json','run_claim.json',
                    'models','initial_states','predictions.jsonl','scores.json','training_summary.json')),'Execution output already exists')
        owned_run=True
        manifest=read(out/'manifest.json'); manifest_sha=sha(out/'manifest.json')
        require(manifest_sha==os.environ.get('E6_MANIFEST_SHA256'),'Launch manifest differs')
        verify_files(out,manifest); cfg=read(out/'prereg.json'); validate_config(cfg)
        audit=read(out/'reference_audit.json')
        require(audit['valid'] is True and audit['independent_consistent'] is True
                and audit['e5_prepared_manifest_sha256']==PARENT_SHA
                and audit['reference_rows_sha256']==sha(out/'reference_rows.jsonl'),'Reference validity gate differs')
        write(out/'run_claim.json',{'at':now(),'pid':os.getpid(),'manifest_sha256':manifest_sha,'resume_allowed':False})
        for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.getsignal(sig); signal.signal(sig,interrupted)
        budget.arm(); set_status('running')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic cuBLAS environment missing')
        uuid=check_idle()
        import torch
        from e6_head_model_v0 import encode_metadata,make_head,state_hash
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
        device=torch.device('cuda:0')
        runtime={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,
                 'gpu_index':0,'gpu_uuid':uuid,'gpu_name':torch.cuda.get_device_name(0),'device':'cuda:0',
                 'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32_matmul':False,
                 'tf32_cudnn':False,'cudnn_benchmark':False,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
                 'manifest_sha256':manifest_sha,'llm_loaded':False,'no_source_date_repair':True}
        write(out/'runtime_environment.json',runtime)
        state['phase']='validate_inputs'; items=lines(out/'items.jsonl'); ordered=read(out/'ordered_ids.json')
        batches=read(out/'batches.json'); eval_sets=read(out/'eval_sets.json')
        pairs=np.load(out/'pairs.npy',mmap_mode='r',allow_pickle=False)
        lookup=validate_population(items,ordered,batches,pairs,eval_sets)
        metadata=encode_metadata([i['phen'] for i in items],[i['dates'] for i in items])
        np.save(out/'metadata_vectors.npy',metadata.numpy(),allow_pickle=False)
        with (out/'metadata_snapshot.jsonl').open('x') as handle:
            for index,item in enumerate(items):
                handle.write(json.dumps({'id':item['id'],'pair_index':index,'phen':item['phen'],'dates':item['dates'],
                                         'encoded_values':metadata[index].tolist()},allow_nan=False)+'\n')
        metadata_pins={name:sha(out/name) for name in ('metadata_vectors.npy','metadata_snapshot.jsonl')}
        initial_root=out/'initial_states'; initial_root.mkdir(); (out/'models').mkdir()
        state['phase']='freeze_all_initial_states'; initial={}
        for seed in SEEDS:
            budget.check(); model=make_head(seed); finite(torch,model.state_dict().values(),'initial state')
            path=initial_root/f'seed{seed}.pt'; torch.save(model.state_dict(),path)
            initial[str(seed)]={'seed':seed,'file_sha256':sha(path),'tensor_sha256':state_hash(model),
                                'n_parameters':sum(p.numel() for p in model.parameters())}
            del model
        write(initial_root/'manifest.json',{'schema':'e6-initial-states-v0','before_any_training':True,'seeds':initial})
        initial_manifest_sha=sha(initial_root/'manifest.json')
        trained=[]
        for seed in SEEDS:
            state.update(phase='training',seed=seed); budget.arm(seed); set_status('running',step=0)
            directory=out/'models'/f'seed{seed}_full_head'; directory.mkdir()
            start=initial_root/f'seed{seed}.pt'; shutil.copyfile(start,directory/'initial.pt')
            require(sha(start)==sha(directory/'initial.pt')==initial[str(seed)]['file_sha256'],'Initial state copy changed')
            model=make_head(seed); model.load_state_dict(torch.load(directory/'initial.pt',map_location='cpu',weights_only=True),strict=True)
            require(state_hash(model)==initial[str(seed)]['tensor_sha256'],'Actual initial tensors differ')
            model=model.to(device); torch.cuda.reset_peak_memory_stats(device)
            write(directory/'training_contract.json',{'seed':seed,'model_arm':'full_head','epochs':3,'batch_size':8,
                'expected_updates':1590,'expected_exposures':12702,'initial':initial[str(seed)],
                'initial_manifest_sha256':initial_manifest_sha,'batches_sha256':sha(out/'batches.json')})
            record=train_model(torch,model,pairs,metadata,items,lookup,batches[str(seed)],device,directory/'steps.jsonl',budget,
                               lambda step,exposures:set_status('running',step=step,exposures=exposures),seed=seed)
            require((record['updates'],record['exposures'])==(1590,12702),'Final training budget differs')
            finite(torch,model.state_dict().values(),'final state'); tensor_hash=state_hash(model)
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},directory/'head.pt')
            final_sha=sha(directory/'head.pt'); loaded=make_head(seed)
            loaded.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True),strict=True)
            require(state_hash(loaded)==tensor_hash,'Saved final checkpoint differs')
            record.update(seed=seed,initial_file_sha256=initial[str(seed)]['file_sha256'],initial_tensor_sha256=initial[str(seed)]['tensor_sha256'],
                          checkpoint_sha256=final_sha,checkpoint_tensor_sha256=tensor_hash,
                          train_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          train_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            write(directory/'training_completed.json',record); trained.append(record)
            del model,loaded; torch.cuda.empty_cache(); budget.finish()
        write(out/'all_training_completed.json',{'n_models':3,'updates':4770,'exposures':38106,'models':trained,
                                               'test_inference_started':False,'initial_manifest_sha256':initial_manifest_sha})
        outcomes=[]; row_count=0; state['phase']='evaluation'
        with (out/'predictions.jsonl').open('x') as aggregate:
            for seed,record in zip(SEEDS,trained):
                state.update(phase='evaluation',seed=seed); budget.arm(seed); set_status('running',predicted=0)
                directory=out/'models'/f'seed{seed}_full_head'
                require(sha(directory/'head.pt')==record['checkpoint_sha256'],'Final checkpoint changed before evaluation')
                model=make_head(seed); model.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True),strict=True)
                require(state_hash(model)==record['checkpoint_tensor_sha256'],'Reloaded evaluation tensors differ')
                model=model.to(device).eval(); torch.cuda.reset_peak_memory_stats(device); eval_started=time.monotonic(); n=0
                with (directory/'predictions.jsonl').open('x') as handle,torch.no_grad():
                    for offset in range(0,1755,8):
                        budget.check(); ids=ordered['test'][offset:offset+8]; indices=[lookup[x] for x in ids]
                        x,m=inputs(torch,pairs,metadata,indices,device); logits=model(x,m); probabilities=torch.sigmoid(logits)
                        finite(torch,[logits,probabilities],'evaluation output')
                        for index,z,p in zip(indices,logits.cpu().tolist(),probabilities.cpu().tolist()):
                            row=prediction_row(seed,items[index],index,z,p); text=json.dumps(row,allow_nan=False)+'\n'
                            handle.write(text); aggregate.write(text); n+=1; row_count+=1
                        handle.flush(); aggregate.flush()
                        if n%200==0:set_status('running',predicted=n)
                    os.fsync(handle.fileno())
                require(n==1755 and state_hash(model)==record['checkpoint_tensor_sha256'],'Evaluation coverage/state changed')
                require(sha(directory/'head.pt')==record['checkpoint_sha256'],'Final checkpoint changed during evaluation')
                outcome={**record,'eval_s':time.monotonic()-eval_started,'predictions_sha256':sha(directory/'predictions.jsonl'),
                         'n_predictions':n,'eval_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                         'eval_peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
                write(directory/'completed.json',outcome); outcomes.append(outcome)
                del model; torch.cuda.empty_cache(); budget.finish()
            os.fsync(aggregate.fileno())
        require(len(outcomes)==3 and row_count==5265,'Final model/output coverage differs')
        state['phase']='final_integrity'; verify_files(out,manifest)
        require(sha(out/'manifest.json')==manifest_sha and sha(initial_root/'manifest.json')==initial_manifest_sha,'Manifest changed during run')
        for name,expected in metadata_pins.items():require(sha(out/name)==expected,'Encoded metadata changed during run')
        for record in outcomes:
            directory=out/'models'/f"seed{record['seed']}_full_head"
            require(sha(initial_root/f"seed{record['seed']}.pt")==record['initial_file_sha256'],'Frozen initial state changed')
            for name,key in [('head.pt','checkpoint_sha256'),('initial.pt','initial_file_sha256'),('steps.jsonl','steps_sha256'),('predictions.jsonl','predictions_sha256')]:
                require(sha(directory/name)==record[key],'Completed artifact changed: '+name)
        summary={'schema':'e6-training-summary-v0','n_models':3,'n_rows':5265,'updates_per_model':1590,
                 'exposures_per_model':12702,'models':outcomes,'all_training_before_test':True,
                 'manifest_sha256':manifest_sha,'initial_manifest_sha256':initial_manifest_sha,
                 'metadata_vectors_sha256':sha(out/'metadata_vectors.npy'),'metadata_snapshot_sha256':sha(out/'metadata_snapshot.jsonl'),
                 'predictions_sha256':sha(out/'predictions.jsonl'),'runtime_sha256':sha(out/'runtime_environment.json'),
                 'code_snapshot_sha256':manifest['code_snapshot_sha256']}
        write(out/'training_summary.json',summary); set_status('inference_completed',n_models=3,n_rows=5265)
        state['phase']='scoring'
        from e6_scoring_v0 import score_run
        scores=score_run(lines(out/'predictions.jsonl'),lines(out/'reference_rows.jsonl'),items,eval_sets)
        scores.update(manifest_sha256=manifest_sha,training_summary_sha256=sha(out/'training_summary.json'),
                      predictions_sha256=summary['predictions_sha256'],reference_rows_sha256=sha(out/'reference_rows.jsonl'),
                      elapsed_s=time.monotonic()-budget.started)
        require(sha(out/'reference_rows.jsonl')==manifest['files_sha256']['reference_rows.jsonl']
                and sha(out/'predictions.jsonl')==summary['predictions_sha256'],'Scoring input changed')
        for name,expected in manifest['code_snapshot_sha256'].items():
            require(sha(relative(out/'code_snapshot',name))==expected,'Source changed during scoring')
        write(out/'scores.json',scores); require(scores.get('valid') is True,'E6 scoring invalid')
        budget.check(); set_status('completed',n_models=3,n_rows=5265,scientifically_valid=True)
        print('E6 COMPLETE descriptive_system_comparison',flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        if owned_run and out.is_dir():
            failure={'schema':'e6-failure-v0','at':now(),**state,'exception_type':type(exc).__name__,'reason':str(exc),
                     'traceback':traceback.format_exc(),'resume_allowed':False,'partial_artifacts_preserved':True,
                     'elapsed_s':time.monotonic()-budget.started}
            if not (out/'failure.json').exists():write(out/'failure.json',failure)
            set_status('invalid',reason=str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
