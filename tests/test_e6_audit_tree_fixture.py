"""One invented artifact-tree integration fixture; pairs array reader is stubbed.

Exercises all file/record paths and deliberate tampering. It is not a dataset,
model run or actual E6 result. Dense EO tensor reads are covered by source audit
plus real finite tests separately; this fixture avoids a 2.35 GB allocation.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import audit_e6_results_v0 as a
from test_audit_e6_results_v0 import fixture


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in values))


def build(root,ref):
    items,sets,ordered,batches,heads,refs=fixture()
    write(root/'ordered_ids.json',ordered);write(root/'eval_sets.json',sets);write(root/'batches.json',batches);rows(root/'items.jsonl',items)
    np.save(root/'pairs.npy',np.zeros(1,dtype=np.float32))
    reference_manifest={'synthetic_parent':True};write(root/'reference_manifest.json',reference_manifest)
    parent_sha=a.digest(root/'reference_manifest.json')
    write(root/'reference_prereg.json',{'synthetic_e5_plan':True});e5plan=a.digest(root/'reference_prereg.json')
    verdict='synthetic_reference_valid'
    for name,value in [('status',{'status':'completed','verdict':verdict}),('scores',{'valid':True,'verdict':verdict}),('training_summary',{}),('inference_completed',{})]:write(root/f'reference_{name}.json',value)
    ref.mkdir();original=copy.deepcopy(refs)
    for arm,mode in [('pair','native'),('later','native'),('delta','native'),('full','full_no_delta')]:
        original.extend([{**row,'model_arm':arm,'eval_arm':mode} for row in refs])
    rows(ref/'predictions.jsonl',original);rows(root/'reference_rows.jsonl',refs)
    remote=Path('/synthetic/remote/e5_equal_budget_v0');audit_remote='/synthetic/remote/e5_audit.json'
    tracked={str(remote/name):a.digest(root/name) for name in ('items.jsonl','pairs.npy','ordered_ids.json','batches.json','eval_sets.json')}
    for name in ('manifest.json','prereg.json','status.json','scores.json','training_summary.json','inference_completed.json'):
        tracked[str(remote/name)]=a.digest(root/('reference_'+name))
    tracked[str(remote/'predictions.jsonl')]=a.digest(ref/'predictions.jsonl')
    independent={'schema':'e5-independent-result-audit-v0','consistent':True,'checkpoint_tensors_loaded_and_checked_on_cpu':True,
                 'audit_code_sha256':a.E5_AUDITOR_SHA,'artifact':str(remote),'n_models':12,'n_steps':19080,'n_training_exposures':152424,
                 'n_answers':26325,'primary_n':902,'primary_events':8,'verdict':verdict,'hashes_verified':dict(tracked)}
    write(root/'e5_independent_audit.json',independent);tracked[audit_remote]=a.digest(root/'e5_independent_audit.json')
    gate={'schema':'e6-reference-gate-v0','valid':True,'independent_consistent':True,'n_rows':5265,'e5_prepared_manifest_sha256':parent_sha,
          'e5_independent_audit_sha256':tracked[audit_remote],'reference_rows_sha256':a.digest(root/'reference_rows.jsonl'),'original_e5_files_sha256':tracked}
    write(root/'reference_audit.json',gate);write(root/'preparation_report.json',{'synthetic':True})
    cfg={'id':'E6-HEAD-v0','source_files':sorted(a.SOURCE_FILES),'reference':{'directory':str(remote),'independent_audit':audit_remote,
         'prereg_sha256':e5plan,'files_sha256':{name:tracked[str(remote/name)] for name in ('items.jsonl','pairs.npy','ordered_ids.json','batches.json','eval_sets.json')}}}
    write(root/'prereg.json',cfg);plan_sha=a.digest(root/'prereg.json')
    sourcepins={}
    for name in sorted(a.SOURCE_FILES):
        path=root/'code_snapshot'/name;path.parent.mkdir(exist_ok=True);path.write_text('# Synthetic never executed\n');sourcepins[name]=a.digest(path)
    manifest={'schema':'e6-no-llm-prepared-v0','parent_e5_prepared_manifest_sha256':parent_sha,'n_items':5989,'n_train':4234,'n_test':1755,
              'plan_sha256':plan_sha,'code_snapshot_sha256':sourcepins,
              'files_sha256':{path.name:a.digest(path) for path in root.iterdir() if path.is_file()}}
    write(root/'manifest.json',manifest);manifest_sha=a.digest(root/'manifest.json')
    metadata=a.metadata_values(items);np.save(root/'metadata_vectors.npy',metadata)
    rows(root/'metadata_snapshot.jsonl',[dict(id=x['id'],pair_index=j,phen=x['phen'],dates=x['dates'],encoded_values=metadata[j].tolist()) for j,x in enumerate(items)])
    runtime={'manifest_sha256':manifest_sha,'gpu_index':0,'device':'cuda:0','deterministic_algorithms':True,'tf32_matmul':False,'tf32_cudnn':False,
             'cudnn_benchmark':False,'cublas_workspace_config':':4096:8','llm_loaded':False,'no_source_date_repair':True}
    write(root/'runtime_environment.json',runtime);write(root/'run_claim.json',{'manifest_sha256':manifest_sha,'resume_allowed':False})
    state={name:torch.zeros(shape,dtype=torch.int64 if name=='token_types' else torch.float32) for name,shape in a.checkpoint_schema().items()}
    state['token_types']=torch.tensor([0]*64+[1]*64+[3]*64)
    import math
    state['positions']=torch.tensor([[v for c in (r,c) for k in range(32) for v in (math.sin(c/10000**(k/32)),math.cos(c/10000**(k/32)))] for r in range(8) for c in range(8)]*3,dtype=torch.float32)
    initial={};(root/'initial_states').mkdir()
    for seed in (1,2,3):
        path=root/'initial_states'/f'seed{seed}.pt';torch.save(state,path)
        initial[str(seed)]={'seed':seed,'file_sha256':a.digest(path),'tensor_sha256':a.canonical_tensor_hash(state),'n_parameters':367361}
    write(root/'initial_states/manifest.json',{'schema':'e6-initial-states-v0','before_any_training':True,'seeds':initial})
    initial_sha=a.digest(root/'initial_states/manifest.json');trained=[];completed=[];index={x['id']:j for j,x in enumerate(items)}
    for seed in (1,2,3):
        directory=root/'models'/f'seed{seed}_full_head';directory.mkdir(parents=True)
        (directory/'initial.pt').write_bytes((root/'initial_states'/f'seed{seed}.pt').read_bytes())
        final={key:value.clone() for key,value in state.items()};final['classifier.bias'][0]=.1;torch.save(final,directory/'head.pt')
        contract={'seed':seed,'model_arm':'full_head','epochs':3,'batch_size':8,'expected_updates':1590,'expected_exposures':12702,
                  'initial':initial[str(seed)],'initial_manifest_sha256':initial_sha,'batches_sha256':a.digest(root/'batches.json')}
        write(directory/'training_contract.json',contract)
        steps=[];step=exposures=0
        for epoch,ep in enumerate(batches[str(seed)]):
            for batch_index,ids in enumerate(ep):
                step+=1;exposures+=len(ids)
                steps.append(dict(seed=seed,step=step,epoch=epoch,batch_index=batch_index,ids=ids,pair_indices=[index[x] for x in ids],batch_size=len(ids),exposures=exposures,loss=1.,loss_finite=True,gradients_finite=True,parameters_finite=True,elapsed_s=step*.01))
        rows(directory/'steps.jsonl',steps)
        record={'seed':seed,'updates':1590,'exposures':12702,'epoch_example_mean_bce':[1.,1.,1.],
                'train_s':20.,'optimizer_groups':[{'lr':1e-4,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'foreach':False,'fused':False,'amsgrad':False,'maximize':False}],
                'initial_file_sha256':initial[str(seed)]['file_sha256'],'initial_tensor_sha256':initial[str(seed)]['tensor_sha256'],
                'steps_sha256':a.digest(directory/'steps.jsonl'),'checkpoint_sha256':a.digest(directory/'head.pt'),'checkpoint_tensor_sha256':a.canonical_tensor_hash(final)}
        write(directory/'training_completed.json',record);trained.append(record)
        selected=[r for r in heads if r['seed']==seed];rows(directory/'predictions.jsonl',selected)
        complete={**record,'eval_s':1.,'predictions_sha256':a.digest(directory/'predictions.jsonl'),'n_predictions':1755}
        write(directory/'completed.json',complete);completed.append(complete)
    write(root/'all_training_completed.json',{'n_models':3,'updates':4770,'exposures':38106,'models':trained,'test_inference_started':False,'initial_manifest_sha256':initial_sha})
    rows(root/'predictions.jsonl',heads)
    summary={'schema':'e6-training-summary-v0','all_training_before_test':True,'n_models':3,'n_rows':5265,'updates_per_model':1590,'exposures_per_model':12702,
             'models':completed,'code_snapshot_sha256':sourcepins,'manifest_sha256':manifest_sha,'initial_manifest_sha256':initial_sha,
             'metadata_vectors_sha256':a.digest(root/'metadata_vectors.npy'),'metadata_snapshot_sha256':a.digest(root/'metadata_snapshot.jsonl'),
             'predictions_sha256':a.digest(root/'predictions.jsonl'),'runtime_sha256':a.digest(root/'runtime_environment.json')}
    write(root/'training_summary.json',summary)
    _,test,strata=a.population(items,sets,ordered,batches);head,hp=a.predictions(heads,items,True);reference,rp=a.predictions(refs,items,False)
    scores={'schema':'e6-no-llm-scores-v0','valid':True,'verdict':'descriptive_system_comparison','manifest_sha256':manifest_sha,
            'training_summary_sha256':a.digest(root/'training_summary.json'),'predictions_sha256':a.digest(root/'predictions.jsonl'),
            'reference_rows_sha256':a.digest(root/'reference_rows.jsonl'),'metrics':a.recompute_metrics(test,strata,head,reference),
            'parse_fail_rates':{'full/native':rp,'full_head/native':hp},'coverage':{'expected_per_system':5265,'received_head':5265,'received_reference':5265,'n_items':5989,'n_train':4234,'n_test':1755,'n_primary':902,'seeds':[1,2,3]},
            'bootstrap':{'scope':'primary contrast separately per seed only','unit':'paired whole event','n_events':8,'draws':5000,'rng_seed':20260925,'quantile_method':'linear','event_order':'sorted string event IDs'},'elapsed_s':70.}
    write(root/'scores.json',scores);write(root/'status.json',{'status':'completed','scientifically_valid':True,'n_models':3,'n_rows':5265})
    return plan_sha,parent_sha,manifest_sha


class TreeTest(unittest.TestCase):
    def test_relocated_synthetic_tree_and_exact_pin_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'e6';root.mkdir();ref=Path(temp).resolve()/'e5'
            plan,parent,manifest=build(root,ref)
            original_load=np.load
            def load(path,*args,**kwargs):
                if Path(path)==root/'pairs.npy':return np.broadcast_to(np.zeros((2,64,768),dtype=np.float32),(5989,2,64,768))
                return original_load(path,*args,**kwargs)
            with patch.object(a,'PLAN_SHA',plan),patch.object(a,'E5_MANIFEST_SHA',parent),patch.object(a.np,'load',side_effect=load):
                report=a.audit(root,manifest,ref)
                self.assertTrue(report['consistent']);self.assertEqual(report['n_steps'],4770)
                self.assertEqual(report['reference_original_root'],'/synthetic/remote/e5_equal_budget_v0')
                self.assertEqual(report['reference_prediction_replica_root'],str(ref))
                self.assertTrue(report['checkpoint_tensors_loaded_and_checked_on_cpu'])
                self.assertIn(str(root/'scores.json'),report['hashes_verified'])
                (root/'reference_rows.jsonl').write_text('{}\n')
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):a.audit(root,manifest,ref)


if __name__=='__main__':unittest.main()
