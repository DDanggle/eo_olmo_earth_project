#!/usr/bin/env python3
"""Freeze actual manual pilot protocol after source/input preparation, before GPU."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

SERVER=Path('/home/work/data/olmoearth/oe11_training_pilot_v0')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=a.repo;artifact=r/'artifacts/oe11_training_pilot_20260929';snap=artifact/'snapshot'
    assert not a.out.exists()
    paths={name:str(SERVER/path) for name,path in {
        'worker':'snapshot/oe11_training_worker_v0/worker.py','matching_code':'snapshot/oe11_matching_v0',
        'loader_code':'snapshot/oe11_loader_v0','text_mask_code':'snapshot/oe10_text_mask_v2',
        'contexts':'snapshot/oe11_context_v0/contexts.json','cases':'config/train_pilot_cases_v1.json',
        'case_preflight':'actual_input_preflight_v0.json','token_audit':'token_audit_v0.json'}.items()}
    paths.update(eo_source='/home/work/data/olmoearth/oe4_native_v12_v0/source',
        deps='/home/work/data/olmoearth/oe4_native_v12_v0/deps',
        eo_weights='/home/work/data/olmoearth/oe4_native_v12_v0/models/OlmoEarth-v1_2-Base',
        qwen_weights='/home/work/data/olmoearth/models/Qwen3-VL-8B-Instruct',
        prepared='/home/work/data/olmoearth/oe11_source_expand_v0/prepared_v0',
        episodes='/home/work/data/olmoearth/oe11_source_expand_v0/catalog_pooled_v0',
        model_identity='/home/work/data/olmoearth/oe10_text_mask_identity_v2/identity_preparation_v1/identity_manifest.json')
    pins={str(SERVER/'snapshot'/f.relative_to(snap)):sha(f) for f in sorted(snap.rglob('*'))
          if f.is_file() and '__pycache__' not in f.parts}
    for name,local in [('cases',artifact/'train_pilot_cases_v1.json'),('case_preflight',artifact/'actual_input_preflight_v0.json'),
                       ('token_audit',artifact/'token_audit_v0.json'),
                       ('model_identity',r/'artifacts/oe10_connection_identity_20260928/identity_preparation_v1/identity_manifest.json')]:
        pins[paths[name]]=sha(local)
    cases=json.loads((artifact/'train_pilot_cases_v1.json').read_text());pins.update(cases['input_hashes'])
    assert Path(paths['worker']).as_posix() in pins
    assert len(cases['episode_ids'])==24 and cases['total_steps']==48 and cases['split_step']==24
    pf=json.loads((artifact/'actual_input_preflight_v0.json').read_text());assert pf['status']=='PASS'
    for field,key in [('cases_sha256','cases'),('contexts_sha256','contexts'),('model_identity_sha256','model_identity')]:assert pf[field]==pins[paths[key]]
    value={'schema':'oe11_training_pilot_v0','backend':'actual','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Actual mask-only optimization and fresh-process resume engineering, before any comparative efficacy run',
        'seed':290929,'total_steps':48,'split_step':24,'max_stage_seconds':1800,'resume_tolerance':1e-6,
        'max_text_tokens':1024,'max_text_cache_entries':32,
        'optimizer':{'encoder_lr':1e-5,'head_lr':1e-3,'weight_decay':.01,'eps':1e-6,'clip_grad_norm':1.0},
        'optimizer_policy':'Fixed LR; AdamW foreach=false,fused=false; no old96-step warmup copied into48 steps.',
        'allowed_arms':['B0','B2'],'allowed_conditions':['names_only','matched_knowledge'],
        'execution':{'arm':'B2','condition':'matched_knowledge','stages':['uninterrupted','split','resume'],
            'gpu_preference':[1,0],'occupied_gpu_policy':'Refuse both-busy state without reservation or queued execution.'},
        'paths':paths,'file_sha256':pins,
        'sample':{'base_queries':12,'source_regions':3,'targets':[1,3,8,14],
            'strata':['target_present','confuser_only','neither'],'training_K':[1,8],
            'training_episodes':24,'fit_diagnostic_K':1,'fit_diagnostic_cases':12,
            'order':'Two precomputed seed290929 permutations of all24 identities; no score-based change',
            'observations':{'query':[2,5],'support':'all8 per object'},'split':'train_only'},
        'budget':{'ledger':'/home/work/data/olmoearth/oe10_text_mask_identity_v2/pilot_budget_ledger_v0.json',
            'maximum_cumulative_seconds':7200,'prior_charged_seconds_at_preparation':48,
            'job_seconds':1800,'all_three_stages_share_one_job_cap':True,'all_retries_and_inference_count':True,
            'job_id':'oe11_b2_facts_resume_01','total_optimizer_updates_across_three_processes':96,
            'automatic_retry':False,'automatic_scheduler_enabled':False},
        'computational_gates':['All loss, gradients, post-update parameters and optimizer moments finite.',
            'B2 encoder and head actually change; B0 encoder and reader/unused connector remain frozen.',
            'At least one K8 training step establishes query-key/object-key/text-key gradients; K1 zero selection gradient is expected.',
            'Model/AdamW/RNG exact restoration at split; fresh-process next-step and final model, optimizer, logits/loss match within1e-6, integer/RNG states exactly.',
            'Actual CUDA UUID equals prechecked physical0/1; other jobs never terminated.',
            'Original source/input hashes unchanged and total timeout honored.'],
        'learning_diagnostic_signals':{'not_computational_pass_criteria':True,'relative_train_loss_drop':.05,
            'mean_present_train_iou_increase':.02,'present_class_non_decrease_count':3,
            'max_absent_train_positive_fraction_increase':.02,
            'interpretation':'Descriptive signal on12K1 training cases only. Missing signal means trainability not established within this short budget, not method falsification.'},
        'text_diagnostics':'Name/fact/removal/wrong-fact swap sensitivity on fixed train input; no semantic-knowledge proof.',
        'limits':['No development/final-region image or label payloads or model predictions used; shared prepared manifest metadata is read. No performance superiority or transfer claim.',
            'Mask-only training: no official native replay, language CE or generated explanation evaluation.',
            'Public agronomy cards and synthetic supports are not human expert corrections.',
            'The source cohort includes former development region; final-region and external-region claims require separate evaluation.',
            'K8 is trained and gradient-checked, but before/after fit diagnostics here useK1 only.',
            'Passing this48-step engineering run does not establish sufficiently trained E0-E3 baselines.']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'protocol':str(a.out),'sha256':sha(a.out),'pinned_files':len(pins)}))

if __name__=='__main__':main()
