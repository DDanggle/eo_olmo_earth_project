#!/usr/bin/env python3
"""Prepare E6 only after complete E5 and its independent CPU tensor audit.

No model is loaded or trained. Copy the five exact prepared input files and
freeze the one preselected reference (full/native), source, and specification.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

PLAN_SHA = '66f820db8bc6a3d03c07d66676adcbe6170c012649a824e83b02a94d03194840'
PARENT_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
AUDITOR_SHA = 'a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825'
ROOT = Path('/home/work/data/olmoearth')
SOURCE_FILES = ('e6_prepare_v0.py','e6_head_model_v0.py','e6_train_v0.py','e6_scoring_v0.py','run_e6_when_idle_v0.py')
INPUT_FILES = ('items.jsonl','pairs.npy','ordered_ids.json','batches.json','eval_sets.json')
REFERENCE_FILES = ('manifest.json','prereg.json','status.json','scores.json','training_summary.json','inference_completed.json','predictions.jsonl')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bad_constant(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=bad_constant)


def lines(path):
    return [json.loads(line, parse_constant=bad_constant) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def validate_reference_report(audit, artifact):
    require(audit.get('schema')=='e5-independent-result-audit-v0' and audit.get('consistent') is True,
            'Independent E5 audit has not passed')
    require(audit.get('artifact')==str(artifact.resolve()), 'Audit refers to another reference root')
    require(audit.get('checkpoint_tensors_loaded_and_checked_on_cpu') is True, 'Independent checkpoint tensor checks required')
    require(audit.get('audit_code_sha256')==AUDITOR_SHA, 'Unexpected independent E5 auditor source')
    require(tuple(audit.get(k) for k in ('n_models','n_steps','n_training_exposures','n_answers','primary_n','primary_events'))
            ==(12,19080,152424,26325,902,8), 'Reference audit population/budget differs')
    require(isinstance(audit.get('hashes_verified'),dict), 'Missing independent E5 hash pins')


def select_reference_rows(all_rows, items):
    require(len(all_rows)==26325 and len(items)==5989, 'Reference/input population differs')
    source={i['id']:i for i in items}
    index={i['id']:j for j,i in enumerate(items)}
    require(len(source)==5989, 'Duplicate input IDs')
    test={i['id'] for i in items if i['partition']=='test'}
    require(len(test)==1755, 'Reference test population differs')
    selected=[row for row in all_rows if row.get('model_arm')=='full' and row.get('eval_arm')=='native']
    expected={(seed,key) for seed in (1,2,3) for key in test}
    actual=[]
    for row in selected:
        require(type(row.get('seed')) is int, 'Reference seed must be an integer')
        key=(row['seed'],row.get('id'))
        require(key in expected,'Reference ID/seed not in fixed test population')
        item=source[row['id']]
        require(all(row.get(k)==item[k] for k in ('tile','cluster','phen','kind')), 'Reference metadata changed')
        require(row.get('source_gold')==item['answer'] and row.get('transformed_gold') is None,
                'Reference source labels changed')
        require(type(row.get('pair_index')) is int and row['pair_index']==index[row['id']], 'Reference global pair index differs')
        require(row.get('parsed') in ('yes','no',None) and isinstance(row.get('answer_raw'),str), 'Reference answer schema differs')
        actual.append(key)
    require(len(selected)==len(set(actual))==5265 and set(actual)==expected, 'Incomplete/duplicate full/native reference coverage')
    return selected


def snapshot_sources(bundle, destination):
    digests={name:sha(bundle/name) for name in SOURCE_FILES}
    destination.mkdir(exist_ok=False)
    for name,expected in digests.items():
        shutil.copyfile(bundle/name,destination/name)
        require(sha(destination/name)==expected and sha(bundle/name)==expected, 'Source changed during snapshot: '+name)
    return digests


def prepare(out, bundle):
    out,bundle=Path(out).resolve(),Path(bundle).resolve()
    require(out.parent==ROOT and out.name=='e6_no_llm_v0', 'Dedicated E6 output required')
    require(not out.exists(), 'E6 output already exists; no silent resume/retry')
    plan_path=bundle/'e6_no_llm_prereg_v0.json'
    require(sha(plan_path)==PLAN_SHA,'Frozen E6 scientific specification changed')
    cfg=read(plan_path)
    require(cfg['source_files']==list(SOURCE_FILES), 'Source membership changed')
    require(cfg['root']==str(ROOT) and cfg['output_directory']==out.name, 'Plan root/output differs')
    ref=Path(cfg['reference']['directory']).resolve()
    require(ref==ROOT/'e5_equal_budget_v0', 'Unexpected E5 reference')
    require(sha(ref/'manifest.json')==PARENT_SHA==cfg['reference']['prepared_manifest_sha256'], 'E5 prepared manifest changed')
    require(not (ref/'failure.json').exists(), 'E5 failed; do not prepare comparison')
    status,scores=read(ref/'status.json'),read(ref/'scores.json')
    require(status.get('status')=='completed' and scores.get('valid') is True, 'E5 must complete valid before preparation')
    audit_path=Path(cfg['reference']['independent_audit']).resolve()
    require(audit_path==ROOT/'e5_independent_audit_20260925.json', 'Unexpected E5 audit path')
    audit=read(audit_path)
    validate_reference_report(audit,ref)
    tracked={str(audit_path):sha(audit_path)}
    for name in INPUT_FILES+REFERENCE_FILES:
        path=ref/name
        digest=sha(path)
        require(audit['hashes_verified'].get(str(path))==digest, 'E5 file differs from independent audit: '+name)
        if name in INPUT_FILES:
            require(cfg['reference']['files_sha256'][name]==digest, 'E6 planned parent input changed: '+name)
        tracked[str(path)]=digest
    require(tracked[str(ref/'prereg.json')]==cfg['reference']['prereg_sha256'], 'Reference plan changed')
    require(scores['verdict']==status['verdict']==audit['verdict'], 'Reference result/audit decision differs')
    reference_rows=select_reference_rows(lines(ref/'predictions.jsonl'),lines(ref/'items.jsonl'))
    require(shutil.disk_usage(ROOT).free>sum((ref/n).stat().st_size for n in INPUT_FILES)+512*1024**2,
            'Not enough space for unchanged input copy and records')
    started=time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        write(out/'status.json',{'status':'preparing','at':now()})
        source_pins=snapshot_sources(bundle,out/'code_snapshot')
        files=[]
        for name in INPUT_FILES:
            shutil.copyfile(ref/name,out/name)
            require(sha(out/name)==tracked[str(ref/name)], 'Input copy mismatch: '+name)
            files.append(name)
        shutil.copyfile(plan_path,out/'prereg.json');files.append('prereg.json')
        require(sha(out/'prereg.json')==PLAN_SHA,'Plan changed during preparation')
        shutil.copyfile(audit_path,out/'e5_independent_audit.json');files.append('e5_independent_audit.json')
        for name in REFERENCE_FILES:
            if name=='predictions.jsonl':
                continue  # Keep full-reference original hash; copy only the fixed full/native subset.
            dest='reference_'+name
            shutil.copyfile(ref/name,out/dest)
            require(sha(out/dest)==tracked[str(ref/name)], 'Reference copy mismatch: '+name)
            files.append(dest)
        with (out/'reference_rows.jsonl').open('x') as f:
            for row in reference_rows:
                f.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        files.append('reference_rows.jsonl')
        write(out/'reference_audit.json',{
            'schema':'e6-reference-gate-v0','valid':True,'independent_consistent':True,
            'e5_prepared_manifest_sha256':PARENT_SHA,
            'e5_independent_audit_sha256':tracked[str(audit_path)],
            'reference_rows_sha256':sha(out/'reference_rows.jsonl'),
            'reference_selection':'full/native only, specified before E5 final scores; all3seeds/all1755test',
            'n_rows':5265,'original_e5_files_sha256':tracked,
            'limits':'An independently audited saved reference, not a replay of its training or a fresh data test.'})
        files.append('reference_audit.json')
        for name,expected in source_pins.items():
            require(sha(bundle/name)==expected and sha(out/'code_snapshot'/name)==expected,'Source changed during preparation')
        for path,expected in tracked.items():
            require(sha(path)==expected,'E5 source changed during preparation: '+path)
        require(sha(out/'e5_independent_audit.json')==tracked[str(audit_path)],'Independent audit copy differs')
        write(out/'preparation_report.json',{'at':now(),'elapsed_s':time.monotonic()-started,
            'n_items':5989,'n_train':4234,'n_test':1755,'n_reference_rows':5265,
            'source_model_loaded':False,'new_date_metadata_applied':False,'e5_artifact_written':False,
            'source_sha256':tracked,'copied_pairs_without_repooling':True,
            'remaining_before_train':'Runner rechecks finite matrices, exact item/index/batch support, CPU initialization snapshots and GPU/lease identity.'})
        files.append('preparation_report.json')
        manifest={'schema':'e6-no-llm-prepared-v0','prepared_at':now(),
            'parent_e5_prepared_manifest_sha256':PARENT_SHA,'n_items':5989,'n_train':4234,'n_test':1755,
            'files_sha256':{name:sha(out/name) for name in files},'code_snapshot_sha256':source_pins,
            'pairs_shape':[5989,2,64,768],'pairs_dtype':'float32',
            'expected_updates_per_model':1590,'expected_exposures_per_model':12702,
            'plan_sha256':PLAN_SHA,'all_initial_states_before_training':True,'all_three_trainings_before_test_inference':True}
        write(out/'manifest.json',manifest)
        write(out/'status.json',{'status':'prepared','at':now(),'manifest_sha256':sha(out/'manifest.json')})
        return {'status':'prepared','manifest_sha256':sha(out/'manifest.json'),'out':str(out)}
    except BaseException as error:
        write(out/'failure.json',{'phase':'preparation','at':now(),'error':str(error),'traceback':traceback.format_exc()})
        write(out/'status.json',{'status':'failed','phase':'preparation','at':now(),'error':str(error)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'e6_no_llm_v0')
    args=p.parse_args()
    def expired(signum,frame):
        raise TimeoutError('E6 preparation exceeded10minutes')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,600)
    try:
        print(json.dumps(prepare(args.out,Path(__file__).resolve().parent)),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
