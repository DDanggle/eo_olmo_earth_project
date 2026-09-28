#!/usr/bin/env python3
"""Independent read-only local export validation; no original scorer imported.
Uses exported scoring counts, not raw labels, for aggregation. This is not
independent regeneration of IoU from source ground truth or model inference.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def macro(rows, k, metric):
    queries = defaultdict(list)
    for r in rows:
        if r['k'] != k:
            continue
        if metric == 'iou':
            if not r['target_present']:
                continue
            value = r['intersection'] / r['union']
        else:
            if r['target_present']:
                continue
            fraction = r['predicted_valid_pixels'] / r['valid_pixels']
            value = fraction if metric == 'fp_area' else float(fraction > .001)
        queries[(r['parent'], r['target_class'], r['query_patch_id'])].append(value)
    classes = defaultdict(list)
    for (parent, c, _), values in queries.items():
        classes[(parent, c)].append(mean(values))
    parents = defaultdict(list)
    by_class = {}
    for (parent, c), values in classes.items():
        value = mean(values)
        parents[parent].append(value)
        by_class[f'{parent}:{c}'] = {'value': value, 'unique_queries': len(values)}
    return mean(mean(v) for v in parents.values()), by_class


def auc(curve):
    ks = [1, 2, 4, 8]
    return sum((b-a) * (curve[str(a)] + curve[str(b)]) / 2 for a,b in zip(ks,ks[1:])) / 7


def review(export, gate):
    errors=[]
    manifest=load(export/'export_manifest.json')
    seen=set()
    for row in manifest['files']:
        rel=row['path']
        if rel in seen:
            errors.append(f'duplicate export path {rel}')
        seen.add(rel)
        p=export/rel
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            errors.append(f'export file mismatch {rel}')
    collection=load(export/'partial_collection_20260928_0539.json')
    if len(collection['verified_runs']) != 4 or collection['p2_decision_made']:
        errors.append('wrong partial completed count or premature P2 decision')
    protocol=load(export/'config/oe10_p2_execution_protocol_20260927.json')
    if sha(gate)!=protocol['gate_config_sha256']:
        errors.append('gate identity mismatch')
    gate_data=load(gate)['p2']
    allowed_cohort_hash=gate_data['cohort']['cohort_base_ids_sha256']
    runs={}
    npz_total=0
    cohort_by_run={}
    for verified in collection['verified_runs']:
        tag=f"{verified['arm']}_{verified['seed']}"
        root=export/'training_v0'/f'{tag}_train'
        receipt=load(root/'receipt.json')
        if sha(root/'receipt.json')!=verified['run_receipt_sha256']:
            errors.append(f'{tag} collector receipt mismatch')
        score=load(root/'score_step_002304.json')
        rows=score['per_episode_scoring_only']
        row_by_id={r['episode_id']:r for r in rows}
        bases=sorted(set(r['base_id'] for r in rows))
        cohort_hash=hashlib.sha256(('\n'.join(bases)+'\n').encode()).hexdigest()
        if len(rows)!=384 or len(row_by_id)!=384 or len(bases)!=96 or cohort_hash!=allowed_cohort_hash:
            errors.append(f'{tag} cohort mismatch')
        cohort_by_run[tag]=cohort_hash
        if score['probability_threshold'] != .5 or score['probability_comparison']!='strictly_greater_than':
            errors.append(f'{tag} threshold changed')
        if set(r['parent'] for r in rows)!={'t31tfm'} or set(r['target_class'] for r in rows)!={1,2,3}:
            errors.append(f'{tag} wrong class or parent')
        for r in rows:
            if not (0 <= r['intersection'] <= r['target_pixels'] <= r['valid_pixels'] <= 128*128):
                errors.append(f'{tag} invalid target/intersection count')
            if r['union']!=r['target_pixels']+r['predicted_valid_pixels']-r['intersection']:
                errors.append(f'{tag} union identity failed')
            if r['target_present'] and not math.isclose(r['intersection']/r['union'],r['target_iou'],abs_tol=1e-12):
                errors.append(f'{tag} row IoU mismatch')
        curve={}
        classes={}
        fp_area={}
        fp_case={}
        for k in [1,2,4,8]:
            curve[str(k)], classes[str(k)] = macro(rows,k,'iou')
            fp_area[str(k)],_=macro(rows,k,'fp_area')
            fp_case[str(k)],_=macro(rows,k,'fp_case')
            for actual, field in [(curve,'target_iou_by_k'),(fp_area,'absent_fp_area_by_k'),(fp_case,'absent_fp_case_rate_by_k')]:
                if not math.isclose(actual[str(k)], score[field][str(k)], abs_tol=1e-12):
                    errors.append(f'{tag} {field} K{k} aggregate mismatch')
        value=auc(curve)
        if not math.isclose(value,score['target_iou_auc'],abs_tol=1e-12):
            errors.append(f'{tag} final AUC mismatch')
        windows=[]
        for step in [384,768,1152,1536,1920,2304]:
            d=load(root/f'score_step_{step:06d}.json')
            c={str(k):macro(d['per_episode_scoring_only'],k,'iou')[0] for k in [1,2,4,8]}
            a=auc(c)
            if not math.isclose(a,d['target_iou_auc'],abs_tol=1e-12):
                errors.append(f'{tag} intermediate step {step} AUC mismatch')
            windows.append({'step':step,'auc':a})
        last=windows[-3:]
        span=max(r['auc'] for r in last)-min(r['auc'] for r in last)
        predroot=root/'predictions_step_002304'
        predictions=[json.loads(line) for line in (predroot/'predictions.jsonl').read_text().splitlines()]
        if len(predictions)!=384 or len({p['episode_id'] for p in predictions})!=384:
            errors.append(f'{tag} prediction manifest size/uniqueness mismatch')
        min_p=1.;max_p=0.
        for pred in predictions:
            p=predroot/pred['npz_path']
            if sha(p)!=pred['sha256'] or pred['sha256']!=row_by_id[pred['episode_id']]['prediction_npz_sha256']:
                errors.append(f'{tag} prediction identity mismatch {pred["episode_id"]}')
            with np.load(p,allow_pickle=False) as archive:
                if archive.files!=['probability']:
                    errors.append(f'{tag} unexpected NPZ keys {p.name}')
                arr=archive['probability']
                if arr.shape!=(128,128) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
                    errors.append(f'{tag} probability content violation {p.name}')
                min_p=min(min_p,float(arr.min()));max_p=max(max_p,float(arr.max()))
                # Raw label valid-pixel masks are deliberately not opened; all-pixel
                # positive count is only a bound on exported valid-positive count.
                if int((arr>.5).sum()) < row_by_id[pred['episode_id']]['predicted_valid_pixels']:
                    errors.append(f'{tag} valid-positive count exceeds total positives')
            npz_total+=1
        wheat={}
        for k in [1,2,4,8]:
            w=[r for r in rows if r['k']==k and r['target_class']==2 and r['target_present']]
            wheat[str(k)]={'iou':classes[str(k)]['t31tfm:2']['value'], 'rows':len(w), 'unique_queries':len({r['query_patch_id'] for r in w}), 'empty_valid_predictions':sum(r['predicted_valid_pixels']==0 for r in w)}
        runs[tag]={'final_auc':value,'target_iou_by_k':curve,'target_classes_by_k':classes,'absent_fp_area_by_k':fp_area,'absent_fp_case_rate_by_k':fp_case,'recomputed_evaluation_windows':windows,'last3_auc_range':span,'last3_range_limit':gate_data['training_adequacy']['maximum_last3_auc_range'],'stability_pass':span<=gate_data['training_adequacy']['maximum_last3_auc_range'],'collector_adequacy_checks':verified['training_adequacy'],'prediction_count':len(predictions),'probability_min':min_p,'probability_max':max_p,'winter_wheat':wheat}
    pairs=[]
    for seed in [270927,270928]:
        b0=runs[f'B0_{seed}'];b2=runs[f'B2_{seed}']
        delta=b2['final_auc']-b0['final_auc']
        pairs.append({'seed':seed,'B0_auc':b0['final_auc'],'B2_auc':b2['final_auc'],'delta':delta,'delta_percentage_points':100*delta,'frozen_practical_margin':gate_data['practical_margin_absolute_auc'],'margin_exceeded':delta>=gate_data['practical_margin_absolute_auc'],'both_stability_pass':b0['stability_pass'] and b2['stability_pass'],'K8_class_delta':{k:b2['target_classes_by_k']['8'][k]['value']-b0['target_classes_by_k']['8'][k]['value'] for k in b0['target_classes_by_k']['8']}})
    return {'schema_version':'oe10_second_pair_independent_review_v1','created_utc':datetime.now(timezone.utc).isoformat(),'export':str(export.resolve()),'review_source_sha256':sha(Path(__file__)),'gate_sha256':sha(gate),'export_manifest_sha256':sha(export/'export_manifest.json'),'export_files':len(seen),'export_bytes':sum(r['bytes'] for r in manifest['files']),'verified_completed_count':4,'full_decision':False,'all_prediction_npz_checked':npz_total,'cohort_hashes':cohort_by_run,'runs':runs,'pairs':pairs,'descriptive_pair_mean_delta':mean(p['delta'] for p in pairs),'errors':errors,'status':'pass' if not errors else 'fail','limits':['Independent reaggregation uses exported per-episode counts; source raw labels were not opened and IoU counts were not independently regenerated.','NPZ integrity, dtype, shape, finiteness, probability bounds and valid-positive upper bound checked; no model forward or checkpoint deserialization.','Single development parent, three target classes, synthetic support, no human correction responses or VLM explanation evaluation.','Two of three seed pairs completed, all four completed runs miss frozen last-three-window stability. No full gate or scientific superiority conclusion.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.export,args.gate)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','export_files','export_bytes','all_prediction_npz_checked','pairs','errors']},indent=2))
    raise SystemExit(0 if not result['errors'] else 1)
