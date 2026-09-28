#!/usr/bin/env python3
"""Read-only six-run collector. Missing evidence stays false or partial."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def require(value, message):
    if not value: raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same(values):
    return bool(values) and bool(values[0]) and all(v == values[0] for v in values)


def run_metrics(folder, arm, seed, protocol, gate, sources, protocol_sha, resume_ok):
    r = read(folder/'receipt.json'); logs = rows(folder/'train_log.jsonl'); n = protocol['updates_per_run']
    require((r['status'], r['arm'], r['seed'], r['mode']) == ('training_completed', arm, seed, 'train'), 'run unfinished/identity mismatch')
    require(r['protocol_sha256'] == protocol_sha and r['source_hashes'] == sources, 'run protocol/source mismatch')
    require(r['completed_updates'] == n and [v['step'] for v in logs] == list(range(1, n+1)), 'incomplete/duplicate training updates')
    order_sha = hashlib.sha256(('\n'.join(v['episode_id'] for v in logs)+'\n').encode()).hexdigest()
    require(order_sha == r['episode_order_sha256'], 'actual episode order differs from receipt')
    require(sha(folder/'final.pt') == r['checkpoint_sha256'], 'final checkpoint file hash mismatch')
    windows = []; scores = []; interval = protocol['evaluation_interval_updates']
    require([w['step'] for w in r['evaluation_windows']] == list(range(interval, n+1, interval)), 'missing/repeated evaluation windows')
    for w in r['evaluation_windows']:
        step = w['step']; path = folder/f'score_step_{step:06d}.json'; score = read(path)
        require(path.name == Path(w['score_path']).name and sha(path) == w['score_sha256'], 'score receipt hash mismatch')
        require(score['status'] == 'independent_probability_scoring_completed', 'score incomplete')
        require(score['scorer_source_sha256'] == sources['score_predictions.py'] and score['episode_loader_source_sha256'] == sources['episode_loader.py'], 'scorer/loader identity mismatch')
        require(score['gate_config_sha256'] == gate['_sha256'], 'score gate config mismatch')
        for key, expected in gate['p2']['cohort'].items():
            if key in ('base_count','query_patch_count','target_present_base_count','target_absent_base_count','target_classes','development_parent_ids','cohort_base_ids_sha256','source_scoring_sha256'):
                require(score['cohort'][key] == expected, 'score cohort mismatch: '+key)
        predroot = folder/f'predictions_step_{step:06d}'; manifest = predroot/'predictions.jsonl'
        require(sha(manifest) == score['predictions_manifest_sha256'], 'prediction manifest hash mismatch')
        predictions = rows(manifest); scored = {v['episode_id']: v for v in score['per_episode_scoring_only']}
        require(len(predictions) == 384 and len(scored) == 384 and len({v['episode_id'] for v in predictions}) == 384, 'prediction count/duplicates')
        require({v['episode_id'] for v in predictions} == set(scored), 'prediction/scoring episode set mismatch')
        for v in predictions:
            path = (predroot/v['npz_path']).resolve()
            require(path.is_relative_to(predroot.resolve()) and sha(path) == v['sha256'] == scored[v['episode_id']]['prediction_npz_sha256'], 'prediction file lineage mismatch')
        for name in ('target_iou_by_k','absent_fp_area_by_k','absent_fp_case_rate_by_k'):
            require(set(score[name]) == {'1','2','4','8'} and all(finite(v) and 0 <= v <= 1 for v in score[name].values()), 'invalid metric curve')
        c = score['target_iou_by_k']; auc = sum((b-a)*(c[str(a)]+c[str(b)])/2 for a,b in ((1,2),(2,4),(4,8)))/7
        require(abs(auc-score['target_iou_auc']) < 1e-12 and abs(auc-w['auc']) < 1e-12, 'AUC/window mismatch')
        windows.append({'step':step,'auc':auc}); scores.append(score)
    require(bool(scores) and windows[-1]['step'] == n, 'final update has not been scored')
    last = scores[-1]; p = gate['p2']; m = r['model_costs']; costs = r['costs']
    finite_logs = all(all(finite(v[k]) for k in ('native_loss','mask_loss','language_loss','total_loss','grad_norm','seconds')) for v in logs)
    nonzero = all(v['nonzero_gradients']['head'] > 0 and v['nonzero_gradients']['connector'] > 0 for v in logs)
    arm_grads = all((v['nonzero_gradients']['encoder'] == 0 if arm == 'B0' else v['nonzero_gradients']['encoder'] > 0 and v['native_encoder_gradient_tensors'] > 0) for v in logs)
    deltas = list(r['encoder_anchor_max_abs_deltas'].values())
    arm_changes = bool(deltas) and all(finite(x) for x in deltas) and (all(x == 0 for x in deltas) if arm == 'B0' else any(x > 0 for x in deltas))
    lrs = protocol.get('base_learning_rates', {}); lr_ok = protocol.get('optimizer_lr_recipe_count') == 1 and bool(lrs)
    warmup = protocol['warmup_updates']
    for v in logs:
        step = v['step']; factor = min(1.,step/warmup) if step <= warmup else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,n-warmup))))
        expected_roles = {'readouts'} if arm == 'B0' else {'encoder','native_decoder','readouts'}
        lr_ok = lr_ok and set(v['learning_rates']) == expected_roles and all(role in lrs and finite(rate) and abs(rate-lrs[role]*factor) < 1e-12 for role,rate in v['learning_rates'].items())
    present = [v for v in last['per_episode_scoring_only'] if v['target_present']]
    nondegenerate = any(0 < v['predicted_valid_pixels'] < v['valid_pixels'] for v in present) and last['target_iou_auc'] > 0
    minimum = protocol.get('minimum_updates_per_run', n)
    adequate = {'training_schedule_completed': True, 'optimizer_and_lr_budget_completed': bool(lr_ok),
        'save_resume_verified': resume_ok, 'finite_loss_and_gradients': finite_logs and nonzero,
        'nondegenerate_supervised_readout': nondegenerate, 'encoder_freeze_or_update_matches_arm': arm_grads and arm_changes,
        'completed_updates': n, 'predeclared_minimum_updates': minimum, 'predeclared_eval_interval_updates': interval,
        'last_evaluation_windows': windows,
        'budget_cap_reached_while_improving': len(windows) < 3 or windows[-1]['auc']-min(w['auc'] for w in windows[-3:]) > p['training_adequacy']['maximum_last3_auc_range']}
    for value in (r['elapsed_seconds'], costs['evaluation_seconds'], costs['training_observation_exposures']): require(finite(value) and value >= 0, 'invalid cost')
    result = {'arm':arm,'seed':seed,'checkpoint_sha256':r['checkpoint_sha256'],'run_receipt_sha256':sha(folder/'receipt.json'),
        'cohort_base_ids_sha256':last['cohort']['cohort_base_ids_sha256'],'training_adequacy':adequate,
        **{k:last[k] for k in ('target_iou_by_k','absent_fp_area_by_k','absent_fp_case_rate_by_k')},
        'costs':{'total_gpu_seconds_including_feature_precompute':r['elapsed_seconds'],
                 'training_observation_exposures':costs['training_observation_exposures']+costs.get('native_raw_patch_date_instances',0),
                 'inference_gpu_seconds':costs['evaluation_seconds'],
                 'inference_distinct_query_observations':last['cohort']['query_patch_count']*2,
                 'inference_query_observation_instances':len(windows)*384*2,
                 'inference_support_observations':len(windows)*96*sum(16*k for k in (1,2,4,8))},
        'model_costs_cumulative':m,
        'cost_note':'worker elapsed is whole-process GPU reservation time including load/hash/cache/training/evaluation; inference time is a subset and also includes CPU I/O/scoring, not additive GPU kernel time',
        'checkpoint_inspection':'file hash verified; checkpoint not deserialized by collector'}
    return result, r, logs, last


def collect(protocol_path, runroot, engineering, source_dir, gate_path, evidence_path=None):
    errors = []; verified = []
    report = {'schema_version':'oe10_partial_run_collection_v1','status':'partial_or_unverified','verified_runs':verified,'blockers':errors,'p2_decision_made':False}
    try:
        protocol = read(protocol_path); gate = read(gate_path); gate['_sha256'] = sha(gate_path)
        seeds = gate['p2']['paired_seed_ids']; require(protocol['paired_seed_ids'] == seeds, 'protocol seed list differs from locked gates')
        require(protocol['locked_before_development_results'] is True, 'protocol not locked')
        controller = read(runroot/'status.json'); sources = controller['source_hashes']; ph = sha(protocol_path)
        require(all(sha(source_dir/k) == v for k,v in sources.items()), 'current source snapshot differs')
        require(controller.get('protocol_sha256') == ph, 'controller protocol mismatch')
        require(controller.get('stage') == 'training', 'collector requires training controller')
        eng = read(engineering/'status.json')
        require(eng['status'] == 'completed' and eng['protected_and_snapshot_unchanged'] is True and eng['source_hashes'] == sources, 'engineering source/integrity mismatch')
        resume = {}
        for arm in ('B0','B2'):
            ref = read(engineering/f'{arm}_270927_reference'/'receipt.json'); rs = read(engineering/f'{arm}_270927_resume'/'receipt.json')
            resume[arm] = rs['status'] == 'cold_resume_passed' and rs['pass_result'] is True and rs['fresh_process_resume'] is True and rs['reference_pid'] != rs['current_pid'] and ref['status'] == 'reference_completed' and rs['source_hashes'] == ref['source_hashes'] == sources
        details = []
        for seed in seeds:
            for arm in ('B0','B2'):
                name = f'{arm}_{seed}_train'
                try:
                    result, receipt, logs, score = run_metrics(runroot/name, arm, seed, protocol, gate, sources, ph, resume[arm])
                    worker = [w for w in controller['workers'] if w['name'] == name]
                    require(len(worker) == 1 and worker[0]['status'] == 'completed' and worker[0]['exit_code'] == 0, 'controller has not confirmed worker completion')
                    verified.append(result); details.append((receipt, logs, score))
                except (OSError, ValueError, KeyError, TypeError) as e: errors.append(name+': '+str(e))
        require(len(verified) == 6 and controller['status'] == 'completed' and controller['protected_and_snapshot_unchanged'] is True, 'six completed workers/controller integrity not yet verified')
        evidence = read(evidence_path) if evidence_path else {}; checks = {}
        for check in gate['p2']['fairness_checks_required']:
            record = evidence.get(check, {}); ok = False
            if record.get('path') and record.get('sha256'):
                path = Path(record['path'])
                if path.is_file() and sha(path) == record['sha256']:
                    proof = read(path); pinned = proof.get('source_hashes', {})
                    ok = proof.get('status') == 'passed' and check in proof.get('checks_passed', []) and bool(pinned) and all(sources.get(k) == v for k,v in pinned.items())
            checks[check] = ok
        receipts = [x[0] for x in details]; logs = [v for x in details for v in x[1]]
        paired_orders = all(receipts[i]['episode_order_sha256'] == receipts[i+1]['episode_order_sha256'] for i in (0,2,4))
        checks.update(same_common_cohort=True, same_support_draws_and_prefixes=paired_orders,
            same_initial_encoder_checkpoint=same([r['native_identity']['checkpoint_hashes'] for r in receipts]),
            same_reader_checkpoint_and_trainability=same([r['vlm_identity'] for r in receipts]) and checks['same_reader_checkpoint_and_trainability'],
            same_connector_and_dense_head_capacity=checks['same_connector_and_dense_head_capacity'],
            same_allowed_images_and_observation_access=checks['same_allowed_images_and_observation_access'] and all(v['model_metrics']['query_dates'] == 2 and v['model_metrics']['support_date_instances'] == 16*v['k'] for v in logs),
            same_label_and_question_supervision=paired_orders and all(v['model_metrics']['language_supervision'] is True for v in logs),
            same_predeclared_hyperparameter_search_budget=all(r['training_adequacy']['optimizer_and_lr_budget_completed'] for r in verified),
            all_feature_precompute_and_training_costs_recorded=True)
        false_checks = [k for k,v in checks.items() if not v]
        return {'schema_version':'oe9_p2_scorer_summary_v1','gate_config_sha256':gate['_sha256'],
            'protocol_locked_before_results':True,'seed_ids_locked_before_results':True,'paired_seed_ids':seeds,
            'scorer_source_sha256':sources['score_predictions.py'],
            'predictions_manifest_sha256':hashlib.sha256(('\n'.join(x[2]['predictions_manifest_sha256'] for x in details)+'\n').encode()).hexdigest(),
            'predictions_manifest_digest_kind':'ordered composite of six final prediction manifest hashes',
            'aggregation':gate['p2']['aggregation'],'absent_case_area_threshold':.001,'cohort':details[0][2]['cohort'],
            'comparison_axis':protocol.get('comparison_axis','undeclared'),'fairness_checks':checks,'runs':verified,
            'collector_blockers':['unverified_fairness:'+k for k in false_checks], 'p2_decision_made':False}
    except (OSError, ValueError, KeyError, TypeError) as e:
        errors.append(str(e)); return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('protocol','run-root','engineering-root','source-dir','gate-config','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--evidence-json',type=Path); a = ap.parse_args()
    result = collect(a.protocol,a.run_root,a.engineering_root,a.source_dir,a.gate_config,a.evidence_json)
    with a.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'schema_version':result['schema_version'],'status':result.get('status'),'completed_verified_runs':len(result.get('runs',result.get('verified_runs',[]))),'blockers':result.get('blockers',result.get('collector_blockers',[]))}))
