#!/usr/bin/env python3
"""Independent saved-receipt/metric audit; no production imports or model execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ('full', 'pool4_nearest', 'pool8_nearest', 'coordinates', 'train_mean')
FOLD_SALT = 'oe2-signal-probe-fivefold-v0|20260926|'
ROOT = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
INPUT = Path('/private/tmp/oe2_problem_audit_20260926/remote_results/cpu20_v0')
OUT = Path('/private/tmp/oe2_problem_audit_20260926/independent_probe_audit.json')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def close(x, y):
    return math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-11)


def run(source, dest, repo):
    hashes = {}
    def read(p):
        hashes[str(p)] = sha(p)
        return json.loads(p.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))
    launch = read(source/'launch_manifest.json')
    status = read(source/'launch_status.json')
    after = read(source/'protected_code_after.json')
    child = read(source/'probe/status.json')
    summary = read(source/'probe/summary.json')
    folds = read(source/'probe/folds.json')
    raw = read(source/'probe/raw_receipts.json')
    tokens = read(source/'probe/token_receipts.json')
    probes = read(source/'probe/probe_receipts.json')
    rows = read(source/'probe/scene_metrics.json')
    candidates = read(source/'code_snapshot/candidates20_manifest.json')
    plan = read(source/'code_snapshot/probe_plan.json')
    expected_names = {'signal_probe.py','launch_signal_probe.py','eo_model.py','candidates20_manifest.json','probe_plan.json'}
    require(set(launch['source_snapshot_sha256']) == expected_names, 'Unexpected snapshot file set')
    source_matches = {}
    for name, digest in launch['source_snapshot_sha256'].items():
        snap = source/'code_snapshot'/name
        original = repo/'code/oe2_problem_audit_v0/probe'/name
        hashes[str(snap)] = sha(snap); hashes[str(original)] = sha(original)
        require(hashes[str(snap)] == digest == hashes[str(original)], 'Snapshot/original/launch hash differs: '+name)
        source_matches[name] = {'sha256': digest, 'snapshot': str(snap), 'local_original': str(original)}
    server_snapshot = Path(summary['arguments']['manifest']).parent
    for remote, digest in summary['source_sha256'].items():
        rel = Path(remote).relative_to(server_snapshot)
        require(len(rel.parts) == 1 and rel.name in expected_names and digest == launch['source_snapshot_sha256'][rel.name], 'Summary source pin mismatch')
    require(status['status'] == child['status'] == 'completed' and status['valid'] is True and status['returncode'] == 0, 'Run not completed')
    require(summary['status'] == 'completed_development_diagnostic', 'Wrong analysis status')
    require(launch['protected_code_before'] == after and len(after) == 4, 'Protected source before/after differs')
    require(launch['device'] == summary['environment']['device'] == summary['arguments']['device'] == plan['device'] == 'cpu', 'Device mismatch')
    require(launch['thread_limit'] == plan['thread_limit'] == 4, 'Thread contract')
    require(summary['encoder_unchanged'] is True and summary['encoder_gradients_absent'] is True, 'Encoder not frozen')
    require(plan['candidate_manifest_sha256'] == launch['source_snapshot_sha256']['candidates20_manifest.json'], 'Candidate pin mismatch')
    require(plan['encoder_weights_sha256'] == summary['model_provenance']['checkpoint_sha256']['weights.pth'], 'Checkpoint receipt mismatch')
    require(0 < summary['elapsed_seconds'] <= child['elapsed_seconds'] < plan['hard_timeout_seconds'] == 540, 'Runtime bounds')
    require(summary['ridge']['fold_count'] == 5 and summary['ridge']['train_scenes_per_fold'] == 16 and summary['ridge']['held_out_scenes_per_fold'] == 4 and summary['ridge']['alpha'] == 1.0, 'Probe setting mismatch')
    require(summary['ridge']['fold_salt'] == FOLD_SALT and summary['arguments']['save_features'] is False, 'Fold/features contract')
    source_file_checks = {}
    for key, value in candidates['source_files'].items():
        p = Path(value['path']); hashes[str(p)] = sha(p)
        require(hashes[str(p)] == value['sha256'] and p.stat().st_size == value['bytes'], 'Candidate original source changed: '+key)
        source_file_checks[key] = {'path': str(p), 'sha256': hashes[str(p)]}
    prepared = read(Path(candidates['source_files']['prepared_manifest']['path']))
    questions_path = Path(candidates['source_files']['train_questions']['path'])
    questions = [json.loads(l) for l in questions_path.read_text().splitlines()]
    run_cfg = read(Path(candidates['source_files']['oe1_run_config']['path']))
    meta_byid = {x['patch_id']: x for x in prepared['patches'] if x['split'] == 'train'}
    trainq = {x['id']: x for x in questions}
    cases = candidates['candidates']; ids = [x['case_id'] for x in cases]
    require(len(ids) == len(set(ids)) == 20, 'Scene count/duplicates')
    cm = {x['case_id']: x for x in cases}
    groups = {i: cm[i]['original_patch_metadata']['mgrs'] for i in ids}
    require(len(set(groups.values())) == 20, 'MGRS groups reused')
    ranked = sorted(meta_byid, key=lambda i: (hashlib.sha256((candidates['sampling_rule']['salt']+i).encode()).hexdigest(), i))
    selected, selected_groups = [], set()
    for i in ranked:
        if meta_byid[i]['mgrs'] not in selected_groups:
            selected.append(i); selected_groups.add(meta_byid[i]['mgrs'])
            if len(selected) == 20:
                break
    require(ids == selected, '20-case deterministic selection differs')
    for c in cases:
        i = c['case_id']; m = c['original_patch_metadata']
        require(m == meta_byid[i] and all(m[k] == 'train' for k in ('split','official_split','geobench_split')), 'Source train metadata differs')
        require(len(c['original_questions']) == 2 and all(q == trainq[q['id']] for q in c['original_questions']), 'Original question linkage differs')
        expected = hashlib.sha256((candidates['sampling_rule']['salt']+i).encode()).hexdigest()
        require(expected == c['selection_sha256'], 'Selection rank differs')
        require(c['raw_image']['expected_npy_file_sha256'] == run_cfg['data_files_sha256'][c['raw_image']['server_path_from_historical_receipt']], 'Historical raw NPY hash differs')
    rawmap = {r['case_id']: r for r in raw}; tokmap = {r['case_id']: r for r in tokens}
    require(len(raw) == len(tokens) == 20 and set(rawmap) == set(tokmap) == set(ids), 'Raw/token population differs')
    allvalidhash = hashlib.sha256(bytes([1])*900).hexdigest()
    for i in ids:
        rr, tt, cc = rawmap[i], tokmap[i], cm[i]
        require(rr['file_sha256'] == cc['raw_image']['expected_npy_file_sha256'] and rr['array_sha256'] == cc['raw_image']['expected_array_sha256'] == cc['original_patch_metadata']['image_array_sha256'], 'Raw receipt linkage differs')
        require(rr['path'] == rr['historical_server_path'] == cc['raw_image']['server_path_from_historical_receipt'], 'Raw location differs')
        require(rr['shape'] == [12,120,120] and tt['native_shape'] == [1,30,30,1,3,192] and tt['native_dtype'] == 'float32', 'Native shape/dtype differs')
        require(tt['pixel_count'] == tt['valid_pixels'] == 14400 and tt['native_cells'] == tt['valid_native_cells'] == 900 and tt['patch_shape'] == [4,4], 'Target validity/shape')
        require(tt['valid_mask_sha256'] == allvalidhash, 'All-valid mask byte hash differs')
        for k in ('native_sha256','target_sha256','valid_mask_sha256'):
            require(len(tt[k]) == 64 and all(c in '0123456789abcdef' for c in tt[k]), 'Malformed tensor digest')
    fold_order = sorted(ids, key=lambda i: (hashlib.sha256((FOLD_SALT+i).encode()).hexdigest(), i))
    assignment = {i: j % 5 for j, i in enumerate(fold_order)}
    require(len(folds) == 5 and {f['fold'] for f in folds} == set(range(5)), 'Fold count')
    for f in folds:
        tr = [i for i in ids if assignment[i] != f['fold']]
        ho = [i for i in ids if assignment[i] == f['fold']]
        require(f['train'] == tr and f['held_out'] == ho and len(tr) == 16 and len(ho) == 4, 'Fold membership/order mismatch')
        require(not {groups[i] for i in tr} & {groups[i] for i in ho}, 'Fold MGRS leakage')
    require(Counter(i for f in folds for i in f['held_out']) == Counter(ids), 'Held-out coverage')
    require(len(rows) == 100, 'Scene metric count')
    by = {(r['case_id'],r['method']):r for r in rows}
    require(len(by) == 100 and set(by) == {(i,m) for i in ids for m in METHODS}, 'Duplicate/missing scene-method row')
    for (i,m), r in by.items():
        require(r['fold'] == assignment[i] and r['mgrs'] == groups[i] and r['valid_cells'] == 900, 'Metric identity/support')
        require(all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) for k in ('mse','mae','signed_bias','r2_within_scene')), 'Nonfinite scene metric')
        require(r['mse'] >= 0 and r['mae'] >= 0 and r['r2_within_scene'] <= 1+1e-10 and abs(r['signed_bias']) <= r['mae']+1e-10 and r['mae']**2 <= r['mse']+1e-10, 'Metric inequality failed')
    pm = {(p['fold'],p['method']):p for p in probes}
    require(len(probes) == len(pm) == 20 and set(pm) == {(f,m) for f in range(5) for m in METHODS[:-1]}, 'Readout receipt coverage')
    for (f,m), p in pm.items():
        require(p['feature_dim'] == (2 if m == 'coordinates' else 192) and math.isfinite(p['intercept']), 'Readout dimension/intercept')
        require(close(p['intercept'],pm[f,'full']['intercept']), 'Intercept method mismatch')
    # Independent moment identities: constant baseline provides scene variance,
    # intercept/bias provide scene mean; all five methods must share those targets.
    target_moments = {}
    for i in ids:
        r = by[i,'train_mean']; foldmean = pm[assignment[i],'full']['intercept']
        mean = foldmean-r['signed_bias']; variance = r['mse']-r['signed_bias']**2
        require(variance > 0 and -1-1e-10 <= mean <= 1+1e-10 and variance <= 1-mean**2+1e-10, 'Invalid ratio target moments')
        for m in METHODS:
            rr = by[i,m]
            require(close(1-rr['mse']/variance,rr['r2_within_scene']), 'Scene shared-variance R2 identity differs')
        target_moments[i] = {'mean_inferred_from_constant_baseline':mean, 'variance_inferred_from_constant_baseline':variance, 'std_inferred':math.sqrt(variance)}
    for f in folds:
        require(close(statistics.mean(target_moments[i]['mean_inferred_from_constant_baseline'] for i in f['train']),pm[f['fold'],'full']['intercept']), 'Training-only mean/intercept identity differs')
    recomputed = {}
    for m in METHODS:
        rr = [by[i,m] for i in ids]
        ss = {'scene_count':20,'mean_scene_mse':statistics.mean(x['mse'] for x in rr),
              'mean_scene_mae':statistics.mean(x['mae'] for x in rr),'median_scene_mae':statistics.median(x['mae'] for x in rr),
              'mean_scene_bias':statistics.mean(x['signed_bias'] for x in rr),'mean_within_scene_r2':statistics.mean(x['r2_within_scene'] for x in rr)}
        require(all(close(v,summary['summary'][m][k]) for k,v in ss.items()), 'Saved aggregate mismatch: '+m)
        recomputed[m] = ss | {'median_within_scene_r2':statistics.median(x['r2_within_scene'] for x in rr),
                            'positive_within_scene_r2_scenes':sum(x['r2_within_scene']>0 for x in rr)}
    differences = {}
    for m in METHODS[1:]:
        delta = [by[i,m]['mae']-by[i,'full']['mae'] for i in ids]
        differences[m+'_minus_full'] = {'same_scene_count':20, 'mean_mae_difference':statistics.mean(delta),
            'median_mae_difference':statistics.median(delta),
            'relative_mean_mae_increase_over_full':statistics.mean(delta)/recomputed['full']['mean_scene_mae'],
            'full_relative_mean_mae_reduction_over_comparator':statistics.mean(delta)/recomputed[m]['mean_scene_mae'],
            'full_lower_mae_scenes':sum(x>0 for x in delta),'comparator_lower_mae_scenes':sum(x<0 for x in delta),'ties':sum(x==0 for x in delta),
            'scene_differences':[{'case_id':i,'mae_difference':d} for i,d in zip(ids,delta)]}
    worst = min((by[i,'full'] for i in ids),key=lambda r:r['r2_within_scene'])
    r2_note = {'worst_full_scene':worst,'target_moments':target_moments[worst['case_id']],
        'fraction_of_sum_full_mse_from_worst_r2_scene':worst['mse']/sum(by[i,'full']['mse'] for i in ids),
        'fraction_of_sum_full_mae_from_worst_r2_scene':worst['mae']/sum(by[i,'full']['mae'] for i in ids),
        'interpretation':'R2 uses each held-out scene own target mean as oracle reference. Low within-scene target variance plus cross-scene prediction bias can give very negative R2. The average is outlier-sensitive; primary scene MAE, all per-scene values and median R2 must remain visible. Negative R2 is not an invalid-metric flag or evidence that no scene has signal.'}
    report = {'schema':'oe2-independent-cpu20-receipt-aggregate-audit-v0','checked_at':datetime.now(timezone.utc).isoformat(),
        'valid_within_scope':True,'consistent':True,'artifact':str(source),'script_sha256':sha(Path(__file__)),
        'scope':'Independent validation of saved source/array receipts, original local metadata, fold membership, metric identities and scene aggregation. No original sensor arrays, native features, fitted coefficients or per-cell predictions are present in this replica; no encoder rerun, readout refit or independent pixelwise metric reproduction.',
        'local_array_presence':{'raw_npy':sum(Path(c['raw_image']['local_path']).exists() for c in cases),'native_feature_archive':(source/'probe/native_tokens_and_targets.npz').exists()},
        'source_snapshot_matches':source_matches,'candidate_original_source_matches':source_file_checks,
        'checks':{'completed_cpu_execution_receipt':True,'snapshot_matches_local_originals':True,'protected_before_after_exact':True,
                  'candidate_hash_selection_reconstructed':True,'20_raw_receipts_match_historical_pins':True,
                  '20_native_shape_receipts_1_30_30_1_3_192':True,'all_20_valid_mask_hashes_are_900_true_bytes':True,
                  'five_16_train_4_heldout_folds_mixed_disjoint_mgrs':True,'all_scenes_once_heldout':True,
                  '100_finite_scene_metrics':True,'20_fixed_readout_receipts':True,
                  'source_target_shared_mean_variance_identities':True,'all_saved_aggregates_recomputed':True},
        'token_contract':{'raw_sensor_shape':[12,120,120],'native_receipt_shape':[1,30,30,1,3,192],
                          'bandset_mean_full_spatial_tokens':900,'pool4_spatial_tokens':16,'pool8_spatial_tokens':64,
                          'pointwise_feature_dimension_all_three':192,'target_cells_per_scene':900,
                          'target':'Mean raw DN B08/B04 ratio over each 4x4 pixel native cell; sensor signal only'},
        'recomputed':recomputed,'paired_same_scene_differences':differences,'r2_diagnostic':r2_note,
        'target_moments_algebraically_inferred':target_moments,
        'protected_before_after':after,'input_sha256':hashes,
        'limitations':summary['limitations']+[
            'Full means bandset-averaged native spatial grid, not every native bandset token retained separately.',
            'Readout ridge coefficients ARE fitted on each fold training scenes; only the EO encoder is frozen. No VLM is evaluated.',
            'These descriptive differences compare restricted nearest-restored pointwise readouts and do not establish irrecoverable information loss.',
            'Array hashes and frozen-encoder claims are execution receipts cross-linked to local pins, not a new local raw-array/tensor verification.',
            'Fold training sets overlap; twenty scene differences are not twenty independent training runs, and different MGRS does not ensure geographic independence.',
        ]}
    for name,digest in hashes.items():
        require(sha(Path(name)) == digest,'Input changed during audit')
    require(not dest.exists(),'Refusing to overwrite audit output')
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(dest),'consistent':True,'raw_receipt_count':len(raw),'native_receipt_count':len(tokens),
                      'scene_metric_count':len(rows),'mean_mae':{m:recomputed[m]['mean_scene_mae'] for m in METHODS},
                      'full_lower_mae_scenes':{m:v['full_lower_mae_scenes'] for m,v in differences.items()}},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=INPUT);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--repo',type=Path,default=ROOT)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.repo.resolve())
