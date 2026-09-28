#!/usr/bin/env python3
"""Independent reanalysis of completed E2 outputs; no model or original file writes."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path
import sys

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def binary_metrics(rows, gold='text_gold'):
    pos = [r for r in rows if r[gold] == 'yes']
    neg = [r for r in rows if r[gold] == 'no']
    recall = sum(r['parsed'] == 'yes' for r in pos) / len(pos) if pos else None
    fpr = sum(r['parsed'] == 'yes' for r in neg) / len(neg) if neg else None
    tn_rate = sum(r['parsed'] == 'no' for r in neg) / len(neg) if neg else None
    return {'n': len(rows), 'n_positive': len(pos), 'n_negative': len(neg),
            'n_predicted_yes': sum(r['parsed'] == 'yes' for r in rows),
            'recall': recall, 'fpr': fpr,
            'balanced_acc': (recall + tn_rate) / 2 if pos and neg else None,
            'd': recall - fpr if pos and neg else None,
            'parse_failures': sum(r['parsed'] not in {'yes', 'no'} for r in rows)}


def counts(rows, gold):
    return np.array([sum(r[gold]=='yes' and r['parsed']=='yes' for r in rows),
                     sum(r[gold]=='yes' for r in rows),
                     sum(r[gold]=='no' and r['parsed']=='yes' for r in rows),
                     sum(r[gold]=='no' for r in rows)], dtype=float)


def clustered_ci(rows, key, gold, n=2000, seed=20260925):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    groups = {g: groups[g] for g in sorted(groups)}
    keys = list(groups)
    c = np.stack([counts(groups[g], gold) for g in keys])
    rng = random.Random(seed)
    idx = np.array([[rng.randrange(len(keys)) for _ in keys] for _ in range(n)])
    sums = c[idx].sum(1)
    valid = (sums[:,1] > 0) & (sums[:,3] > 0)
    vals = np.sort(sums[valid,0] / sums[valid,1] - sums[valid,2] / sums[valid,3])
    return {'clusters': len(keys), 'n_bootstrap_valid': int(valid.sum()),
            'ci95': [float(vals[int(.025*len(vals))]), float(vals[int(.975*len(vals))-1])],
            'cluster_sizes': {str(g): len(groups[g]) for g in keys}}


def delta_ba_cluster_ci(reader, blind, key, n=10000):
    by_r, by_b = collections.defaultdict(list), collections.defaultdict(list)
    for r in reader: by_r[key(r)].append(r)
    for r in blind: by_b[key(r)].append(r)
    keys = sorted(by_r)
    assert set(by_r) == set(by_b)
    rc = np.stack([counts(by_r[k], 'text_gold') for k in keys])
    bc = np.stack([counts(by_b[k], 'text_gold') for k in keys])
    rng = random.Random(20260925)
    idx = np.array([[rng.randrange(len(keys)) for _ in keys] for _ in range(n)])
    r, b = rc[idx].sum(1), bc[idx].sum(1)
    valid = (r[:,1]>0) & (r[:,3]>0) & (b[:,1]>0) & (b[:,3]>0)
    vals = np.sort(.5*(r[valid,0]/r[valid,1]-r[valid,2]/r[valid,3]-b[valid,0]/b[valid,1]+b[valid,2]/b[valid,3]))
    return [float(vals[int(.025*len(vals))]), float(vals[int(.975*len(vals))-1])]


def pair_outcomes(rows, gold='emb_gold'):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['tile']][r[gold]] = r['parsed']
    assert all(set(d)=={'yes','no'} for d in by.values())
    return dict(collections.Counter(f"{d['yes']}|{d['no']}" for d in by.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = repo/'artifacts/e2_multi_reader_v0'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = json.loads((root/'manifest.json').read_text())
    final = json.loads((root/'final.json').read_text())
    hash_results = {'verified': [], 'mismatches': [], 'not_local': []}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root/name
        if not path.exists(): hash_results['not_local'].append(name)
        elif sha(path) != expected: hash_results['mismatches'].append(name)
        else: hash_results['verified'].append(name)
    assert not hash_results['mismatches']
    assert manifest == final['manifest']
    code_hashes = {name: sha(repo/'code'/name) for name in ['e2_multi_reader_v0.py', 'earthtalk_content_controls_v0.py']}
    assert code_hashes['e2_multi_reader_v0.py'] == manifest['code_sha256']
    assert code_hashes['earthtalk_content_controls_v0.py'] == manifest['e0_code_sha256']
    sys.path.insert(0,str(repo/'code'))
    from earthtalk_content_controls_v0 import parse, cross_tile_donor
    metadata = {r['id']:r for r in read_jsonl(repo/'artifacts/streaming_review_20260909/kurosiwo_s1_cache/meta.jsonl')}
    all_rows, saved_scores = {}, {}
    invariants = []
    for arm in ('reader','blind'):
        for seed in (1,2,3):
            key = f'{arm}_seed{seed}'
            d = root/key
            by_arm = {p.stem.removeprefix('answers_'):read_jsonl(p) for p in sorted(d.glob('answers_*.jsonl'))}
            expected_arms = set(manifest['plan']) if arm=='reader' else {'real_all'}
            assert set(by_arm)==expected_arms
            real = {r['id']:r for r in by_arm['real_all']}
            assert len(real)==manifest['plan']['real_all']
            for mode, rows in by_arm.items():
                assert len(rows)==manifest['plan'][mode]
                assert len(rows)==len({r['id'] for r in rows})
                for r in rows:
                    assert parse('Q1',r['answer_raw']) == r['parsed']
                    src = real[r['id']]
                    for field in ('tile','fold','phen','kind','text_gold'):
                        assert r[field]==src[field]
                    if mode=='zero_embedding':
                        assert r['emb_item'] is None and r['emb_gold'] is None
                    else:
                        donor=real[r['emb_item']]
                        assert r['emb_gold']==donor['text_gold']
                        assert r['phen']==donor['phen'] and r['fold']==donor['fold']
                        if mode=='real_all': assert r['emb_item']==r['id']
                        else:
                            assert r['kind'] in {'pos','neg'}
                            assert r['emb_gold']!=r['text_gold']
                            if mode=='swap_within_tile': assert r['tile']==donor['tile']
                            else: assert r['tile']!=donor['tile']
                if mode=='swap_cross_tile':
                    for phen in ('landslide','flood'):
                        pool=[{**r,'answer':r['text_gold']} for r in real.values() if r['phen']==phen and r['kind'] in {'pos','neg'}]
                        for r in (r for r in rows if r['phen']==phen):
                            source={**real[r['id']],'answer':r['text_gold']}
                            expected=cross_tile_donor(source,pool,lambda s,c:c['answer']!=s['answer'])
                            assert r['emb_item']==expected['id']
            for mode, rows in by_arm.items():
                identity=[{k:v for k,v in r.items() if k not in {'parsed','answer_raw'}} for r in rows]
                identity_path=(mode,arm)
                if seed>1:
                    prev=[{k:v for k,v in r.items() if k not in {'parsed','answer_raw'}} for r in all_rows[f'{arm}_seed1'][mode]]
                    assert identity==prev
            all_rows[key]=by_arm
            saved_scores[key]=json.loads((d/'scores.json').read_text())
            assert saved_scores[key]['per_phenomenon']==final['table'][key]
            invariants.append(key)
    results={}
    pass_counts=collections.Counter()
    for seed in (1,2,3):
        reader,blind=all_rows[f'reader_seed{seed}'],all_rows[f'blind_seed{seed}']
        for phen in ('landslide','flood'):
            R={a:[r for r in rows if r['phen']==phen] for a,rows in reader.items()}
            B=[r for r in blind['real_all'] if r['phen']==phen]
            metrics={a:binary_metrics(rows,'emb_gold' if a.startswith('swap_') else 'text_gold') for a,rows in R.items()}
            blind_metrics=binary_metrics(B)
            expected=final['table'][f'reader_seed{seed}'][phen]
            assert abs(metrics['real_all']['balanced_acc']-expected['balanced_acc'])<1e-12
            assert abs(blind_metrics['balanced_acc']-final['table'][f'blind_seed{seed}'][phen]['balanced_acc'])<1e-12
            control_ci={}
            for a,field in [('real_all','d_real'),('swap_within_tile','d_swap'),('swap_cross_tile','d_cross')]:
                rows=[r for r in R[a] if r['kind'] in {'pos','neg'}]
                gold='emb_gold'
                current=binary_metrics(rows,gold)
                ci=clustered_ci(rows,lambda r:r['tile'],gold)
                assert abs(current['d']-expected[field]['value'])<1e-12
                assert np.allclose(ci['ci95'],expected[field]['ci95'],rtol=0,atol=1e-12)
                control_ci[a]=ci
            delta=metrics['real_all']['balanced_acc']-blind_metrics['balanced_acc']
            passed=metrics['swap_within_tile']['d']>=.1 and control_ci['swap_within_tile']['ci95'][0]>0 and delta>=.05
            pass_counts[phen]+=passed
            real_by_id={r['id']:r for r in R['real_all']}
            donor_prediction_consistency=sum(r['parsed']==real_by_id[r['emb_item']]['parsed'] for r in R['swap_within_tile'])/len(R['swap_within_tile'])
            key=lambda r:str(metadata[r['tile']]['actid']) if phen=='flood' else r['fold']
            subgroups={}
            for group in sorted({key(r) for r in R['real_all']}):
                real=[r for r in R['real_all'] if key(r)==group]
                bl=[r for r in B if key(r)==group]
                swap=[r for r in R['swap_within_tile'] if key(r)==group]
                subgroups[group]={'real':binary_metrics(real),'blind':binary_metrics(bl),
                                 'swap':binary_metrics(swap,'emb_gold'),
                                 'kinds':dict(collections.Counter(r['kind'] for r in real)),
                                 'FPR_by_negative_kind':{kind:binary_metrics([r for r in real if r['kind']==kind])['fpr'] for kind in ('neg','hard_neg')}}
            entry={'metrics':metrics,'blind':blind_metrics,'reader_minus_blind_balanced_acc':delta,
                   'registered_gate_pass':passed,'registered_tile_bootstrap':control_ci,
                   'zero_answer_counts':dict(collections.Counter(r['parsed'] for r in R['zero_embedding'])),
                   'pair_outcomes_real':pair_outcomes([r for r in R['real_all'] if r['kind'] in {'pos','neg'}]),
                   'pair_outcomes_swap':pair_outcomes(R['swap_within_tile']),
                   'swap_predictions_matching_donor_real_predictions':donor_prediction_consistency,
                   'by_event_or_region':subgroups}
            if phen=='flood':
                entry['posthoc_event_cluster_d_swap']=clustered_ci(R['swap_within_tile'],key,'emb_gold',10000)
                entry['posthoc_event_cluster_BA_gain_ci95']=delta_ba_cluster_ci(R['real_all'],B,key)
                ds=[g['swap']['d'] for g in subgroups.values() if g['swap']['d'] is not None]
                bas=[g['real']['balanced_acc'] for g in subgroups.values() if g['real']['balanced_acc'] is not None]
                entry['posthoc_equal_event_macro_d_swap']=float(np.mean(ds))
                entry['posthoc_equal_event_macro_balanced_acc']=float(np.mean(bas))
                entry['events_with_d_swap_le_zero']=[k for k,v in subgroups.items() if v['swap']['d'] is not None and v['swap']['d']<=0]
            results[f'{phen}_seed{seed}']=entry
    assert dict(pass_counts)==final['seeds_passing_per_phenomenon']
    real=all_rows['reader_seed1']['real_all']
    test_flood_ids={r['tile'] for r in real if r['phen']=='flood'}
    events={str(metadata[t]['actid']) for t in test_flood_ids}
    previous=json.loads((repo/'artifacts/streaming_review_20260909/summary.json').read_text())
    old_events=set(previous['kuro']['event_delta_gru_minus_teacher_seed_mean'])
    out={'scope':'Post-hoc independent reanalysis only; no new model experiment or modified prereg',
         'source_sha256':{str(p.relative_to(repo)):sha(p) for p in [repo/'config/e2_multi_reader_prereg_v0.json',root/'manifest.json',root/'final.json',repo/'code/e2_multi_reader_v0.py',repo/'code/earthtalk_content_controls_v0.py',repo/'artifacts/streaming_review_20260909/kurosiwo_s1_cache/meta.jsonl']},
         'file_hash_verification':hash_results,'arms_with_complete_nonduplicate_consistent_rows':invariants,
         'saved_point_estimates_and_tile_CIs_reproduced':True,'registered_verdict_reproduced':final['verdict'],
         'passing_seeds':dict(pass_counts),
         'all_raw_answer_counts':dict(collections.Counter(r['answer_raw'] for d in all_rows.values() for rows in d.values() for r in rows)),
         'total_answer_rows':sum(len(rows) for d in all_rows.values() for rows in d.values()),
         'flood_unique_tiles':len(test_flood_ids),'flood_test_events':sorted(events),
         'flood_events_previously_evaluated_in_20260909':sorted(events&old_events),
         'previous_known_nonfinite_test_tile_in_E2': 'ks_05418' in test_flood_ids,
         'training_and_eval_seconds':{k:{t:v[t] for t in ['train_s','eval_s']} for k,v in saved_scores.items()},
         'results':results}
    (output/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for key,s in results.items():
        print(key,'BA',round(s['metrics']['real_all']['balanced_acc'],4),'gain',round(s['reader_minus_blind_balanced_acc'],4),'d_swap',round(s['metrics']['swap_within_tile']['d'],4),'PASS',s['registered_gate_pass'])
        if 'posthoc_event_cluster_d_swap' in s:
            print(' eventCI',s['posthoc_event_cluster_d_swap']['ci95'],'eventmacro',s['posthoc_equal_event_macro_d_swap'],'gainCI',s['posthoc_event_cluster_BA_gain_ci95'],'nonpositive events',s['events_with_d_swap_le_zero'])
        else:
            print(' region metrics',[(g,x['real']['balanced_acc'],x['real']['recall'],x['real']['fpr'],x['swap']['d']) for g,x in s['by_event_or_region'].items()])
    print('verified',len(hash_results['verified']),'notlocal',len(hash_results['not_local']),'rows',out['total_answer_rows'],'old events',sorted(events&old_events))


if __name__=='__main__':main()
