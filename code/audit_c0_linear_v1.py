#!/usr/bin/env python3
"""Read-only independent C0 artifact audit; NumPy only, no C0 code imports.

Usage: python audit_c0_linear_v1.py ARTIFACT_DIR --out /tmp/c0_audit.json
Keep global_features.npz with the artifacts: it is required to independently
recompute logits, training normalization, objective and convergence gradients.
Optional --reference-root ROOT also verifies all three historical E2 test files.
Exit 0 = consistent valid run; 1 = consistent but invalid run; 2 = audit failure.
The only file written is the explicitly requested --out path (outside artifacts).
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np

ARMS = ('earlier', 'later', 'pair')
PHEN = ('flood', 'landslide')
COUNTS = {'train|landslide|yes': 518, 'train|landslide|no': 518,
          'train|flood|yes': 1066, 'train|flood|no': 2132,
          'test|landslide|yes': 192, 'test|landslide|no': 192,
          'test|flood|yes': 457, 'test|flood|no': 914}
E2_HASHES = {
    'e2_multi_reader_v0/manifest.json': 'e46e4bcd7ecf9a3fb3e325358c2881ac3bbf79c8f942039351914032c60448f5',
    'e2_multi_reader_v0/reader_seed1/answers_real_all.jsonl': 'b36041a82de5f8cee44c3f5971cbfd8fb40c980ddc6d62cfb08e2c9bedd23d12',
    'e2_multi_reader_v0/reader_seed2/answers_real_all.jsonl': '0aa4b84dda0fd71aa5da49524da9203f2c7b24e996bf6453bde727eecbaed3b9',
    'e2_multi_reader_v0/reader_seed3/answers_real_all.jsonl': 'c063117b3999ee346c7b176379dff870ecb7606fcf7986ef528f69a0b664079a'}


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reject_constant(value):
    raise AuditFailure('Nonfinite JSON constant: ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(value):
    return json.loads(value, parse_constant=reject_constant, object_pairs_hook=unique_object)


def read_json(path):
    return decode(Path(path).read_text())


def read_rows(path):
    return [decode(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest_ids(ids):
    return hashlib.sha256(json.dumps(sorted(ids), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def close(actual, expected, name, atol=1e-10, rtol=1e-10):
    """Compare recursively, including support counts, missing keys and null CIs."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': key mismatch')
        for key in expected:
            close(actual[key], expected[key], name + '.' + key, atol, rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ': list mismatch')
        for index, value in enumerate(expected):
            close(actual[index], value, name + '[' + str(index) + ']', atol, rtol)
    elif expected is None or isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected, name + ': value mismatch')
    elif isinstance(expected, (int, np.integer)):
        require(type(actual) is int and actual == expected, name + ': integer mismatch')
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(float(expected))
                and math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol),
                name + ': numeric mismatch: ' + str(actual) + ' vs ' + str(expected))


def metric_table(items, answers):
    """Compute from independent confusion counts, without using reported metrics."""
    counts = collections.defaultdict(collections.Counter)
    dry = collections.defaultdict(collections.Counter)
    support = collections.Counter()
    for item, answer in zip(items, answers):
        cluster, kind, gold = str(item['cluster']), item['kind'], item['answer']
        support[f'{cluster}|{kind}|{gold}'] += 1
        if kind == 'hard_neg':
            dry[cluster]['n'] += 1
            dry[cluster]['fp'] += answer == 'yes'
        else:
            counts[cluster]['pos' if gold == 'yes' else 'neg'] += 1
            counts[cluster]['tp' if gold == 'yes' else 'fp'] += answer == 'yes'
    events = {}
    for cluster, c in sorted(counts.items()):
        require(c['pos'] > 0 and c['neg'] > 0, 'Missing paired class: ' + cluster)
        recall, fpr = c['tp'] / c['pos'], c['fp'] / c['neg']
        events[cluster] = {'ba': (recall + 1.0 - fpr) / 2.0,
                           'n': c['pos'] + c['neg'], 'pos': c['pos'], 'neg': c['neg'],
                           'recall': recall, 'fpr': fpr}
    hard_events = {cluster: {'n': c['n'], 'fpr': c['fp'] / c['n']} for cluster, c in sorted(dry.items())}
    n_hard = sum(c['n'] for c in dry.values())
    return {'event_macro_ba': math.fsum(v['ba'] for v in events.values()) / len(events) if events else None,
            'events': events,
            'hard_negative': {'events': hard_events, 'n': n_hard,
                              'pooled_fpr': sum(c['fp'] for c in dry.values()) / n_hard if n_hard else None,
                              'event_macro_fpr': math.fsum(v['fpr'] for v in hard_events.values()) / len(hard_events) if hard_events else None},
            'tuple_counts': dict(sorted(support.items()))}


def arm_contrast(later, pair, phenomenon):
    require(set(later['events']) == set(pair['events']) and pair['events'], 'Unpaired comparison events')
    deltas = {c: later['events'][c]['ba'] - pair['events'][c]['ba'] for c in sorted(pair['events'])}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    ci = None
    if phenomenon == 'flood':
        # Same NumPy generator specification, independently computed from event differences.
        draws = np.random.default_rng(20260925).integers(0, len(values), (5000, len(values)))
        bootstrap = np.sum(values[draws], axis=1) / len(values)
        ci = np.quantile(bootstrap, [0.025, 0.975], method='linear').tolist()
    return {'delta': math.fsum(deltas.values()) / len(deltas), 'ci95_delta': ci,
            'event_deltas': deltas, 'unit': 'event' if phenomenon == 'flood' else 'two_regions_descriptive_only'}


def logistic_recompute(train, labels, test, model):
    """Analytic NumPy objective/gradient; never optimizes or invokes torch."""
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    mu, sigma = train.mean(axis=0), train.std(axis=0, ddof=0)
    sigma[sigma < 1e-6] = 1.0
    require(np.allclose(model['mean'], mu, atol=1e-12, rtol=1e-12), 'Saved mean is not train-only mean')
    require(np.allclose(model['scale'], sigma, atol=1e-12, rtol=1e-12), 'Saved scale is not train-only std')
    x = (train - mu) / sigma
    w, b = model['weights'], float(model['bias'])
    train_logits = x @ w + b
    test_logits = ((test - mu) / sigma) @ w + b
    labels = np.asarray(labels, dtype=np.float64)
    positives, n = int(labels.sum()), len(labels)
    require(0 < positives < n, 'Training lacks a binary class')
    sample_weights = np.where(labels == 1, n / (2 * positives), n / (2 * (n - positives)))
    # logaddexp avoids overflow; sigmoid uses a stable log-probability expression.
    loss = float(np.mean(sample_weights * (np.logaddexp(0, train_logits) - labels * train_logits)) + 0.005 * (w @ w))
    residual = sample_weights * (np.exp(-np.logaddexp(0, -train_logits)) - labels) / n
    grad_w, grad_b = x.T @ residual + 0.01 * w, float(residual.sum())
    gradient = max(float(np.max(np.abs(grad_w))), abs(grad_b))
    initial = float(np.mean(sample_weights) * math.log(2.0))
    require(np.isfinite(test_logits).all() and math.isfinite(loss) and math.isfinite(gradient), 'Nonfinite analytic recomputation')
    return test_logits, {'initial_loss': initial, 'final_loss': loss, 'gradient_inf': gradient,
                         'converged': gradient <= 1e-5 and loss <= initial + 1e-10}


def audit(directory, reference_root=None):
    directory = Path(directory).resolve()
    report = {'schema': 'independent-c0-audit-v1', 'artifact_dir': str(directory),
              'auditor_sha256': sha(__file__), 'numpy': np.__version__, 'consistent': False,
              'errors': [], 'hashes_verified': {}, 'recomputed_metrics': {}, 'model_checks': {},
              'scope': 'Artifact arithmetic and convergence audit only; no scientific or E3 verdict.',
              'limitations': ['Historical train membership and original cache generation are not independently reconstructed.']}
    try:
        manifest, results, config = (read_json(directory / f) for f in ('manifest.json', 'results.json', 'prereg.json'))
        status = read_json(directory / 'status.json')
        require(manifest['schema'] == 'c0-prepared-v1' and results['schema'] == 'c0-linear-view-probe-v1', 'Expected C0 v1 artifacts')
        require(config['expected_counts'] == COUNTS, 'Unexpected preregistered population counts')
        require(config['fit']['lambda'] == 0.01, 'Unexpected regularization')

        def check_hash(filename, expected):
            observed = sha(directory / filename)
            require(observed == expected, 'SHA256 mismatch: ' + filename)
            report['hashes_verified'][filename] = observed

        for filename, key in [('items.jsonl', 'items_sha256'), ('prereg.json', 'prereg_sha256'),
                              ('source.py', 'code_sha256'), ('global_features.npz', 'features_sha256'),
                              ('eligibility_audit.json', 'eligibility_sha256')]:
            check_hash(filename, manifest[key])
        check_hash('predictions.jsonl', results['predictions_sha256'])
        check_hash('manifest.json', results['input_manifest_sha256'])
        report['input_results_sha256'] = sha(directory / 'results.json')
        for filename, expected in E2_HASHES.items():
            require(config['expected_source_sha256'][filename] == expected
                    and manifest['source_sha256'][filename] == expected, 'Historical E2 reference hash declaration changed')

        items = read_rows(directory / 'items.jsonl')
        require(len(items) == 5989 and len({x['id'] for x in items}) == len(items), 'Expected 5,989 unique population items')
        all_by_id = {x['id']: x for x in items}
        observed_counts = collections.Counter(f"{x['partition']}|{x['phen']}|{x['answer']}" for x in items)
        require(dict(observed_counts) == COUNTS and manifest['counts'] == COUNTS, 'Population class counts mismatch')
        population_support = collections.Counter()
        paired = collections.defaultdict(list)
        for item in items:
            require(item['partition'] in ('train', 'test') and item['phen'] in PHEN and item['type'] == 'Q1', 'Unknown population partition/phen/type')
            require(item['kind'] in ('pos', 'neg', 'hard_neg') and item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Invalid source kind/label')
            require(str(item['cluster']) == (str(item['event']) if item['phen'] == 'flood' else item['fold']), 'Item source cluster mismatch')
            idx = item['indices']
            require(len(idx) == 2 and all(type(v) is int for v in idx) and 0 <= idx[0] < idx[1] < (3 if item['phen'] == 'flood' else 12), 'Invalid frozen feature indices')
            if item['phen'] == 'flood':
                expected_slots = ['pre_1', 'pre_2'] if item['kind'] == 'neg' else ['pre_2', 'post']
                require(item['slots'] == expected_slots and idx == ([0, 1] if item['kind'] == 'neg' else [1, 2]), 'Flood kind/slot mismatch')
            population_support[f"{item['partition']}|{item['phen']}|{item['cluster']}|{item['kind']}|{item['answer']}"] += 1
            if item['kind'] != 'hard_neg':
                paired[(item['partition'], item['phen'], item['tile'])].append(item)
        require(dict(population_support) == manifest['tuple_counts'], 'Manifest population support mismatch')
        for pair in paired.values():
            require(len(pair) == 2 and {x['kind'] for x in pair} == {'pos', 'neg'} and len({x['cluster'] for x in pair}) == 1, 'Incomplete tile pair')
        for phenomenon in PHEN:
            train = [x for x in items if x['partition'] == 'train' and x['phen'] == phenomenon]
            test = [x for x in items if x['partition'] == 'test' and x['phen'] == phenomenon]
            require(not ({x['tile'] for x in train} & {x['tile'] for x in test}), 'Train/test tile overlap')
            require(not ({x['cluster'] for x in train} & {x['cluster'] for x in test}), 'Train/test event/fold overlap')
        for split in ('train', 'test'):
            require(digest_ids(x['id'] for x in items if x['partition'] == split) == manifest['population_ids_sha256'][split], 'Population ID digest mismatch')
        test_items = [x for x in items if x['partition'] == 'test']
        test_by_id = {x['id']: x for x in test_items}
        require(len(test_items) == 1755, 'Expected exactly 1,755 test items')
        land_folds = collections.Counter(x['fold'] for x in test_items if x['phen'] == 'landslide')
        require(sorted(land_folds.values()) == [12, 372], 'Expected two landslide folds with 6/186 paired tiles')
        require(len({x['cluster'] for x in test_items if x['phen'] == 'flood' and x['kind'] != 'hard_neg'}) == 10, 'Expected 10 flood events')

        eligibility = read_json(directory / 'eligibility_audit.json')
        candidates = eligibility['items']
        candidate_keys = [(x['partition'], x['phen'], x['id']) for x in candidates]
        require(len(set(candidate_keys)) == len(candidate_keys), 'Duplicate eligibility candidate')
        eligible = [x for x in candidates if x['eligible']]
        require({x['id'] for x in eligible} == set(all_by_id) and len(eligible) == len(items), 'Eligibility/retained population mismatch')
        for candidate in candidates:
            require(type(candidate['eligible']) is bool and isinstance(candidate['reasons'], list), 'Malformed eligibility decision')
            require(candidate['eligible'] == (not candidate['reasons']), 'Eligibility/reason mismatch')
            if candidate['eligible']:
                item = all_by_id[candidate['id']]
                require(all(candidate[k] == item[k] for k in ('tile', 'partition', 'phen', 'cache_path')), 'Retained eligibility metadata mismatch')
        require(eligibility['excluded'] == [x for x in candidates if not x['eligible']], 'Exclusion disclosure mismatch')
        require(eligibility['candidate_counts'] == dict(collections.Counter(f"{x['partition']}|{x['phen']}" for x in candidates)), 'Candidate support mismatch')
        require(eligibility['eligible_counts'] == dict(collections.Counter(f"{x['partition']}|{x['phen']}" for x in eligible)), 'Eligible support mismatch')
        report['exclusions'] = eligibility['excluded']
        report['coverage'] = {'population': len(items), 'test_items': len(test_items), 'prediction_rows': 5265,
                              'landslide_fold_support': dict(land_folds), 'flood_events': 10}

        if reference_root is not None:
            root = Path(reference_root)
            for filename, expected in E2_HASHES.items():
                require(sha(root / filename) == expected, 'Historical reference bytes mismatch: ' + filename)
                if filename.endswith('.jsonl'):
                    rows = read_rows(root / filename)
                    require(len(rows) == 1755 and len({x['id'] for x in rows}) == 1755, 'E2 reference coverage/duplicates')
                    require({x['id'] for x in rows} == set(test_by_id), 'E2 reference test IDs differ')
                    for row in rows:
                        item = test_by_id[row['id']]
                        require(all(row[k] == item[k] for k in ('tile', 'phen', 'kind', 'fold')) and row['text_gold'] == item['answer'], 'E2 reference metadata differs')
            report['historical_test_references_verified'] = True
        else:
            report['historical_test_references_verified'] = False
            report['limitations'].append('Historical E2 hashes are pinned, but reference files were not supplied for independent ID/label cross-check.')

        rows = read_rows(directory / 'predictions.jsonl')
        require(len(rows) == 5265, 'Expected exactly 1,755 x 3 = 5,265 predictions')
        keys = [(x['id'], x['arm']) for x in rows]
        require(len(set(keys)) == len(keys) and set(keys) == {(i, a) for i in test_by_id for a in ARMS}, 'Prediction coverage/duplicate failure')
        prediction_by_key = dict(zip(keys, rows))
        for row in rows:
            item = test_by_id[row['id']]
            require(all(row[k] == item[k] for k in ('tile', 'phen', 'kind', 'cluster')) and row['source_gold'] == item['answer'], 'Prediction metadata/source label mismatch')
            require(type(row['logit']) in (int, float) and math.isfinite(row['logit']), 'Nonfinite/non-numeric prediction logit')
            require(row['prediction'] == ('yes' if row['logit'] >= 0 else 'no'), 'Prediction/logit sign mismatch')

        with np.load(directory / 'global_features.npz', allow_pickle=False) as archive:
            require(set(archive.files) == {'pairs', 'ids'}, 'Unexpected feature archive fields')
            pairs, ids = archive['pairs'], archive['ids'].tolist()
        require(ids == [x['id'] for x in items] and pairs.shape == (5989, 2, 768) and pairs.dtype == np.float32 and np.isfinite(pairs).all(), 'Feature archive shape/dtype/order/finite contract')
        require(manifest['feature_shape'] == list(pairs.shape), 'Manifest feature shape mismatch')
        caches = {x['cache_path'] for x in items}
        require(set(manifest['cache_sha256']) == caches and manifest['n_unique_caches'] == len(caches), 'Cache provenance coverage mismatch')

        expected_models = {p + '_' + a for p in PHEN for a in ARMS}
        require(set(results['fits']) == expected_models and set(results['model_sha256']) == expected_models, 'Expected exactly six fitted models')
        all_converged = True
        for phenomenon in PHEN:
            train_idx = [j for j, x in enumerate(items) if x['partition'] == 'train' and x['phen'] == phenomenon]
            test_idx = [j for j, x in enumerate(items) if x['partition'] == 'test' and x['phen'] == phenomenon]
            eval_items = [items[j] for j in test_idx]
            labels = [items[j]['answer'] == 'yes' for j in train_idx]
            report['recomputed_metrics'][phenomenon] = {}
            for arm in ARMS:
                key, d = phenomenon + '_' + arm, 1536 if arm == 'pair' else 768
                check_hash(key + '.npz', results['model_sha256'][key])
                with np.load(directory / (key + '.npz'), allow_pickle=False) as archive:
                    require(set(archive.files) == {'weights', 'bias', 'mean', 'scale'}, 'Unexpected model archive fields: ' + key)
                    model = {name: archive[name] for name in archive.files}
                require(all(a.dtype == np.float64 and np.isfinite(a).all() for a in model.values()), 'Nonfinite/non-float64 model: ' + key)
                require(all(model[name].shape == (d,) for name in ('weights', 'mean', 'scale')) and model['bias'].shape == () and (model['scale'] > 0).all(), 'Model dimensions/scales: ' + key)
                x = pairs.reshape(len(items), 1536) if arm == 'pair' else pairs[:, 0 if arm == 'earlier' else 1, :]
                logits, analytic = logistic_recompute(x[train_idx], labels, x[test_idx], model)
                predictions = [prediction_by_key[(item['id'], arm)] for item in eval_items]
                saved_logits = np.array([p['logit'] for p in predictions])
                require(np.allclose(logits, saved_logits, atol=1e-8, rtol=1e-8), 'Saved logits differ from frozen model: ' + key)
                require(np.array_equal(logits >= 0, saved_logits >= 0), 'Recomputed decision sign differs: ' + key)
                diag = results['fits'][key]
                for name in ('initial_loss', 'final_loss', 'gradient_inf'):
                    close(diag[name], analytic[name], key + '.' + name, atol=1e-9, rtol=1e-8)
                require(type(diag['converged']) is bool and diag['converged'] == analytic['converged'], 'Convergence flag differs: ' + key)
                require(diag['train_n'] == len(train_idx) and diag['n_features'] == d and diag['parameter_count'] == d + 1, 'Reported fit dimensions: ' + key)
                require(type(diag['iterations']) is int and 0 <= diag['iterations'] <= 200 and type(diag['function_evaluations']) is int and diag['function_evaluations'] >= 1, 'Malformed optimizer counters')
                require(isinstance(diag['runtime_s'], (int, float)) and math.isfinite(diag['runtime_s']) and diag['runtime_s'] >= 0, 'Malformed fit runtime')
                all_converged &= analytic['converged']
                table = metric_table(eval_items, [p['prediction'] for p in predictions])
                report['recomputed_metrics'][phenomenon][arm] = table
                report['model_checks'][key] = dict(analytic, max_abs_logit_difference=float(np.max(np.abs(logits - saved_logits))), train_n=len(train_idx), test_n=len(test_idx), parameter_count=d + 1)
            by_arm = report['recomputed_metrics'][phenomenon]
            by_arm['later_minus_pair'] = arm_contrast(by_arm['later'], by_arm['pair'], phenomenon)
        close(results['metrics'], report['recomputed_metrics'], 'results.metrics')
        require(type(results['valid']) is bool and results['valid'] == bool(all_converged), 'Overall validity differs from six convergence checks')
        expected_failures = [key + ': optimizer did not meet frozen convergence criterion' for key in results['fits'] if not results['fits'][key]['converged']]
        require(results['failures'] == expected_failures, 'Reported convergence failures differ')
        require(status['status'] == ('complete' if all_converged else 'invalid') and status['valid'] == bool(all_converged), 'Final status differs')
        report.update(consistent=True, all_six_converged=bool(all_converged), run_valid=bool(all_converged))
    except (AuditFailure, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_dir', type=Path)
    parser.add_argument('--out', type=Path, help='Optional JSON report path outside the input artifact directory')
    parser.add_argument('--reference-root', type=Path, help='Root containing historical e2_multi_reader_v0 outputs')
    args = parser.parse_args()
    if args.out is not None:
        require(not args.out.resolve().is_relative_to(args.artifact_dir.resolve()), '--out must be outside read-only input artifacts')
    report = audit(args.artifact_dir, args.reference_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out is not None:
        args.out.write_text(rendered)
    print(rendered, end='')
    return 2 if not report['consistent'] else (0 if report['run_valid'] else 1)


if __name__ == '__main__':
    sys.exit(main())
