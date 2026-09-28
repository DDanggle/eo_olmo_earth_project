"""Synthetic 5,989-item audit fixtures; no C0 implementation or real data imports."""
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_c0_linear_v1 as audit


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def sh(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(sorted(ids), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fixture(root):
    items = []
    def tile(partition, phen, cluster, index, hard=False):
        name = f'{partition}_{phen}_{cluster}_{index}' + ('_dry' if hard else '')
        for kind in (('hard_neg',) if hard else ('pos', 'neg')):
            item = dict(id=name + '_' + kind, tile=name, phen=phen, partition=partition,
                        cluster=str(cluster), fold=partition if phen == 'flood' else str(cluster),
                        kind=kind, answer='yes' if kind == 'pos' else 'no', type='Q1',
                        cache_path='/not/read/' + name + '.npy', indices=[0, 1] if kind == 'neg' else [1, 2])
            if phen == 'flood':
                item.update(event=cluster, slots=['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'])
            items.append(item)
    for i in range(518): tile('train', 'landslide', 'train_region', i)
    for i in range(1066):
        tile('train', 'flood', 100, i)
        tile('train', 'flood', 100, i, hard=True)
    for cluster, n in [('holdout_hiroshima', 186), ('holdout_indonesia', 6)]:
        for i in range(n): tile('test', 'landslide', cluster, i)
    for event, n in zip(range(200, 210), [261, 101, 3, 5, 48, 21, 1, 9, 2, 6]):
        for i in range(n):
            tile('test', 'flood', event, i)
            tile('test', 'flood', event, i, hard=True)
    items.sort(key=lambda x: (x['partition'], x['phen'], x['id']))
    assert len(items) == 5989
    write_rows(root / 'items.jsonl', items)
    (root / 'source.py').write_text('# Hash-only source fixture; this file is never imported.\n')
    config = {'expected_counts': dict(audit.COUNTS), 'fit': {'lambda': .01}, 'expected_source_sha256': dict(audit.E2_HASHES)}
    write(root / 'prereg.json', config)
    pairs = np.zeros((len(items), 2, 768), dtype=np.float32)
    np.savez_compressed(root / 'global_features.npz', pairs=pairs, ids=np.array([x['id'] for x in items]))
    candidates = [dict((k, it[k]) for k in ('id', 'tile', 'partition', 'phen', 'cache_path')) | {'eligible': True, 'reasons': [], 'source_path': 'fixture'} for it in items]
    candidate_counts = dict(audit.collections.Counter(f"{x['partition']}|{x['phen']}" for x in candidates))
    write(root / 'eligibility_audit.json', {'schema': 'c0-e2-eligibility-audit-v1', 'items': candidates, 'excluded': [], 'candidate_counts': candidate_counts, 'eligible_counts': candidate_counts})
    manifest = {'schema': 'c0-prepared-v1', 'counts': dict(audit.COUNTS),
                'source_sha256': dict(audit.E2_HASHES), 'cache_sha256': {x['cache_path']: '0' * 64 for x in items},
                'population_ids_sha256': {s: ids_sha(x['id'] for x in items if x['partition'] == s) for s in ('train', 'test')},
                'tuple_counts': dict(audit.collections.Counter(f"{x['partition']}|{x['phen']}|{x['cluster']}|{x['kind']}|{x['answer']}" for x in items)),
                'feature_shape': [len(items), 2, 768], 'n_unique_caches': len({x['cache_path'] for x in items})}
    for filename, key in [('items.jsonl', 'items_sha256'), ('prereg.json', 'prereg_sha256'), ('source.py', 'code_sha256'), ('global_features.npz', 'features_sha256'), ('eligibility_audit.json', 'eligibility_sha256')]:
        manifest[key] = sh(root / filename)
    write(root / 'manifest.json', manifest)
    rows, fits, model_hashes, metrics = [], {}, {}, {}
    for phen in ('flood', 'landslide'):
        metrics[phen] = {}
        train_n = 3198 if phen == 'flood' else 1036
        test = [x for x in items if x['partition'] == 'test' and x['phen'] == phen]
        for arm in ('earlier', 'later', 'pair'):
            key, d = phen + '_' + arm, 1536 if arm == 'pair' else 768
            np.savez_compressed(root / (key + '.npz'), weights=np.zeros(d, np.float64), bias=np.array(0., np.float64), mean=np.zeros(d, np.float64), scale=np.ones(d, np.float64))
            model_hashes[key] = sh(root / (key + '.npz'))
            fits[key] = {'converged': True, 'initial_loss': math.log(2), 'final_loss': math.log(2), 'gradient_inf': 0., 'iterations': 0, 'function_evaluations': 1, 'train_n': train_n, 'n_features': d, 'parameter_count': d + 1, 'runtime_s': 0.}
            events, dry = {}, {}
            for item in test:
                rows.append({k: item[k] for k in ('id', 'tile', 'phen', 'cluster', 'kind')} | dict(source_gold=item['answer'], arm=arm, logit=0., prediction='yes'))
                if item['kind'] == 'hard_neg':
                    dry.setdefault(item['cluster'], {'n': 0, 'fpr': 1.})['n'] += 1
                else:
                    e = events.setdefault(item['cluster'], dict(ba=.5, n=0, pos=0, neg=0, recall=1., fpr=1.))
                    e['n'] += 1; e[item['kind']] += 1
            hard_n = sum(v['n'] for v in dry.values())
            metrics[phen][arm] = {'event_macro_ba': .5, 'events': events, 'hard_negative': {'events': dry, 'n': hard_n, 'pooled_fpr': 1. if hard_n else None, 'event_macro_fpr': 1. if hard_n else None}, 'tuple_counts': dict(audit.collections.Counter(f"{x['cluster']}|{x['kind']}|{x['answer']}" for x in test))}
        metrics[phen]['later_minus_pair'] = {'delta': 0., 'ci95_delta': [0., 0.] if phen == 'flood' else None, 'event_deltas': {c: 0. for c in events}, 'unit': 'event' if phen == 'flood' else 'two_regions_descriptive_only'}
    write_rows(root / 'predictions.jsonl', rows)
    result = {'schema': 'c0-linear-view-probe-v1', 'valid': True, 'fits': fits, 'metrics': metrics, 'model_sha256': model_hashes, 'predictions_sha256': sh(root / 'predictions.jsonl'), 'input_manifest_sha256': sh(root / 'manifest.json'), 'failures': []}
    write(root / 'results.json', result)
    write(root / 'status.json', {'status': 'complete', 'valid': True})


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='c0_independent_audit_synthetic_')
        cls.root = Path(cls.tmp.name)
        fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @contextlib.contextmanager
    def replace_json(self, name, edit):
        path = self.root / name; original = path.read_bytes()
        obj = json.loads(original); edit(obj); write(path, obj)
        try: yield
        finally: path.write_bytes(original)

    @contextlib.contextmanager
    def replace_predictions(self, edit):
        path = self.root / 'predictions.jsonl'; original = path.read_bytes()
        rows = [json.loads(x) for x in original.splitlines()]; edit(rows); write_rows(path, rows)
        try:
            with self.replace_json('results.json', lambda r: r.update(predictions_sha256=sh(path))): yield
        finally: path.write_bytes(original)

    def assert_bad(self, substring):
        report = audit.audit(self.root)
        self.assertFalse(report['consistent'])
        self.assertIn(substring, '\n'.join(report['errors']))

    def test_full_valid_fixture(self):
        report = audit.audit(self.root)
        self.assertTrue(report['consistent'], report['errors'])
        self.assertTrue(report['all_six_converged'])
        self.assertEqual(len(report['model_checks']), 6)
        self.assertEqual(report['coverage']['prediction_rows'], 5265)
        self.assertFalse(report['historical_test_references_verified'])

    def test_duplicate_predictions_rejected_even_after_rehash(self):
        with self.replace_predictions(lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0]))):
            self.assert_bad('coverage/duplicate')

    def test_sign_mismatch_rejected_even_after_rehash(self):
        with self.replace_predictions(lambda rows: rows[0].update(logit=-1.)):
            self.assert_bad('sign mismatch')

    def test_source_cluster_mismatch_rejected(self):
        with self.replace_predictions(lambda rows: rows[0].update(cluster='wrong')):
            self.assert_bad('metadata/source label')

    def test_wrong_metric_rejected(self):
        with self.replace_json('results.json', lambda r: r['metrics']['flood']['earlier'].update(event_macro_ba=.6)):
            self.assert_bad('event_macro_ba')

    def test_invented_landslide_ci_rejected(self):
        with self.replace_json('results.json', lambda r: r['metrics']['landslide']['later_minus_pair'].update(ci95_delta=[0., 0.])):
            self.assert_bad('ci95_delta')

    def test_reported_convergence_must_match_analytic(self):
        with self.replace_json('results.json', lambda r: r['fits']['flood_earlier'].update(gradient_inf=.1)):
            self.assert_bad('gradient_inf')

    def test_analytic_gradient_known_unfitted_case(self):
        x = np.array([[-1.], [1.]])
        _, out = audit.logistic_recompute(x, [0, 1], x, dict(weights=np.zeros(1), bias=np.array(0.), mean=np.zeros(1), scale=np.ones(1)))
        self.assertEqual(out['gradient_inf'], .5)
        self.assertAlmostEqual(out['final_loss'], math.log(2))
        self.assertFalse(out['converged'])

    def test_duplicate_json_object_keys_rejected(self):
        with self.assertRaises(audit.AuditFailure): audit.decode('{"valid": true, "valid": false}')

    def test_event_macro_and_dry_support_are_independent(self):
        rows, answers = [], []
        for event, n, correct in [('large', 9, True), ('small', 1, False)]:
            for kind, gold in [('pos', 'yes'), ('neg', 'no')]:
                for _ in range(n):
                    rows.append(dict(cluster=event, kind=kind, answer=gold))
                    answers.append(gold if correct else ('no' if gold == 'yes' else 'yes'))
        for event, n, prediction in [('large', 3, 'yes'), ('small', 1, 'no')]:
            for _ in range(n):
                rows.append(dict(cluster=event, kind='hard_neg', answer='no'))
                answers.append(prediction)
        result = audit.metric_table(rows, answers)
        self.assertEqual(result['event_macro_ba'], .5)
        self.assertEqual(result['hard_negative']['n'], 4)
        self.assertEqual(result['hard_negative']['pooled_fpr'], .75)
        self.assertEqual(result['hard_negative']['event_macro_fpr'], .5)

    def test_nonzero_paired_event_bootstrap(self):
        later = {'events': {'b': {'ba': 0.}, 'a': {'ba': 1.}}}
        pair = {'events': {'a': {'ba': .5}, 'b': {'ba': .5}}}
        result = audit.arm_contrast(later, pair, 'flood')
        self.assertEqual(result['event_deltas'], {'a': .5, 'b': -.5})
        self.assertEqual(result['delta'], 0.)
        self.assertEqual(result['ci95_delta'], [-.5, .5])
        self.assertIsNone(audit.arm_contrast(later, pair, 'landslide')['ci95_delta'])

    def test_stable_loss_and_l2_gradient_at_large_logit(self):
        x = np.array([[-1.], [1.]])
        _, out = audit.logistic_recompute(x, [0, 1], x, dict(weights=np.array([1000.]), bias=np.array(0.), mean=np.zeros(1), scale=np.ones(1)))
        self.assertEqual(out['final_loss'], 5000.)
        self.assertEqual(out['gradient_inf'], 10.)
        self.assertFalse(out['converged'])


if __name__ == '__main__':
    unittest.main()
