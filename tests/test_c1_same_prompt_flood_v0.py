"""Independent synthetic C1 tests; never reads historical E2 prediction files.

Run against an explicit checkout:
  python c1_same_prompt_tests_ready.py --code-dir /repo/code -v
Or during unittest discovery:
  C1_CODE_DIR=/repo/code python -m unittest c1_same_prompt_tests_ready -v
When installed under repo/tests, repo/code is selected automatically.
All fixtures and expected metrics below are hand constructed. The full entrypoint
fixtures contain 1,755 fake rows solely to exercise its fixed coverage contracts.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

_explicit_code_dir = os.environ.get('C1_CODE_DIR')
if '--code-dir' in sys.argv:
    _index = sys.argv.index('--code-dir')
    if _index + 1 >= len(sys.argv):
        raise ValueError('--code-dir requires a directory')
    _explicit_code_dir = sys.argv[_index + 1]
    del sys.argv[_index:_index + 2]
_code_dirs = ([Path(_explicit_code_dir)] if _explicit_code_dir else
              [Path(__file__).resolve().parent.parent / 'code', Path(__file__).resolve().parent])
_module_paths = [p / 'c1_same_prompt_flood_v0.py' for p in _code_dirs]
MODULE_PATH = next((p.resolve() for p in _module_paths if p.is_file()), None)
if MODULE_PATH is None:
    raise FileNotFoundError('C1 module not found in requested code directories: ' + str(_code_dirs))
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location('c1_under_test', MODULE_PATH)
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)


def item(name, kind, event='1', dates=('2020-01-01', '2020-01-13')):
    return dict(id=name, tile='tile_' + name, event=event, dates=list(dates),
                slots=['pre_2', 'post'], kind=kind,
                answer='yes' if kind == 'pos' else 'no', partition='test',
                fold='test', phen='flood', type='Q1')


def predictions(items, reader=None, blind=None):
    result = {}
    for arm in ('reader', 'blind'):
        for seed in (1, 2, 3):
            rule = reader if arm == 'reader' else blind
            result[(arm, seed)] = {it['id']: {'parsed': rule(seed, it) if rule else 'no'} for it in items}
    return result


def add_pair(items, event, name, count, dates=('2020-01-01', '2020-01-13')):
    for index in range(count):
        items.append(item(f'{name}_p{index}', 'pos', event, dates))
        items.append(item(f'{name}_n{index}', 'hard_neg', event, dates))


def raw_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def raw_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in values))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AggregationTests(unittest.TestCase):
    def test_exact_prompt_ignores_event_tile_and_label_but_preserves_dates(self):
        positive = item('p', 'pos', 'e1')
        negative = item('n', 'hard_neg', 'e2')
        expected = ('These are 2 Sentinel-1 observations of the same area in chronological order, '
                    'taken on 2020-01-01, 2020-01-13: <EO> Did a flood occur between the two observations? Answer with yes or no.')
        self.assertEqual(c1.prompt(positive), expected)
        self.assertEqual(c1.prompt(negative), expected)
        negative['dates'][0] = '2020-01-02'
        self.assertNotEqual(c1.prompt(positive), c1.prompt(negative))
        negative['slots'] = ['pre_1', 'post']
        with self.assertRaisesRegex(ValueError, 'slots/dates'):
            c1.prompt(negative)

    def test_imbalanced_classes_still_give_constant_blind_ba_half(self):
        items = [item('p' + str(i), 'pos') for i in range(9)] + [item('n', 'hard_neg')]
        groups, excluded = c1.strata(items)
        self.assertFalse(excluded)
        for decision in ('yes', 'no'):
            with self.subTest(decision=decision):
                result = c1.event_metrics(groups, {it['id']: {'parsed': decision} for it in items})['1']
                self.assertEqual(result['ba'], .5)
                self.assertEqual(result['decision_contrast'], 0.)
                self.assertEqual((result['n_pos'], result['n_hard_neg']), (9, 1))

    def test_events_get_equal_weight_despite_tile_count(self):
        items = []
        add_pair(items, 'large', 'large', 9)
        add_pair(items, 'small', 'small', 1)
        def reader(seed, it):
            return it['answer'] if it['event'] == 'large' else ('no' if it['answer'] == 'yes' else 'yes')
        result = c1.evaluate_subset(items, predictions(items, reader=reader))
        # Pooled classification accuracy would be .9; equal-event BA is .5.
        self.assertEqual(result['per_seed']['1']['reader_macro_ba'], .5)
        self.assertEqual(result['per_seed']['1']['reader_minus_blind'], 0.)
        self.assertEqual(result['seed_mean_reader_minus_blind'], 0.)
        self.assertEqual(result['seed_mean_ci95_delta'], [-.5, .5])
        self.assertEqual(result['per_seed']['1']['arms']['reader']['large']['ba'], 1.)
        self.assertEqual(result['per_seed']['1']['arms']['reader']['small']['ba'], 0.)

    def test_prompt_strata_get_equal_weight_within_event(self):
        items = []
        add_pair(items, 'same_event', 'many', 9)
        add_pair(items, 'same_event', 'few', 1, dates=('2020-02-01', '2020-02-13'))
        def reader(seed, it):
            return it['answer'] if it['id'].startswith('many') else ('no' if it['answer'] == 'yes' else 'yes')
        result = c1.evaluate_subset(items, predictions(items, reader=reader))
        event = result['per_seed']['1']['arms']['reader']['same_event']
        self.assertEqual(event['n_strata'], 2)
        self.assertEqual(event['ba'], .5)
        self.assertEqual(event['recall'], .5)
        self.assertEqual(event['fpr'], .5)
        self.assertEqual(event['decision_contrast'], 0.)

    def test_missing_class_strata_are_fully_disclosed_not_fake_zero(self):
        items = [item('good_p', 'pos', 'e1'), item('good_n', 'hard_neg', 'e1'),
                 item('lone_p', 'pos', 'e2'),
                 item('other_date_n', 'hard_neg', 'e1', dates=('2021-01-01', '2021-01-13'))]
        result = c1.evaluate_subset(items, predictions(items))
        self.assertEqual(result['candidate_count'], 4)
        self.assertEqual(result['events'], ['e1'])
        self.assertEqual(result['supported_n_pos'], 1)
        self.assertEqual(result['supported_n_hard_neg'], 1)
        excluded = result['unsupported_strata']
        self.assertEqual(len(excluded), 2)
        self.assertEqual({i for x in excluded for key in ('pos_ids', 'hard_neg_ids') for i in x[key]}, {'lone_p', 'other_date_n'})
        self.assertTrue(all(x['reason'] == 'missing_one_source_class' for x in excluded))
        self.assertEqual(result['per_seed']['1']['blind_macro_ba'], .5)
        self.assertFalse(result['support_sufficient_for_interpretation'])

    def test_blind_consistency_is_global_across_events_and_unsupported_items(self):
        # Both events lack a class; the same reconstructed prompt still must agree.
        items = [item('p', 'pos', 'e1'), item('n', 'hard_neg', 'e2')]
        self.assertFalse(c1.strata(items)[0])
        result = c1.blind_consistency(items, {'p': {'parsed': 'yes'}, 'n': {'parsed': 'no'}})
        self.assertFalse(result['consistent'])
        self.assertEqual(result['n_reconstructed_prompts'], 1)
        self.assertEqual(result['violations'][0]['ids'], ['n', 'p'])
        self.assertEqual(result['violations'][0]['outputs'], ['no', 'yes'])

    def test_seed_mean_bootstrap_uses_events_not_six_seed_event_rows(self):
        items = []
        add_pair(items, 'a', 'a', 1)
        add_pair(items, 'b', 'b', 1)
        def reader(seed, it):
            if seed == 2: return 'no'
            correct = (seed == 1 and it['event'] == 'a') or (seed == 3 and it['event'] == 'b')
            return it['answer'] if correct else ('no' if it['answer'] == 'yes' else 'yes')
        result = c1.evaluate_subset(items, predictions(items, reader=reader))
        self.assertEqual(result['seed_mean_delta_by_event'], {'a': 0., 'b': 0.})
        self.assertEqual(result['seed_mean_ci95_delta'], [0., 0.])
        self.assertEqual(result['per_seed']['1']['ci95_delta'], [-.5, .5])
        self.assertEqual(result['per_seed']['3']['ci95_delta'], [-.5, .5])
        # Flattening the six seed-event rows incorrectly gives a nonzero interval.
        self.assertNotEqual(c1.bootstrap([.5, -.5, 0., 0., -.5, .5]), [0., 0.])

    def test_zero_or_single_event_has_no_invented_interval(self):
        self.assertIsNone(c1.bootstrap([]))
        self.assertIsNone(c1.bootstrap([.5]))
        empty = c1.evaluate_subset([item('unsupported', 'pos')], predictions([item('unsupported', 'pos')]))
        self.assertEqual(empty['n_events'], 0)
        self.assertIsNone(empty['seed_mean_reader_minus_blind'])
        self.assertIsNone(empty['seed_mean_ci95_delta'])
        self.assertEqual(len(empty['unsupported_strata']), 1)


class StrictEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='c1_synthetic_only_')
        self.root = Path(self.tmp.name)
        self.repo, self.quality = self.root / 'repo', self.root / 'quality'
        self.items = []
        for index in range(457):
            for kind in ('pos', 'hard_neg'):
                self.items.append(item(f'{kind}_{index}', kind, 100 if index < 456 else 200))
            neg = item('neg_' + str(index), 'neg', 100 if index < 456 else 200)
            neg['slots'] = ['pre_1', 'pre_2']
            self.items.append(neg)
        for index in range(384):
            it = item('land_' + str(index), 'pos' if index % 2 else 'neg', 'land_event')
            it.update(phen='landslide', fold='holdout_synthetic')
            self.items.append(it)
        self.assertEqual(len(self.items), 1755)
        items_rel = 'artifacts/c0_synthetic/items.jsonl'
        raw_rows(self.repo / items_rel, self.items)
        self.targets = [x for x in self.items if x['phen'] == 'flood' and x['kind'] in ('pos', 'hard_neg')]
        quality_rows = []
        for it in self.targets:
            excluded = it['id'] == 'pos_0'
            quality_rows.append({k: it[k] for k in ('id', 'tile', 'dates', 'slots', 'kind')} |
                                dict(event=str(it['event']), source_answer=it['answer'], valid_frac=.8 if excluded else 1.,
                                     flood_frac=.5 if it['kind'] == 'pos' else 0.,
                                     flood_px=50 if it['kind'] == 'pos' else 0,
                                     eligible_symmetric_quality=not excluded))
        raw_rows(self.quality / 'quality.jsonl', quality_rows)
        raw_json(self.quality / 'manifest.json', {'quality_sha256': sha(self.quality / 'quality.jsonl'), 'items_sha256': sha(self.repo / items_rel)})
        raw_json(self.quality / 'status.json', {'status': 'complete'})
        inputs = [items_rel]
        for arm in ('reader', 'blind'):
            for seed in (1, 2, 3):
                rel = f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'
                answer_rows = []
                for it in self.items:
                    answer_rows.append({k: it[k] for k in ('id', 'tile', 'fold', 'phen', 'kind')} |
                                       dict(emb_item=it['id'], text_gold=it['answer'], emb_gold=it['answer'], answer_raw='no', parsed='no'))
                raw_rows(self.repo / rel, answer_rows)
                inputs.append(rel)
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(items_path=items_rel, scope='synthetic-only no historical predictions', limits=['fixture'],
                         inputs_sha256={rel: sha(self.repo / rel) for rel in inputs})
        raw_json(self.plan_path, self.plan)
        self.args = SimpleNamespace(repo=str(self.repo), plan=str(self.plan_path), quality=str(self.quality))

    def tearDown(self):
        self.tmp.cleanup()

    def alter_predictions(self, arm, seed, edit, refresh_hash=True):
        rel = f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'
        path = self.repo / rel
        values = [json.loads(x) for x in path.read_text().splitlines()]
        edit(values)
        raw_rows(path, values)
        if refresh_hash:
            self.plan['inputs_sha256'][rel] = sha(path)
            raw_json(self.plan_path, self.plan)

    def run_fixture(self, name):
        out = self.root / name
        out.mkdir()
        return c1.run(self.args, out), out

    def test_invalid_parse_rejected_even_for_quality_excluded_item(self):
        # Guard the pure helper too: an invalid string must not become Boolean no.
        helper_items = [item('p', 'pos'), item('n', 'hard_neg')]
        groups, _ = c1.strata(helper_items)
        with self.assertRaises(ValueError):
            c1.event_metrics(groups, {'p': {'parsed': 'abstain'}, 'n': {'parsed': 'no'}})
        def edit(values):
            next(x for x in values if x['id'] == 'pos_0')['parsed'] = 'abstain'
        self.alter_predictions('reader', 1, edit)
        with self.assertRaisesRegex(ValueError, 'Unparsed answer: no implicit negative'):
            self.run_fixture('invalid_parse')

    def test_quality_exclusion_and_selection_do_not_depend_on_predictions(self):
        first, first_out = self.run_fixture('before_predictions')
        self.assertTrue(first['valid'])
        self.assertEqual(first['quality_excluded_ids'], ['pos_0'])
        quality_subset = first['subsets']['quality_symmetric']
        self.assertEqual((quality_subset['supported_n_pos'], quality_subset['supported_n_hard_neg']), (456, 457))
        self.assertEqual(quality_subset['per_seed']['1']['blind_macro_ba'], .5)
        selected_before = (first_out / 'selected_ids.json').read_bytes()
        def edit(values):
            for value in values:
                value['parsed'] = 'yes'
                value['answer_raw'] = 'yes'
        for seed in (1, 2, 3): self.alter_predictions('reader', seed, edit)
        after, after_out = self.run_fixture('after_predictions')
        self.assertTrue(after['valid'])
        self.assertEqual(after['quality_excluded_ids'], ['pos_0'])
        self.assertEqual((after_out / 'selected_ids.json').read_bytes(), selected_before)
        self.assertEqual(after['subsets']['all_original_targets']['candidate_count'], 914)
        self.assertEqual(after['subsets']['quality_symmetric']['candidate_count'], 913)

    def test_quality_excluded_blind_violation_still_invalidates_entire_analysis(self):
        def edit(values):
            value = next(x for x in values if x['id'] == 'pos_0')
            value['parsed'] = 'yes'; value['answer_raw'] = 'yes'
        self.alter_predictions('blind', 2, edit)
        result, out = self.run_fixture('excluded_blind_violation')
        self.assertFalse(result['valid'])
        self.assertEqual(result['quality_excluded_ids'], ['pos_0'])
        self.assertFalse(result['blind_consistency']['2']['consistent'])
        violation_ids = result['blind_consistency']['2']['violations'][0]['ids']
        self.assertEqual(len(violation_ids), 914)
        self.assertIn('pos_0', violation_ids)
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'invalid')
        # The violating row remains in original-target evaluation; it is not deleted.
        self.assertEqual(result['subsets']['all_original_targets']['supported_n_pos'], 457)

    def test_frozen_prediction_hash_guard_rejects_mutation(self):
        self.alter_predictions('reader', 1, lambda values: values[0].update(parsed='yes'), refresh_hash=False)
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            self.run_fixture('hash_failure')


if __name__ == '__main__':
    unittest.main()
