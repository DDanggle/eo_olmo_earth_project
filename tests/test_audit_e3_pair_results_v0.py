import copy
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import audit_e3_pair_results_v0 as audit


def fixture():
    items = []
    for phen, n in (('flood', 10), ('landslide', 2)):
        for event in range(n):
            for kind in ('pos', 'neg'):
                items.append({'id': f'{phen}_{event}_{kind}', 'tile': f'{phen}_{event}',
                              'cluster': str(event), 'phen': phen, 'kind': kind,
                              'answer': 'yes' if kind == 'pos' else 'no',
                              'allowed_arms': list(audit.ARMS), 'pair_key': f'pair_{len(items)}'})
    rows = []
    for seed in (1, 2, 3):
        for item in items:
            for arm in item['allowed_arms']:
                rows.append({'seed': seed, 'arm': arm, **{k: item[k] for k in ('id', 'tile', 'cluster', 'phen', 'kind', 'pair_key')},
                             'source_gold': item['answer'], 'transformed_gold': None,
                             'parsed': item['answer'], 'answer_raw': item['answer']})
    return items, rows


class IndependentAuditTests(unittest.TestCase):
    def test_sufficient_requires_both_arms_within_two_seeds(self):
        items, rows = fixture()
        index, _ = audit.index_answers(items, rows)
        score = audit.recompute(items, index)
        self.assertEqual(score['verdict'], 'single_second_view_sufficient_under_intervention')
        self.assertEqual(score['sufficient_seeds'], [1, 2, 3])
        self.assertEqual(score['metrics']['1']['flood']['contrasts']['later_only']['ci95_delta'], [0., 0.])
        self.assertIsNone(score['metrics']['1']['landslide']['contrasts']['later_only']['ci95_delta'])

    def test_history_sensitive_constant_event_drop(self):
        items, rows = fixture()
        for row in rows:
            if row['arm'] in ('later_only', 'repeat_later'):
                row['parsed'] = row['answer_raw'] = 'no'
        index, _ = audit.index_answers(items, rows)
        score = audit.recompute(items, index)
        self.assertEqual(score['verdict'], 'history_sensitive_under_intervention')
        contrast = score['metrics']['1']['flood']['contrasts']['later_only']
        self.assertEqual(contrast['delta'], -.5)
        self.assertEqual(contrast['ci95_delta'], [-.5, -.5])

    def test_cross_seed_arm_passing_does_not_create_a_joint_pass(self):
        items, rows = fixture()
        for row in rows:
            fail = ((row['seed'] == 1 and row['arm'] == 'repeat_later') or
                    (row['seed'] == 2 and row['arm'] == 'later_only') or
                    (row['seed'] == 3 and row['arm'] in ('later_only', 'repeat_later')))
            if fail:
                row['parsed'] = row['answer_raw'] = 'no'
        index, _ = audit.index_answers(items, rows)
        score = audit.recompute(items, index)
        self.assertEqual(score['verdict'], 'mixed_or_inconclusive')
        self.assertEqual(score['sufficient_seeds'], [])
        self.assertEqual(score['history_sensitive_seeds'], [3])

    def test_gold_parse_and_coverage_tamper_are_rejected(self):
        items, rows = fixture()
        for change in ('gold', 'parsed', 'missing', 'duplicate'):
            altered = copy.deepcopy(rows)
            if change == 'gold': altered[0]['transformed_gold'] = 'no'
            if change == 'parsed': altered[0]['parsed'] = 'no'
            if change == 'missing': altered.pop()
            if change == 'duplicate': altered.append(altered[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.index_answers(items, altered)

    def test_reproduction_failure_prevents_a_scientific_verdict(self):
        items, rows = fixture()
        index, _ = audit.index_answers(items, rows)
        saved = {str(seed): {item['id']: item['answer'] for item in items} for seed in (1, 2, 3)}
        audit.reproduction(items, index, saved)
        index[(1, 'real', 'flood_0_pos')]['parsed'] = 'no'
        with self.assertRaises(ValueError):
            audit.reproduction(items, index, saved)


if __name__ == '__main__':
    unittest.main()
