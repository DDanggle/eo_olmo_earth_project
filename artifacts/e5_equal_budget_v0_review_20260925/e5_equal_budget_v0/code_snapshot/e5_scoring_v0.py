"""Pure E5 input transforms and exposed-data equal-budget source-label scoring.

No file access, fitting, model execution, or label editing occurs here. The
runner owns frozen data/code provenance and equal training-budget verification.
"""
from collections import Counter, defaultdict
import re

import numpy as np

ARMS = ('full', 'pair', 'later', 'delta')
SEEDS = (1, 2, 3)
EVALUATIONS = tuple((arm, 'native') for arm in ARMS) + (('full', 'full_no_delta'),)
EPS = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def transform_pair(pair, arm):
    """192 slots for every model; full_no_delta uses the 'pair' transform."""
    require(isinstance(arm, str) and arm in ARMS, 'Unknown E5 model/input arm')
    require(isinstance(pair, np.ndarray) and pair.shape == (2, 64, 768)
            and pair.dtype == np.float32, 'Expected float32 pair shape (2,64,768)')
    require(np.isfinite(pair).all(), 'Nonfinite source pair')
    tokens = np.zeros((192, 768), dtype=np.float32)
    if arm in ('full', 'pair'):
        tokens[:64] = pair[0]
    if arm in ('full', 'pair', 'later'):
        tokens[64:128] = pair[1]
    if arm in ('full', 'delta'):
        with np.errstate(over='ignore', invalid='ignore'):
            tokens[128:] = pair[1] - pair[0]
        require(np.isfinite(tokens).all(), 'Nonfinite difference block')
    types = np.repeat(np.array([0, 1, 3], dtype=np.int64), 64)
    return tokens, types


def interval(differences):
    values = np.asarray(differences, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), 'Invalid paired event differences')
    draws = np.random.default_rng(20260925).integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975], method='linear').tolist()


def _prepare(items, eval_sets):
    require(isinstance(items, list) and len(items) == 5989, 'Expected unchanged 5989 C0 items')
    require(all(isinstance(item, dict) and isinstance(item.get('id'), str) for item in items), 'Invalid source item')
    source = {item['id']: item for item in items}
    require(len(source) == len(items), 'Duplicate source ID')
    expected_counts = {
        ('train', 'flood', 'pos'): 1066, ('train', 'flood', 'neg'): 1066, ('train', 'flood', 'hard_neg'): 1066,
        ('train', 'landslide', 'pos'): 518, ('train', 'landslide', 'neg'): 518,
        ('test', 'flood', 'pos'): 457, ('test', 'flood', 'neg'): 457, ('test', 'flood', 'hard_neg'): 457,
        ('test', 'landslide', 'pos'): 192, ('test', 'landslide', 'neg'): 192,
    }
    require(Counter((item.get('partition'), item.get('phen'), item.get('kind')) for item in items) == expected_counts,
            'Source population/category counts differ')
    tiles = defaultdict(list)
    for item in items:
        require(item.get('answer') == ('yes' if item['kind'] == 'pos' else 'no'), 'Original source kind/gold mismatch')
        require(isinstance(item.get('tile'), str) and isinstance(item.get('cluster'), (str, int))
                and type(item['cluster']) is not bool, 'Invalid tile/event identifier')
        require(isinstance(item.get('dates'), list) and len(item['dates']) == 2
                and all(isinstance(date, str) and date for date in item['dates']), 'Invalid source date pair')
        if item['phen'] == 'flood':
            expected_slots = ['pre_1', 'pre_2'] if item['kind'] == 'neg' else ['pre_2', 'post']
            require(item.get('slots') == expected_slots and str(item.get('event')) == str(item['cluster']),
                    'Flood source slots/event differ')
        tiles[(item['phen'], item['tile'])].append(item)
    for group in tiles.values():
        require(len({item['partition'] for item in group}) == 1, 'Train/test tile overlap')
        kinds = [item['kind'] for item in group]
        if 'hard_neg' in kinds:
            require(kinds == ['hard_neg'], 'Hard-negative tile overlaps another item')
        else:
            require(len(group) == 2 and set(kinds) == {'pos', 'neg'}
                    and len({str(item['cluster']) for item in group}) == 1, 'Incomplete same-tile source pair')
    require(isinstance(eval_sets, dict) and set(eval_sets) == {'all_test', 'primary_same_prompt', 'paired_flood',
            'hard_negative_flood', 'landslide', 'e3_subset'}, 'Unexpected evaluation-set keys')
    for name, ids in eval_sets.items():
        require(isinstance(ids, list) and all(isinstance(value, str) for value in ids)
                and len(ids) == len(set(ids)), 'Invalid/duplicate evaluation IDs: ' + name)
    test_ids = {item['id'] for item in items if item['partition'] == 'test'}
    require(set(eval_sets['all_test']) == test_ids and len(test_ids) == 1755, 'Test support differs or train ID leaked into evaluation')
    for name, phen, kinds, expected_n in (
        ('paired_flood', 'flood', ('pos', 'neg'), 914),
        ('hard_negative_flood', 'flood', ('hard_neg',), 457),
        ('landslide', 'landslide', ('pos', 'neg'), 384),
    ):
        expected_ids = {item['id'] for item in items if item['partition'] == 'test' and item['phen'] == phen and item['kind'] in kinds}
        require(set(eval_sets[name]) == expected_ids and len(expected_ids) == expected_n, 'Descriptive evaluation support differs: ' + name)
    require(len(eval_sets['e3_subset']) == 209 and set(eval_sets['e3_subset']) <= test_ids, 'E3 descriptive subset support differs')
    primary_ids = set(eval_sets['primary_same_prompt'])
    require(len(primary_ids) == 902 and primary_ids <= test_ids, 'Expected 902 primary test IDs')
    primary = [source[item_id] for item_id in sorted(primary_ids)]
    require(Counter((item['phen'], item['kind']) for item in primary) == {('flood', 'pos'): 445, ('flood', 'hard_neg'): 457},
            'Primary same-prompt class counts differ')
    strata = defaultdict(list)
    for item in primary:
        strata[(str(item['cluster']), tuple(item['dates']), tuple(item['slots']))].append(item)
    require(len({key[0] for key in strata}) == 8, 'Expected eight primary supported events')
    for group in strata.values():
        require({item['kind'] for item in group} == {'pos', 'hard_neg'}, 'Primary stratum lacks a source class; no post-hoc exclusion allowed')
    test = [item for item in items if item['partition'] == 'test']
    for phen, count in (('flood', 10), ('landslide', 2)):
        require(len({str(item['cluster']) for item in test if item['phen'] == phen and item['kind'] != 'hard_neg'}) == count,
                'Original paired event support differs: ' + phen)
    return source, test, strata


def _index(rows, items, source, test):
    require(isinstance(rows, list), 'Rows must be a list')
    pair_index = {item['id']: index for index, item in enumerate(items)}
    expected = {(seed, arm, evaluation, item['id']) for seed in SEEDS
                for arm, evaluation in EVALUATIONS for item in test}
    answers = {}
    for row in rows:
        require(isinstance(row, dict) and type(row.get('seed')) is int, 'Invalid row/seed type')
        key = (row['seed'], row.get('model_arm'), row.get('eval_arm'), row.get('id'))
        require(key in expected and key not in answers, 'Unexpected/duplicate evaluation row or train ID')
        item = source[row['id']]
        require(all(row.get(field) == item[field] for field in ('tile', 'phen', 'kind'))
                and str(row.get('cluster')) == str(item['cluster']), 'Output/source metadata mismatch')
        require(type(row.get('pair_index')) is int and row['pair_index'] == pair_index[row['id']], 'Pair index differs from original global item order')
        require(row.get('source_gold') == item['answer'] and 'transformed_gold' in row and row['transformed_gold'] is None,
                'Source gold changed or transformed physical gold assigned')
        require('parsed' in row and row['parsed'] in ('yes', 'no', None) and isinstance(row.get('answer_raw'), str),
                'Invalid parsed/raw answer')
        match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (match[1] if match else None), 'Raw/parsed answer mismatch')
        answers[key] = row['parsed']
    require(len(rows) == 26325 and set(answers) == expected, 'Incomplete 26325-row evaluation coverage')
    rates = {}
    for seed in SEEDS:
        for arm, evaluation in EVALUATIONS:
            for phen in ('flood', 'landslide'):
                group = [item for item in test if item['phen'] == phen]
                failed = sum(answers[(seed, arm, evaluation, item['id'])] is None for item in group)
                rate = failed / len(group)
                require(rate <= .01 + EPS, f'Parse failure exceeds .01: {seed}/{arm}/{evaluation}/{phen}')
                rates[f'{seed}|{arm}|{evaluation}|{phen}'] = {'n': len(group), 'failed': failed, 'rate': rate}
    return answers, rates


def _binary_group(group, prediction, negative_kind):
    pos = [item for item in group if item['kind'] == 'pos']
    neg = [item for item in group if item['kind'] == negative_kind]
    require(pos and neg and len(pos) + len(neg) == len(group), 'Unsupported binary source group')
    recall = sum(prediction[item['id']] == 'yes' for item in pos) / len(pos)
    specificity = sum(prediction[item['id']] == 'no' for item in neg) / len(neg)
    fpr = sum(prediction[item['id']] == 'yes' for item in neg) / len(neg)
    return {'n_pos': len(pos), 'n_negative': len(neg), 'negative_kind': negative_kind,
            'recall': recall, 'specificity': specificity, 'fpr': fpr, 'ba': (recall + specificity) / 2,
            'parse_failures': sum(prediction[item['id']] is None for item in group)}


def _primary(strata, prediction):
    events = defaultdict(list)
    for (event, dates, slots), group in sorted(strata.items()):
        events[event].append({'dates': list(dates), 'slots': list(slots), **_binary_group(group, prediction, 'hard_neg')})
    output = {}
    for event, groups in sorted(events.items()):
        output[event] = {key: float(np.mean([group[key] for group in groups])) for key in ('ba', 'recall', 'specificity', 'fpr')}
        output[event].update({'n_strata': len(groups), 'n_pos': sum(group['n_pos'] for group in groups),
                             'n_hard_neg': sum(group['n_negative'] for group in groups),
                             'parse_failures': sum(group['parse_failures'] for group in groups), 'strata': groups})
    return {'macro_ba': float(np.mean([group['ba'] for group in output.values()])), 'n_events': len(output),
            'n_strata': len(strata), 'n_items': sum(len(group) for group in strata.values()), 'events': output}


def _paired_source(test, prediction, phen):
    groups = defaultdict(list)
    for item in test:
        if item['phen'] == phen and item['kind'] != 'hard_neg':
            groups[str(item['cluster'])].append(item)
    events = {event: _binary_group(group, prediction, 'neg') for event, group in sorted(groups.items())}
    return {'macro_ba': float(np.mean([group['ba'] for group in events.values()])), 'n_events': len(events),
            'n_items': sum(len(group) for group in groups.values()), 'events': events}


def _hard_negative(test, prediction):
    groups = defaultdict(list)
    for item in test:
        if item['phen'] == 'flood' and item['kind'] == 'hard_neg':
            groups[str(item['cluster'])].append(item)
    events = {event: {'n': len(group), 'fpr': sum(prediction[item['id']] == 'yes' for item in group) / len(group),
                      'yes_count': sum(prediction[item['id']] == 'yes' for item in group),
                      'parse_failures': sum(prediction[item['id']] is None for item in group)}
              for event, group in sorted(groups.items())}
    n = sum(group['n'] for group in events.values())
    return {'n': n, 'n_events': len(events), 'events': events,
            'fpr_pooled': sum(group['yes_count'] for group in events.values()) / n,
            'fpr_event_macro': float(np.mean([group['fpr'] for group in events.values()])),
            'parse_failures': sum(group['parse_failures'] for group in events.values())}


def _contrasts(evaluations, with_interval):
    output = {}
    definitions = [('pair_minus_full', 'pair/native', 'full/native'),
                   ('later_minus_full', 'later/native', 'full/native'),
                   ('delta_minus_full', 'delta/native', 'full/native'),
                   ('full_no_delta_minus_full', 'full/full_no_delta', 'full/native'),
                   ('pair_minus_full_no_delta', 'pair/native', 'full/full_no_delta')]
    for name, left, right in definitions:
        lhs, rhs = evaluations[left]['events'], evaluations[right]['events']
        require(set(lhs) == set(rhs), 'Comparison event support differs')
        changes = {event: lhs[event]['ba'] - rhs[event]['ba'] for event in sorted(lhs)}
        output[name] = {'delta': float(np.mean(list(changes.values()))), 'event_deltas': changes,
                        'ci95_delta': interval(list(changes.values())) if with_interval else None,
                        'left': left, 'right': right}
    return output


def _score(rows, items, eval_sets):
    source, test, strata = _prepare(items, eval_sets)
    answers, parse_rates = _index(rows, items, source, test)
    metrics, decisions = {}, {}
    for seed in SEEDS:
        primary, paired, hard = {}, {'flood': {}, 'landslide': {}}, {}
        for arm, evaluation in EVALUATIONS:
            key = f'{arm}/{evaluation}'
            prediction = {item['id']: answers[(seed, arm, evaluation, item['id'])] for item in test}
            primary[key] = _primary(strata, prediction)
            for phen in paired:
                paired[phen][key] = _paired_source(test, prediction, phen)
            hard[key] = _hard_negative(test, prediction)
        primary_contrasts = _contrasts(primary, True)
        metrics[str(seed)] = {
            'primary_same_prompt': {'evaluations': primary, 'contrasts': primary_contrasts},
            'paired_source': {phen: {'evaluations': values, 'contrasts': _contrasts(values, phen == 'flood')}
                              for phen, values in paired.items()},
            'hard_negative': hard,
        }
        full, pair = primary['full/native'], primary['pair/native']
        delta = primary_contrasts['pair_minus_full']
        eligible = full['n_events'] >= 5 and full['macro_ba'] >= .60 - EPS
        preserves = eligible and pair['macro_ba'] >= .60 - EPS and delta['ci95_delta'][0] >= -.05 - EPS
        helps = eligible and delta['delta'] <= -.10 + EPS and delta['ci95_delta'][1] < -EPS
        decisions[str(seed)] = {'eligible': bool(eligible), 'pair_preserves': bool(preserves),
                               'explicit_difference_helps': bool(helps)}
    preserves = [seed for seed in SEEDS if decisions[str(seed)]['pair_preserves']]
    helps = [seed for seed in SEEDS if decisions[str(seed)]['explicit_difference_helps']]
    verdict = ('pair_preserves_source_discrimination_at_equal_budget' if len(preserves) >= 2 else
               'explicit_difference_helps_at_this_budget' if len(helps) >= 2 else 'mixed_or_inconclusive')
    return {'schema': 'e5-equal-budget-scores-v0', 'valid': True, 'verdict': verdict,
            'coverage': {'expected': 26325, 'received': len(rows), 'n_items': 5989, 'n_train': 4234, 'n_test': 1755, 'n_primary': 902},
            'parse_fail_rates': parse_rates, 'metrics': metrics, 'seed_decisions': decisions,
            'pair_preserves_seeds': preserves, 'explicit_difference_helps_seeds': helps,
            'interpretation': 'Exploratory source-label discrimination on exposed E2/C1 development data. Only trained pair versus trained full is primary; later/delta and frozen full_no_delta are descriptive. D contains both observations; no memory-necessity or physical counterfactual claim.'}


def score_run(rows, items, eval_sets):
    """Return valid=False without changing a failed population or its thresholds."""
    try:
        return _score(rows, items, eval_sets)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError) as error:
        return {'schema': 'e5-equal-budget-scores-v0', 'valid': False, 'verdict': 'invalid', 'invalid_reason': str(error)}
