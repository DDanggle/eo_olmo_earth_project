"""Standalone E6 full-head versus fixed E5 full/native descriptive scorer.

Pure in-memory scoring only: no files, fitting, GPU, model imports, E5 source
imports, or result selection. The runner separately verifies frozen bytes,
completed training, and independent E5 audit. Population/metric helper logic
retains E5's source-label and unparsed-response contracts.
"""
from collections import Counter, defaultdict
import math
import re

import numpy as np

SEEDS = (1, 2, 3)
EPS = 1e-12
SYSTEMS = ('full/native', 'full_head/native')

def require(value, message):
    if not value:
        raise ValueError(message)


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


def _sigmoid(logit):
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    value = math.exp(logit)
    return value / (1.0 + value)


def _index_system(rows, items, source, test, is_head):
    require(isinstance(rows, list) and len(rows) == 5265, 'Expected exactly 5265 rows per system')
    indices = {item['id']: index for index, item in enumerate(items)}
    expected = {(seed, item['id']) for seed in SEEDS for item in test}
    answers = {}
    for row in rows:
        require(isinstance(row, dict) and type(row.get('seed')) is int, 'Invalid row/seed type')
        require(row.get('model_arm') == ('full_head' if is_head else 'full') and row.get('eval_arm') == 'native',
                'Only full_head/native or fixed reference full/native is allowed')
        key = (row['seed'], row.get('id'))
        require(key in expected and key not in answers, 'Unexpected/duplicate row, seed or train ID')
        item = source[row['id']]
        require(all(row.get(k) == item[k] for k in ('tile', 'phen', 'kind'))
                and str(row.get('cluster')) == str(item['cluster']), 'Output/source metadata mismatch')
        require(type(row.get('pair_index')) is int and row['pair_index'] == indices[item['id']],
                'Pair index differs from original global item order')
        require(row.get('source_gold') == item['answer'] and 'transformed_gold' in row
                and row['transformed_gold'] is None, 'Changed source label or assigned transformed physical gold')
        if is_head:
            for field in ('logit', 'probability'):
                require(type(row.get(field)) in (int, float) and math.isfinite(row[field]), 'Nonfinite/non-numeric head ' + field)
            require(0 <= row['probability'] <= 1 and abs(row['probability'] - _sigmoid(row['logit'])) <= 1e-6,
                    'Head probability differs from sigmoid(logit)')
            predicted = 'yes' if row['logit'] >= 0 else 'no'
            require(row.get('prediction') == predicted, 'Head prediction differs from fixed logit>=0 threshold')
        else:
            require('parsed' in row and row['parsed'] in ('yes', 'no', None) and isinstance(row.get('answer_raw'), str),
                    'Invalid reference parsed/raw answer')
            match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
            predicted = match[1] if match else None
            require(row['parsed'] == predicted, 'Reference raw/parsed answer mismatch')
        answers[key] = predicted
    require(set(answers) == expected, 'Incomplete same-seed test coverage')
    parse_rates = {}
    for seed in SEEDS:
        for phen in ('flood', 'landslide'):
            group = [item for item in test if item['phen'] == phen]
            failed = sum(answers[(seed, item['id'])] is None for item in group)
            rate = failed / len(group)
            require(rate <= .01 + EPS, f'Reference parse failure exceeds .01: {seed}/{phen}')
            parse_rates[f'{seed}|{phen}'] = {'n': len(group), 'failed': failed, 'rate': rate}
    return answers, parse_rates


def _contrast(reference, head, with_interval):
    left, right = reference['events'], head['events']
    require(set(left) == set(right), 'Comparison event support differs')
    differences = {event: left[event]['ba'] - right[event]['ba'] for event in sorted(left)}
    return {'delta': float(np.mean(list(differences.values()))), 'event_deltas': differences,
            'ci95_delta': interval(list(differences.values())) if with_interval else None,
            'left': 'full/native', 'right': 'full_head/native', 'direction': 'E5_full_minus_E6_head'}


def _score(head_rows, reference_rows, items, eval_sets):
    source, test, strata = _prepare(items, eval_sets)
    # Frozen evaluator lists retain the C0 item order. Row arrival order may differ,
    # but every row's pair_index must still identify that original global index.
    for name, ids in eval_sets.items():
        members = set(ids)
        require(ids == [item['id'] for item in test if item['id'] in members], 'Evaluation item order differs: ' + name)
    for phen, train_n, test_n in (('flood', 27, 10), ('landslide', 7, 2)):
        groups = {part: {str(item['cluster']) for item in items if item['phen'] == phen and item['partition'] == part}
                  for part in ('train', 'test')}
        require(len(groups['train']) == train_n and len(groups['test']) == test_n
                and not groups['train'] & groups['test'], 'Train/test event or region support differs: ' + phen)
    head, head_parse = _index_system(head_rows, items, source, test, True)
    reference, reference_parse = _index_system(reference_rows, items, source, test, False)
    metrics = {}
    for seed in SEEDS:
        predictions = {'full/native': {item['id']: reference[(seed, item['id'])] for item in test},
                       'full_head/native': {item['id']: head[(seed, item['id'])] for item in test}}
        primary = {name: _primary(strata, values) for name, values in predictions.items()}
        paired = {phen: {name: _paired_source(test, values, phen) for name, values in predictions.items()}
                  for phen in ('flood', 'landslide')}
        metrics[str(seed)] = {
            'primary_same_prompt': {'evaluations': primary,
                                   'contrasts': {'full_minus_head': _contrast(primary['full/native'], primary['full_head/native'], True)}},
            'paired_source': {phen: {'evaluations': values,
                                    'contrasts': {'full_minus_head': _contrast(values['full/native'], values['full_head/native'], False)}}
                              for phen, values in paired.items()},
            'hard_negative': {name: _hard_negative(test, values) for name, values in predictions.items()},
        }
    return {'schema': 'e6-no-llm-scores-v0', 'valid': True, 'verdict': 'descriptive_system_comparison',
            'coverage': {'expected_per_system': 5265, 'received_head': len(head_rows), 'received_reference': len(reference_rows),
                         'n_items': len(items), 'n_train': 4234, 'n_test': len(test), 'n_primary': 902, 'seeds': list(SEEDS)},
            'parse_fail_rates': {'full/native': reference_parse, 'full_head/native': head_parse},
            'metrics': metrics,
            'bootstrap': {'scope': 'primary contrast separately per seed only', 'unit': 'paired whole event', 'n_events': 8,
                          'draws': 5000, 'rng_seed': 20260925, 'quantile_method': 'linear', 'event_order': 'sorted string event IDs'},
            'interpretation': 'Descriptive source-label system comparison on exposed development data, with fixed E5 full/native as reference. No success threshold, seed-averaged interval, best-arm selection, equivalence test, isolated LLM-effect estimate, physical change-onset or memory-necessity claim. Training completion, hashes and independent E5 validity are external runner gates.'}


def score_run(head_rows, reference_rows, items, eval_sets):
    """Return invalid without excluding a row, repairing labels, or choosing an arm."""
    try:
        return _score(head_rows, reference_rows, items, eval_sets)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError, OverflowError) as error:
        return {'schema': 'e6-no-llm-scores-v0', 'valid': False, 'verdict': 'invalid', 'invalid_reason': str(error)}
