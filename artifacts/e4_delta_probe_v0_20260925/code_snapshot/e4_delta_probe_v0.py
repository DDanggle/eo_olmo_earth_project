"""Pure E4 difference-block interventions and pre-result exploratory scoring.

D=B-A already contains information from BOTH observation slots. Sufficiency of
[0,0,D] cannot establish that historical observations are unnecessary.
No model, filesystem, label editing, inference, or training occurs in this module.
"""
from collections import defaultdict
import re
import numpy as np

ARMS = ('real', 'delta_only', 'delta_sign_flip', 'delta_feature_permute')
SEEDS = (1, 2, 3)
FEATURE_SHIFT = 257
EPS = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_permutation():
    """One predeclared bijection with no fixed feature coordinates; no RNG/search."""
    return (np.arange(768, dtype=np.int64) + FEATURE_SHIFT) % 768


def transform_pair(pair, arm):
    require(isinstance(arm, str) and arm in ARMS, 'Unknown E4 arm')
    require(isinstance(pair, np.ndarray) and pair.shape == (2, 64, 768), 'Expected pair shape (2,64,768)')
    require(pair.dtype in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')),
            'Expected float16/float32/float64 pair')
    require(np.isfinite(pair).all(), 'Nonfinite source pair')
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.subtract(pair[1], pair[0])
    require(np.isfinite(delta).all(), 'Nonfinite difference block')
    tokens = np.zeros((192, 768), dtype=pair.dtype)
    if arm != 'delta_only':
        tokens[:64], tokens[64:128] = pair[0], pair[1]
    if arm == 'delta_sign_flip':
        tokens[128:] = -delta
    elif arm == 'delta_feature_permute':
        tokens[128:] = delta[:, feature_permutation()]
    else:
        tokens[128:] = delta
    return tokens, np.repeat(np.array([0, 1, 3], dtype=np.int64), 64)


def interval(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), 'Invalid event contrast')
    indices = np.random.default_rng(20260925).integers(0, len(values), (5000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975], method='linear').tolist()


def _score(rows, items, references):
    """All four arms apply to every retained E3 item, including hard negatives.

    ``references`` contains e2_real/e3_real, each mapping string seed to an
    exact item-ID -> parsed yes/no/null map. Runner verifies their byte hashes
    and executes all real arms before generating any E4 intervention.
    """
    source = {item['id']: item for item in items}
    require(len(source) == len(items) and items, 'Empty or duplicate source items')
    pairs = defaultdict(list)
    for item in items:
        require(item['phen'] in ('flood', 'landslide'), 'Unknown phenomenon')
        require(item['kind'] in ('pos', 'neg', 'hard_neg') and item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'),
                'Source kind/answer mismatch')
        if item['kind'] != 'hard_neg':
            pairs[(item['phen'], item['tile'])].append(item)
        else:
            require(item['phen'] == 'flood', 'Non-flood hard negative')
    for pair in pairs.values():
        require(len(pair) == 2 and {item['kind'] for item in pair} == {'pos', 'neg'}
                and len({str(item['cluster']) for item in pair}) == 1, 'Incomplete source pair')
    require(all((item['phen'], item['tile']) not in pairs for item in items if item['kind'] == 'hard_neg'),
            'Hard-negative/paired-tile overlap')
    expected = {(seed, arm, item['id']) for seed in SEEDS for arm in ARMS for item in items}
    answers = {}
    for row in rows:
        require(type(row.get('seed')) is int, 'Seed must be integer')
        key = (row['seed'], row.get('arm'), row.get('id'))
        require(key in expected and key not in answers, 'Unexpected or duplicate output key')
        item = source[row['id']]
        require(all(row.get(k) == item[k] for k in ('tile', 'phen', 'kind'))
                and str(row.get('cluster')) == str(item['cluster']), 'Output metadata differs')
        require(row.get('source_gold') == item['answer'] and 'transformed_gold' in row and row['transformed_gold'] is None,
                'Original source gold changed or transformed-scene gold assigned')
        require('parsed' in row and row['parsed'] in ('yes', 'no', None), 'Invalid parsed output')
        require(isinstance(row.get('answer_raw'), str), 'Missing raw answer')
        match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (match.group(1) if match else None), 'Raw/parsed answer mismatch')
        answers[key] = row['parsed']
    require(set(answers) == expected and len(rows) == len(expected), 'Missing output coverage')
    require(set(references) == {'e2_real', 'e3_real'}, 'Both frozen E2 and E3 real references are required')
    for name, ref in references.items():
        require(set(ref) == {'1', '2', '3'}, 'Reference seed coverage differs: ' + name)
        require(all(set(ref[seed]) == set(source) and all(v in ('yes', 'no', None) for v in ref[seed].values())
                    for seed in ref), 'Reference item coverage differs: ' + name)
    metrics, reproduction, parse_rates, decisions = {}, {}, {}, {}
    for seed in SEEDS:
        metrics[str(seed)], reproduction[str(seed)] = {}, {}
        for phen in ('flood', 'landslide'):
            all_items = [item for item in items if item['phen'] == phen]
            require(all_items, 'Missing phenomenon')
            paired = [item for item in all_items if item['kind'] != 'hard_neg']
            events = sorted({str(item['cluster']) for item in paired})
            require(events, 'Missing paired event support')
            per_arm, contrasts, agreement, hard_metrics = {}, {}, {}, {}
            for arm in ARMS:
                failure = sum(answers[(seed, arm, item['id'])] is None for item in all_items) / len(all_items)
                require(failure <= .01 + EPS, 'Parse failure exceeds .01')
                parse_rates[f'{seed}|{phen}|{arm}'] = failure
                event_values = {}
                for event in events:
                    pos = [item for item in paired if str(item['cluster']) == event and item['kind'] == 'pos']
                    neg = [item for item in paired if str(item['cluster']) == event and item['kind'] == 'neg']
                    recall = sum(answers[(seed, arm, item['id'])] == 'yes' for item in pos) / len(pos)
                    specificity = sum(answers[(seed, arm, item['id'])] == 'no' for item in neg) / len(neg)
                    event_values[event] = {'n_pos': len(pos), 'n_neg': len(neg), 'ba': (recall + specificity) / 2,
                                           'recall': recall, 'specificity': specificity}
                per_arm[arm] = {'events': event_values, 'n_events': len(events),
                                'macro_ba': float(np.mean([event_values[event]['ba'] for event in events]))}
                real = [answers[(seed, 'real', item['id'])] for item in all_items]
                current = [answers[(seed, arm, item['id'])] for item in all_items]
                matched = sum(a in ('yes', 'no') and a == b for a, b in zip(real, current))
                both_parsed = sum(a in ('yes', 'no') and b in ('yes', 'no') for a, b in zip(real, current))
                agreement[arm] = {'n_items': len(all_items), 'matched_valid_decisions': matched,
                    'n_both_parsed': both_parsed, 'agreement_all_items': matched / len(all_items),
                    'agreement_both_parsed': matched / both_parsed if both_parsed else None,
                    'agreement_ge_0_99_descriptive_only': matched / len(all_items) >= .99,
                    'source_yes_rate': sum(value == 'yes' for value in current) / len(current)}
                hard = [item for item in all_items if item['kind'] == 'hard_neg']
                if hard:
                    fprs = {event: sum(answers[(seed, arm, item['id'])] == 'yes' for item in hard if str(item['cluster']) == event)
                            / sum(str(item['cluster']) == event for item in hard)
                            for event in sorted({str(item['cluster']) for item in hard})}
                    hard_metrics[arm] = {'n': len(hard), 'fpr_by_event': fprs,
                        'fpr_event_macro': float(np.mean(list(fprs.values()))),
                        'fpr_pooled': sum(answers[(seed, arm, item['id'])] == 'yes' for item in hard) / len(hard),
                        'parse_failures': sum(answers[(seed, arm, item['id'])] is None for item in hard)}
            for arm in ARMS:
                changes = {event: per_arm[arm]['events'][event]['ba'] - per_arm['real']['events'][event]['ba'] for event in events}
                contrasts[arm] = {'event_deltas': changes, 'delta': float(np.mean(list(changes.values()))),
                                  'ci95_delta': interval(list(changes.values())) if phen == 'flood' else None}
            reproduction[str(seed)][phen] = {}
            for name, reference in references.items():
                rate = sum(answers[(seed, 'real', item['id'])] == reference[str(seed)][item['id']] for item in all_items) / len(all_items)
                require(rate >= .99, 'Real reproduction below .99: ' + name)
                reproduction[str(seed)][phen][name] = {'n': len(all_items), 'reproduction_rate': rate}
            metrics[str(seed)][phen] = {'paired': per_arm, 'contrasts': contrasts,
                                       'agreement_with_real': agreement, 'hard_negative': hard_metrics}
        flood = metrics[str(seed)]['flood']
        real = flood['paired']['real']
        delta = flood['contrasts']['delta_only']
        eligible = real['n_events'] >= 5 and real['macro_ba'] >= .60 - EPS
        sufficient = eligible and flood['paired']['delta_only']['macro_ba'] >= .60 - EPS and delta['ci95_delta'][0] >= -.05 - EPS
        degraded = eligible and delta['delta'] <= -.10 + EPS and delta['ci95_delta'][1] < -EPS
        decisions[str(seed)] = {'eligible': bool(eligible), 'delta_only_sufficient': bool(sufficient),
                               'delta_only_degrades': bool(degraded)}
    passing = [seed for seed in SEEDS if decisions[str(seed)]['delta_only_sufficient']]
    degrading = [seed for seed in SEEDS if decisions[str(seed)]['delta_only_degrades']]
    verdict = ('difference_block_sufficient_under_intervention' if len(passing) >= 2 else
               'delta_only_degrades_source_agreement' if len(degrading) >= 2 else 'mixed_or_inconclusive')
    return {'schema': 'e4-delta-probe-scores-v0', 'valid': True, 'verdict': verdict,
            'coverage': {'expected': len(expected), 'received': len(rows)}, 'parse_fail_rates': parse_rates,
            'reproduction': reproduction, 'metrics': metrics, 'seed_decisions': decisions,
            'sufficient_seeds': passing, 'degraded_seeds': degrading,
            'interpretation': 'Post-E3 exploratory source-label agreement. D contains both observations; no history-unnecessary or physical counterfactual claim.'}


def score_run(rows, items, references):
    try:
        return _score(rows, items, references)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        return {'schema': 'e4-delta-probe-scores-v0', 'valid': False, 'verdict': 'invalid', 'invalid_reason': str(error)}
