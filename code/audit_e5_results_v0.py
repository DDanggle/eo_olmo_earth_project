#!/usr/bin/env python3
"""Independent E5 postrun audit; no production module or model is executed.

Default dependencies are stdlib and NumPy. --check-tensors additionally uses
CPU torch.load(weights_only=True) to inspect saved initial/final checkpoints.
The primary same-prompt statistic and decision are recomputed independently.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ARMS = ('full', 'pair', 'later', 'delta')
SEEDS = (1, 2, 3)
EVALUATIONS = tuple((arm, 'native') for arm in ARMS) + (('full', 'full_no_delta'),)
CORE_SHA = '2eb0be5e7fb8ac5d6534716a25c9a07c8f046b56ea8a55c7f2912b51b7e6af7d'
REVIEWED_SOURCES = {
    'e5_prepare_v0.py': '508f444a512fcbd3f6937dd51b8ae2755c1f4deb334bdaed118c98e78840fd61',
    'e5_train_v0.py': '3f33873711e6b94a8bb54d50ee4dbd5b5492da22c41d848ef4f664c0894b398b',
    'e5_scoring_v0.py': CORE_SHA,
    'run_e5_when_idle_v0.py': '4701462c818117683eb3587e83d91eea4629fcfc381d50b9341f5b12cf386eb6',
}
SCIENCE_SHA = 'bff66ca45d0fac6f64f276aa0a31506d3a6c56fea799c413481ccaa5882bbbad'
SCIENCE_KEYS = ('population', 'inputs', 'arms', 'training', 'evaluation', 'primary_analysis',
                'secondary_descriptive', 'validity', 'parents', 'pool_reproduction')
EPS = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('Nonfinite JSON token: ' + value)


def loads(text):
    return json.loads(text, parse_constant=invalid_constant, object_pairs_hook=object_pairs)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def unique(rows):
    by_id = {row['id']: row for row in rows}
    require(len(rows) == len(by_id), 'Duplicate item ID')
    return by_id


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def compare(actual, expected, location='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Object support differs: ' + location)
        for key, value in expected.items():
            compare(actual[key], value, location + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Array support differs: ' + location)
        for index, value in enumerate(expected):
            compare(actual[index], value, location + '/' + str(index))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), 'Number differs: ' + location)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value/type differs: ' + location)


def verify(artifact, relative, expected, tracked):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe artifact path')
    destination = (artifact / path).resolve()
    require(destination.is_relative_to(artifact.resolve()), 'Artifact symlink escapes root')
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Malformed SHA256')
    require(sha(destination) == expected, 'File SHA differs: ' + str(relative))
    tracked[str(destination)] = expected
    return destination


def track(artifact, relative, tracked):
    return verify(artifact, relative, sha(artifact / relative), tracked)


def expected_schedule(ids, seed):
    rng = np.random.default_rng(seed)
    schedule = []
    for epoch in range(3):
        order = [ids[int(index)] for index in rng.permutation(len(ids))]
        schedule.append([order[index:index + 8] for index in range(0, len(order), 8)])
    return schedule


def validate_population(items, ordered, eval_sets, batches):
    source = unique(items)
    require(len(source) == 5989 and set(ordered) == {'train', 'test'}, 'Prepared item/partition support differs')
    expected_counts = {('train', 'flood', 'pos'): 1066, ('train', 'flood', 'neg'): 1066,
        ('train', 'flood', 'hard_neg'): 1066, ('train', 'landslide', 'pos'): 518, ('train', 'landslide', 'neg'): 518,
        ('test', 'flood', 'pos'): 457, ('test', 'flood', 'neg'): 457, ('test', 'flood', 'hard_neg'): 457,
        ('test', 'landslide', 'pos'): 192, ('test', 'landslide', 'neg'): 192}
    require(Counter((row['partition'], row['phen'], row['kind']) for row in items) == expected_counts, 'Source category counts differ')
    tile_groups = defaultdict(list)
    for item in items:
        require(item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Source kind/answer differs')
        require(isinstance(item['dates'], list) and len(item['dates']) == 2, 'Source dates differ')
        if item['phen'] == 'flood':
            require(item['slots'] == (['pre_1', 'pre_2'] if item['kind'] == 'neg' else ['pre_2', 'post'])
                    and str(item['event']) == str(item['cluster']), 'Source flood slots/event differs')
        tile_groups[(item['phen'], item['tile'])].append(item)
    for group in tile_groups.values():
        require(len({item['partition'] for item in group}) == 1, 'Train/test tile overlap')
        kinds = [item['kind'] for item in group]
        require(kinds == ['hard_neg'] or (len(kinds) == 2 and set(kinds) == {'pos', 'neg'}), 'Broken paired source tile')
    for split, count in (('train', 4234), ('test', 1755)):
        require(ordered[split] == [item['id'] for item in items if item['partition'] == split]
                and len(ordered[split]) == count, 'Frozen global source order differs')
    for phen, support in (('flood', (27, 10)), ('landslide', (7, 2))):
        groups = [{str(item['cluster']) for item in items if item['partition'] == split and item['phen'] == phen}
                  for split in ('train', 'test')]
        require(groups[0].isdisjoint(groups[1]) and tuple(map(len, groups)) == support, 'Train/test cluster support differs')
    require(set(batches) == {'1', '2', '3'}, 'Training schedule seed support differs')
    for seed in SEEDS:
        compare(batches[str(seed)], expected_schedule(ordered['train'], seed), f'batches/{seed}')
    test = [source[item_id] for item_id in ordered['test']]
    expected_sets = {'all_test': ordered['test'],
                    'paired_flood': [item['id'] for item in test if item['phen'] == 'flood' and item['kind'] != 'hard_neg'],
                    'hard_negative_flood': [item['id'] for item in test if item['kind'] == 'hard_neg'],
                    'landslide': [item['id'] for item in test if item['phen'] == 'landslide']}
    require(set(eval_sets) == set(expected_sets) | {'primary_same_prompt', 'e3_subset'}, 'Evaluation-set names differ')
    for name, values in expected_sets.items():
        compare(eval_sets[name], values, 'eval_sets/' + name)
    for name, count in (('primary_same_prompt', 902), ('e3_subset', 209)):
        require(len(eval_sets[name]) == len(set(eval_sets[name])) == count
                and set(eval_sets[name]) <= set(ordered['test']), 'Evaluation ID support differs: ' + name)
    primary = [source[item_id] for item_id in eval_sets['primary_same_prompt']]
    require(Counter((item['phen'], item['kind']) for item in primary) == {('flood', 'pos'): 445, ('flood', 'hard_neg'): 457},
            'Primary source classes differ')
    require(len({str(item['cluster']) for item in primary}) == 8, 'Expected eight primary events')
    return source, test, primary


def validate_selection(artifact, items, eval_sets):
    source = unique(items)
    quality = unique(lines(artifact / 'parent_snapshot/c1_quality.jsonl'))
    target = {item['id']: item for item in items if item['partition'] == 'test' and item['phen'] == 'flood'
              and item['kind'] in ('pos', 'hard_neg')}
    require(set(quality) == set(target), 'Quality source coverage differs')
    strata = defaultdict(list)
    for item_id, row in quality.items():
        item = target[item_id]
        require(all(row[field] == item[field] for field in ('tile', 'kind', 'dates', 'slots'))
                and str(row['event']) == str(item['cluster']) and row['source_answer'] == item['answer'], 'Quality source metadata differs')
        require(type(row['eligible_symmetric_quality']) is bool
                and row['eligible_symmetric_quality'] == (row['valid_frac'] >= .9), 'Quality selection flag differs')
        if row['eligible_symmetric_quality']:
            strata[(str(item['cluster']), tuple(item['dates']), tuple(item['slots']))].append(item)
    supported = {item['id'] for group in strata.values() if {item['kind'] for item in group} == {'pos', 'hard_neg'} for item in group}
    require(supported == set(eval_sets['primary_same_prompt']), 'Primary selection differs from source-only quality/support rule')
    previous = unique(lines(artifact / 'parent_snapshot/e3_items.jsonl'))
    require(set(previous) == set(eval_sets['e3_subset']), 'E3 subset selection differs')
    for item_id, item in previous.items():
        require(all(item[field] == source[item_id][field] for field in ('tile', 'phen', 'kind', 'dates', 'indices', 'answer', 'cluster')),
                'E3 subset source metadata differs')


def index_predictions(items, rows):
    source = unique(items)
    indices = {item['id']: index for index, item in enumerate(items)}
    test = [item for item in items if item['partition'] == 'test']
    expected = {(seed, arm, evaluation, item['id']) for seed in SEEDS for arm, evaluation in EVALUATIONS for item in test}
    indexed = {}
    for row in rows:
        require(type(row.get('seed')) is int, 'Invalid seed type')
        key = (row['seed'], row.get('model_arm'), row.get('eval_arm'), row.get('id'))
        require(key in expected and key not in indexed, 'Unexpected/duplicate row or evaluation train ID')
        item = source[row['id']]
        require(all(row.get(field) == item[field] for field in ('tile', 'phen', 'kind'))
                and str(row.get('cluster')) == str(item['cluster']), 'Answer source metadata differs')
        require(type(row.get('pair_index')) is int and row['pair_index'] == indices[row['id']], 'Global pair index differs')
        require(row.get('source_gold') == item['answer'] and 'transformed_gold' in row and row['transformed_gold'] is None,
                'Source gold changed or physical counterfactual gold assigned')
        require(isinstance(row.get('answer_raw'), str) and row.get('parsed') in ('yes', 'no', None) and 'parsed' in row,
                'Invalid answer representation')
        match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (match[1] if match else None), 'Saved parsed answer differs from raw text')
        indexed[key] = row
    require(set(indexed) == expected and len(rows) == 26325, 'Incomplete 26325-row coverage')
    rates = {}
    for seed in SEEDS:
        for arm, evaluation in EVALUATIONS:
            for phen in ('flood', 'landslide'):
                group = [item for item in test if item['phen'] == phen]
                failed = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] is None for item in group)
                require(failed / len(group) <= .01 + EPS, 'Parse failure exceeds .01')
                rates[f'{seed}|{arm}|{evaluation}|{phen}'] = {'n': len(group), 'failed': failed, 'rate': failed / len(group)}
    return indexed, rates


def event_interval(differences):
    values = np.asarray(differences, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), 'Invalid event contrast')
    indices = np.random.default_rng(20260925).integers(len(values), size=(5000, len(values)))
    return np.quantile(np.mean(values[indices], axis=1), [.025, .975], method='linear').tolist()


def primary_metrics(primary, indexed):
    strata = defaultdict(list)
    for item in primary:
        strata[(str(item['cluster']), tuple(item['dates']), tuple(item['slots']))].append(item)
    for group in strata.values():
        require({item['kind'] for item in group} == {'pos', 'hard_neg'}, 'Unsupported primary stratum')
    result, decisions = {}, {}
    for seed in SEEDS:
        evaluations = {}
        for arm, evaluation in EVALUATIONS:
            event_strata = defaultdict(list)
            for (event, dates, slots), group in sorted(strata.items()):
                groups = {kind: [item for item in group if item['kind'] == kind] for kind in ('pos', 'hard_neg')}
                correct = {kind: sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == item['answer'] for item in subset)
                           for kind, subset in groups.items()}
                recall, specificity = correct['pos'] / len(groups['pos']), correct['hard_neg'] / len(groups['hard_neg'])
                fp = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == 'yes' for item in groups['hard_neg']) / len(groups['hard_neg'])
                event_strata[event].append({'dates': list(dates), 'slots': list(slots), 'n_pos': len(groups['pos']),
                    'n_negative': len(groups['hard_neg']), 'negative_kind': 'hard_neg', 'recall': recall,
                    'specificity': specificity, 'fpr': fp, 'ba': (recall + specificity) / 2,
                    'parse_failures': sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] is None for item in group)})
            events = {}
            for event, groups in sorted(event_strata.items()):
                events[event] = {key: float(np.mean([group[key] for group in groups])) for key in ('ba', 'recall', 'specificity', 'fpr')}
                events[event].update(n_strata=len(groups), n_pos=sum(group['n_pos'] for group in groups),
                    n_hard_neg=sum(group['n_negative'] for group in groups),
                    parse_failures=sum(group['parse_failures'] for group in groups), strata=groups)
            evaluations[f'{arm}/{evaluation}'] = {'macro_ba': float(np.mean([event['ba'] for event in events.values()])),
                'n_events': len(events), 'n_strata': len(strata), 'n_items': len(primary), 'events': events}
        definitions = [('pair_minus_full', 'pair/native', 'full/native'), ('later_minus_full', 'later/native', 'full/native'),
                       ('delta_minus_full', 'delta/native', 'full/native'), ('full_no_delta_minus_full', 'full/full_no_delta', 'full/native'),
                       ('pair_minus_full_no_delta', 'pair/native', 'full/full_no_delta')]
        contrasts = {}
        for name, left, right in definitions:
            changes = {event: evaluations[left]['events'][event]['ba'] - evaluations[right]['events'][event]['ba']
                       for event in sorted(evaluations[left]['events'])}
            contrasts[name] = {'delta': float(np.mean(list(changes.values()))), 'event_deltas': changes,
                               'ci95_delta': event_interval(list(changes.values())), 'left': left, 'right': right}
        result[str(seed)] = {'evaluations': evaluations, 'contrasts': contrasts}
        full, pair, delta = evaluations['full/native'], evaluations['pair/native'], contrasts['pair_minus_full']
        eligible = full['n_events'] >= 5 and full['macro_ba'] >= .60 - EPS
        preserves = eligible and pair['macro_ba'] >= .60 - EPS and delta['ci95_delta'][0] >= -.05 - EPS
        helps = eligible and delta['delta'] <= -.10 + EPS and delta['ci95_delta'][1] < -EPS
        decisions[str(seed)] = {'eligible': bool(eligible), 'pair_preserves': bool(preserves), 'explicit_difference_helps': bool(helps)}
    preserves = [seed for seed in SEEDS if decisions[str(seed)]['pair_preserves']]
    helps = [seed for seed in SEEDS if decisions[str(seed)]['explicit_difference_helps']]
    verdict = ('pair_preserves_source_discrimination_at_equal_budget' if len(preserves) >= 2 else
               'explicit_difference_helps_at_this_budget' if len(helps) >= 2 else 'mixed_or_inconclusive')
    return {'metrics': result, 'seed_decisions': decisions, 'pair_preserves_seeds': preserves,
            'explicit_difference_helps_seeds': helps, 'verdict': verdict}


def secondary_metrics(test, indexed):
    """Descriptive original pos/pre-negative BA and separate source hard FPR."""
    output = {}
    definitions = [('pair_minus_full', 'pair/native', 'full/native'), ('later_minus_full', 'later/native', 'full/native'),
                   ('delta_minus_full', 'delta/native', 'full/native'), ('full_no_delta_minus_full', 'full/full_no_delta', 'full/native'),
                   ('pair_minus_full_no_delta', 'pair/native', 'full/full_no_delta')]
    for seed in SEEDS:
        paired, hard = {}, {}
        for phen in ('flood', 'landslide'):
            groups = defaultdict(list)
            for item in test:
                if item['phen'] == phen and item['kind'] in ('pos', 'neg'):
                    groups[str(item['cluster'])].append(item)
            evaluations = {}
            for arm, evaluation in EVALUATIONS:
                events = {}
                for event, group in sorted(groups.items()):
                    pos = [item for item in group if item['kind'] == 'pos']
                    neg = [item for item in group if item['kind'] == 'neg']
                    require(pos and neg, 'Paired descriptive source class missing')
                    correct_pos = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == 'yes' for item in pos)
                    correct_neg = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == 'no' for item in neg)
                    false_pos = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == 'yes' for item in neg)
                    recall, specificity = correct_pos / len(pos), correct_neg / len(neg)
                    events[event] = {'n_pos': len(pos), 'n_negative': len(neg), 'negative_kind': 'neg',
                        'recall': recall, 'specificity': specificity, 'fpr': false_pos / len(neg),
                        'ba': (recall + specificity) / 2,
                        'parse_failures': sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] is None for item in group)}
                evaluations[f'{arm}/{evaluation}'] = {'macro_ba': float(np.mean([event['ba'] for event in events.values()])),
                    'n_events': len(events), 'n_items': sum(len(group) for group in groups.values()), 'events': events}
            contrasts = {}
            for name, left, right in definitions:
                differences = {event: evaluations[left]['events'][event]['ba'] - evaluations[right]['events'][event]['ba']
                               for event in sorted(groups)}
                contrasts[name] = {'delta': float(np.mean(list(differences.values()))), 'event_deltas': differences,
                    'ci95_delta': event_interval(list(differences.values())) if phen == 'flood' else None, 'left': left, 'right': right}
            paired[phen] = {'evaluations': evaluations, 'contrasts': contrasts}
        hard_groups = defaultdict(list)
        for item in test:
            if item['phen'] == 'flood' and item['kind'] == 'hard_neg':
                hard_groups[str(item['cluster'])].append(item)
        for arm, evaluation in EVALUATIONS:
            events = {}
            for event, group in sorted(hard_groups.items()):
                yes = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] == 'yes' for item in group)
                failed = sum(indexed[(seed, arm, evaluation, item['id'])]['parsed'] is None for item in group)
                events[event] = {'n': len(group), 'fpr': yes / len(group), 'yes_count': yes, 'parse_failures': failed}
            n = sum(event['n'] for event in events.values())
            hard[f'{arm}/{evaluation}'] = {'n': n, 'n_events': len(events), 'events': events,
                'fpr_pooled': sum(event['yes_count'] for event in events.values()) / n,
                'fpr_event_macro': float(np.mean([event['fpr'] for event in events.values()])),
                'parse_failures': sum(event['parse_failures'] for event in events.values())}
        output[str(seed)] = {'paired_source': paired, 'hard_negative': hard}
    return output


def validate_input_audit(artifact, plan, items, eval_sets):
    record = read(artifact / 'input_audit.json')
    require(record['no_exclusions'] is True and record['n_items'] == 5989 and record['n_unique_caches'] == 3756,
            'Prepared input audit population/exclusion claim differs')
    expected_pool = {'expected_items': 209, 'matched_items': 209, 'bit_exact': True,
                     'items_sha256': plan['pool_reproduction']['items_sha256'],
                     'pairs_sha256': plan['pool_reproduction']['pairs_sha256']}
    compare(record['pool_reproduction'], expected_pool, 'input_audit/pool_reproduction')
    compare(record['eval_set_counts'], {name: len(ids) for name, ids in eval_sets.items()}, 'input_audit/eval_set_counts')
    c0 = read(artifact / 'parent_snapshot/c0_manifest.json')
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[item['cache_path']].append((index, item))
    require(set(record['cache_sources']) == set(groups) == set(c0['cache_sha256']) == set(c0['cache_stat'])
            and len(groups) == 3756, 'Prepared cache-source support differs')
    for path, entries in groups.items():
        cache = record['cache_sources'][path]
        require(cache['sha256'] == c0['cache_sha256'][path], 'Prepared cache pin differs')
        compare(cache['before'], c0['cache_stat'][path], 'cache/before')
        compare(cache['after'], c0['cache_stat'][path], 'cache/after')
        compare(cache['item_rows'], [index for index, item in entries], 'cache/item_rows')
        compare(cache['indices_used'], sorted({frame for _, item in entries for frame in item['indices']}), 'cache/indices_used')
        require(cache['finite_scope'] == 'all referenced frames', 'Prepared finite-scope claim differs')


def audit_steps(records, schedule):
    flattened = [(epoch, batch_index, ids) for epoch, batches in enumerate(schedule) for batch_index, ids in enumerate(batches)]
    require(len(records) == len(flattened) == 1590, 'Expected exactly 1590 optimizer-step records')
    exposures, previous_elapsed = 0, 0.
    losses = defaultdict(list)
    for step, (record, (epoch, batch_index, ids)) in enumerate(zip(records, flattened), 1):
        exposures += len(ids)
        expected = {'step': step, 'epoch': epoch, 'batch_index': batch_index, 'ids': ids,
                    'batch_size': len(ids), 'exposures': exposures, 'finite_loss_grad_parameters': True}
        for key, value in expected.items():
            compare(record[key], value, 'step/' + str(step) + '/' + key)
        require(type(record['loss']) in (int, float) and math.isfinite(record['loss']), 'Nonfinite step loss')
        require(type(record['elapsed_s']) in (int, float) and math.isfinite(record['elapsed_s'])
                and record['elapsed_s'] >= previous_elapsed, 'Training step time is nonfinite or reverses')
        previous_elapsed = record['elapsed_s']
        losses[epoch].append(record['loss'])
    require(exposures == 12702 and sorted(losses) == [0, 1, 2], 'Expected exactly 12702 training exposures')
    return {'epoch_mean_batch_loss': [float(np.mean(losses[epoch])) for epoch in range(3)],
            'last_elapsed_s': previous_elapsed, 'updates': 1590, 'exposures': exposures}


def tensor_hash(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state, 'Invalid tensor state')
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        require(isinstance(key, str) and isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()), 'Invalid/nonfinite checkpoint tensor')
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
    return h.hexdigest(), sum(value.numel() for value in state.values())


def audit_training(artifact, manifest, batches, indexed, tracked, check_tensors=False):
    initial = read(track(artifact, 'initial_states/manifest.json', tracked))
    require(initial['all_generated_before_any_training'] is True and set(initial['seeds']) == {'1', '2', '3'}, 'Initial freeze record differs')
    expected_directories = {f'seed{seed}_{arm}' for seed in SEEDS for arm in ARMS}
    require({path.name for path in (artifact / 'models').iterdir()} == expected_directories, 'Completed model directory support differs')
    outcomes, total_steps = [], 0
    for seed in SEEDS:
        init = initial['seeds'][str(seed)]
        compare(read(track(artifact, f'initial_states/seed{seed}.json', tracked)), init, 'initial_seed_record')
        initial_path = verify(artifact, f'initial_states/seed{seed}.pt', init['file_sha256'], tracked)
        require(init['seed'] == seed and re.fullmatch(r'[0-9a-f]{64}', init['tensor_sha256']), 'Initial seed/hash differs')
        if check_tensors:
            state_hash, n_params = tensor_hash(initial_path)
            require(state_hash == init['tensor_sha256'] and n_params == init['parameter_count'], 'Canonical initial tensors differ')
        for arm in ARMS:
            relative = f'models/seed{seed}_{arm}'
            contract = read(track(artifact, relative + '/training_contract.json', tracked))
            record = read(track(artifact, relative + '/training_completed.json', tracked))
            outcome = read(track(artifact, relative + '/completed.json', tracked))
            for key in record:
                compare(outcome[key], record[key], relative + '/' + key)
            expected_contract = {'seed': seed, 'model_arm': arm, 'initial_file_sha256': init['file_sha256'],
                'initial_tensor_sha256': init['tensor_sha256'], 'epochs': 3, 'batch_size': 8,
                'expected_updates': 1590, 'expected_exposures': 12702,
                'ordered_ids_sha256': manifest['files_sha256']['ordered_ids.json'],
                'batches_sha256': manifest['files_sha256']['batches.json']}
            for key, value in expected_contract.items():
                compare(contract[key], value, relative + '/contract/' + key)
            groups = contract['optimizer_groups']
            require(len(groups) == 1 and groups[0]['lr'] == 1e-4 and groups[0]['weight_decay'] == .01
                    and groups[0]['betas'] == [.9, .999] and groups[0]['eps'] == 1e-8
                    and groups[0]['maximize'] is False, 'Optimizer contract differs')
            require(record['seed'] == seed and record['model_arm'] == arm and record['updates'] == 1590 and record['exposures'] == 12702,
                    'Completed training budget/identity differs')
            require(record['initial_file_sha256'] == init['file_sha256'] and record['initial_tensor_sha256'] == init['tensor_sha256'],
                    'Condition used a different initial state')
            verify(artifact, relative + '/initial.pt', init['file_sha256'], tracked)
            checkpoint = verify(artifact, relative + '/projector.pt', record['checkpoint_sha256'], tracked)
            if check_tensors:
                checkpoint_tensor_hash, n_params = tensor_hash(checkpoint)
                require(checkpoint_tensor_hash == record['checkpoint_tensor_sha256'] and n_params == init['parameter_count'],
                        'Final checkpoint tensor identity/count differs')
            step_path = verify(artifact, relative + '/steps.jsonl', record['steps_sha256'], tracked)
            calculated = audit_steps(lines(step_path), batches[str(seed)])
            compare(record['epoch_mean_batch_loss'], calculated['epoch_mean_batch_loss'], relative + '/epoch_losses')
            require(type(record['train_s']) in (int, float) and math.isfinite(record['train_s'])
                    and calculated['last_elapsed_s'] <= record['train_s'] <= 45 * 60, 'Training elapsed time differs')
            total_steps += calculated['updates']
            evaluation_arms = ['native', 'full_no_delta'] if arm == 'full' else ['native']
            require(set(outcome['answers_sha256']) == set(evaluation_arms) and set(outcome['parse']) == set(evaluation_arms), 'Completed evaluation support differs')
            for evaluation in evaluation_arms:
                answer_path = verify(artifact, relative + f'/answers_{evaluation}.jsonl', outcome['answers_sha256'][evaluation], tracked)
                records = lines(answer_path)
                require(len(records) == 1755, 'Per-model answer count differs')
                found = set()
                for row in records:
                    key = (seed, arm, evaluation, row['id'])
                    require(key in indexed and key not in found, 'Per-model answer coverage differs')
                    compare(row, indexed[key], relative + '/saved_answer/' + row['id'])
                    found.add(key)
                for phen in ('flood', 'landslide'):
                    subset = [row for row in records if row['phen'] == phen]
                    failed = sum(row['parsed'] is None for row in subset)
                    compare(outcome['parse'][evaluation][phen], {'n': len(subset), 'unparsed': failed, 'rate': failed / len(subset)},
                            relative + '/parse/' + evaluation + '/' + phen)
            compare(read(track(artifact, relative + '/parse_audit.json', tracked)), outcome['parse'], relative + '/parse_audit')
            require(type(outcome['model_elapsed_s']) in (int, float) and math.isfinite(outcome['model_elapsed_s'])
                    and record['train_s'] <= outcome['model_elapsed_s'] <= 45 * 60, 'Model time cap differs')
            require(type(outcome['eval_s']) in (int, float) and math.isfinite(outcome['eval_s'])
                    and 0 <= outcome['eval_s'] <= outcome['model_elapsed_s'], 'Evaluation time differs')
            outcomes.append(outcome)
    require(total_steps == 19080, 'Twelve-model optimizer-step total differs')
    compare(read(track(artifact, 'partial_models.json', tracked)), outcomes, 'partial_models')
    return outcomes


def audit(artifact, check_tensors=False):
    artifact = Path(artifact).resolve()
    tracked = {}
    manifest = read(track(artifact, 'manifest.json', tracked))
    status = read(track(artifact, 'status.json', tracked))
    scores = read(track(artifact, 'scores.json', tracked))
    require(status['status'] == 'completed' and scores['valid'] is True and not (artifact / 'failure.json').exists(), 'Require complete valid E5 result')
    require(manifest['schema'] == 'e5-equal-budget-prepared-v0', 'Manifest schema differs')
    for filename, expected in manifest['files_sha256'].items():
        verify(artifact, filename, expected, tracked)
    require(set(manifest['files_sha256']) == {'items.jsonl', 'pairs.npy', 'ordered_ids.json', 'batches.json', 'eval_sets.json', 'prereg.json', 'input_audit.json'},
            'Prepared file support differs')
    plan = read(artifact / 'prereg.json')
    require(canonical_hash({key: plan[key] for key in SCIENCE_KEYS}) == SCIENCE_SHA, 'Preregistered scientific contract differs')
    require(manifest['parent_snapshot_sha256'] == plan['parents'], 'Parent snapshot pins differ')
    for filename, expected in manifest['parent_snapshot_sha256'].items():
        verify(artifact, 'parent_snapshot/' + filename, expected, tracked)
    require(manifest['files_sha256']['items.jsonl'] == plan['parents']['items.jsonl'], 'C0 item bytes changed')
    require(set(manifest['code_snapshot_sha256']) == set(plan['source_files']), 'Source snapshot support differs')
    compare(manifest['code_snapshot_sha256'], REVIEWED_SOURCES, 'reviewed_source_versions')
    for filename, expected in manifest['code_snapshot_sha256'].items():
        verify(artifact, 'code_snapshot/' + filename, expected, tracked)
    require(manifest['code_snapshot_sha256']['e5_scoring_v0.py'] == CORE_SHA, 'Unreviewed scoring/transform code')
    e4 = read(artifact / 'parent_snapshot/e4_manifest.json')
    require(manifest['llm_files_sha256'] == e4['llm_files_sha256'], 'LLM/tokenizer source pins differ from E4')
    items, ordered = lines(artifact / 'items.jsonl'), read(artifact / 'ordered_ids.json')
    batches, eval_sets = read(artifact / 'batches.json'), read(artifact / 'eval_sets.json')
    source, test, primary = validate_population(items, ordered, eval_sets, batches)
    validate_selection(artifact, items, eval_sets)
    validate_input_audit(artifact, plan, items, eval_sets)
    pairs = np.load(artifact / 'pairs.npy', mmap_mode='r', allow_pickle=False)
    require(pairs.shape == (5989, 2, 64, 768) and pairs.dtype == np.float32, 'Prepared pair shape/dtype differs')
    for index, item in enumerate(items):
        require(np.isfinite(pairs[index]).all(), 'Nonfinite source pair: ' + item['id'])
        with np.errstate(over='ignore', invalid='ignore'):
            difference = pairs[index, 1] - pairs[index, 0]
        require(np.isfinite(difference).all(), 'Nonfinite source difference: ' + item['id'])
    del pairs
    summary = read(track(artifact, 'training_summary.json', tracked))
    inference = read(track(artifact, 'inference_completed.json', tracked))
    require(summary['n_models'] == inference['n_models'] == 12 and inference['n_rows'] == 26325
            and inference['n_prompts'] == 5989, 'Final training/inference counts differ')
    require(summary['updates_per_model'] == 1590 and summary['exposures_per_model'] == 12702, 'Final training budget differs')
    require(summary['manifest_sha256'] == scores['manifest_sha256'] == sha(artifact / 'manifest.json')
            and summary['code_snapshot_sha256'] == manifest['code_snapshot_sha256'], 'Final freeze identity differs')
    require(scores['training_summary_sha256'] == sha(artifact / 'training_summary.json'), 'Scores use different training summary')
    verify(artifact, 'initial_states/manifest.json', summary['initial_manifest_sha256'], tracked)
    for filename, key in (('predictions.jsonl', 'predictions_sha256'), ('prompts.jsonl', 'prompts_sha256'), ('runtime_environment.json', 'runtime_sha256')):
        verify(artifact, filename, summary[key], tracked)
        require(summary[key] == inference[key], 'Inference/summary file identity differs')
        if key in scores:
            require(scores[key] == summary[key], 'Score file identity differs')
    runtime = read(artifact / 'runtime_environment.json')
    require(runtime['manifest_sha256'] == summary['manifest_sha256']
            and runtime['train_order_is_historical_e2_replay'] is False, 'Runtime provenance/replay claim differs')
    require(type(runtime['hidden_size']) is int and runtime['hidden_size'] > 0
            and type(runtime['embedding_rms']) in (int, float) and math.isfinite(runtime['embedding_rms'])
            and runtime['embedding_rms'] > 0 and isinstance(runtime['gpu_uuid'], str) and runtime['gpu_uuid'],
            'Runtime model/device metadata is invalid')
    prompts = lines(artifact / 'prompts.jsonl')
    require(len(prompts) == 5989 and [row['id'] for row in prompts] == [item['id'] for item in items], 'Saved prompt order/support differs')
    for index, (prompt, item) in enumerate(zip(prompts, items)):
        sensor = 'Sentinel-1' if item['phen'] == 'flood' else 'Sentinel-2'
        user = f"These are 2 {sensor} observations of the same area in chronological order, taken on {', '.join(item['dates'])}: <EO> Did a {item['phen']} occur between the two observations? Answer with yes or no."
        require(prompt['user_text'] == user and user in prompt['chat_text'] and prompt['chat_text'].count('<EO>') == 1
                and prompt['source_gold'] == item['answer'] and prompt['pair_index'] == index and prompt['n_eo_tokens'] == 192,
                'Saved source prompt/label/token budget differs')
        require(all(isinstance(prompt[key], list) and prompt[key] and all(type(token) is int and token >= 0 for token in prompt[key])
                    for key in ('prefix_ids', 'suffix_ids', 'answer_ids')), 'Invalid saved token sequence')
    indexed, parse_rates = index_predictions(items, lines(artifact / 'predictions.jsonl'))
    compare(scores['coverage'], {'expected': 26325, 'received': 26325, 'n_items': 5989, 'n_train': 4234, 'n_test': 1755, 'n_primary': 902}, 'coverage')
    compare(scores['parse_fail_rates'], parse_rates, 'parse_fail_rates')
    recalculated = primary_metrics(primary, indexed)
    secondary = secondary_metrics(test, indexed)
    for seed in SEEDS:
        compare(scores['metrics'][str(seed)]['primary_same_prompt'], recalculated['metrics'][str(seed)], f'primary/{seed}')
        for key in ('paired_source', 'hard_negative'):
            compare(scores['metrics'][str(seed)][key], secondary[str(seed)][key], f'secondary/{seed}/{key}')
    for key in ('seed_decisions', 'pair_preserves_seeds', 'explicit_difference_helps_seeds', 'verdict'):
        compare(scores[key], recalculated[key], key)
    require(status['verdict'] == recalculated['verdict'], 'Final status verdict differs')
    outcomes = audit_training(artifact, manifest, batches, indexed, tracked, check_tensors)
    compare(summary['models'], outcomes, 'summary/models')
    compare(inference['models'], outcomes, 'inference/models')
    for value in (scores['elapsed_s'], inference['elapsed_s']):
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 240 * 60, 'Total model-work time cap differs')
    for path, expected in tracked.items():
        require(sha(path) == expected, 'Artifact changed during audit: ' + path)
    return {'schema': 'e5-independent-result-audit-v0', 'consistent': True, 'artifact': str(artifact),
            'n_models': 12, 'n_steps': 19080, 'n_training_exposures': 152424, 'n_answers': 26325,
            'primary_n': 902, 'primary_events': 8, 'verdict': recalculated['verdict'],
            'seed_decisions': recalculated['seed_decisions'], 'primary_metrics': recalculated['metrics'],
            'secondary_metrics': secondary,
            'checkpoint_tensors_loaded_and_checked_on_cpu': check_tensors,
            'hashes_verified': tracked, 'audit_code_sha256': sha(__file__),
            'limits': ['Exploratory exposed-data follow-up, not fresh confirmation or evidence of memory-system benefit.',
                'Primary and secondary numeric summaries, all output identities, schedules, training records and hashes are independently checked; these still measure agreement with source labels.',
                'Logged finite flags and step counts cannot independently prove every GPU gradient/optimizer operation occurred; the reviewed frozen runner supplies that execution evidence.',
                'Final records cannot independently prove chronological initialization freeze or absence of answer tokens in evaluation; reviewed control flow enforces those rules.',
                'Saved token IDs are type/support checked; the tokenizer is not reloaded for retokenization.',
                'Original remote cache and LLM bytes are not rehashed here; prepared-pair bytes and inherited frozen source pins are checked.',
                'D contains both observations; neither delta performance nor source-label discrimination alone establishes temporal reasoning.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-tensors', action='store_true', help='Additionally load saved checkpoints with CPU torch weights_only=True')
    args = parser.parse_args()
    require(not args.out.exists(), 'Audit report already exists')
    try:
        report = audit(args.artifact, args.check_tensors)
        code = 0
    except Exception as error:
        report = {'schema': 'e5-independent-result-audit-v0', 'consistent': False,
                  'error_type': type(error).__name__, 'error': str(error), 'audit_code_sha256': sha(__file__)}
        code = 1
    report['checked_at'] = datetime.now(timezone.utc).isoformat()
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'consistent': report['consistent'], 'verdict': report.get('verdict'), 'error': report.get('error'), 'out': str(args.out)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
