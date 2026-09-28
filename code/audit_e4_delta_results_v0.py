#!/usr/bin/env python3
"""Independently audit E4 saved results; no experimental source is imported.

NumPy and the standard library suffice. This is a retrospective consistency
audit of frozen records, not a second model execution or fresh validation set.
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

ARMS = ('real', 'delta_only', 'delta_sign_flip', 'delta_feature_permute')
SEEDS = (1, 2, 3)
EPS = 1e-12
CORE_SHA256 = '1e282051908fe3194a08805bbfab33ecc0bff7a13f95ea1239e3fdaa40b49c73'
SCIENCE_PLAN_SHA256 = '1f9cd4519d245cb842d33b45c3c1b3623015e6c5bf1594b80413b2170455aca7'
RUNNER_SHA256 = '311201365c1871a1bf4a69a24cb4ffa53f209877733f4ce888115914c2de1288'
PARENT_MANIFEST_SHA256 = '68d6c339e81ec9357559ff28d0a85dcb1a881d35fede99a63736b34be54e80fd'
PARENT_ITEMS_SHA256 = 'f3ae0abc245b3e800f1a39b4ca5a250b36f4cf503fe364f3003a556c7f907502'
PARENT_PAIRS_SHA256 = '3f9d693993ce72504a241b8ac266ab3bcf5663f50693d63c3b3b8947d126807a'
REFERENCE_NAMES = ('e2_real', 'e3_real')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _nonfinite(value):
    raise ValueError('Nonfinite JSON token: ' + value)


def _pairs_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def loads(text):
    return json.loads(text, parse_constant=_nonfinite, object_pairs_hook=_pairs_object)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def read_lines(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def unique(rows):
    result = {row['id']: row for row in rows}
    require(len(result) == len(rows), 'Duplicate source ID')
    return result


def compare(actual, expected, path='value'):
    """Compare independently derived science fields with round-off tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Dictionary support differs: ' + path)
        for key, value in expected.items():
            compare(actual[key], value, path + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List support differs: ' + path)
        for index, value in enumerate(expected):
            compare(actual[index], value, path + '/' + str(index))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), 'Numeric mismatch: ' + path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value/type mismatch: ' + path)


def validate_items(items, frozen_population=False):
    require(isinstance(items, list) and items, 'Empty item population')
    source = unique(items)
    tile_groups = defaultdict(list)
    for item in items:
        require(item['phen'] in ('flood', 'landslide'), 'Unknown phenomenon')
        require(item['kind'] in ('pos', 'neg', 'hard_neg'), 'Unknown item kind')
        require(item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Source kind/answer mismatch')
        require(isinstance(item['id'], str) and isinstance(item['tile'], str)
                and isinstance(item['pair_key'], str), 'Invalid source identifier type')
        require(item['kind'] != 'hard_neg' or item['phen'] == 'flood', 'Landslide hard negative is not allowed')
        tile_groups[(item['phen'], item['tile'])].append(item)
    for group in tile_groups.values():
        kinds = [item['kind'] for item in group]
        if 'hard_neg' in kinds:
            require(kinds == ['hard_neg'], 'Hard-negative tile overlaps another item')
        else:
            require(len(group) == 2 and set(kinds) == {'pos', 'neg'}
                    and len({str(item['cluster']) for item in group}) == 1, 'Incomplete paired tile')
    require(len({item['pair_key'] for item in items}) == len(items), 'Duplicate pair archive key')
    if frozen_population:
        require(len(items) == 209, 'Expected unchanged 209 source items')
        counts = Counter((item['phen'], item['kind']) for item in items)
        require(counts == {('flood', 'pos'): 57, ('flood', 'neg'): 57, ('flood', 'hard_neg'): 51,
                           ('landslide', 'pos'): 22, ('landslide', 'neg'): 22}, 'Frozen item category counts differ')
        for phen, count in (('flood', 10), ('landslide', 2)):
            require(len({str(item['cluster']) for item in items if item['phen'] == phen and item['kind'] != 'hard_neg'}) == count,
                    'Frozen event support differs: ' + phen)
    return source


def index_answers(items, rows):
    source = validate_items(items)
    expected = {(seed, arm, item_id) for seed in SEEDS for arm in ARMS for item_id in source}
    indexed = {}
    for row in rows:
        require(type(row.get('seed')) is int, 'Seed must be an integer')
        key = (row['seed'], row.get('arm'), row.get('id'))
        require(key in expected and key not in indexed, 'Unexpected or duplicate answer key: ' + str(key))
        item = source[row['id']]
        require(all(row.get(field) == item[field] for field in ('tile', 'phen', 'kind', 'pair_key'))
                and str(row.get('cluster')) == str(item['cluster']), 'Answer/source metadata differs')
        require(row.get('source_gold') == item['answer'], 'Source gold changed')
        require('transformed_gold' in row and row['transformed_gold'] is None, 'Transformed gold must be explicit null')
        require('parsed' in row and row['parsed'] in ('yes', 'no', None), 'Invalid parsed answer')
        require(isinstance(row.get('answer_raw'), str), 'Missing raw answer')
        match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (match[1] if match else None), 'Raw/parsed answer mismatch')
        indexed[key] = row['parsed']
    require(set(indexed) == expected and len(rows) == len(expected), 'Missing output coverage')
    return indexed


def event_interval(event_differences):
    x = np.array(event_differences, dtype=np.float64)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), 'Invalid bootstrap event differences')
    positions = np.random.default_rng(20260925).integers(0, len(x), size=(5000, len(x)))
    averages = np.mean(x[positions], axis=1)
    return [float(v) for v in np.quantile(averages, (0.025, 0.975), method='linear')]


def recompute(items, rows, references):
    """Recompute the full scored schema from individual answer decisions."""
    values = index_answers(items, rows)
    ids = {item['id'] for item in items}
    require(isinstance(references, dict) and set(references) == set(REFERENCE_NAMES), 'Missing E2/E3 references')
    for name in REFERENCE_NAMES:
        reference = references[name]
        require(set(reference) == {'1', '2', '3'}, 'Reference seed support differs: ' + name)
        require(all(set(mapping) == ids and all(value in ('yes', 'no', None) for value in mapping.values())
                    for mapping in reference.values()), 'Reference item support/value differs: ' + name)
    metrics, parse_rates, reproduced, decisions = {}, {}, {}, {}
    for seed in SEEDS:
        metrics[str(seed)], reproduced[str(seed)] = {}, {}
        for phen in ('flood', 'landslide'):
            population = [item for item in items if item['phen'] == phen]
            require(population, 'Missing phenomenon')
            grouped = defaultdict(list)
            hard = defaultdict(list)
            for item in population:
                (hard if item['kind'] == 'hard_neg' else grouped)[str(item['cluster'])].append(item)
            require(grouped, 'No paired event support')
            events = sorted(grouped)
            per_arm, agreement, hard_metrics, contrasts = {}, {}, {}, {}
            for arm in ARMS:
                predictions = {item['id']: values[(seed, arm, item['id'])] for item in population}
                parse_rate = sum(value is None for value in predictions.values()) / len(population)
                require(parse_rate <= .01 + EPS, f'Parse-failure threshold exceeded: {seed}/{phen}/{arm}')
                parse_rates[f'{seed}|{phen}|{arm}'] = parse_rate
                event_metrics = {}
                for event in events:
                    category_counts = Counter(item['kind'] for item in grouped[event])
                    correct = Counter(item['kind'] for item in grouped[event]
                                      if predictions[item['id']] == item['answer'])
                    recall = correct['pos'] / category_counts['pos']
                    specificity = correct['neg'] / category_counts['neg']
                    event_metrics[event] = {'n_pos': category_counts['pos'], 'n_neg': category_counts['neg'],
                                            'ba': (recall + specificity) / 2,
                                            'recall': recall, 'specificity': specificity}
                per_arm[arm] = {'events': event_metrics, 'n_events': len(events),
                                'macro_ba': float(np.mean([event_metrics[event]['ba'] for event in events]))}
                n_match = n_parsed = 0
                for item in population:
                    before, after = values[(seed, 'real', item['id'])], predictions[item['id']]
                    n_match += before in ('yes', 'no') and before == after
                    n_parsed += before in ('yes', 'no') and after in ('yes', 'no')
                agreement[arm] = {'n_items': len(population), 'matched_valid_decisions': n_match,
                    'n_both_parsed': n_parsed, 'agreement_all_items': n_match / len(population),
                    'agreement_both_parsed': n_match / n_parsed if n_parsed else None,
                    'agreement_ge_0_99_descriptive_only': n_match / len(population) >= .99,
                    'source_yes_rate': sum(value == 'yes' for value in predictions.values()) / len(population)}
                if hard:
                    fp_by_event = {event: sum(predictions[item['id']] == 'yes' for item in hard[event]) / len(hard[event])
                                   for event in sorted(hard)}
                    hard_items = [item for group in hard.values() for item in group]
                    hard_metrics[arm] = {'n': len(hard_items), 'fpr_by_event': fp_by_event,
                        'fpr_event_macro': float(np.mean(list(fp_by_event.values()))),
                        'fpr_pooled': sum(predictions[item['id']] == 'yes' for item in hard_items) / len(hard_items),
                        'parse_failures': sum(predictions[item['id']] is None for item in hard_items)}
            for arm in ARMS:
                differences = {event: per_arm[arm]['events'][event]['ba'] - per_arm['real']['events'][event]['ba']
                               for event in events}
                contrasts[arm] = {'event_deltas': differences, 'delta': float(np.mean(list(differences.values()))),
                                  'ci95_delta': event_interval(list(differences.values())) if phen == 'flood' else None}
            reproduced[str(seed)][phen] = {}
            for name in REFERENCE_NAMES:
                correct = sum(values[(seed, 'real', item['id'])] == references[name][str(seed)][item['id']] for item in population)
                rate = correct / len(population)
                require(rate >= .99, f'Real reproduction below .99: {name}/{seed}/{phen}')
                reproduced[str(seed)][phen][name] = {'n': len(population), 'reproduction_rate': rate}
            metrics[str(seed)][phen] = {'paired': per_arm, 'contrasts': contrasts,
                                       'agreement_with_real': agreement, 'hard_negative': hard_metrics}
        flood = metrics[str(seed)]['flood']
        real, delta = flood['paired']['real'], flood['contrasts']['delta_only']
        eligible = real['n_events'] >= 5 and real['macro_ba'] >= .60 - EPS
        sufficient = eligible and flood['paired']['delta_only']['macro_ba'] >= .60 - EPS and delta['ci95_delta'][0] >= -.05 - EPS
        degrades = eligible and delta['delta'] <= -.10 + EPS and delta['ci95_delta'][1] < -EPS
        decisions[str(seed)] = {'eligible': bool(eligible), 'delta_only_sufficient': bool(sufficient),
                               'delta_only_degrades': bool(degrades)}
    sufficient = [seed for seed in SEEDS if decisions[str(seed)]['delta_only_sufficient']]
    degraded = [seed for seed in SEEDS if decisions[str(seed)]['delta_only_degrades']]
    verdict = ('difference_block_sufficient_under_intervention' if len(sufficient) >= 2 else
               'delta_only_degrades_source_agreement' if len(degraded) >= 2 else 'mixed_or_inconclusive')
    return {'schema': 'e4-delta-probe-scores-v0', 'valid': True, 'verdict': verdict,
            'coverage': {'expected': len(items) * 12, 'received': len(rows)},
            'parse_fail_rates': parse_rates, 'reproduction': reproduced, 'metrics': metrics,
            'seed_decisions': decisions, 'sufficient_seeds': sufficient, 'degraded_seeds': degraded}


def validate_pair_archive(path, items):
    errors = {}
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == {item['pair_key'] for item in items},
                'Pair archive key coverage differs')
        for item in items:
            pair = archive[item['pair_key']]
            require(pair.shape == (2, 64, 768) and pair.dtype == np.float32 and np.isfinite(pair).all(),
                    'Pair shape/dtype/finiteness invalid: ' + item['id'])
            with np.errstate(over='ignore', invalid='ignore'):
                difference = pair[1] - pair[0]
            require(np.isfinite(difference).all(), 'Difference overflow: ' + item['id'])
            # Independently verify the fixed shift is a bijection; reductions
            # are tolerant, because permutation changes summation order.
            changed = difference[:, (np.arange(768) + 257) % 768].astype(np.float64)
            original = difference.astype(np.float64)
            require(np.array_equal(np.sort(original, axis=1), np.sort(changed, axis=1)), 'Permutation changed values')
            max_error = 0.
            for statistic in (lambda x: np.mean(x, axis=1), lambda x: np.var(x, axis=1), lambda x: np.linalg.norm(x, axis=1)):
                before, after = statistic(original), statistic(changed)
                require(np.allclose(before, after, rtol=1e-6, atol=1e-6),
                        'Permutation scalar invariant changed')
                max_error = max(max_error, float(np.max(np.abs(before - after))))
            errors[item['id']] = max_error
    return errors


def verify_file(artifact, relative, expected, tracked):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact-relative path')
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Malformed SHA256')
    path = artifact / relative
    require(sha(path) == expected, 'Frozen SHA mismatch: ' + str(relative))
    tracked[str(path)] = expected
    return path


def validate_lineage(artifact, manifest, plan, tracked):
    require(manifest['schema'] == 'e4-delta-prepared-v0', 'Unknown prepared manifest schema')
    for filename, key in (('items.jsonl', 'items_sha256'), ('pairs.npz', 'pairs_sha256'),
                          ('references.json', 'references_sha256'), ('prereg.json', 'prereg_sha256'),
                          ('input_audit.json', 'input_audit_sha256')):
        verify_file(artifact, filename, manifest[key], tracked)
    science_plan = {key: value for key, value in plan.items() if key != 'compute'}
    science_hash = hashlib.sha256(json.dumps(science_plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    require(science_hash == SCIENCE_PLAN_SHA256, 'Scientific preregistration fields differ from reviewed plan')
    require(plan['compute']['training'] is False and plan['compute']['max_runtime_minutes'] == 45,
            'Training/runtime contract changed')
    require(manifest['n_items'] == 209 and manifest['n_generations'] == 2508, 'Prepared population changed')
    require(manifest['scope'] == plan['scope'], 'Manifest scope differs from preregistration')
    require(manifest['items_sha256'] == PARENT_ITEMS_SHA256 and manifest['pairs_sha256'] == PARENT_PAIRS_SHA256,
            'Copied items/pairs differ from frozen E3 bytes')
    expected_sources = {'e4_delta_runner_v0.py', 'e4_delta_probe_v0.py', 'run_e4_when_idle_v0.py'}
    require(set(manifest['code_snapshot_sha256']) == expected_sources, 'Source snapshot support differs')
    for name, expected in manifest['code_snapshot_sha256'].items():
        require(Path(name).name == name, 'Unsafe source snapshot filename')
        verify_file(artifact, 'code_snapshot/' + name, expected, tracked)
    require(manifest['code_snapshot_sha256']['e4_delta_probe_v0.py'] == CORE_SHA256, 'Unreviewed scientific core version')
    require(manifest['code_snapshot_sha256']['e4_delta_runner_v0.py'] == RUNNER_SHA256, 'Unreviewed runner version')
    snapshots = manifest['parent_snapshot_sha256']
    require(set(snapshots) == {'manifest.json', 'scores.json', 'status.json', 'prereg.json'}, 'Parent snapshot support differs')
    for name, expected in snapshots.items():
        verify_file(artifact, 'parent_snapshot/' + name, expected, tracked)
    require(snapshots['manifest.json'] == PARENT_MANIFEST_SHA256 == plan['parent']['manifest_sha256'], 'Parent manifest pin differs')
    require(snapshots['scores.json'] == plan['parent']['scores_sha256'], 'Parent scores pin differs')
    parent = read(artifact / 'parent_snapshot/manifest.json')
    require(snapshots['prereg.json'] == parent['prereg_sha256'], 'Parent preregistration pin differs')
    require(read(artifact / 'parent_snapshot/status.json')['status'] == 'completed'
            and read(artifact / 'parent_snapshot/scores.json')['valid'] is True, 'Parent run is not complete and valid')
    for field in ('items_sha256', 'pairs_sha256', 'models', 'llm_files_sha256'):
        require(manifest[field] == parent[field], 'E3 inherited field differs: ' + field)
    require(parent['saved_e2_real_sha256'] == plan['parent']['saved_e2_real_sha256'], 'Saved E2 pin differs')
    require(set(manifest['models']) == {'1', '2', '3'}
            and {seed: model['sha256'] for seed, model in manifest['models'].items()} == plan['checkpoints'],
            'Projector checkpoint pins differ')
    return parent


def original_references(repo, items, references, plan, parent_manifest, tracked):
    """Verify full frozen E2 files and original E3 real rows, including metadata."""
    repo = Path(repo)
    source = unique(items)
    e3 = repo / 'artifacts/e3_pair_dependence_v1_20260925'
    for filename, key in (('manifest.json', 'manifest_sha256'), ('scores.json', 'scores_sha256'),
                          ('items.jsonl', 'items_sha256'), ('pairs.npz', 'pairs_sha256'),
                          ('saved_e2_real.json', 'saved_e2_real_sha256')):
        verify_file(e3, filename, plan['parent'][key], tracked)
    compare(read_lines(e3 / 'items.jsonl'), items, 'original_E3/items')
    compare(read(e3 / 'saved_e2_real.json'), references['e2_real'], 'original_E3/saved_E2')
    for seed in SEEDS:
        seed_s = str(seed)
        e3_path = verify_file(e3, f'answers_seed{seed}_real.jsonl', plan['parent']['real_answer_sha256'][seed_s], tracked)
        e3_rows = read_lines(e3_path)
        e3_index = unique(e3_rows)
        require(set(e3_index) == set(source), 'Original E3 real population differs')
        for item_id, item in source.items():
            row = e3_index[item_id]
            require(row['seed'] == seed and row['arm'] == 'real' and row['source_gold'] == item['answer']
                    and 'transformed_gold' in row and row['transformed_gold'] is None, 'Original E3 real identity/gold differs')
            require(all(row[field] == item[field] for field in ('tile', 'phen', 'kind', 'pair_key'))
                    and str(row['cluster']) == str(item['cluster']), 'Original E3 real metadata differs')
            match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
            require(row['parsed'] == (match[1] if match else None) == references['e3_real'][seed_s][item_id],
                    'Original E3 real raw/parsed reference differs')
        e2_base = repo / f'artifacts/e2_multi_reader_v0/reader_seed{seed}'
        e2_path = verify_file(e2_base, 'answers_real_all.jsonl', parent_manifest['e2_reference_sha256'][seed_s], tracked)
        e2_rows = read_lines(e2_path)
        e2_index = unique(e2_rows)
        require(len(e2_index) == 1755 and set(source) <= set(e2_index), 'Original E2 population differs')
        for item_id, item in source.items():
            row = e2_index[item_id]
            require(all(row[field] == item[field] for field in ('tile', 'fold', 'phen', 'kind'))
                    and row['text_gold'] == item['answer'], 'Original E2 source metadata/gold differs')
            match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
            require(row['parsed'] == (match[1] if match else None) == references['e2_real'][seed_s][item_id],
                    'Original E2 raw/parsed reference differs')


def runtime_reproduction(scores):
    result = {}
    for seed in SEEDS:
        seed_s = str(seed)
        result[seed_s] = {}
        for phen in ('flood', 'landslide'):
            result[seed_s][phen] = {}
            for name in REFERENCE_NAMES:
                result[seed_s][phen][name] = {**scores['reproduction'][seed_s][phen][name],
                    'parse_failure_rate': scores['parse_fail_rates'][f'{seed}|{phen}|real']}
    return result


def audit(artifact, repo=None):
    artifact = Path(artifact).resolve()
    tracked = {str(artifact / name): sha(artifact / name)
               for name in ('manifest.json', 'scores.json', 'status.json', 'reproduction.json')}
    manifest, plan, status, scores = (read(artifact / name) for name in ('manifest.json', 'prereg.json', 'status.json', 'scores.json'))
    require(status['status'] == 'completed' and scores['valid'] is True, 'Require a completed valid E4 run')
    require(not (artifact / 'failure.json').exists(), 'Completed artifact retains a failure record')
    parent = validate_lineage(artifact, manifest, plan, tracked)
    items = read_lines(artifact / 'items.jsonl')
    source = validate_items(items, frozen_population=True)
    counts = dict(Counter(f"{item['phen']}|{item['cluster']}|{item['kind']}" for item in items))
    require(counts == parent['selected_counts'], 'Original E3 per-event selected counts changed')
    permutation_errors = validate_pair_archive(artifact / 'pairs.npz', items)
    inputs = read(artifact / 'input_audit.json')
    require(inputs['n_items'] == 209 and isinstance(inputs['all'], list), 'Prepared input-audit population differs')
    input_index = unique(inputs['all'])
    require(set(input_index) == set(source), 'Prepared input-audit ID coverage differs')
    for item_id, item in source.items():
        entry = input_index[item_id]
        require(entry['pair_key'] == item['pair_key'] and entry['all_four_finite'] is True,
                'Prepared input-audit source/finiteness differs')
        error = entry['permutation_stat_max_abs_error']
        require(type(error) in (float, int) and math.isfinite(error) and error >= 0, 'Invalid prepared permutation error')
        compare(error, permutation_errors[item_id], 'input_audit/permutation_error/' + item_id)
    expected_files = {f'answers_seed{seed}_{arm}.jsonl' for seed in SEEDS for arm in ARMS}
    require({path.name for path in artifact.glob('answers_*.jsonl')} == expected_files, 'Answer file support differs')
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            path = artifact / f'answers_seed{seed}_{arm}.jsonl'
            tracked[str(path)] = sha(path)
            chunk = read_lines(path)
            require(len(chunk) == 209 and all(row.get('seed') == seed and row.get('arm') == arm for row in chunk),
                    'Answer file has wrong identity/population')
            rows.extend(chunk)
    references = read(artifact / 'references.json')
    # Frozen E2/E3 actual reference maps had no parse failures. This also makes
    # final reproduction equality agree with the runtime valid-answer gate.
    require(all(value in ('yes', 'no') for mapping in references.values()
                for seed_values in mapping.values() for value in seed_values.values()), 'Unexpected unparsed frozen reference')
    recalculated = recompute(items, rows, references)
    require(recalculated['coverage'] == {'expected': 2508, 'received': 2508}, 'Expected exact 2508 output rows')
    for key, value in recalculated.items():
        compare(scores[key], value, 'scores/' + key)
    runtime = runtime_reproduction(recalculated)
    compare(read(artifact / 'reproduction.json'), runtime, 'reproduction.json')
    compare(scores['runtime_reproduction'], runtime, 'scores/runtime_reproduction')
    require(status['verdict'] == recalculated['verdict'], 'Status verdict differs')
    require(scores['manifest_sha256'] == tracked[str(artifact / 'manifest.json')]
            and scores['prereg_sha256'] == manifest['prereg_sha256'], 'Scores refer to different frozen inputs')
    verify_file(artifact, 'prompt_snapshot.json', scores['prompt_snapshot_sha256'], tracked)
    prompts = read(artifact / 'prompt_snapshot.json')
    require(set(prompts) == set(source), 'Prompt snapshot ID support differs')
    for item_id, item in source.items():
        prompt = prompts[item_id]
        sensor = 'Sentinel-1' if item['phen'] == 'flood' else 'Sentinel-2'
        expected_user = (f"These are 2 {sensor} observations of the same area in chronological order, taken on "
                         + ', '.join(item['dates']) + f": <EO> Did a {item['phen']} occur between the two observations? Answer with yes or no.")
        require(prompt['user_text'] == expected_user and isinstance(prompt['chat_text'], str)
                and expected_user in prompt['chat_text'] and prompt['chat_text'].count('<EO>') == 1
                and prompt['n_eo_tokens'] == 192, 'Prompt snapshot content/token budget differs')
        require(all(isinstance(prompt[field], list) and prompt[field]
                    and all(type(token) is int and token >= 0 for token in prompt[field])
                    for field in ('prefix_ids', 'suffix_ids')), 'Invalid prompt token ID sequence')
    require(type(scores['elapsed_s']) in (int, float) and math.isfinite(scores['elapsed_s'])
            and 0 <= scores['elapsed_s'] <= plan['compute']['max_runtime_minutes'] * 60,
            'Runtime is not finite or exceeds the declared budget')
    originals_checked = repo is not None
    if originals_checked:
        original_references(repo, items, references, plan, parent, tracked)
    for path, expected in tracked.items():
        require(sha(path) == expected, 'File changed during independent audit: ' + path)
    return {'schema': 'e4-independent-result-audit-v0', 'consistent': True,
            'artifact': str(artifact), 'n_items': 209, 'n_answers': 2508, 'flood_events': 10,
            'verdict': recalculated['verdict'], 'seed_decisions': recalculated['seed_decisions'],
            'reproduction': recalculated['reproduction'], 'metrics': recalculated['metrics'],
            'original_full_E2_and_E3_references_revalidated': originals_checked,
            'frozen_core_sha256': CORE_SHA256, 'frozen_runner_sha256': RUNNER_SHA256,
            'hashes_verified': tracked, 'audit_code_sha256': sha(__file__),
            'scientific_review': [
                'All four arms cover all 209 original items in all three seeds; transformed_gold is explicit null.',
                'Primary BA averages the ten original flood events; hard negatives are separate. Landslide intervals remain null.',
                'Paired event differences are bootstrapped 5000 times with NumPy seed 20260925 and linear percentiles.',
                'Valid-decision agreement uses all-item denominators; even null/null is not an agreed decision.',
                'No substantive numeric error was found in the reviewed score definitions.'],
            'limits': [
                'Post-E3 exploratory diagnosis on exposed data; this audit provides no fresh confirmation.',
                'D contains both observations. Delta-only sufficiency cannot imply that history is unnecessary.',
                'Sign/permutation interventions are out of distribution; sensitivity is not proof of physical change reasoning.',
                'Permutation preserves raw D statistics only; the learned projector can change projected norms or downstream gating.',
                'Server checkpoint and LLM bytes are not rehashed by this local audit; their parent/plan/manifest pins are checked.',
                'Projected tensors and generation logits are not saved; intermediate finite checks remain runner evidence.',
                'Chronological real-before-controls execution cannot be proved from final JSONL files alone; it is enforced by the reviewed runner.',
                'Prompt user text/dates and saved token sequence types are checked, but the tokenizer is not reloaded to retokenize them.',
                'Source scripts are hashed against the preparation manifest; the scientific core and reviewed runner are additionally pinned.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--repo', type=Path, help='Local repository with full frozen E2 and E3 artifacts')
    parser.add_argument('--out', type=Path, required=True, help='New audit JSON; parent directory must already exist')
    args = parser.parse_args()
    # Refuse overwrite before reading a potentially expensive pair archive.
    require(not args.out.exists(), 'Audit report already exists')
    try:
        report = audit(args.artifact, args.repo)
        exit_code = 0
    except Exception as error:
        report = {'schema': 'e4-independent-result-audit-v0', 'consistent': False,
                  'error_type': type(error).__name__, 'error': str(error), 'audit_code_sha256': sha(__file__)}
        exit_code = 1
    report['checked_at'] = datetime.now(timezone.utc).isoformat()
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'consistent': report['consistent'], 'verdict': report.get('verdict'),
                      'error': report.get('error'), 'out': str(args.out)}, ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
