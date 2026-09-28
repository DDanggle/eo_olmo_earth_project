#!/usr/bin/env python3
"""Independent E3 result audit: frozen lineage, coverage, reproduction and verdict.

Only NumPy and the standard library are used. The original scoring module is
never imported. Primary event/tile balanced agreement, paired event bootstrap,
hard-negative specificity and decisions are recomputed from saved answer rows.
Counterfactual answers are compared with source labels, never physical-scene gold.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ARMS = ('real', 'earlier_only', 'later_only', 'repeat_earlier', 'repeat_later', 'no_delta', 'reverse')
HARD_ARMS = ('real', 'later_only', 'repeat_later')
SEEDS = (1, 2, 3)
EPS = 1e-12
SCORER_SHA = '59861d43601fbaa0e29f4d9e6998350cfb493feb397f9d0fe5363963b2dca38b'
TRANSFORM_SHA = 'd051a0a872bb998bcc9dce40d0b0da49fac6994fb32133025a74e7ec5267d346'
GPU_REPLACEMENTS = (
    ("['nvidia-smi','-i','1','--query-gpu=uuid'", "['nvidia-smi','-i','0','--query-gpu=uuid'"),
    ("RuntimeError('GPU1 occupied; do not allocate')", "RuntimeError('GPU0 occupied; do not allocate')"),
    ("os.environ.get('CUDA_VISIBLE_DEVICES')!='1'", "os.environ.get('CUDA_VISIBLE_DEVICES')!='0'"),
    ("RuntimeError('Explicit GPU1 mapping required')", "RuntimeError('Explicit GPU0 mapping required')"),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def read_lines(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return [json.loads(line, parse_constant=invalid) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(items):
    result = {item['id']: item for item in items}
    require(len(result) == len(items), 'Duplicate item IDs')
    return result


def compare(actual, expected, path='value'):
    """Allow numeric round-off only; preserve booleans, null and exact supports."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Dictionary support differs: ' + path)
        for key, value in expected.items():
            compare(actual[key], value, path + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List support differs: ' + path)
        for index, value in enumerate(expected):
            compare(actual[index], value, path + '/' + str(index))
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), 'Numeric mismatch: ' + path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value/type mismatch: ' + path)


def validate_items(items):
    by_id = unique(items)
    pairs = defaultdict(list)
    for item in items:
        require(item['phen'] in ('flood', 'landslide') and item['kind'] in ('pos', 'neg', 'hard_neg'), 'Unexpected item category')
        require(item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Source kind/answer mismatch')
        hard = item['kind'] == 'hard_neg'
        require(not hard or item['phen'] == 'flood', 'Non-flood hard negative')
        require(item['allowed_arms'] == list(HARD_ARMS if hard else ARMS), 'Prepared allowed-arm contract differs')
        if not hard:
            pairs[(item['phen'], item['tile'])].append(item)
    for pair in pairs.values():
        require(len(pair) == 2 and {item['kind'] for item in pair} == {'pos', 'neg'}
                and len({str(item['cluster']) for item in pair}) == 1, 'Broken same-tile source pair')
    require(all((item['phen'], item['tile']) not in pairs for item in items if item['kind'] == 'hard_neg'), 'Hard negative overlaps positive pair')
    require(len({item['pair_key'] for item in items}) == len(items), 'Duplicate prepared pair key')
    return by_id


def index_answers(items, rows):
    source = validate_items(items)
    expected = {(seed, arm, item['id']) for seed in SEEDS for item in items for arm in item['allowed_arms']}
    indexed = {}
    parse_groups = defaultdict(list)
    for row in rows:
        require(type(row.get('seed')) is int, 'Invalid answer seed type')
        key = (row['seed'], row.get('arm'), row.get('id'))
        require(key in expected and key not in indexed, 'Unexpected/duplicate answer key: ' + str(key))
        item = source[row['id']]
        require(all(row.get(field) == item[field] for field in ('tile', 'phen', 'kind', 'pair_key'))
                and str(row.get('cluster')) == str(item['cluster']), 'Answer metadata differs from item')
        require(row.get('source_gold') == item['answer'], 'Changed original source gold')
        require('transformed_gold' in row and row['transformed_gold'] is None, 'A transformed physical gold was assigned')
        require('parsed' in row and row['parsed'] in ('yes', 'no', None), 'Invalid parsed answer')
        require(isinstance(row.get('answer_raw'), str), 'Missing raw model answer')
        match = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (match.group(1) if match else None), 'Saved parsed answer differs from raw answer')
        indexed[key] = row
        parse_groups[(row['seed'], item['phen'], row['arm'])].append(row['parsed'])
    require(set(indexed) == expected and len(rows) == len(expected), 'Incomplete E3 answer coverage')
    rates = {}
    for key, values in sorted(parse_groups.items()):
        failed = sum(value is None for value in values)
        rate = failed / len(values)
        require(rate <= .01 + EPS, 'Parse-failure threshold exceeded: ' + str(key))
        rates['|'.join(map(str, key))] = {'n': len(values), 'failed': failed, 'rate': rate}
    return indexed, rates


def bootstrap(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), 'Empty/nonfinite event contrast')
    # One event index is shared by real and intervention because only their
    # already-paired event differences are resampled.
    draws = np.random.default_rng(20260925).integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975], method='linear').tolist()


def recompute(items, answers):
    metrics, decisions = {}, {}
    for seed in SEEDS:
        metrics[str(seed)] = {}
        for phen in ('flood', 'landslide'):
            paired = [item for item in items if item['phen'] == phen and item['kind'] != 'hard_neg']
            require(paired, 'Missing phenomenon')
            events = sorted({str(item['cluster']) for item in paired})
            per_arm, contrasts, hard_metrics = {}, {}, {}
            for arm in ARMS:
                event_metrics, tile_metrics = {}, {}
                for event in events:
                    subset = [item for item in paired if str(item['cluster']) == event]
                    positive = [item for item in subset if item['kind'] == 'pos']
                    negative = [item for item in subset if item['kind'] == 'neg']
                    n_yes = sum(answers[(seed, arm, item['id'])]['parsed'] == 'yes' for item in positive)
                    n_no = sum(answers[(seed, arm, item['id'])]['parsed'] == 'no' for item in negative)
                    for tile in sorted({item['tile'] for item in subset}):
                        pair = [item for item in subset if item['tile'] == tile]
                        tile_metrics[tile] = {'cluster': event, 'ba': sum(answers[(seed, arm, item['id'])]['parsed'] == item['answer'] for item in pair) / 2}
                    # Exactly one pos/neg per tile was independently validated.
                    ba = float(np.mean([tile_metrics[tile]['ba'] for tile in sorted({item['tile'] for item in subset})]))
                    event_metrics[event] = {'n_items': len(subset), 'n_tiles': len(positive),
                        'n_yes': len(positive), 'n_no': len(negative), 'ba': ba,
                        'yes_recall': n_yes / len(positive), 'no_recall': n_no / len(negative),
                        'fpr_yes': sum(answers[(seed, arm, item['id'])]['parsed'] == 'yes' for item in negative) / len(negative),
                        'parse_fail_count': sum(answers[(seed, arm, item['id'])]['parsed'] is None for item in subset)}
                per_arm[arm] = {'macro_ba': float(np.mean([event_metrics[event]['ba'] for event in events])),
                    'event_count': len(events), 'n_items': len(paired), 'n_tiles': len(tile_metrics),
                    'events': event_metrics, 'tile_ba': tile_metrics,
                    'parse_fail_count': sum(answers[(seed, arm, item['id'])]['parsed'] is None for item in paired)}
            for arm in ARMS:
                delta = {event: per_arm[arm]['events'][event]['ba'] - per_arm['real']['events'][event]['ba'] for event in events}
                contrasts[arm] = {'delta': float(np.mean(list(delta.values()))), 'event_deltas': delta,
                    'ci95_delta': bootstrap(list(delta.values())) if phen == 'flood' else None,
                    'uncertainty_unit': 'flood_event' if phen == 'flood' else 'descriptive_regions_only',
                    'bootstrap_draws': 5000 if phen == 'flood' else None, 'bootstrap_seed': 20260925 if phen == 'flood' else None}
            hard = [item for item in items if item['phen'] == phen and item['kind'] == 'hard_neg']
            for arm in HARD_ARMS if hard else ():
                eh = {}
                for event in sorted({str(item['cluster']) for item in hard}):
                    subset = [item for item in hard if str(item['cluster']) == event]
                    yes = sum(answers[(seed, arm, item['id'])]['parsed'] == 'yes' for item in subset)
                    eh[event] = {'n_items': len(subset), 'false_positive_yes_count': yes, 'fpr': yes / len(subset),
                        'parse_fail_count': sum(answers[(seed, arm, item['id'])]['parsed'] is None for item in subset)}
                hard_metrics[arm] = {'n_items': len(hard), 'event_count': len(eh), 'events': eh,
                    'fpr_pooled': sum(e['false_positive_yes_count'] for e in eh.values()) / len(hard),
                    'fpr_event_macro': float(np.mean([e['fpr'] for e in eh.values()])),
                    'parse_fail_count': sum(e['parse_fail_count'] for e in eh.values()),
                    'note': 'hard_negative_only; unparsed outputs reported separately and not counted as yes'}
            metrics[str(seed)][phen] = {'paired': per_arm, 'contrasts': contrasts, 'hard_neg': hard_metrics}
        flood = metrics[str(seed)]['flood']
        support = flood['paired']['real']['event_count']
        real_ba = flood['paired']['real']['macro_ba']
        eligible = support >= 5 and real_ba >= .60 - EPS
        sufficient = eligible and all(flood['paired'][arm]['macro_ba'] >= .60 - EPS
                                     and flood['contrasts'][arm]['ci95_delta'][0] >= -.05 - EPS
                                     for arm in ('later_only', 'repeat_later'))
        sensitive = eligible and all(flood['contrasts'][arm]['delta'] <= -.10 + EPS
                                    and flood['contrasts'][arm]['ci95_delta'][1] < -EPS
                                    for arm in ('later_only', 'repeat_later'))
        decisions[str(seed)] = {'flood_events': support, 'real_macro_ba': real_ba, 'eligible': bool(eligible),
                               'both_arms_sufficient': bool(sufficient), 'both_arms_history_sensitive': bool(sensitive)}
    sufficient = [seed for seed in SEEDS if decisions[str(seed)]['both_arms_sufficient']]
    sensitive = [seed for seed in SEEDS if decisions[str(seed)]['both_arms_history_sensitive']]
    verdict = ('single_second_view_sufficient_under_intervention' if len(sufficient) >= 2 else
               'history_sensitive_under_intervention' if len(sensitive) >= 2 else 'mixed_or_inconclusive')
    return {'metrics': metrics, 'seed_decisions': decisions, 'verdict': verdict,
            'sufficient_seeds': sufficient, 'history_sensitive_seeds': sensitive}


def reproduction(items, answers, saved):
    expected_ids = {item['id'] for item in items}
    require(set(saved) == {'1', '2', '3'} and all(set(saved[seed]) == expected_ids for seed in saved), 'Saved E2 reproduction ID coverage differs')
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        for phen in ('flood', 'landslide'):
            group = [item for item in items if item['phen'] == phen]
            reproduced = sum(answers[(seed, 'real', item['id'])]['parsed'] == saved[str(seed)][item['id']] for item in group) / len(group)
            failed = sum(answers[(seed, 'real', item['id'])]['parsed'] is None for item in group) / len(group)
            require(reproduced >= .99 and failed <= .01, f'Real reproduction/parse gate failed: seed{seed}/{phen}')
            result[str(seed)][phen] = {'n': len(group), 'reproduction_rate': reproduced, 'parse_failure_rate': failed}
    return result


def lineage(artifact, manifest, cfg, tracked):
    def verify_file(relative, expected):
        path = artifact / relative
        require(sha(path) == expected, 'Frozen artifact SHA mismatch: ' + relative)
        tracked[str(path)] = expected
    for filename, key in (('items.jsonl', 'items_sha256'), ('pairs.npz', 'pairs_sha256'),
                          ('saved_e2_real.json', 'saved_e2_real_sha256'), ('prereg.json', 'prereg_sha256')):
        verify_file(filename, manifest[key])
    for name, expected in manifest['code_snapshot_sha256'].items():
        require(Path(name).name == name, 'Snapshot path is not a filename')
        verify_file('code_snapshot/' + name, expected)
    require(manifest['code_snapshot_sha256']['e3_pair_scoring.py'] == SCORER_SHA, 'Unreviewed scoring source version')
    require(manifest['code_snapshot_sha256']['e3_pair_transforms_v0.py'] == TRANSFORM_SHA, 'Unreviewed transformation version')
    require(cfg['validity']['real_reproduction_min'] == .99 and cfg['validity']['max_parse_fail'] == .01
            and cfg['flood_decision']['minimum_real_macro_BA'] == .6, 'Validity/decision contract changed')
    require(set(cfg['arms']) == set(ARMS) and cfg['sampling']['selection_uses_predictions'] is False
            and manifest['selection_uses_predictions'] is False, 'Arm/selection contract changed')
    require({seed: info['sha256'] for seed, info in manifest['models'].items()} == cfg['checkpoints'], 'Manifest checkpoint hashes differ from preregistration')
    amendment_result = None
    if 'operational_amendment' in manifest:
        operational = manifest['operational_amendment']
        require(operational['path'] == 'amendment.json', 'Unexpected amendment path')
        verify_file('amendment.json', operational['sha256'])
        amendment = read(artifact / 'amendment.json')
        verify_file('parent_manifest.json', operational['parent_manifest_sha256'])
        parent = read(artifact / 'parent_manifest.json')
        verify_file('parent_prereg.json', parent['prereg_sha256'])
        original_cfg = read(artifact / 'parent_prereg.json')
        expected_cfg = copy.deepcopy(original_cfg)
        expected_cfg['compute']['gpu'] = '0 only after no other process, with GPU0 and legacy GPU1 research locks'
        expected_cfg['compute']['outputs'] = amendment['output'] + '/'
        require(cfg == expected_cfg, 'Operational amendment changed non-operational preregistration fields')
        for key in ('items_sha256', 'pairs_sha256', 'saved_e2_real_sha256', 'models', 'llm_files_sha256',
                    'selected_cache_sha256', 'e2_reference_sha256', 'selected_counts', 'n_items', 'n_generations'):
            require(manifest[key] == parent[key], 'Operational amendment changed prepared science: ' + key)
        for name, expected in parent['code_snapshot_sha256'].items():
            verify_file('parent_code_snapshot/' + name, expected)
            if name != 'e3_pair_dependence_v0.py':
                require(expected == manifest['code_snapshot_sha256'][name], 'Operational amendment changed scientific helper')
        original = (artifact / 'parent_code_snapshot/e3_pair_dependence_v0.py').read_text()
        for before, after in GPU_REPLACEMENTS:
            require(original.count(before) == 1, 'Ambiguous GPU-only source amendment')
            original = original.replace(before, after)
        require((artifact / 'code_snapshot/e3_pair_dependence_v0.py').read_text() == original, 'Runner differs beyond four GPU guard literals')
        verify_file('operational_handoff.json', amendment['handoff_sha256'])
        verify_file('input_audit.json', amendment['parent_input_audit_sha256'])
        verify_file('launcher_snapshot/run_e3_pair_when_idle_v1.py', amendment['launcher_sha256'])
        verify_file('launcher_snapshot/prepare_e3_gpu0_v1.py', amendment['prepare_helper_sha256'])
        handoff = read(artifact / 'operational_handoff.json')
        require(handoff['status'] == 'stopped_waiter_before_inference' and handoff['no_child_or_model_stopped'] is True,
                'Invalid operational handoff evidence')
        amendment_result = {'gpu': 0, 'science_bytes_unchanged_except_four_gpu_guard_literals': True,
                            'parent_manifest_sha256': operational['parent_manifest_sha256']}
    return amendment_result


def audit(artifact, repo=None):
    artifact = Path(artifact).resolve()
    manifest, cfg, status, scores = (read(artifact / name) for name in ('manifest.json', 'prereg.json', 'status.json', 'scores.json'))
    tracked = {str(artifact / name): sha(artifact / name) for name in ('manifest.json', 'prereg.json', 'status.json', 'scores.json', 'reproduction.json')}
    require(status['status'] == 'completed' and scores['valid'] is True, 'A complete valid E3 run is required')
    amendment = lineage(artifact, manifest, cfg, tracked)
    items = read_lines(artifact / 'items.jsonl')
    source = validate_items(items)
    require(len(items) == manifest['n_items'] == 209, 'Expected frozen 209-item selection')
    expected_rows = 3 * sum(len(item['allowed_arms']) for item in items)
    require(expected_rows == manifest['n_generations'] == 3777, 'Expected frozen 3777-generation coverage')
    counts = dict(Counter(f"{item['phen']}|{item['cluster']}|{item['kind']}" for item in items))
    require(counts == manifest['selected_counts'], 'Selected-count manifest mismatch')
    require(len({str(item['cluster']) for item in items if item['phen'] == 'flood' and item['kind'] != 'hard_neg'}) == 10,
            'Original ten flood events must remain in E3 primary analysis')
    with np.load(artifact / 'pairs.npz', allow_pickle=False) as pairs:
        require(set(pairs.files) == {item['pair_key'] for item in items}, 'Prepared pair archive coverage mismatch')
        for item in items:
            pair = pairs[item['pair_key']]
            require(pair.shape == (2, 64, 768) and pair.dtype == np.float32 and np.isfinite(pair).all(), 'Invalid prepared pair tensor: ' + item['id'])
    filenames = {f'answers_seed{seed}_{arm}.jsonl' for seed in SEEDS for arm in ARMS}
    require({p.name for p in artifact.glob('answers_*.jsonl')} == filenames, 'Missing or extra answer files')
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            path = artifact / f'answers_seed{seed}_{arm}.jsonl'
            tracked[str(path)] = sha(path)
            chunk = read_lines(path)
            require(all(row.get('seed') == seed and row.get('arm') == arm for row in chunk), 'Row does not match its answer file')
            rows.extend(chunk)
    indexed, parse_rates = index_answers(items, rows)
    expected_coverage = {'expected_rows': 3777, 'received_rows': 3777, 'accepted_unique_rows': 3777,
                         'missing_count': 0, 'duplicate_count': 0, 'unexpected_count': 0,
                         'missing_examples': [], 'duplicate_examples': [], 'unexpected_examples': [],
                         'parse_fail_by_seed_phen_arm': parse_rates}
    compare(scores['coverage'], expected_coverage, 'coverage')
    saved = read(artifact / 'saved_e2_real.json')
    reproduced = reproduction(items, indexed, saved)
    compare(read(artifact / 'reproduction.json'), reproduced, 'reproduction.json')
    compare(scores['reproduction'], reproduced, 'scores/reproduction')
    recomputed = recompute(items, indexed)
    for seed in SEEDS:
        for phen in ('flood', 'landslide'):
            for key in ('paired', 'contrasts', 'hard_neg'):
                compare(scores['metrics'][str(seed)][phen][key], recomputed['metrics'][str(seed)][phen][key], f'metrics/{seed}/{phen}/{key}')
    for key in ('seed_decisions', 'verdict', 'sufficient_seeds', 'history_sensitive_seeds'):
        compare(scores[key], recomputed[key], key)
    require(scores['invalid_reasons'] == [] and status['verdict'] == recomputed['verdict'], 'Final status/verdict differs')
    require(scores['manifest_sha256'] == tracked[str(artifact / 'manifest.json')]
            and scores['prereg_sha256'] == manifest['prereg_sha256'], 'Scores reference different frozen inputs')
    e2_checked = False
    if repo is not None:
        repo = Path(repo)
        for seed in SEEDS:
            path = repo / f'artifacts/e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl'
            h = sha(path)
            require(h == manifest['e2_reference_sha256'][str(seed)], 'Original E2 reference bytes differ')
            tracked[str(path)] = h
            original = unique(read_lines(path))
            require(len(original) == 1755 and set(source) <= set(original), 'Original E2 population mismatch')
            for key, item in source.items():
                row = original[key]
                require(all(row[field] == item[field] for field in ('tile', 'fold', 'phen', 'kind'))
                        and row['text_gold'] == item['answer'] and row['parsed'] == saved[str(seed)][key],
                        'Original E2 subset metadata/decision differs')
        e2_checked = True
    for path, expected in tracked.items():
        require(sha(path) == expected, 'Artifact changed during independent audit: ' + path)
    return {'schema': 'e3-independent-result-audit-v0', 'consistent': True,
            'artifact': str(artifact), 'n_items': 209, 'n_answers': 3777, 'flood_events': 10,
            'verdict': recomputed['verdict'], 'seed_decisions': recomputed['seed_decisions'],
            'reproduction': reproduced, 'primary_metrics': recomputed['metrics'],
            'original_full_E2_references_revalidated': e2_checked, 'operational_amendment': amendment,
            'hashes_verified': tracked, 'audit_code_sha256': sha(__file__),
            'scientific_review': ['Primary BA includes paired source-positive tiles only; hard negatives remain separate.',
                'Event bootstrap uses 5000 paired event resamples with seed 20260925; two intervention arms must pass in the same seed.',
                'transformed_gold is required to be explicit null; this is stricter than the original standalone scorer.',
                'No substantive numeric error found in the reviewed primary scorer; its labels remain source-label agreement, not physical counterfactual accuracy.'],
            'limits': ['Checkpoint and LLM raw bytes on the server are not independently re-read here; their frozen manifest consistency is checked.',
                       'Projected tensors and intermediate generation logits are not saved; their finite checks remain runner-side evidence.',
                       'Secondary behaviour/transition summaries are not recomputed by this bounded audit.',
                       'Wall-clock generation order is not independently provable from JSONL contents; pre-intervention reproduction ordering is enforced by the reviewed frozen runner.',
                       'An intervention-sensitive outcome may reflect distribution shift and does not establish temporal reasoning.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Audit report JSON; parent directory must exist')
    parser.add_argument('--repo', type=Path, help='Optional repository for original full E2 answer SHA revalidation')
    args = parser.parse_args()
    try:
        report = audit(args.artifact, args.repo)
        code = 0
    except Exception as exc:
        report = {'schema': 'e3-independent-result-audit-v0', 'consistent': False,
                  'error_type': type(exc).__name__, 'error': str(exc), 'audit_code_sha256': sha(__file__)}
        code = 1
    report['checked_at'] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'consistent': report['consistent'], 'verdict': report.get('verdict'),
                      'error': report.get('error'), 'out': str(args.out)}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
