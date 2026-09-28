#!/usr/bin/env python3
"""Post-hoc, train-fitted question/class-prior audit of immutable OE1 outputs.
No production scorer imports, fitting on dev, image access, or new inference.
The 20260926 directory names identify the source run; checked_at is execution UTC.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

DEFAULT_ROOT = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe1_bentxt_v0_20260926')
PINNED = {
    'data/manifest.json': 'dd90a82744317e35f851b4c94a8a92f3207f74843960db7a21cde33050c422a5',
    'data/train.jsonl': 'ce35d09f5fe59242ea936783f423814cf540b65e6eeb5fe605bf1d904aa51f26',
    'data/dev.jsonl': 'b76abb71e3ac1a889d7c93931bf41e7b09efbac6fc85df32d54be2c04b6bef2a',
    'pilot_v1/training/manifest.json': '385bef8757575d08e4cc05905a35ac235e17949e805eda3509899ff8d525304b',
    'pilot_v1/training/status.json': '9ecdc441b954530a1f491ae3a6a4d1388b639ec6c516ec9b05c5b87a740420a8',
    'result_receipt_audit.json': '2a2f574f5d3a9d86937d5b889f91be96729864912abd834c91ac8de527575cd0',
}
# These five target phrases occur in TRAIN questions; no answer or image is used
# to map phrases to semantic class. This normalization is exploratory, not preregistered.
ALIASES = {
    'transitional woodlands or shrubs': 'Transitional woodland, shrub',
    'natural grasslands or sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'land principally occupied by agriculture with significant areas of natural vegetation':
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'moors, heathland, or sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'beaches, dunes, or sands': 'Beaches, dunes, sands',
}
ARMS = ('blind', 'frozen', 'joint')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, msg):
    if not value:
        raise ValueError(msg)


def metric(rows, predictions, gold_field='answer'):
    support = Counter(x[gold_field] for x in rows)
    require(set(support) <= {'yes', 'no'}, 'Nonbinary source label')
    correct = {y: sum(predictions[x['id']] == y for x in rows if x[gold_field] == y)
               for y in ('yes', 'no')}
    recalls = {y: correct[y] / support[y] if support[y] else None for y in ('yes', 'no')}
    return {'n': len(rows), 'support': {y: support[y] for y in ('yes', 'no')},
            'correct_by_label': correct, 'correct': sum(correct.values()),
            'accuracy': sum(correct.values()) / len(rows) if rows else None,
            'balanced_accuracy': sum(recalls.values()) / 2 if all(support[y] for y in ('yes', 'no')) else None,
            'recall_by_label': recalls,
            'invalid_count': sum(predictions[x['id']] not in ('yes', 'no') for x in rows)}


def majority_table(train, key):
    counts = defaultdict(Counter)
    for x in train:
        k = key(x)
        if k is not None:
            counts[k][x['answer']] += 1
    return {k: {'yes': c['yes'], 'no': c['no'], 'n': sum(c.values()),
                'prediction': 'yes' if c['yes'] > c['no'] else 'no',
                'tie': c['yes'] == c['no']} for k, c in sorted(counts.items())}


def grouping(rows, key):
    groups = defaultdict(list)
    for x in rows:
        groups[key(x)].append(x)
    return dict(groups)


def mixed_groups(rows, key):
    groups = grouping(rows, key)
    return {k: v for k, v in groups.items() if {x['answer'] for x in v} == {'yes', 'no'}}


def paired_outcomes(rows, first, second):
    c = Counter((first[x['id']] == x['answer'], second[x['id']] == x['answer']) for x in rows)
    a, b = metric(rows, first), metric(rows, second)
    return {'same_id_n': len(rows), 'both_correct': c[True, True], 'both_wrong': c[False, False],
            'first_only_correct': c[True, False], 'second_only_correct': c[False, True],
            'net_correct': c[True, False] - c[False, True],
            'pooled_ba_difference': a['balanced_accuracy'] - b['balanced_accuracy']
                if a['balanced_accuracy'] is not None and b['balanced_accuracy'] is not None else None}


def population(rows, predictions):
    return {'ids': [x['id'] for x in rows], 'n_patches': len({x['patch_id'] for x in rows}),
            'n_mgrs': len({x['mgrs'] for x in rows}),
            'metrics': {k: metric(rows, p) for k, p in predictions.items()},
            'paired_joint_minus': {k: paired_outcomes(rows, predictions['joint'], p)
                                   for k, p in predictions.items() if k != 'joint'}}


def conditional_report(all_rows, key, predictions):
    groups = grouping(all_rows, key)
    supported = mixed_groups(all_rows, key)
    ids = {x['id'] for v in supported.values() for x in v}
    rows = [x for x in all_rows if x['id'] in ids]
    base = population(rows, predictions)
    base.update({'all_group_count': len(groups), 'supported_group_count': len(supported),
                 'excluded_group_count': len(groups) - len(supported),
                 'excluded_ids': [x['id'] for x in all_rows if x['id'] not in ids],
                 'selection': 'Both source-answer classes occur in this dev stratum; no predictions used.',
                 'equal_stratum_macro_ba': {a: sum(metric(v, p)['balanced_accuracy'] for v in supported.values()) / len(supported)
                     if supported else None for a, p in predictions.items()},
                 'strata': [{'key': list(k) if isinstance(k, tuple) else k, 'ids': [x['id'] for x in v],
                             'metrics': {a: metric(v, p) for a, p in predictions.items()}}
                            for k, v in sorted(supported.items())],
                 'excluded_strata': [{'key': list(k) if isinstance(k, tuple) else k,
                                      'ids': [x['id'] for x in v], 'support': dict(Counter(x['answer'] for x in v))}
                                     for k, v in sorted(groups.items()) if k not in supported]})
    return base


def selftests():
    fixture = [{'id': str(i), 'answer': a, 'key': 'x'} for i, a in enumerate(['yes', 'yes', 'yes', 'no'])]
    assert metric(fixture, {x['id']: 'yes' for x in fixture})['balanced_accuracy'] == .5
    assert metric(fixture, {x['id']: None for x in fixture})['invalid_count'] == 4
    assert metric(fixture, {x['id']: None for x in fixture})['balanced_accuracy'] == 0
    assert majority_table(fixture[:1] + fixture[3:], lambda x: x['key'])['x']['prediction'] == 'no'
    assert metric(fixture[:1], {'0': 'yes'})['balanced_accuracy'] is None
    assert majority_table(fixture, lambda x: None) == {}
    assert len(mixed_groups(fixture, lambda x: x['key'])) == 1
    assert len(mixed_groups(fixture[:3], lambda x: x['key'])) == 0
    assert paired_outcomes(fixture, {x['id']: x['answer'] for x in fixture}, {x['id']: 'no' for x in fixture})['net_correct'] == 3
    return 9


def run(root, out):
    checked = {}
    def read(rel):
        path = root / rel
        checked[rel] = sha(path)
        if rel in PINNED:
            require(checked[rel] == PINNED[rel], f'Pinned source changed: {rel}')
        return path.read_text()
    def js(rel):
        return json.loads(read(rel))
    def jl(rel):
        return [json.loads(x) for x in read(rel).splitlines() if x.strip()]
    tests_passed = selftests()
    m = js('pilot_v1/training/manifest.json')
    status = js('pilot_v1/training/status.json')
    require(m['valid'] is True and status['valid'] is True and status['status'] == 'completed', 'Incomplete source run')
    require(status['manifest_sha256'] == checked['pilot_v1/training/manifest.json'], 'Status manifest mismatch')
    receipt = js('result_receipt_audit.json')
    require(receipt['valid_within_scope'] is True, 'Source receipt audit invalid')
    dm = js('data/manifest.json'); train = jl('data/train.jsonl'); dev = jl('data/dev.jsonl')
    items = js('pilot_v1/training/items.json')
    prompts = js('pilot_v1/training/prompts.json')
    summary = js('pilot_v1/training/summary.json')
    donors = js('pilot_v1/training/donor_plan.json')['mapping']
    run_config = js('pilot_v1/training/run_config.json')
    rows = jl('pilot_v1/training/predictions.jsonl')
    read('pilot_v1/code_snapshot/train_pilot.py')
    read('pilot_v1/code_snapshot/eo_model.py')
    for rel, digest in checked.items():
        if rel.startswith('data/'):
            require(digest == m['source_files_sha256']['/home/work/data/olmoearth/oe1_bentxt_v0/' + rel], 'Training source hash mismatch')
        if rel.startswith('pilot_v1/training/') and Path(rel).name not in ('manifest.json', 'status.json'):
            require(digest == m['files_sha256'][Path(rel).name], 'Training output hash mismatch')
            require(digest == receipt['verified_result_files_sha256'][Path(rel).name], 'Receipt hash mismatch')
    require((len(train), len(dev), len(rows)) == (1024, 256, 1740), 'Unexpected populations')
    require(len(items) == len(train) + len(dev), 'Items length')
    ids = [x['id'] for x in train + dev]
    require(len(set(ids)) == len(ids), 'Duplicate IDs')
    for x, it in zip(train + dev, items):
        require(all(it[k] == v for k, v in x.items()), 'Saved item/source row differs')
        require(x['input'] == x['question'] and x['output'] == x['answer'] and x['answer'] in ('yes', 'no'), 'Question/answer aliases')
    require(not {x['patch_id'] for x in train} & {x['patch_id'] for x in dev}, 'Shared train/dev patch')
    require(not {x['mgrs'] for x in train} & {x['mgrs'] for x in dev}, 'Shared train/dev MGRS')
    prompt_map = {x['id']: x for x in prompts['prompts']}
    require(set(prompt_map) == set(ids), 'Prompt ID coverage')
    for x in train + dev:
        p = prompt_map[x['id']]
        require(p['original_input'] == x['input'], 'Changed saved original question')
        require(p['user_text'] == 'Sentinel-2 satellite observation: <EO>\n' + x['input'] + '\nAnswer with yes or no.', 'Unexpected prompt')
    known = {x['query_class'] for x in train if x['query_class'] is not None}
    phrases = {k.lower(): k for k in known} | ALIASES
    require(len(phrases) == 19 and all(any(p in x['question'].lower() for x in train) for p in phrases), 'Phrase dictionary not covered by train')
    def semantic_class(x):
        hits = [label for phrase, label in phrases.items() if phrase in x['question'].lower()]
        require(len(hits) == 1, 'Ambiguous/uncovered question phrase')
        require(x['query_class'] is None or x['query_class'] == hits[0], 'Recorded class differs from question phrase')
        return hits[0]
    for x in train + dev:
        semantic_class(x)
    dev_byid = {x['id']: x for x in dev}
    bycondition = defaultdict(dict)
    for p in rows:
        require(p['arm'] in ARMS and p['control'] in ('real', 'zero', 'observation_swap'), 'Unexpected condition')
        require(p['id'] in dev_byid and p['presented_id'] in dev_byid, 'Non-dev prediction')
        src, shown = dev_byid[p['id']], dev_byid[p['presented_id']]
        require(p['source_gold'] == src['answer'] and p['eval_gold'] == shown['answer'], 'Wrong scored labels')
        require(p['patch_id'] == src['patch_id'] and p['presented_patch_id'] == shown['patch_id'] and p['presented_date'] == shown['date'], 'Prediction metadata mismatch')
        parsed = p['raw_token'].strip().lower()
        require(p['parsed'] == (parsed if parsed in ('yes', 'no') else None), 'Free-vocabulary parse differs')
        require(p['image_tokens_used'] == (p['arm'] != 'blind' and p['control'] != 'zero'), 'Image-control flag mismatch')
        if p['control'] == 'observation_swap':
            require(donors[p['id']] == p['presented_id'] and src['question'] == shown['question'] and src['answer'] != shown['answer'], 'Invalid donor mapping')
        else:
            require(p['id'] == p['presented_id'], 'Native/zero item mismatch')
        key = p['arm'] + '/' + p['control']
        require(p['id'] not in bycondition[key], 'Duplicate condition ID')
        bycondition[key][p['id']] = p
    original_metrics = {}
    for key, records in bycondition.items():
        expected = set(donors) if key.endswith('/observation_swap') else set(dev_byid)
        require(set(records) == expected, 'Incomplete condition population')
        mm = metric(list(records.values()), {i: p['parsed'] for i, p in records.items()}, 'eval_gold')
        for name in ('accuracy', 'balanced_accuracy', 'recall_by_label', 'invalid_count'):
            require(mm[name] == summary['arms'][key]['parsed'][name], 'Original summary metric differs')
        original_metrics[key] = mm
    tables = {'class_majority': majority_table(train, semantic_class),
              'exact_question_majority': majority_table(train, lambda x: x['question']),
              'recorded_class_majority': majority_table(train, lambda x: x['query_class'])}
    key_functions = {'class_majority': semantic_class, 'exact_question_majority': lambda x: x['question'],
                     'recorded_class_majority': lambda x: x['query_class']}
    predictions = {a: {i: p['parsed'] for i, p in bycondition[a + '/real'].items()} for a in ARMS}
    for name, table in tables.items():
        predictions[name] = {x['id']: table.get(key_functions[name](x), {'prediction': 'no'})['prediction'] for x in dev}
    exact_seen = [x for x in dev if x['question'] in tables['exact_question_majority']]
    recorded_known = [x for x in dev if x['query_class'] is not None]
    conditional = conditional_report(dev, lambda x: (semantic_class(x), x['question']), predictions)
    require(set(conditional['ids']) == set(donors), 'Conditional population differs from original donor eligibility')
    consistency = []
    for question, group in sorted(grouping(dev, lambda x: x['question']).items()):
        outputs = {a: dict(Counter(predictions[a][x['id']] for x in group)) for a in ARMS}
        consistency.append({'question': question, 'ids': [x['id'] for x in group], 'outputs': outputs,
                            'blind_identical': len(outputs['blind']) == 1})
    per_class = []
    for cls, group in sorted(grouping(dev, semantic_class).items()):
        per_class.append({'class': cls, 'train_counts': tables['class_majority'][cls],
                          'dev_support': dict(Counter(x['answer'] for x in group)),
                          'same_rows_comparison': population(group, predictions)})
    decomposition = {}
    for baseline in ('class_majority', 'exact_question_majority', 'blind', 'frozen'):
        decomposition[baseline] = {}
        for correct in (True, False):
            subset = [x for x in dev if (predictions[baseline][x['id']] == x['answer']) == correct]
            decomposition[baseline]['baseline_correct' if correct else 'baseline_wrong'] = population(subset, predictions)
    matched_controls = {}
    for a in ARMS:
        native = {i: p['parsed'] for i, p in bycondition[a + '/real'].items()}
        zero = {i: p['parsed'] for i, p in bycondition[a + '/zero'].items()}
        mixed = [x for x in dev if x['id'] in set(conditional['ids'])]
        matched_controls[a] = {'all_dev_real_minus_zero': paired_outcomes(dev, native, zero),
                               'mixed_question_real_minus_zero': paired_outcomes(mixed, native, zero)}
    report = {
        'schema': 'oe2-posthoc-question-shortcut-audit-v0', 'valid_within_scope': True,
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Post-hoc CPU reaggregation of existing source labels and saved free-vocabulary first-token responses. No inference, image/tensor reload, training, or preregistration change.',
        'artifact_root': str(root), 'script_sha256': sha(Path(__file__)), 'input_sha256': checked,
        'synthetic_selftests_passed': tests_passed,
        'method': {'fit_partition': 'train only', 'tie_prediction': 'no', 'unseen_or_missing_prediction': 'no',
                   'semantic_class_parser': phrases,
                   'normalization_note': 'Five obvious target-phrase aliases repair missing metadata for analysis only. All 19 phrases occur in train; no source answer, patch label, or imagery used to choose a target class. Original files unchanged.',
                   'primary_saved_prediction_field': 'parsed', 'invalid_predictions': 'Wrong, retained in denominator',
                   'conditional_weighting': 'Within exact question + semantic class, average positive and negative recall, then equally average supported strata.',
                   'no_uncertainty_claim': 'No confirmatory test or iid-binomial CI. One training seed; 128 patches in 11 dev MGRS groups; conditional support is sparse and selected post hoc by source labels.'},
        'population': {'train_questions': len(train), 'dev_questions': len(dev),
                       'train_patches': len({x['patch_id'] for x in train}), 'dev_patches': len({x['patch_id'] for x in dev}),
                       'train_mgrs': len({x['mgrs'] for x in train}), 'dev_mgrs': len({x['mgrs'] for x in dev}),
                       'train_question_strings': len({x['question'] for x in train}), 'dev_question_strings': len({x['question'] for x in dev}),
                       'recorded_query_class_missing_train': sum(x['query_class'] is None for x in train),
                       'recorded_query_class_missing_dev': sum(x['query_class'] is None for x in dev),
                       'normalized_class_coverage_train': len(train), 'normalized_class_coverage_dev': len(dev),
                       'exact_question_seen_dev': len(exact_seen), 'exact_question_unseen_dev': len(dev) - len(exact_seen),
                       'cross_split_minimum_center_km_from_manifest': dm['cross_split_minimum_center_km'],
                       'training_seed': run_config['training_seed']},
        'train_fitted_tables': tables,
        'all_dev': population(dev, predictions),
        'exact_question_seen_in_train': population(exact_seen, predictions),
        'exact_question_unseen_in_train': population([x for x in dev if x not in exact_seen], predictions),
        'recorded_class_available': population(recorded_known, predictions),
        'conditional_both_labels_same_class': conditional_report(dev, semantic_class, predictions),
        'conditional_both_labels_same_question_and_class': conditional,
        'per_class': per_class,
        'per_mgrs': [{'mgrs': k, 'comparison': population(v, predictions)} for k, v in sorted(grouping(dev, lambda x: x['mgrs']).items())],
        'joint_paired_error_decomposition': decomposition,
        'saved_original_all_control_metrics_verified': original_metrics,
        'matched_real_zero_controls': matched_controls,
        'same_prompt_blind_consistency': {'all_groups': len(consistency),
            'duplicate_groups': sum(len(x['ids']) > 1 for x in consistency),
            'violations': sum(not x['blind_identical'] for x in consistency), 'groups': consistency},
        'limitations': dm['limitations'] + [
            'Class-majority equals frozen pooled BA but need not make identical predictions.',
            'Question-conditioned support excludes one-label strata and is not a full-dev performance estimate.',
            'Real EO features depend on multispectral observations AND acquisition-date encoding; this analysis does not isolate pixels from season/date or prove fine-grained spatial reasoning.',
            'Blind has its own trained projector; joint-minus-blind is a descriptive system contrast, not a pure intervention on one trained system.',
            'Zero EO features can be out of distribution; source-label agreement after zeroing is not accuracy for a new physical scene.',
            'This normalizer and analysis were designed after OE1 results existed; no new claim of preregistered success.',
            'One patch provides two questions and MGRS groups may be correlated. No claim of 256 independent observations.',
        ],
    }
    for rel, digest in checked.items():
        require(sha(root / rel) == digest, 'Input changed during audit')
    out.mkdir(parents=True, exist_ok=True)
    output = out / 'shortcut_audit.json'
    require(not output.exists(), 'Refusing to overwrite audit result')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (out / 'dev_predictions.jsonl').open('x') as stream:
        for x in dev:
            record = {'id': x['id'], 'patch_id': x['patch_id'], 'mgrs': x['mgrs'], 'question': x['question'],
                      'source_answer': x['answer'], 'recorded_query_class': x['query_class'],
                      'normalized_query_class': semantic_class(x),
                      'exact_question_seen_in_train': x['question'] in tables['exact_question_majority'],
                      'mixed_question_and_class_supported': x['id'] in set(conditional['ids']),
                      'predictions': {a: p[x['id']] for a, p in predictions.items()}}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'all_dev_ba': {a: p['balanced_accuracy'] for a, p in report['all_dev']['metrics'].items()},
                      'conditional_question_macro': conditional['equal_stratum_macro_ba'],
                      'conditional_n': len(conditional['ids']), 'conditional_groups': conditional['supported_group_count'],
                      'script_sha256': report['script_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.artifact.resolve(), args.out.resolve())
