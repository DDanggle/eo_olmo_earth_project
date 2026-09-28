#!/usr/bin/env python3
"""Independent read-only C1 output audit. Imports no C1 runner or quality code.

Uses existing output artifacts only; no model inference, fitting, or selection.
CLI reports JSON to stdout and optionally --out outside the analyzed directory.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

FROZEN_PLAN_SHA = 'c7f6a51226d2222c870851b62b4ab4ad468aba6d1cf6055b547b9b2fff492324'
FROZEN_CODE_SHA = '0678ac93ec2e09f3fa03dd4f0046eb605d4684882b01c4576887da3e6f8cdaf4'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def indexed(items):
    check(len(items) == len({i['id'] for i in items}), 'Duplicate source/answer ID')
    return {i['id']: i for i in items}


def compare(actual, expected, location='result'):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), location + ': keys differ')
        for key in expected:
            compare(actual[key], expected[key], location + '.' + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), location + ': list length differs')
        for index, value in enumerate(expected):
            compare(actual[index], value, location + '[' + str(index) + ']')
    elif isinstance(expected, float):
        check(type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), location + ': numeric value differs')
    else:
        check(type(actual) is type(expected) and actual == expected, location + ': value differs')


def user_text(item):
    check(item['slots'] == ['pre_2', 'post'] and len(item['dates']) == 2, 'Target dates/slots mismatch')
    return ('These are 2 Sentinel-1 observations of the same area in chronological order, '
            'taken on ' + ', '.join(item['dates']) + ': <EO> Did a flood occur between the two observations? Answer with yes or no.')


def independently_group(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[(str(item['event']), user_text(item), tuple(item['slots']))].append(item)
    supported, unsupported = [], []
    for (event, text, slots), members in sorted(grouped.items()):
        pos = sorted(i['id'] for i in members if i['kind'] == 'pos')
        dry = sorted(i['id'] for i in members if i['kind'] == 'hard_neg')
        entry = dict(event=event, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     reconstructed_prompt=text, slots=list(slots), n_pos=len(pos), n_hard_neg=len(dry),
                     pos_ids=pos, hard_neg_ids=dry)
        if pos and dry:
            supported.append(entry)
        else:
            unsupported.append(dict(entry, reason='missing_one_source_class'))
    return supported, unsupported


def independent_events(groups, answers):
    event_strata = defaultdict(list)
    for group in groups:
        n_pos, n_dry = len(group['pos_ids']), len(group['hard_neg_ids'])
        tp = sum(answers[i]['parsed'] == 'yes' for i in group['pos_ids'])
        tn = sum(answers[i]['parsed'] == 'no' for i in group['hard_neg_ids'])
        recall, specificity = tp / n_pos, tn / n_dry
        event_strata[group['event']].append(dict(group, recall=recall, fpr=1 - specificity,
                                                ba=(recall + specificity) / 2,
                                                decision_contrast=recall + specificity - 1))
    output = {}
    for event, strata in sorted(event_strata.items()):
        output[event] = {metric: math.fsum(s[metric] for s in strata) / len(strata)
                         for metric in ('ba', 'decision_contrast', 'recall', 'fpr')}
        output[event].update(n_strata=len(strata), n_pos=sum(s['n_pos'] for s in strata),
                             n_hard_neg=sum(s['n_hard_neg'] for s in strata), strata=strata)
    return output


def interval(event_deltas):
    if len(event_deltas) < 2:
        return None
    values = np.asarray(event_deltas, dtype=float)
    draw = np.random.default_rng(20260925).integers(0, len(values), size=(5000, len(values)))
    means = np.sum(values[draw], axis=1) / len(values)
    return np.quantile(means, [.025, .975], method='linear').tolist()


def quality_artifact_audit(repo, quality_dir, target, manifest):
    """Reconcile record arithmetic and byte-hash links, without rereading raw masks."""
    records = rows(quality_dir / 'quality.jsonl')
    source = indexed(target)
    check(len(records) == 914 and len({r['tile'] for r in records}) == 914, 'Quality distinct tile count')
    check([r['id'] for r in records] == sorted(source), 'Quality record ordering/coverage')
    observed_id_digest = hashlib.sha256(json.dumps([r['id'] for r in records], separators=(',', ':')).encode()).hexdigest()
    check(observed_id_digest == manifest['selected_ids_sha256'], 'Quality selected ID digest')
    quality_plan = read(quality_dir / 'prereg.json')
    check(quality_plan['no_model_scores'] is True and manifest['no_model_scores'] is True, 'Quality role changed')
    compare(quality_plan['thresholds'], {'flood_min': .02, 'valid_min': .90, 'rounded_qa_abs_tolerance': .00005}, 'quality thresholds')
    checked_files = {}
    for filename, key in [('source.py', 'code_sha256'), ('prereg.json', 'prereg_sha256'),
                          ('quality.jsonl', 'quality_sha256'), ('summary.json', 'summary_sha256'),
                          ('symmetric_quality_ids.json', 'symmetric_ids_sha256')]:
        check(sha(quality_dir / filename) == manifest[key], 'Quality artifact hash differs: ' + filename)
        checked_files[filename] = manifest[key]
    # The server's historical directory name maps to the downloaded frozen C0 artifact.
    c0_dir = repo / 'artifacts/c0_linear_view_probe_v1_20260925'
    metadata = [('manifest.json', 'manifest_sha256'), ('items.jsonl', 'items_sha256'),
                ('source.py', 'code_sha256'), ('prereg.json', 'prereg_sha256')]
    expected_sources = {}
    for filename, key in metadata:
        relative = quality_plan['input_directory'] + '/' + filename
        observed = sha(c0_dir / filename)
        check(observed == quality_plan['expected_c0'][key] == manifest['source_files'][relative], 'Quality C0 source hash mismatch: ' + filename)
        expected_sources[relative] = observed
    check(manifest['c0_manifest_sha256'] == quality_plan['expected_c0']['manifest_sha256'], 'Quality C0 manifest link')
    check(manifest['c0_code_sha256'] == quality_plan['expected_c0']['code_sha256'], 'Quality C0 code link')
    check(manifest['c0_prereg_sha256'] == quality_plan['expected_c0']['prereg_sha256'], 'Quality C0 prereg link')
    by_event = defaultdict(Counter)
    all_ids, kept_ids, excluded_ids = [], [], []
    count_names = {f'mask_{m}|valid_{v}' for m in range(4) for v in range(2)}
    for record in records:
        item = source[record['id']]
        all_ids.append(item['id'])
        check(record['kind'] == item['kind'] and record['source_qa_flood_frac'] == item['flood_frac'], 'Quality original rounded flood fraction differs')
        counts = record['class_valid_counts']
        check(set(counts) == count_names and all(type(v) is int and 0 <= v <= 36864 for v in counts.values()), 'Invalid class-valid cross counts')
        check(sum(counts.values()) == record['total_px'] == 36864, 'Class-valid pixel total differs')
        labelled = sum(counts[f'mask_{m}|valid_1'] for m in (1, 2, 3))
        flood = counts['mask_3|valid_1']
        valid = labelled + counts['mask_0|valid_1']
        check(record['labelled_px'] == labelled and record['flood_px'] == flood, 'Quality pixel numerator differs')
        check(record['valid_but_unlabelled_px'] == counts['mask_0|valid_1'], 'Unlabelled valid pixels differ')
        compare(record['valid_frac'], labelled / 36864, 'valid fraction')
        compare(record['source_valid_frac'], valid / 36864, 'source valid fraction')
        compare(record['flood_frac'], flood / max(labelled, 1), 'flood fraction')
        check(abs(record['flood_frac'] - item['flood_frac']) <= .00005, 'Rounded QA fraction mismatch')
        check(np.issubdtype(np.dtype(record['mask_dtype']), np.integer)
              and np.issubdtype(np.dtype(record['valid_dtype']), np.integer), 'Noninteger source dtype')
        position = record['position']
        check(position['axis_order'] == 'row_y_then_column_x; original cache pixel grid', 'Position axis convention differs')
        for axis in ('rows', 'columns'):
            labelled_axis, flood_axis = position[f'labelled_{axis}_px'], position[f'flood_{axis}_px']
            check(len(labelled_axis) == len(flood_axis) == 192, 'Position support length differs')
            check(all(type(v) is int and 0 <= v <= 192 for v in labelled_axis + flood_axis), 'Position counts invalid')
            check(sum(labelled_axis) == labelled and sum(flood_axis) == flood, 'Position sum differs from cross counts')
            check(all(f <= l for f, l in zip(flood_axis, labelled_axis)), 'Flood support exceeds labelled support')
        for name, directory in [('mask', 'mask_u8'), ('valid', 'valid_u8')]:
            path = f'kurosiwo_s1_cache/{directory}/{item["tile"]}.npy'
            digest = record[name + '_sha256']
            check(record[name + '_path'] == path and re.fullmatch(r'[a-f0-9]{64}', digest), 'Invalid mask/valid source path/hash')
            check(manifest['source_files'][path] == digest, 'Record/manifest source hash mismatch')
            expected_sources[path] = digest
        accepted = labelled / 36864 >= .9
        check(record['eligible_symmetric_quality'] is accepted, 'Quality eligibility disagrees with counts')
        (kept_ids if accepted else excluded_ids).append(item['id'])
        by_event[str(item['event'])][item['kind']] += 1
        if accepted:
            by_event[str(item['event'])]['symmetric_' + item['kind']] += 1
    compare(manifest['source_files'], expected_sources, 'Quality exact source-file coverage')
    check(manifest['n'] == 914, 'Quality manifest count')
    stored_ids = read(quality_dir / 'symmetric_quality_ids.json')
    compare(stored_ids['ids'], sorted(kept_ids), 'Quality kept IDs')
    compare(stored_ids['excluded_ids'], sorted(excluded_ids), 'Quality excluded IDs')
    expected_by_kind = {kind: sorted(i for i in kept_ids if source[i]['kind'] == kind) for kind in ('pos', 'hard_neg')}
    compare(stored_ids['by_kind'], expected_by_kind, 'Quality kept IDs by kind')
    check(stored_ids['source_items_sha256'] == manifest['items_sha256'] and stored_ids['no_model_scores'] is True, 'Quality ID provenance')
    summary = read(quality_dir / 'summary.json')
    expected_scalars = {'n': 914, 'n_unique_tiles': 914, 'no_model_scores': True,
                       'counts': {'pos': 457, 'hard_neg': 457},
                       'symmetric_counts': {k: len(v) for k, v in expected_by_kind.items()},
                       'symmetric_n': len(kept_ids), 'events': {e: dict(c) for e, c in sorted(by_event.items())},
                       'thresholds': quality_plan['thresholds'], 'all_source_labels_verified': True}
    for key, value in expected_scalars.items():
        compare(summary[key], value, 'quality summary.' + key)
    for kind in ('pos', 'hard_neg'):
        for metric in ('valid_frac', 'source_valid_frac', 'flood_frac', 'flood_px'):
            values = np.array([r[metric] for r in records if r['kind'] == kind], dtype=float)
            q25, median, q75 = np.quantile(values, [.25, .5, .75], method='linear')
            expected = dict(n=len(values), mean=float(values.mean()), min=float(values.min()), q25=float(q25),
                            median=float(median), q75=float(q75), max=float(values.max()))
            compare(summary['by_kind'][kind][metric], expected, 'quality distribution.' + kind + '.' + metric)
    return {'consistent': True, 'n_records': 914, 'class_counts': expected_scalars['counts'],
            'symmetric_counts': expected_scalars['symmetric_counts'], 'symmetric_n': len(kept_ids),
            'quality_excluded': [{'id': i, 'event': str(source[i]['event']),
                                  'valid_frac': next(r['valid_frac'] for r in records if r['id'] == i)} for i in sorted(excluded_ids)],
            'events': expected_scalars['events'], 'position_vectors_checked': 914 * 4,
            'record_source_hash_links_checked': 914 * 2, 'c0_metadata_files_hashed': 4,
            'raw_array_bytes_reread': 0, 'artifact_hashes_verified': checked_files,
            'limit': 'Count/position arithmetic and 1828 record-to-manifest source hash links verified; server-only mask/valid bytes were not independently reread locally.'}


def independent_subset(items, predictions):
    groups, unsupported = independently_group(items)
    events = sorted({g['event'] for g in groups})
    seed_results, event_by_seed = {}, []
    for seed in (1, 2, 3):
        arms = {arm: independent_events(groups, predictions[arm, seed]) for arm in ('reader', 'blind')}
        deltas = [arms['reader'][e]['ba'] - arms['blind'][e]['ba'] for e in events]
        event_by_seed.append(deltas)
        seed_results[str(seed)] = dict(arms=arms,
            reader_macro_ba=math.fsum(arms['reader'][e]['ba'] for e in events) / len(events) if events else None,
            blind_macro_ba=math.fsum(arms['blind'][e]['ba'] for e in events) / len(events) if events else None,
            reader_minus_blind=math.fsum(deltas) / len(events) if events else None,
            ci95_delta=interval(deltas))
    event_means = [math.fsum(event_by_seed[s][j] for s in range(3)) / 3 for j in range(len(events))]
    return dict(candidate_count=len(items), n_events=len(events), events=events,
        support_sufficient_for_interpretation=len(events) >= 5,
        supported_n_pos=sum(g['n_pos'] for g in groups), supported_n_hard_neg=sum(g['n_hard_neg'] for g in groups),
        unsupported_strata=unsupported, per_seed=seed_results,
        seed_mean_reader_minus_blind=math.fsum(event_means) / len(events) if events else None,
        seed_mean_ci95_delta=interval(event_means), seed_mean_delta_by_event=dict(zip(events, event_means)))


def audit(repo, quality_dir, output_dir, expected_plan_sha=FROZEN_PLAN_SHA):
    repo, quality_dir, output_dir = map(Path, (repo, quality_dir, output_dir))
    report = {'schema': 'c1-independent-audit-v0', 'consistent': False, 'errors': [],
              'auditor_sha256': sha(__file__), 'recomputed_subsets': {},
              'limits': ['Recomputes existing binary decisions only; this is not new inference or confirmation.',
                         'Quality JSON hashes and contracts are checked; large server-only mask arrays are not reread here.']}
    try:
        plan = read(output_dir / 'analysis_plan.json')
        check(sha(output_dir / 'analysis_plan.json') == expected_plan_sha, 'Frozen analysis plan changed')
        check(sha(output_dir / 'source.py') == FROZEN_CODE_SHA, 'Frozen analysis source changed')
        result = read(output_dir / 'results.json')
        check(result['schema'] == 'c1-same-prompt-flood-diagnostic-v0' and result['primary_verdict'] is None, 'Unexpected schema/primary verdict')
        compare(result['scope'], plan['scope'], 'scope')
        compare(result['limits'], plan['limits'], 'limits')
        for relative, digest in plan['inputs_sha256'].items():
            check(sha(repo / relative) == digest, 'Frozen input bytes changed: ' + relative)
        quality_manifest = read(quality_dir / 'manifest.json')
        check(read(quality_dir / 'status.json')['status'] == 'complete', 'Quality audit incomplete')
        check(sha(quality_dir / 'quality.jsonl') == quality_manifest['quality_sha256'], 'Quality bytes changed')
        check(sha(repo / plan['items_path']) == quality_manifest['items_sha256'], 'Quality/input membership source differs')
        all_items = [i for i in rows(repo / plan['items_path']) if i['partition'] == 'test']
        item_map = indexed(all_items)
        check(len(all_items) == 1755, 'Expected 1755 full test items')
        target = [i for i in all_items if i['phen'] == 'flood' and i['kind'] in ('pos', 'hard_neg')]
        check(Counter(i['kind'] for i in target) == {'pos': 457, 'hard_neg': 457}, 'Target support changed')
        quality = indexed(rows(quality_dir / 'quality.jsonl'))
        check(set(quality) == {i['id'] for i in target}, 'Quality ID coverage differs')
        report['quality_audit'] = quality_artifact_audit(repo, quality_dir, target, quality_manifest)
        eligible, excluded = [], []
        for item in target:
            q = quality[item['id']]
            check(all(q[k] == item[k] for k in ('tile', 'dates', 'slots', 'kind'))
                  and str(q['event']) == str(item['event']) and q['source_answer'] == item['answer'], 'Quality/source metadata differ')
            check(0 <= q['valid_frac'] <= 1 and 0 <= q['flood_frac'] <= 1, 'Invalid quality fractions')
            accepted = q['valid_frac'] >= .9
            check(type(q['eligible_symmetric_quality']) is bool and q['eligible_symmetric_quality'] == accepted, 'Quality selection differs from fixed .90')
            check((item['kind'] == 'pos' and item['answer'] == 'yes' and q['flood_frac'] >= .02)
                  or (item['kind'] == 'hard_neg' and item['answer'] == 'no' and q['flood_px'] == 0 and accepted), 'Source class/quality mismatch')
            (eligible if accepted else excluded).append(item)
        candidate_sets = {'quality_symmetric': eligible, 'all_original_targets': target}
        expected_ids = {name: sorted(i['id'] for i in its) for name, its in candidate_sets.items()}
        compare(read(output_dir / 'selected_ids.json'), expected_ids, 'selected_ids')
        compare(result['quality_excluded_ids'], sorted(i['id'] for i in excluded), 'quality_excluded_ids')
        predictions = {}
        for arm in ('reader', 'blind'):
            for seed in (1, 2, 3):
                data = indexed(rows(repo / f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'))
                check(set(data) == set(item_map), 'Prediction full ID coverage mismatch')
                for key, item in item_map.items():
                    d = data[key]
                    check(all(d[k] == item[k] for k in ('tile', 'fold', 'phen', 'kind'))
                          and d['emb_item'] == key and d['text_gold'] == d['emb_gold'] == item['answer'], 'Prediction source metadata differ')
                    check(d['parsed'] in ('yes', 'no'), 'Invalid parse must not become no')
                predictions[arm, seed] = data
        blind_groups = defaultdict(list)
        for item in target:
            blind_groups[user_text(item)].append(item['id'])
        blind_checks = {}
        for seed in (1, 2, 3):
            violations = []
            for text, ids in blind_groups.items():
                values = {predictions['blind', seed][i]['parsed'] for i in ids}
                if len(values) > 1:
                    violations.append(dict(prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), ids=sorted(ids), outputs=sorted(values)))
            blind_checks[str(seed)] = dict(n_reconstructed_prompts=len(blind_groups), violations=violations, consistent=not violations)
        compare(result['blind_consistency'], blind_checks, 'blind_consistency')
        recomputed = {name: independent_subset(its, predictions) for name, its in candidate_sets.items()}
        compare(result['subsets'], recomputed, 'subsets')
        is_valid = all(v['consistent'] for v in blind_checks.values())
        compare(result['valid'], is_valid, 'valid')
        if not is_valid:
            check(bool(result.get('failure')), 'Missing blind-control failure disclosure')
        compare(read(output_dir / 'status.json'), {'status': 'complete' if is_valid else 'invalid', 'valid': is_valid}, 'status')
        compare(result['provenance'], dict(plan_sha256=sha(output_dir / 'analysis_plan.json'),
            code_sha256=FROZEN_CODE_SHA, quality_manifest_sha256=sha(quality_dir / 'manifest.json'),
            quality_sha256=sha(quality_dir / 'quality.jsonl'), selected_ids_sha256=sha(output_dir / 'selected_ids.json'),
            frozen_input_sha256=plan['inputs_sha256']), 'provenance')
        report.update(consistent=True, controls_valid=is_valid, quality_excluded_ids=sorted(i['id'] for i in excluded),
                      blind_consistency=blind_checks, recomputed_subsets=recomputed,
                      results_sha256=sha(output_dir / 'results.json'), plan_sha256=expected_plan_sha,
                      source_sha256=FROZEN_CODE_SHA)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out:
        check(not args.out.resolve().is_relative_to(args.analysis.resolve()), 'Report must be outside analyzed artifact directory')
    report = audit(args.repo, args.quality, args.analysis)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(text)
        summary = {'report': str(args.out), 'consistent': report['consistent'],
                   'controls_valid': report.get('controls_valid'), 'errors': report['errors'],
                   'subsets': {k: {field: value[field] for field in ('candidate_count', 'n_events', 'supported_n_pos', 'supported_n_hard_neg', 'seed_mean_reader_minus_blind', 'seed_mean_ci95_delta')}
                               for k, value in report['recomputed_subsets'].items()}}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(text, end='')
    return 2 if not report['consistent'] else (0 if report['controls_valid'] else 1)


if __name__ == '__main__':
    sys.exit(main())
