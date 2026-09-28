#!/usr/bin/env python3
"""Attach audited, already saved E5 answers to existing flood cases in new v11.

No network, inference, voting, confidence, new scores, date repair, or labeling.
Copies the audited source JSON/JSONL and records exact prediction line numbers.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

V10_SHA = '0662a7399cfeb40b6cbb67680bb25f2c1d3bb4b469db1777f8941ce6f0b08136'
PLAN_SHA = 'fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
PARENT_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
AUDITOR_SHA = 'a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825'
SEEDS = ('1', '2', '3')
EVALUATIONS = ('full/native', 'pair/native', 'later/native', 'delta/native', 'full/full_no_delta')
SOURCE_NAMES = ('manifest.json', 'prereg.json', 'status.json', 'scores.json', 'predictions.jsonl',
                'items.jsonl', 'eval_sets.json', 'prompts.jsonl', 'parent_snapshot/c1_quality.jsonl', 'parent_snapshot/c1_selected_ids.json')
EXTRAS = {'v10_build_manifest.json', 'http_readback_checks_v10.json', 'browser_checks_v10.json'}
VERDICTS = {'pair_preserves_source_discrimination_at_equal_budget', 'explicit_difference_helps_at_this_budget', 'mixed_or_inconclusive'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw):
    def unique(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def child(root, name):
    p = Path(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts, 'Unsafe relative source path')
    target = root / p
    require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root), 'Missing/escaping source: ' + name)
    return target


def pin(path, expected, tracking):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid source hash')
    require(sha(path) == expected, 'Source hash mismatch: ' + str(path))
    tracking[str(path)] = expected


def aware(value):
    require(isinstance(value, str) and datetime.fromisoformat(value.replace('Z', '+00:00')).tzinfo is not None,
            'Timezone-aware completion/audit timestamp required')


def unique(rows):
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'] and row['id'] not in result,
                'Missing/duplicate source ID')
        result[row['id']] = row
    return result


def load_sources(e5, audit_path, tracking):
    pin(audit_path, sha(audit_path), tracking); audit = read(audit_path)
    require(audit.get('schema') == 'e5-independent-result-audit-v0' and audit.get('consistent') is True
            and audit.get('checkpoint_tensors_loaded_and_checked_on_cpu') is True
            and audit.get('audit_code_sha256') == AUDITOR_SHA, 'Completed independent CPU tensor audit required')
    require(tuple(audit.get(k) for k in ('n_models', 'n_steps', 'n_training_exposures', 'n_answers', 'primary_n', 'primary_events'))
            == (12, 19080, 152424, 26325, 902, 8), 'Independent E5 population/budget differs')
    aware(audit['checked_at']); original = Path(audit['artifact'])
    require(original.is_absolute() and '..' not in original.parts, 'Original artifact path must be absolute')
    require(not (e5 / 'failure.json').exists(), 'E5 failure record present')
    loaded, original_pins = {}, {}
    for name in SOURCE_NAMES:
        expected = audit['hashes_verified'].get(str(original / name))
        pin(child(e5, name), expected, tracking); original_pins[str(original / name)] = expected
        raw = (e5 / name).read_bytes()
        loaded[name] = ([decode(line) for line in raw.splitlines() if line.strip()] if name.endswith('.jsonl') else decode(raw))
    manifest, plan, status, scores = [loaded[n] for n in SOURCE_NAMES[:4]]
    require(original == Path(plan['root']) / plan['output_directory'], 'Original artifact differs from fixed plan')
    require(tracking[str(e5 / 'manifest.json')] == PARENT_SHA and tracking[str(e5 / 'prereg.json')] == PLAN_SHA,
            'Fixed E5 manifest/preregistration differs')
    require(manifest['schema'] == 'e5-equal-budget-prepared-v0' and manifest['files_sha256']['prereg.json'] == PLAN_SHA,
            'Prepared E5 schema/plan differs')
    for name in ('items.jsonl', 'eval_sets.json'):
        require(manifest['files_sha256'][name] == tracking[str(e5 / name)], 'Prepared source identity differs: ' + name)
    require(manifest['parent_snapshot_sha256'] == plan['parents'], 'Frozen parent pins differ')
    for name in ('c1_quality.jsonl', 'c1_selected_ids.json'):
        require(plan['parents'][name] == tracking[str(e5 / 'parent_snapshot' / name)], 'C1 parent identity differs')
    require(status['status'] == 'completed' and status['n_models'] == 12 and status['n_rows'] == 26325,
            'E5 is not complete')
    aware(status['at'])
    require(scores['schema'] == 'e5-equal-budget-scores-v0' and scores['valid'] is True
            and scores['manifest_sha256'] == PARENT_SHA, 'E5 scored result invalid')
    require(scores['predictions_sha256'] == tracking[str(e5 / 'predictions.jsonl')], 'Scores use another prediction file')
    if 'prompts_sha256' in scores:
        require(scores['prompts_sha256'] == tracking[str(e5 / 'prompts.jsonl')], 'Scores use another prompt file')
    require(scores['coverage'] == {'expected': 26325, 'received': 26325, 'n_items': 5989,
            'n_train': 4234, 'n_test': 1755, 'n_primary': 902}, 'Scored coverage differs')
    require(scores['verdict'] in VERDICTS and scores['verdict'] == status['verdict'] == audit['verdict'], 'Registered verdict differs')
    require(set(scores['metrics']) == set(audit['primary_metrics']) == set(SEEDS)
            and {s: scores['metrics'][s]['primary_same_prompt'] for s in SEEDS} == audit['primary_metrics'],
            'Independent primary metrics disagree')
    require(scores['seed_decisions'] == audit['seed_decisions'], 'Independent seed decisions disagree')
    return loaded, audit, original_pins


def population(loaded):
    items = loaded['items.jsonl']; source = unique(items)
    require(len(source) == 5989, 'Expected unchanged 5989 source items')
    prompts = loaded['prompts.jsonl']
    require(len(prompts) == 5989 and [p['id'] for p in prompts] == [x['id'] for x in items], 'Saved prompt order/coverage differs')
    for index, (saved, item) in enumerate(zip(prompts, items)):
        require(type(saved['pair_index']) is int and saved['pair_index'] == index
                and saved['source_gold'] == item['answer'] and saved['n_eo_tokens'] == 192
                and isinstance(saved['user_text'], str), 'Saved prompt index/label/token budget differs')
    counts = {('train', 'flood', 'pos'): 1066, ('train', 'flood', 'neg'): 1066, ('train', 'flood', 'hard_neg'): 1066,
              ('train', 'landslide', 'pos'): 518, ('train', 'landslide', 'neg'): 518,
              ('test', 'flood', 'pos'): 457, ('test', 'flood', 'neg'): 457, ('test', 'flood', 'hard_neg'): 457,
              ('test', 'landslide', 'pos'): 192, ('test', 'landslide', 'neg'): 192}
    require(Counter((x['partition'], x['phen'], x['kind']) for x in items) == counts, 'Source partition/category counts differ')
    require(all(x['answer'] == ('yes' if x['kind'] == 'pos' else 'no') for x in items), 'Source kind/label mismatch')
    for item in items:
        if item['phen'] == 'flood':
            require(str(item['event']) == str(item['cluster']) and item['slots'] ==
                    (['pre_1', 'pre_2'] if item['kind'] == 'neg' else ['pre_2', 'post']),
                    'Flood event/cluster/slots differ')
    test = {i: x for i, x in source.items() if x['partition'] == 'test'}
    target = {i: x for i, x in test.items() if x['phen'] == 'flood' and x['kind'] in ('pos', 'hard_neg')}
    require(len(target) == len({x['tile'] for x in target.values()}) == 914, 'Target tile coverage differs')
    train_tiles = {x['tile'] for x in items if x['partition'] == 'train'}
    require(not train_tiles & {x['tile'] for x in test.values()}, 'Train/test tile leakage')
    ev = loaded['eval_sets.json']
    require(set(ev) == {'all_test', 'primary_same_prompt', 'paired_flood', 'hard_negative_flood', 'landslide', 'e3_subset'},
            'Evaluation keys differ')
    for name, ids in ev.items():
        require(isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids))
                and set(ids) <= set(test), 'Invalid/duplicate/out-of-test evaluation ID')
        members = set(ids)
        require(ids == [x['id'] for x in items if x['id'] in members], 'Frozen evaluation order differs: ' + name)
    require(set(ev['all_test']) == set(test), 'All-test IDs differ')
    for name, phen, kinds in [('paired_flood', 'flood', ('pos', 'neg')),
            ('hard_negative_flood', 'flood', ('hard_neg',)), ('landslide', 'landslide', ('pos', 'neg'))]:
        require(set(ev[name]) == {i for i, x in test.items() if x['phen'] == phen and x['kind'] in kinds}, 'Evaluation support differs: ' + name)
    require(len(ev['e3_subset']) == 209, 'E3 frozen subset differs')
    primary = set(ev['primary_same_prompt'])
    require(len(primary) == 902 and primary <= set(target)
            and Counter(target[i]['kind'] for i in primary) == {'pos': 445, 'hard_neg': 457}, 'Primary support differs')
    quality = unique(loaded['parent_snapshot/c1_quality.jsonl'])
    require(set(quality) == set(target), 'Quality target coverage differs')
    eligible = set()
    for key, item in target.items():
        q = quality[key]
        require(all(q[k] == item[k] for k in ('tile', 'dates', 'slots', 'kind')) and str(q['event']) == str(item['event'])
                and q['source_answer'] == item['answer'], 'Quality metadata/gold differs')
        require(type(q['total_px']) is int and type(q['labelled_px']) is int and q['total_px'] > 0
                and 0 <= q['labelled_px'] <= q['total_px'], 'Invalid label-coverage counts')
        ratio = q['labelled_px'] / q['total_px']
        require(type(q['valid_frac']) in (int, float) and math.isfinite(q['valid_frac']) and q['valid_frac'] == ratio
                and type(q['eligible_symmetric_quality']) is bool and q['eligible_symmetric_quality'] == (ratio >= .90),
                'Quality eligibility differs from known-label fraction')
        if ratio >= .90:
            eligible.add(key)
    selected = loaded['parent_snapshot/c1_selected_ids.json']
    require(len(eligible) == 907 and selected == {'quality_symmetric': sorted(eligible), 'all_original_targets': sorted(target)},
            'Frozen quality selection differs')
    groups = defaultdict(list)
    for key in eligible:
        x = target[key]; groups[(str(x['event']), tuple(x['dates']), tuple(x['slots']))].append(key)
    supported_groups = {key: ids for key, ids in groups.items() if {target[i]['kind'] for i in ids} == {'pos', 'hard_neg'}}
    require({i for ids in supported_groups.values() for i in ids} == primary
            and len({key[0] for key in supported_groups}) == 8, 'Primary IDs do not equal supported quality strata')
    return items, source, test, target, eligible, primary


def index_predictions(rows, items, source, test):
    require(len(rows) == 26325, 'Expected all 26325 saved E5 predictions')
    indices = {x['id']: j for j, x in enumerate(items)}
    expected = {(s, arm, i) for s in SEEDS for arm in EVALUATIONS for i in test}
    indexed = {}
    for row in rows:
        require(type(row.get('seed')) is int and row['seed'] in (1, 2, 3), 'Invalid prediction seed')
        arm = str(row.get('model_arm')) + '/' + str(row.get('eval_arm'))
        key = (str(row['seed']), arm, row.get('id'))
        require(key in expected and key not in indexed, 'Unexpected/duplicate prediction ID/condition/seed')
        x = source[row['id']]
        require(all(row.get(k) == x[k] for k in ('tile', 'phen', 'kind')) and str(row.get('cluster')) == str(x['cluster']),
                'Prediction source metadata differs')
        require(type(row.get('pair_index')) is int and row['pair_index'] == indices[row['id']], 'Prediction item index differs')
        require(row.get('source_gold') == x['answer'] and 'transformed_gold' in row and row['transformed_gold'] is None,
                'Source label changed or transformed label invented')
        require(isinstance(row.get('answer_raw'), str) and 'parsed' in row and row['parsed'] in ('yes', 'no', None), 'Invalid saved response')
        parsed = re.search(r'\b(yes|no)\b', row['answer_raw'].strip().lower())
        require(row['parsed'] == (parsed[1] if parsed else None), 'Saved parsed/raw response disagrees')
        indexed[key] = row
    require(set(indexed) == expected, 'Missing saved condition/seed/ID')
    return indexed, indices


def attach_cases(reader, catalog, loaded, audit, source_pins, audit_hash):
    items, source, test, target, eligible, primary = population(loaded)
    indexed, indices = index_predictions(loaded['predictions.jsonl'], items, source, test)
    require(reader['schema_version'] == 'eo_reader_cases_v1' and reader['run_id'] == 'E2'
            and reader['source_role'] == 'historical_model_output' and len(reader['cases']) == 914, 'Existing case schema/count differs')
    require('e5_control_provenance' not in reader, 'E5 cases already attached')
    require(set(reader['cases']) == {x['tile'] for x in target.values()}, 'Existing case tiles differ from E5 target tiles')
    catalog_rows = unique(catalog['records'])
    hashes = {name.replace('/', '__').replace('.', '_'): value for name, value in source_pins.items()}
    hashes['independent_audit'] = audit_hash
    output = copy.deepcopy(reader); join_index = []
    original_rows = loaded['predictions.jsonl']
    row_numbers = {(str(row['seed']), row['model_arm'] + '/' + row['eval_arm'], row['id']): j
                   for j, row in enumerate(original_rows, 1)}
    for key, item in sorted(target.items(), key=lambda p: p[1]['tile']):
        tile = item['tile']; case = output['cases'][tile]; record = catalog_rows.get(tile)
        require(record and record['dataset'] == 'kurosiwo' and record['split'] == 'test'
                and str(record['aoi_id']) == str(item['event']), 'Target absent or wrong event in catalog')
        require(case['tile'] == tile and case['question_id'] == key and case['reference_label'] == item['answer']
                and case['run_id'] == 'E2' and case['source_role'] == 'historical_model_output'
                and case['dates'] == item['dates'] and case['slots'] == item['slots'] == ['pre_2', 'post'],
                'Existing case and E5 question/label/date/slots differ')
        event_day = date.fromisoformat(record['event_date'])
        require(item['dates'] == [(event_day - timedelta(days=12)).isoformat(), event_day.isoformat()],
                'Existing synthetic date contract changed')
        expected_prompt = ('These are 2 Sentinel-1 observations of the same area in chronological order, '
            f"taken on {', '.join(item['dates'])}: <EO> Did a flood occur between the two observations? Answer with yes or no.")
        saved_prompt = loaded['prompts.jsonl'][indices[key]]
        require(case['prompt'] == expected_prompt == saved_prompt['user_text']
                and case['prompt_sha256'] == hashlib.sha256(expected_prompt.encode()).hexdigest(),
                'Existing and actual saved E5 source question text differ')
        require(case['source_sha256']['items'] == source_pins['items.jsonl']
                and case['source_sha256']['quality'] == source_pins['parent_snapshot/c1_quality.jsonl'], 'Existing case source identity differs')
        membership = case['c1_membership']
        require(type(membership['quality_eligible']) is bool and membership['quality_eligible'] == (key in eligible)
                and type(membership['supported_stratum']) is bool and membership['supported_stratum'] == (key in primary)
                and membership['scope'] == 'quality_symmetric_primary_subset', 'Existing C1 membership differs')
        require('e5_control' not in case, 'Existing E5 per-case control must not be replaced')
        case['e5_control'] = {'run_id': 'E5-EB-v0', 'question_id': key, 'source_role': 'historical_model_output',
            'source_gold': item['answer'], 'predictions': {arm: {s: indexed[(s, arm, key)]['parsed'] for s in SEEDS} for arm in EVALUATIONS},
            'primary_membership': key in primary, 'dates': copy.deepcopy(item['dates']), 'slots': copy.deepcopy(item['slots']),
            'completed_at': loaded['status.json']['at'], 'source_sha256': copy.deepcopy(hashes)}
        join_index.append({'tile': tile, 'question_id': key, 'pair_index': indices[key],
            'source_gold': item['answer'], 'primary_membership': key in primary,
            'prompt_record_number_1_based': indices[key] + 1,
            'prediction_record_numbers_1_based': {arm: {s: row_numbers[(s, arm, key)] for s in SEEDS} for arm in EVALUATIONS}})
    restored = copy.deepcopy(output)
    for case in restored['cases'].values():
        case.pop('e5_control')
    require(restored == reader, 'Existing E2/E3/E4 reader payload changed')
    return output, join_index


def build(base, e5, audit_path, out, html=None):
    base, e5, audit_path, out = [Path(p).resolve() for p in (base, e5, audit_path, out)]
    html = Path(html).resolve() if html is not None else None
    require(not out.exists(), 'New output required; no overwrite')
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in (base, e5)), 'Output overlaps input tree')
    tracking = {}; pin(base / 'v10_build_manifest.json', V10_SHA, tracking); manifest = read(base / 'v10_build_manifest.json')
    require(manifest['schema'] == 'eo-v10-delta-wording-v0', 'Expected fixed v10 snapshot')
    require(not any(p.is_symlink() for p in base.rglob('*')), 'Symlink in base snapshot')
    actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    require(actual == set(manifest['output_files_sha256']) | EXTRAS, 'Unexpected v10 files')
    for name, digest in manifest['output_files_sha256'].items():
        pin(child(base, name), digest, tracking)
    base_hashes = {name: sha(child(base, name)) for name in sorted(actual)}
    loaded, audit, original_pins = load_sources(e5, audit_path, tracking)
    source_pins = {name: tracking[str(e5 / name)] for name in SOURCE_NAMES}
    reader = read(base / 'reader_cases.json'); catalog = read(base / 'catalog.json')
    cases, join_index = attach_cases(reader, catalog, loaded, audit, source_pins, tracking[str(audit_path)])
    now = datetime.now(timezone.utc).isoformat(); changed = {'reader_cases.json', 'research_sources.json'}
    provenance = {'schema': 'eo-e5-case-attachment-v1', 'checked_at': now, 'run_id': 'E5-EB-v0',
        'source_role': 'historical_model_output', 'n_cases': 914, 'n_saved_answers_attached': 13710,
        'n_original_prediction_rows': 26325, 'primary_membership_count': 902,
        'original_artifact': audit['artifact'], 'source_package': 'e5_cases_v11',
        'source_files_sha256': source_pins, 'independent_audit_sha256': tracking[str(audit_path)],
        'audited_original_files_sha256': original_pins, 'builder_sha256': sha(__file__),
        'limitations': ['Stored E5 responses to exactly the original source question; no new inference or outcome evaluation.',
            'Null means an unparsed stored answer. Missing predictions are rejected rather than fabricated.',
            'Source labels remain separate; full_no_delta is a fixed full-model input intervention, not a fifth trained model.',
            'Three seeds are separate saved answers, not confidence or a vote.',
            'Synthetic dates remain model inputs; T0 published-day metadata does not retroactively repair these answers.',
            'This attachment does not establish physical change, timing, damage, causality, or memory necessity.']}
    cases['e5_control_provenance'] = provenance
    sources = read(base / 'research_sources.json')
    require('e5_cases_v11' not in sources, 'E5 source attachment already present')
    sources['e5_cases_v11'] = copy.deepcopy(provenance)
    if html is not None:
        pin(html, sha(html), tracking); changed.add('index.html')
    shutil.copytree(base, out)
    try:
        write(out / 'reader_cases.json', cases); write(out / 'research_sources.json', sources)
        if html is not None:
            shutil.copyfile(html, out / 'index.html')
            require(sha(out / 'index.html') == tracking[str(html)], 'Reviewed HTML copy changed')
        package = out / 'e5_cases_v11'; package.mkdir()
        for name in SOURCE_NAMES:
            target = package / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(e5 / name, target)
            require(sha(target) == source_pins[name], 'E5 source copy changed')
        shutil.copyfile(audit_path, package / 'independent_audit.json')
        require(sha(package / 'independent_audit.json') == tracking[str(audit_path)], 'Audit copy changed')
        write(package / 'case_join_index.json', {'schema': 'eo-e5-case-join-index-v1', 'records': join_index,
            'numbering': 'One-based JSON record number, excluding blank physical lines in predictions.jsonl',
            'predictions_sha256': source_pins['predictions.jsonl']})
        for name, digest in base_hashes.items():
            require(sha(base / name) == digest, 'Original v10 changed during build')
            if name not in changed:
                require(sha(out / name) == digest, 'Inherited file changed: ' + name)
        for path, digest in tracking.items():
            require(sha(path) == digest, 'Input changed during build: ' + path)
        output_hashes = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
        report = {'schema': 'eo-v11-e5-cases-build-v0', 'created_at': now, 'base': str(base), 'out': str(out),
            'n_cases': 914, 'n_saved_answers_attached': 13710, 'primary_membership_count': 902,
            'changed_existing_files': sorted(changed), 'existing_reader_fields_unchanged': True,
            'catalog_meta_source_dates_previews_research_runs_byte_identical': True,
            'no_new_inference': True, 'no_new_scoring': True, 'source_hashes': tracking,
            'base_files_sha256': base_hashes, 'output_files_sha256': output_hashes, 'builder_sha256': sha(__file__),
            'limits': ['New v11 API/HTTP/browser readback remains required; inherited checks describe previous snapshots.',
                'Only exact existing test cases receive E5 answers; no training or unknown tile answer is generated.',
                'The independent audit is verified by its source pins; this attachment does not reload heavy model tensors.']}
        write(out / 'v11_build_manifest.json', report)
        return report
    except BaseException as error:
        write(out / 'v11_failure.json', {'error': str(error), 'partial_copy_not_valid': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'e5', 'audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--html', type=Path)
    args = parser.parse_args(); result = build(args.base, args.e5, args.audit, args.out, args.html)
    print(json.dumps({k: result[k] for k in ('out', 'n_cases', 'n_saved_answers_attached', 'primary_membership_count')}))
