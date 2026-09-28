#!/usr/bin/env python3
"""Publish verified historical E2 cases after valid C1 completion; no inference or scoring.

--out is an exclusive JSON filename. Raw remote mask hashes are cross-checked
against the completed audit, not presented as locally re-read cache bytes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

TEST_COUNT, TARGET_COUNTS = 1755, {'pos': 457, 'hard_neg': 457}
SEEDS = ('1', '2', '3')
ARMS = ('reader', 'blind')
LIMITATIONS = [
    '과거 저장된 E2 판독이며 현재 상황 판독이나 새 추론이 아닙니다.',
    'dates는 사건일과 사건일−12일로 구성한 합성 시간이며 실제 SAR 취득일이 아닙니다.',
    '참조 라벨은 기존 마스크에서 만든 문항 정답이며 독립 검수한 시간 변화 정답이 아닙니다.',
    '노출된 개발 자료의 사례입니다. 3개 seed를 투표 신뢰도나 새 평가 성능으로 해석하지 않습니다.',
    '농지 피해·경제 손실·인과관계는 이 판독으로 측정하지 않습니다.',
    '원격 mask/valid SHA는 완료된 품질 감사와 대조했으며 이 builder에서 원격 픽셀을 다시 읽지 않았습니다.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def unique(rows, field='id'):
    out = {}
    for row in rows:
        key = row[field]
        require(isinstance(key, str) and key and key not in out, 'Missing/duplicate ' + field)
        out[key] = row
    return out


class Inputs:
    """Hash the exact bytes parsed, then reject any input mutation before return."""
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None, lines=False):
        path = Path(path).resolve()
        raw = path.read_bytes()
        h = digest(raw)
        if expected is not None:
            require(valid_hash(expected) and h == expected, 'SHA256 mismatch: ' + str(path))
        require(path not in self.files or self.files[path] == h, 'Input changed: ' + str(path))
        self.files[path] = h
        if path.suffix == '.py':
            return raw
        def invalid(x):
            raise ValueError('Non-finite JSON number: ' + x)
        parse = lambda s: json.loads(s, parse_constant=invalid)
        return [parse(s) for s in raw.decode().splitlines() if s.strip()] if lines else parse(raw)

    def sha(self, path):
        return self.files[Path(path).resolve()]

    def unchanged(self):
        for path, h in self.files.items():
            require(digest(path.read_bytes()) == h, 'Input changed during build: ' + str(path))


def prompt(item):
    require(item['slots'] == ['pre_2', 'post'] and len(item['dates']) == 2,
            'Unexpected target slots/dates')
    return ('These are 2 Sentinel-1 observations of the same area in chronological order, '
            f"taken on {', '.join(item['dates'])}: <EO> Did a flood occur between the two observations? "
            'Answer with yes or no.')


def groups(items):
    by = defaultdict(list)
    for it in items:
        by[(str(it['event']), prompt(it), tuple(it['slots']))].append(it)
    supported, unsupported = [], []
    for (event, text, slots), its in sorted(by.items()):
        ids = {k: sorted(x['id'] for x in its if x['kind'] == k) for k in TARGET_COUNTS}
        row = {'event': event, 'prompt_sha256': digest(text.encode()),
               'reconstructed_prompt': text, 'slots': list(slots),
               'n_pos': len(ids['pos']), 'n_hard_neg': len(ids['hard_neg']),
               'pos_ids': ids['pos'], 'hard_neg_ids': ids['hard_neg']}
        if ids['pos'] and ids['hard_neg']:
            supported.append(row)
        else:
            row['reason'] = 'missing_one_source_class'
            unsupported.append(row)
    return supported, unsupported


def validate_membership(saved, items):
    """Check saved strata membership without recomputing or interpreting scores."""
    supported, unsupported = groups(items)
    events = sorted({g['event'] for g in supported})
    require(saved['candidate_count'] == len(items) and saved['events'] == events
            and saved['n_events'] == len(events), 'C1 subset population mismatch')
    require(saved['unsupported_strata'] == unsupported, 'C1 unsupported strata mismatch')
    for kind in TARGET_COUNTS:
        require(saved['supported_n_' + kind] == sum(g['n_' + kind] for g in supported),
                'C1 supported count mismatch')
    require(set(saved['per_seed']) == set(SEEDS), 'C1 seed coverage')
    for seed in SEEDS:
        arms = saved['per_seed'][seed]['arms']
        require(set(arms) == set(ARMS), 'C1 arm coverage')
        for arm in ARMS:
            require(set(arms[arm]) == set(events), 'C1 event coverage')
            for event in events:
                wanted = [g for g in supported if g['event'] == event]
                actual = arms[arm][event]['strata']
                require(len(actual) == len(wanted), 'C1 stratum count mismatch')
                for have, want in zip(actual, wanted):
                    require(all(have.get(k) == v for k, v in want.items()), 'C1 stratum metadata mismatch')
    return {i for g in supported for k in ('pos_ids', 'hard_neg_ids') for i in g[k]}, events


def build_cases(repo, c1, quality, catalog):
    repo, c1, quality, catalog = map(Path, (repo, c1, quality, catalog))
    inp = Inputs()
    cs = inp.read(c1 / 'status.json')
    require(cs.get('status') == 'complete' and cs.get('valid') is True, 'C1 is not complete and valid')
    result = inp.read(c1 / 'results.json')
    require(result['schema'] == 'c1-same-prompt-flood-diagnostic-v0' and result['valid'] is True,
            'Invalid C1 result')
    provenance = result['provenance']
    cfg = inp.read(c1 / 'analysis_plan.json', provenance['plan_sha256'])
    inp.read(c1 / 'source.py', provenance['code_sha256'])
    require(cfg['schema'] == 'c1-same-prompt-flood-analysis-plan-v0', 'Unknown C1 plan')
    require(cfg['inputs_sha256'] == provenance['frozen_input_sha256'], 'C1 frozen input manifest mismatch')
    answer_paths = {(arm, seed): f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'
                    for arm in ARMS for seed in SEEDS}
    required = {cfg['items_path'], *answer_paths.values(), 'code/e2_multi_reader_v0.py',
                'code/e1_flood_qa_v0.py', 'code/extract_kurosiwo_s1_cache.py'}
    require(required <= set(cfg['inputs_sha256']), 'Missing frozen input hash')
    loaded = {}
    for rel, h in cfg['inputs_sha256'].items():
        path = (repo / rel).resolve()
        require(not Path(rel).is_absolute() and path.is_relative_to(repo.resolve()), 'Frozen path escapes repo')
        loaded[rel] = inp.read(path, h, lines=path.suffix == '.jsonl')
    items = loaded[cfg['items_path']]
    unique(items)
    test = unique([x for x in items if x['partition'] == 'test'])
    require(len(test) == TEST_COUNT, 'Expected exactly 1755 E2 test IDs')
    target = {i: x for i, x in test.items() if x['phen'] == 'flood' and x['kind'] in TARGET_COUNTS}
    require(Counter(x['kind'] for x in target.values()) == TARGET_COUNTS, 'Expected 457 pos and 457 hard_neg')
    require(len({x['tile'] for x in target.values()}) == 914, 'Expected 914 distinct target tiles')
    train_tiles = {x['tile'] for x in items if x['partition'] == 'train'}
    require(not train_tiles.intersection(x['tile'] for x in target.values()), 'Target overlaps training tile')
    qs = inp.read(quality / 'status.json')
    require(qs.get('status') == 'complete' and qs.get('valid') is True and qs.get('n') == 914,
            'Quality audit is not complete and valid')
    qm = inp.read(quality / 'manifest.json', provenance['quality_manifest_sha256'])
    require(qm['schema'] == 'c1-flood-label-quality-v0' and qm['no_model_scores'] is True
            and qm['n'] == 914, 'Invalid quality manifest')
    require(qm['quality_sha256'] == provenance['quality_sha256'], 'C1 quality SHA mismatch')
    qrows = unique(inp.read(quality / 'quality.jsonl', qm['quality_sha256'], lines=True))
    qplan = inp.read(quality / 'prereg.json', qm['prereg_sha256'])
    inp.read(quality / 'source.py', qm['code_sha256'])
    summary = inp.read(quality / 'summary.json', qm['summary_sha256'])
    qids = inp.read(quality / 'symmetric_quality_ids.json', qm['symmetric_ids_sha256'])
    require(qplan['schema'] == 'c1-flood-label-quality-plan-v0' and qplan['no_model_scores'] is True,
            'Invalid quality plan')
    require(qplan['expected_selected_counts'] == TARGET_COUNTS and qplan['thresholds'] == {
        'flood_min': .02, 'valid_min': .90, 'rounded_qa_abs_tolerance': .00005}, 'Quality plan contract changed')
    c0 = (repo / cfg['items_path']).parent
    c0_fields = [('manifest.json', 'manifest_sha256', 'c0_manifest_sha256'),
                 ('source.py', 'code_sha256', 'c0_code_sha256'),
                 ('prereg.json', 'prereg_sha256', 'c0_prereg_sha256'),
                 ('items.jsonl', 'items_sha256', 'items_sha256')]
    for filename, expected_key, manifest_key in c0_fields:
        h = qplan['expected_c0'][expected_key]
        require(qm[manifest_key] == h and qm['source_files']['c0_linear_view_probe_v1/' + filename] == h,
                'Quality C0 provenance mismatch: ' + filename)
        inp.read(c0 / filename, h, lines=filename.endswith('.jsonl'))
    require(set(qrows) == set(target), 'Quality target ID coverage mismatch')
    require(qm['selected_ids_sha256'] == digest(json.dumps(sorted(target), separators=(',', ':')).encode()),
            'Quality selected ID SHA mismatch')
    cat = unique(inp.read(catalog)['records'])
    eligible = set()
    raw_paths = set()
    for key, item in target.items():
        q = qrows[key]
        require(all(q[k] == item[k] for k in ('tile', 'dates', 'slots', 'kind'))
                and str(q['event']) == str(item['event']) and q['source_answer'] == item['answer']
                and q['source_qa_flood_frac'] == item['flood_frac'],
                'Quality metadata mismatch: ' + key)
        require(type(q['eligible_symmetric_quality']) is bool and
                q['eligible_symmetric_quality'] == (q['valid_frac'] >= .90), 'Quality eligibility mismatch')
        require(0 <= q['valid_frac'] <= 1 and 0 <= q['flood_frac'] <= 1, 'Invalid quality fraction')
        require((item['kind'] == 'pos' and item['answer'] == 'yes' and q['flood_frac'] >= .02) or
                (item['kind'] == 'hard_neg' and item['answer'] == 'no' and q['flood_px'] == 0 and q['valid_frac'] >= .90),
                'Source label inconsistent: ' + key)
        if q['eligible_symmetric_quality']:
            eligible.add(key)
        for name, folder in [('mask', 'mask_u8'), ('valid', 'valid_u8')]:
            path = f"kurosiwo_s1_cache/{folder}/{item['tile']}.npy"
            require(q[name + '_path'] == path and valid_hash(q[name + '_sha256'])
                    and qm['source_files'].get(path) == q[name + '_sha256'], 'Raw audit hash mismatch')
            raw_paths.add(path)
        record = cat.get(item['tile'])
        require(record is not None and record['dataset'] == 'kurosiwo' and record['split'] == 'test',
                'Target absent from test KuroSiwo catalog: ' + item['tile'])
        event_day = date.fromisoformat(record['event_date'])
        require(str(record['aoi_id']) == str(item['event']) and item['fold'] == 'test'
                and item['dates'] == [(event_day - timedelta(days=12)).isoformat(), event_day.isoformat()],
                'Catalog event/date metadata mismatch: ' + key)
        prompt(item)
    require(set(qm['source_files']) == raw_paths | {'c0_linear_view_probe_v1/' + f for f, _, _ in c0_fields},
            'Unexpected quality source manifest coverage')
    excluded = set(target) - eligible
    require(qids['source_items_sha256'] == qm['items_sha256'] and qids['ids'] == sorted(eligible)
            and qids['excluded_ids'] == sorted(excluded), 'Quality selected/excluded IDs mismatch')
    eligible_by_kind = {kind: sorted(i for i in eligible if target[i]['kind'] == kind) for kind in TARGET_COUNTS}
    require(qids['by_kind'] == eligible_by_kind and summary['symmetric_counts'] ==
            {kind: len(ids) for kind, ids in eligible_by_kind.items()}, 'Quality per-kind counts mismatch')
    require(summary['n'] == summary['n_unique_tiles'] == 914 and summary['counts'] == TARGET_COUNTS
            and summary['symmetric_n'] == len(eligible) and summary['all_source_labels_verified'] is True,
            'Quality summary population mismatch')
    selected = inp.read(c1 / 'selected_ids.json', provenance['selected_ids_sha256'])
    require(selected == {'quality_symmetric': sorted(eligible), 'all_original_targets': sorted(target)}
            and result['quality_excluded_ids'] == sorted(excluded), 'C1 selected IDs mismatch')
    supported, events = validate_membership(result['subsets']['quality_symmetric'], [target[i] for i in sorted(eligible)])
    validate_membership(result['subsets']['all_original_targets'], list(target.values()))
    predictions = {}
    for pair, rel in answer_paths.items():
        data = unique(loaded[rel])
        require(set(data) == set(test), 'Historical answer ID coverage mismatch: ' + rel)
        for key, item in test.items():
            d = data[key]
            require(all(d[k] == item[k] for k in ('tile', 'fold', 'phen', 'kind'))
                    and d['emb_item'] == key and d['text_gold'] == d['emb_gold'] == item['answer'],
                    'Historical row metadata/label mismatch: ' + key)
            require(d['parsed'] in ('yes', 'no'), 'Unparsed historical answer: ' + key)
        predictions[pair] = data
    for seed in SEEDS:
        blind = defaultdict(set)
        for key, item in target.items():
            blind[prompt(item)].add(predictions[('blind', seed)][key]['parsed'])
        saved = result['blind_consistency'][seed]
        require(all(len(x) == 1 for x in blind.values()) and saved['consistent'] is True
                and not saved['violations'] and saved['n_reconstructed_prompts'] == len(blind),
                'Historical blind consistency mismatch')
    shared_hashes = {'items': qm['items_sha256'], 'c1_results': inp.sha(c1 / 'results.json'),
                     'quality': qm['quality_sha256'], 'catalog': inp.sha(catalog)}
    shared_hashes.update({arm + '_seed' + seed: cfg['inputs_sha256'][rel]
                         for (arm, seed), rel in answer_paths.items()})
    cases = {}
    for key, item in sorted(target.items(), key=lambda p: p[1]['tile']):
        q = qrows[key]
        cases[item['tile']] = {
            'tile': item['tile'], 'question_id': key, 'dates': item['dates'], 'slots': item['slots'],
            'date_basis': 'synthetic_event_day_minus_12_and_event_day_not_verified_acquisition',
            'prompt': prompt(item), 'prompt_sha256': digest(prompt(item).encode()),
            'reference_label': item['answer'], 'reference_label_role': 'source_mask_derived_qa_label',
            'run_id': 'E2', 'source_role': 'historical_model_output',
            'reader_predictions': {s: predictions[('reader', s)][key]['parsed'] for s in SEEDS},
            'blind_predictions': {s: predictions[('blind', s)][key]['parsed'] for s in SEEDS},
            'source_sha256': dict(shared_hashes, audited_mask=q['mask_sha256'], audited_valid=q['valid_sha256']),
            'limitations': LIMITATIONS,
            'c1_membership': {'quality_eligible': key in eligible, 'supported_stratum': key in supported,
                              'scope': 'quality_symmetric_primary_subset'},
        }
    inp.unchanged()
    return {'schema_version': 'eo_reader_cases_v1', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'source_role': 'historical_model_output', 'run_id': 'E2', 'cases': cases,
            'summary': {'case_count': len(cases), 'source_kind_counts': TARGET_COUNTS,
                        'quality_eligible_count': len(eligible), 'quality_excluded_count': len(excluded),
                        'supported_count': len(supported), 'unsupported_eligible_count': len(eligible - supported),
                        'supported_event_count': len(events), 'supported_events': events,
                        'supported_kind_counts': dict(Counter(target[i]['kind'] for i in supported))},
            'provenance': {'source_sha256': {str(p): h for p, h in sorted(inp.files.items())},
                           'builder_sha256': digest(Path(__file__).read_bytes()),
                           'raw_mask_bytes_revalidated_locally': False,
                           'raw_mask_hash_verification': 'Exact equality of each quality row and completed audit manifest; no remote cache reread',
                           'no_new_inference': True, 'no_new_score_calculation': True},
            'limitations': LIMITATIONS}


def write_cases(payload, out):
    """Serialize first, then exclusively create; never overwrite an existing snapshot."""
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(out).open('x', encoding='utf-8') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'c1', 'quality', 'catalog', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite output: ' + str(args.out))
    payload = build_cases(args.repo, args.c1, args.quality, args.catalog)
    write_cases(payload, args.out)
    print(json.dumps({'out': str(args.out), **payload['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
