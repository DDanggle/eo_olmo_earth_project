#!/usr/bin/env python3
"""Verify historical reader-case artifacts and a small local HTTP sample.

Standard library only. The full 914-case comparison reads local frozen files;
HTTP requests cover representative cases, unavailable cases and invalid routes.
No inference, model fitting, new scoring, or remote mask reads are performed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(values, key='id'):
    result = {row[key]: row for row in values}
    require(len(result) == len(values), 'Duplicate ' + key)
    return result


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(args, report):
    repo = args.repo.resolve()
    case_path = args.cases or repo / 'artifacts/eo_evidence_search_v3_20260925/reader_cases.json'
    case_path = case_path.resolve()
    case_dir = case_path.parent
    c1 = repo / 'artifacts/c1_same_prompt_flood_v0_20260925'
    quality_dir = repo / 'artifacts/c1_flood_label_quality_v0_20260925'
    items_path = repo / 'artifacts/c0_linear_view_probe_v1_20260925/items.jsonl'
    snapshot = read(case_path)
    require(snapshot['schema_version'] == 'eo_reader_cases_v1', 'Wrong reader-case schema')
    require(datetime.fromisoformat(snapshot['checked_at'].replace('Z', '+00:00')).tzinfo is not None,
            'Snapshot checked_at lacks timezone')
    source_items = unique(rows(items_path))
    test = {key: item for key, item in source_items.items() if item['partition'] == 'test'}
    require(len(test) == 1755, 'Expected 1755 frozen test IDs')
    target = {key: item for key, item in test.items() if item['phen'] == 'flood' and item['kind'] in ('pos', 'hard_neg')}
    require(Counter(item['kind'] for item in target.values()) == {'pos': 457, 'hard_neg': 457}, 'Target kind counts changed')
    by_tile = unique(list(target.values()), 'tile')
    cases = snapshot['cases']
    require(len(cases) == 914 and set(cases) == set(by_tile), '914 exact case tiles mismatch')
    catalog = unique(read(case_dir / 'catalog.json')['records'])
    quality = unique(rows(quality_dir / 'quality.jsonl'))
    require(set(quality) == set(target), 'Quality population mismatch')
    selected = read(c1 / 'selected_ids.json')
    eligible = {key for key, q in quality.items() if q['eligible_symmetric_quality']}
    require(selected == {'quality_symmetric': sorted(eligible), 'all_original_targets': sorted(target)},
            'C1 selected ID sets mismatch')
    group_ids = defaultdict(list)
    for key in eligible:
        item = target[key]
        group_ids[(str(item['event']), tuple(item['dates']), tuple(item['slots']))].append(key)
    supported = {key for group in group_ids.values()
                 if {target[i]['kind'] for i in group} == {'pos', 'hard_neg'} for key in group}
    source_hashes = {'items': sha(items_path), 'c1_results': sha(c1 / 'results.json'),
                     'quality': sha(quality_dir / 'quality.jsonl'), 'catalog': sha(case_dir / 'catalog.json')}
    predictions = {}
    for arm in ('reader', 'blind'):
        for seed in ('1', '2', '3'):
            path = repo / f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'
            data = unique(rows(path))
            require(set(data) == set(test), f'{arm} seed{seed} test ID coverage changed')
            for key, item in test.items():
                row = data[key]
                require(all(row[field] == item[field] for field in ('tile', 'fold', 'phen', 'kind'))
                        and row['emb_item'] == key and row['text_gold'] == row['emb_gold'] == item['answer'],
                        f'{arm} seed{seed} source metadata/label mismatch: {key}')
                require(row['parsed'] in ('yes', 'no'), 'Invalid frozen parsed class')
            predictions[(arm, seed)] = data
            source_hashes[f'{arm}_seed{seed}'] = sha(path)
    disagreement_ids = []
    for tile, case in cases.items():
        item = by_tile[tile]
        key = item['id']
        q = quality[key]
        require(catalog[tile]['dataset'] == 'kurosiwo' and catalog[tile]['split'] == 'test', 'Case is outside test catalog')
        require(case['tile'] == tile and case['question_id'] == key and case['dates'] == item['dates']
                and case['slots'] == item['slots'] and case['reference_label'] == item['answer'], 'Case metadata/label mismatch: ' + tile)
        require(case['run_id'] == 'E2' and case['source_role'] == 'historical_model_output'
                and case['reference_label_role'] == 'source_mask_derived_qa_label', 'Case provenance role mismatch')
        require(case['source_sha256'] == dict(source_hashes, audited_mask=q['mask_sha256'], audited_valid=q['valid_sha256']),
                'Case source SHA mismatch: ' + tile)
        require(case['c1_membership']['quality_eligible'] is (key in eligible)
                and case['c1_membership']['supported_stratum'] is (key in supported), 'Case C1 membership mismatch')
        for arm in ('reader', 'blind'):
            expected = {seed: predictions[(arm, seed)][key]['parsed'] for seed in ('1', '2', '3')}
            require(case[arm + '_predictions'] == expected, 'Historical predictions changed: ' + tile)
        if any(value != case['reference_label'] for value in case['reader_predictions'].values()):
            disagreement_ids.append(tile)
    report['checks'].append({'name': 'all_914_cases_match_frozen_sources', 'passed': True, 'n': len(cases),
                              'quality_eligible': len(eligible), 'supported': len(supported),
                              'cases_with_reader_reference_disagreement': len(disagreement_ids)})
    report['artifact_sha256'] = {'reader_cases': sha(case_path), **source_hashes}
    report['checked_at'] = snapshot['checked_at']

    def request(path, expected_status):
        request = Request(args.url.rstrip('/') + path, headers={'Accept': 'application/json'})
        try:
            with urlopen(request, timeout=15) as response:
                status, headers, body = response.status, response.headers, response.read()
        except HTTPError as exc:
            status, headers, body = exc.code, exc.headers, exc.read()
        require(status == expected_status, f'HTTP {status}, expected {expected_status}: {path}')
        require('application/json' in headers.get('Content-Type', ''), 'Expected JSON HTTP response')
        require(headers.get('Cache-Control') == 'no-store', 'HTTP snapshot response must not be cached')
        decoded = body.decode('utf-8')
        require('Traceback' not in decoded and str(repo) not in decoded, 'HTTP response exposed filesystem/traceback')
        return json.loads(decoded)

    chosen = {}
    for kind in ('pos', 'hard_neg'):
        candidates = sorted(tile for tile, item in by_tile.items() if item['kind'] == kind)
        differing = [tile for tile in candidates if tile in disagreement_ids]
        chosen[kind] = differing[0] if differing else candidates[0]
    for kind, tile in chosen.items():
        data = request('/api/reader-case?' + urlencode({'tile': tile}), 200)
        require(data['available'] is True and data['tile'] == tile and data['checked_at'] == snapshot['checked_at'],
                'HTTP snapshot selection mismatch')
        require(data['case'] == cases[tile], 'HTTP case does not equal frozen local case')
        report['checks'].append({'name': 'http_' + kind + '_exact_case', 'passed': True, 'tile': tile})
    require(catalog['ks_00276']['split'] == 'train', 'Expected original demonstration tile to remain train')
    for name, tile in (('training_unavailable', 'ks_00276'), ('unknown_unavailable', '__unknown_reader_tile__'),
                       ('query_traversal_unavailable', '../../asset_routes.private.json')):
        data = request('/api/reader-case?' + urlencode({'tile': tile}), 200)
        require(data['available'] is False and data.get('case') is None, 'Unavailable case leaked a record')
        report['checks'].append({'name': 'http_' + name, 'passed': True})
    tile = chosen['pos']
    for name, path in (('missing_param', '/api/reader-case'),
                       ('blank_param', '/api/reader-case?tile='),
                       ('duplicate_param', '/api/reader-case?tile=' + tile + '&tile=' + tile),
                       ('unexpected_param', '/api/reader-case?tile=' + tile + '&extra=1')):
        request(path, 400)
        report['checks'].append({'name': 'http_' + name, 'passed': True})
    for path in ('/reader_cases.json', '/asset_routes.private.json', '/%2e%2e/asset_routes.private.json'):
        request(path, 404)
        report['checks'].append({'name': 'http_private_path_404', 'path': path, 'passed': True})
    report['http_requests'] = 12
    report['source_scope'] = 'All 914 local cases checked; two representative historical E2 cases requested over HTTP. No new inference.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Running local server, e.g. http://127.0.0.1:8774')
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='Write verification report JSON here')
    parser.add_argument('--cases', type=Path, help='Optional snapshot path; defaults to repository v3 artifact')
    args = parser.parse_args()
    report = {'schema': 'eo-reader-case-http-verification-v0', 'started_at': datetime.now(timezone.utc).isoformat(),
              'passed': False, 'checks': [], 'no_new_inference': True}
    try:
        verify(args, report)
        report['passed'] = True
    except Exception as exc:
        report['error_type'], report['error'] = type(exc).__name__, str(exc)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(report['checks']), 'out': str(args.out),
                      'error': report.get('error')}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
