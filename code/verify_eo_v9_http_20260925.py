#!/usr/bin/env python3
"""Read back v9 HTTP against disk and independently read E5 replica/audit JSON.

Presentation verification only; no production builder imports, model loading,
new inference, or repetition of the independent statistical/tensor audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

PLAN_SHA = 'fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
PARENT_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
AUDITOR_SHA = 'a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825'
NAMES = ('manifest.json', 'prereg.json', 'status.json', 'scores.json', 'training_summary.json', 'inference_completed.json')
VERDICTS = {'pair_preserves_source_discrimination_at_equal_budget',
            'explicit_difference_helps_at_this_budget', 'mixed_or_inconclusive'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def safe(root, name):
    path = Path(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts, 'Unsafe manifest path')
    result = root / path
    require(result.is_file() and not result.is_symlink() and result.resolve().is_relative_to(root), 'Missing/escaping file')
    return result


def verify(snapshot, e5, audit_path, url):
    snapshot, e5, audit_path = [Path(p).resolve() for p in (snapshot, e5, audit_path)]
    base = url.rstrip('/'); u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname in ('127.0.0.1', 'localhost') and not u.username
            and not u.password and u.path in ('', '/') and not u.query and not u.fragment, 'Local HTTP origin only')
    def get(route):
        with urlopen(base + route, timeout=10) as response:
            require(response.status == 200, 'HTTP failure')
            return response.read(), response.headers.get_content_type()
    def api(route):
        raw, mime = get(route)
        require(mime == 'application/json', 'JSON MIME differs')
        return json.loads(raw)
    manifest = read(snapshot / 'v9_build_manifest.json'); manifest_digest = sha(snapshot / 'v9_build_manifest.json')
    require(manifest['schema'] == 'eo-v9-e5-results-build-v0'
            and set(manifest['changed_existing_files']) == {'research_runs.json', 'research_sources.json'}, 'v9 build scope differs')
    for name, digest in manifest['output_files_sha256'].items():
        require(sha(safe(snapshot, name)) == digest, 'v9 disk file changed: ' + name)
    inherited = 0
    for name, digest in manifest['base_files_sha256'].items():
        if name not in ('research_runs.json', 'research_sources.json'):
            require(sha(safe(snapshot, name)) == digest, 'Inherited v8 file changed: ' + name)
            inherited += 1
    audit = read(audit_path)
    require(audit['schema'] == 'e5-independent-result-audit-v0' and audit['consistent'] is True
            and audit['checkpoint_tensors_loaded_and_checked_on_cpu'] is True
            and audit['audit_code_sha256'] == AUDITOR_SHA, 'Independent E5 CPU audit did not pass')
    require((audit['n_models'], audit['n_answers'], audit['primary_n'], audit['primary_events']) == (12, 26325, 902, 8),
            'Independent E5 support differs')
    original = Path(audit['artifact']); require(original.is_absolute(), 'Original E5 artifact must be absolute')
    source_pins = {}
    for name in NAMES:
        expected = audit['hashes_verified'].get(str(original / name))
        require(sha(safe(e5, name)) == expected == sha(safe(snapshot / 'e5_results_v9', name)), 'E5 source/replica copy differs: ' + name)
        source_pins[name] = expected
    require(sha(audit_path) == sha(snapshot / 'e5_results_v9/independent_audit.json'), 'Independent audit copy differs')
    require(source_pins['manifest.json'] == PARENT_SHA and source_pins['prereg.json'] == PLAN_SHA, 'E5 frozen plan/manifest differ')
    plan = read(e5 / 'prereg.json'); require(original == Path(plan['root']) / plan['output_directory'], 'Original E5 path differs from plan')
    status, scores = read(e5 / 'status.json'), read(e5 / 'scores.json')
    require(status['status'] == 'completed' and status['n_models'] == 12 and status['n_rows'] == 26325
            and scores['valid'] is True and not (e5 / 'failure.json').exists(), 'E5 result incomplete/invalid')
    require(scores['verdict'] in VERDICTS and scores['verdict'] == status['verdict'] == audit['verdict']
            == manifest['registered_verdict'], 'Registered verdict differs')
    primary = {s: scores['metrics'][s]['primary_same_prompt'] for s in ('1', '2', '3')}
    require(primary == audit['primary_metrics'], 'Independent primary metrics disagree')
    require(scores['seed_decisions'] == audit['seed_decisions'], 'Seed decision audit disagrees')
    labels = [('full/native', 'full · A, B, D 학습'), ('pair/native', 'pair · A, B 학습'),
              ('later/native', 'later · B 학습'), ('delta/native', 'delta · D 학습'),
              ('full/full_no_delta', 'full_no_delta · 학습된 full에서 D 제외')]
    expected_rows = [[label] + [round(primary[s]['evaluations'][key]['macro_ba'], 4) for s in ('1', '2', '3')]
                     for key, label in labels]
    contrasts = [primary[s]['contrasts']['pair_minus_full'] for s in ('1', '2', '3')]
    require(all(c['left'] == 'pair/native' and c['right'] == 'full/native' for c in contrasts), 'Contrast direction differs')
    expected_rows.append(['주대조 pair − full · BA 차이'] + [round(c['delta'], 4) for c in contrasts])
    expected_rows.append(['주대조 · 사건 bootstrap 95% 구간'] + [f"[{c['ci95_delta'][0]:.4f}, {c['ci95_delta'][1]:.4f}]" for c in contrasts])
    research_disk = read(snapshot / 'research_runs.json'); research = api('/api/research')
    require(research['available'] is True and research['runs'] == research_disk['runs']
            and research['checked_at'] == research_disk['checked_at'], 'HTTP research snapshot differs')
    runs = {r['id']: r for r in research['runs']}; require(len(runs) == len(research['runs']), 'Duplicate research IDs')
    require('T0-DATE' in runs and runs['E5-EB']['status'] == '완료 · 독립 감사 통과', 'T0 missing or E5 not completed')
    card = runs['E5-EB']
    require(card['metrics']['columns'] == ['입력 / 측정값', 'Seed 1', 'Seed 2', 'Seed 3']
            and card['metrics']['rows'] == expected_rows, 'HTTP E5 table differs from independently read scores')
    require('등록 판정: ' + scores['verdict'] in card['findings']
            and 'full_no_delta는 별도 학습 모델이 아닌' in card['metrics']['caption'], 'Registered verdict/evaluation role absent')
    original_base = Path(manifest['base'])
    prior = read(original_base / 'research_runs.json')
    require(sha(original_base / 'research_runs.json') == manifest['base_files_sha256']['research_runs.json'], 'Original v8 research changed')
    require([r for r in research['runs'] if r['id'] != 'E5-EB'] == [r for r in prior['runs'] if r['id'] != 'E5-EB'],
            'Other research runs changed')
    meta = api('/api/meta'); require(meta == read(snapshot / 'meta.json') and meta['preview_count'] == 9, 'Metadata differs')
    catalog = {r['id']: r for r in read(snapshot / 'catalog.json')['records']}
    require(sum('source_date_provenance' in r for r in catalog.values()) == 7000, 'Date provenance coverage differs')
    readers = read(snapshot / 'reader_cases.json')['cases']; cases = {}
    require(all('e5_control' not in c for c in readers.values()), 'Individual E5 outputs unexpectedly mixed in')
    for tile in ('ks_06770', 'ks_05265', 'ks_05229', 'ks_00276'):
        result = api('/api/reader-case?' + urlencode({'tile': tile}))
        require(result['available'] == (tile in readers), 'Reader availability differs')
        if tile in readers:
            require(result['case'] == readers[tile], 'Existing reader case changed')
        else:
            require(result.get('case') is None and result['reason'] == 'no_saved_case', 'Invented training-tile answer')
        cases[tile] = {'reader_available': result['available'], 'matches_frozen_disk_case': True}
    previews = {}
    for name, digest in manifest['output_files_sha256'].items():
        if name.startswith('previews/') and name.endswith('.png'):
            raw, mime = get('/' + name)
            require(mime == 'image/png' and hashlib.sha256(raw).hexdigest() == digest, 'HTTP preview differs: ' + name)
            previews[name] = digest
    require(len(previews) == 9, 'Nine original preview files required')
    q = {'dataset': 'kurosiwo', 'land_cover': 'all', 'aoi_id': '562', 'min_flood_pct': 0, 'limit': 40}
    result = api('/api/search?' + urlencode({'q': json.dumps(q)}))
    require(len(result['records']) == 8, 'Event562 support differs')
    for row in result['records']:
        expected = catalog[row['id']]
        require(all(row.get(k) == v for k, v in expected.items()), 'HTTP catalog/source field differs')
        d = row['source_date_provenance']
        require(d['published_slot_dates'] == {'pre_1': '2021-08-07', 'pre_2': '2021-08-19', 'post': '2022-02-03'}, 'Documented dates differ')
        require(d['model_assumed_slot_dates'] == {'pre_1': '2022-01-05', 'pre_2': '2022-01-17', 'post': '2022-01-29'}, 'Imputed dates changed')
        require(d['actual_acquisition_verified'] is False and all(v is None for v in d['sensor_utc_acquisition_timestamps'].values()), 'Unverified UTC promoted')
        require(d['intervals_days']['pre2_to_post_days'] == 168 and row['event_date'] == '2022-01-29', 'Event/observation dates conflated')
    raw, mime = get('/'); require(mime == 'text/html' and raw == (snapshot / 'index.html').read_bytes(), 'HTTP HTML differs')
    require(sha(snapshot / 'v9_build_manifest.json') == manifest_digest, 'Build manifest changed during readback')
    for name, digest in manifest['output_files_sha256'].items():
        require(sha(safe(snapshot, name)) == digest, 'Snapshot changed during readback')
    require(sha(audit_path) == sha(snapshot / 'e5_results_v9/independent_audit.json'), 'Audit changed during readback')
    for name, digest in source_pins.items():
        require(sha(e5 / name) == digest, 'E5 replica changed during readback')
    return {'schema': 'eo-v9-http-readback-v0', 'valid': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'url': base, 'snapshot': str(snapshot), 'manifest_sha256': manifest_digest,
        'disk_output_hashes_checked': len(manifest['output_files_sha256']), 'inherited_files_byte_identical': inherited,
        'catalog_provenance_count': 7000, 'event_562_dates_verified_over_http': 8, 'cases': cases,
        'preview_http_sha256': previews, 'research_exact': True, 'e5_results_included': True,
        'e5_registered_verdict': scores['verdict'], 'e5_primary_metrics_equal_independent_audit': True,
        'e5_table_seed_count': 3, 'e5_table_shape': [7, 4], 'e5_source_replica_sha256': source_pins,
        'e5_independent_audit_sha256': sha(audit_path), 'individual_e5_predictions_added': False,
        'index_bytes_exact': True, 'script_sha256': sha(__file__),
        'limits': ['This is fresh v9 HTTP/disk readback, not an inherited v8 pass.',
                   'Statistical outcomes are quoted from the already audited saved run; no new scientific result is estimated here.',
                   'Heavy E5 tensor/prediction bytes are not re-audited; original E2/E3/E4 cases and date uncertainty remain unchanged.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'e5', 'audit', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8774')
    args = parser.parse_args(); require(not args.report.exists(), 'New report required')
    report = verify(args.snapshot, args.e5, args.audit, args.url)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))
