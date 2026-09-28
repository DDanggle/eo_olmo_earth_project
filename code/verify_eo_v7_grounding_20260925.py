#!/usr/bin/env python3
"""Independent disk/API-function readback; no HTTP, server, model, or source writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verify(repo, base, out, export):
    manifest = read(out / 'v7_build_manifest.json')
    for rel, digest in manifest['output_files_sha256'].items():
        check(sha(out / rel) == digest, 'Output hash differs: ' + rel)
    for path, digest in manifest['source_sha256'].items():
        check(sha(Path(path)) == digest, 'Input hash differs: ' + path)
    check((base / 'reader_cases.json').read_bytes() == (out / 'reader_cases.json').read_bytes(), 'Historical readers changed')
    before, after = [read(p / 'catalog.json') for p in (base, out)]
    check({k: v for k, v in before.items() if k != 'records'} == {k: v for k, v in after.items() if k != 'records'}, 'Catalog header changed')
    check(len(before['records']) == len(after['records']), 'Record coverage changed')
    case_ids = ('ks_06770', 'ks_05265')
    cases = {}
    for old, new in zip(before['records'], after['records']):
        if old['id'] not in case_ids:
            check(old == new, 'Unrelated catalog record changed')
            continue
        check({k: v for k, v in old.items() if k != 'evidence'} == {k: v for k, v in new.items() if k != 'evidence'}, 'Original case metadata changed')
        check(old['evidence'] == [] and len(new['evidence']) == 1, 'Evidence cardinality differs')
        evidence = new['evidence'][0]
        source = read(export / old['id'] / 'source_manifest.json')
        check(evidence['source'] == source, 'Embedded source manifest differs')
        image_path = out / evidence['image_url'].lstrip('/')
        check(sha(image_path) == source['preview_sha256'], 'Preview differs from export')
        known = source['reference_summary']['labelled_fraction']
        unknown = 100 * (1 - known)
        check(f'라벨 확인 불가 {unknown:.2f}%' in evidence['caption'], 'Unknown support caption differs')
        check('source valid=1' in evidence['caption'] and '실제 취득일은 미검증' in evidence['caption'], 'Validity/date caveat missing')
        cases[old['id']] = {'preview_sha256': sha(image_path), 'unknown_pct': unknown,
                            'c1_quality_eligible': source['eligible_C1_symmetric_quality']}
    check(set(cases) == set(case_ids), 'Wrong grounding case IDs')
    old_meta, new_meta = [read(p / 'meta.json') for p in (base, out)]
    check(new_meta['preview_count'] == old_meta['preview_count'] + 2, 'Preview count differs')
    check({k: v for k, v in old_meta.items() if k != 'preview_count'} ==
          {k: v for k, v in new_meta.items() if k != 'preview_count'}, 'Unrelated metadata changed')
    old_runs, new_runs = [read(p / 'research_runs.json') for p in (base, out)]
    check(new_runs['runs'][:-1] == old_runs['runs'], 'Historical research runs changed')
    e5 = new_runs['runs'][-1]
    check(e5['id'] == 'E5-EB' and 'metrics' not in e5 and '결과 없음' in e5['status'], 'E5 outcome invented')
    check('2026-09-25T06:14:07Z' in e5['findings'][0] and '275 / 1,590' in e5['findings'][0], 'Historical status differs')
    check('실시간 상태가 아닙니다' in e5['findings'][0], 'Live-state ambiguity')
    old_sources, new_sources = [read(p / 'research_sources.json') for p in (base, out)]
    check({k: v for k, v in new_sources.items() if k != 'source_grounding_v7'} == old_sources, 'Prior research sources changed')
    check(new_sources['source_grounding_v7']['sources'] == manifest['source_sha256'], 'Grounding source map differs')
    sys.path.insert(0, str(repo / 'code'))
    spec = importlib.util.spec_from_file_location('eo_v7_readback_api', repo / 'code/eo_evidence_search_v0.py')
    api = importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    result, status = api.read_research_snapshot(out)
    check(status == 200 and result['available'] and result['runs'][-1] == e5, 'Research API rejected snapshot')
    expected_reader = read(base / 'reader_cases.json')['cases']
    catalog_ids = {r['id'] for r in after['records']}
    for tile in (*case_ids, 'ks_05229', 'ks_00276'):
        result, status = api.read_reader_case(out, tile, catalog_ids)
        check(status == 200, 'Reader API status differs')
        if tile in expected_reader:
            check(result['available'] and result['case'] == expected_reader[tile], 'Historical reader API payload differs')
        else:
            check(result['available'] is False, 'Invented reader prediction')
    return {'schema': 'eo-v7-grounding-readback-v0', 'valid': True,
            'checked_at': datetime.now(timezone.utc).isoformat(), 'cases': cases,
            'historical_reader_bytes_unchanged': True, 'noncase_catalog_payload_unchanged': True,
            'prior_research_runs_unchanged': True, 'e5_outcomes_included': False,
            'api_function_readback': True, 'http_or_browser_checked': False,
            'build_manifest_sha256': sha(out / 'v7_build_manifest.json'), 'script_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('repo', 'base', 'out', 'case-export', 'report'):
        parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    check(not args.report.exists(), 'Report already exists')
    result = verify(args.repo.resolve(), args.base.resolve(), args.out.resolve(), args.case_export.resolve())
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
