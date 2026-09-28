#!/usr/bin/env python3
"""Read back the v7 local HTTP service; no inference or scientific scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8774')
    args = parser.parse_args()
    check(not args.report.exists(), 'Report already exists')
    parts = urlsplit(args.url)
    check(parts.scheme == 'http' and parts.hostname in ('127.0.0.1', 'localhost'), 'Local HTTP only')
    base = args.url.rstrip('/')
    def get(route):
        with urlopen(base + route, timeout=10) as response:
            check(response.status == 200, 'HTTP failure: ' + route)
            return response.read(), response.headers.get_content_type()
    def read(name):
        return json.loads((args.snapshot / name).read_text())
    def api(route):
        data, mime = get(route)
        check(mime == 'application/json', 'Wrong JSON MIME')
        return json.loads(data)
    meta = api('/api/meta')
    check(meta == read('meta.json') and meta['preview_count'] == 9, 'Metadata differs')
    research = api('/api/research')
    expected = read('research_runs.json')
    check(research['available'] and research['runs'] == expected['runs'], 'Research differs')
    e5 = research['runs'][-1]
    check(e5['id'] == 'E5-EB' and 'metrics' not in e5 and '결과 없음' in e5['status'], 'Invented E5 scores')
    catalog = {row['id']: row for row in read('catalog.json')['records']}
    readers = read('reader_cases.json')['cases']
    cases = {}
    for tile in ('ks_06770', 'ks_05265', 'ks_05229', 'ks_00276'):
        payload = api('/api/reader-case?' + urlencode({'tile': tile}))
        check(payload['tile'] == tile, 'Wrong reader tile')
        check(payload['available'] == (tile in readers), 'Wrong reader availability')
        if tile in readers:
            check(payload['case'] == readers[tile], 'Historical reader differs')
        else:
            check(payload.get('case') is None and payload['reason'] == 'no_saved_case', 'Invented train answer')
        cases[tile] = {'reader_available': payload['available'], 'exact_reader_match': tile in readers}
        if tile in ('ks_06770', 'ks_05265'):
            evidence = catalog[tile]['evidence'][0]
            data, mime = get(evidence['image_url'])
            digest = hashlib.sha256(data).hexdigest()
            check(mime == 'image/png' and digest == evidence['source']['preview_sha256'], 'Preview mismatch')
            check(data == (args.snapshot / evidence['image_url'].lstrip('/')).read_bytes(), 'Preview bytes differ')
            cases[tile].update(preview_sha256=digest, png_http_verified=True)
    query = {'dataset': 'kurosiwo', 'land_cover': 'all', 'aoi_id': '562', 'min_flood_pct': 0, 'limit': 40}
    result = api('/api/search?' + urlencode({'q': json.dumps(query)}))
    rows = result['records']
    check(len(rows) == 8, 'Expected eight event 562 tiles')
    grounded = {row['id'] for row in rows if row['evidence']}
    check(grounded == {'ks_06770', 'ks_05265'}, 'Unexpected event preview coverage')
    html, mime = get('/')
    check(mime == 'text/html' and html == (args.snapshot / 'index.html').read_bytes(), 'Index differs')
    report = {'schema': 'eo-v7-http-readback-v0', 'valid': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'url': base, 'snapshot': str(args.snapshot.resolve()), 'preview_count': 9, 'cases': cases,
              'event_562_count': len(rows), 'event_562_grounded_cases': sorted(grounded),
              'research_runs_exact': True, 'e5_outcomes_included': False, 'index_bytes_exact': True,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
