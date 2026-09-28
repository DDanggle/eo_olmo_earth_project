#!/usr/bin/env python3
"""Local, provenance-preserving evidence search. No model execution or D1 gate changes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import statistics
import sys
from urllib.parse import parse_qs, unquote, urlsplit

sys.dont_write_bytecode = True
from eo_query_core_v0 import build_kurosiwo_catalog, query_catalog


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build_sn7(repo):
    from sn7_visible_pack_v05 import load_pack
    from sn7_visible_contract_v05 import derive_target
    pack_path = repo / 'labeling_pack/visible_contract_v05_20260922/annotator_pack/pack.json'
    export_path = repo / 'labeling_pack/visible_contract_v05_20260922/exports/sn7v05-198e4ff03e26f004_dongdong.json'
    pack = load_pack(pack_path)  # Includes image hashes; does not interpret image content.
    export = json.loads(export_path.read_text())
    if export['schema'] != 'sn7-visible-annotations-v0.5' or export['pack_id'] != pack['pack_id']:
        raise ValueError('SN7 annotation schema or pack mismatch')
    eps = {ep['id']: ep for ep in pack['episodes']}
    seen, records, seconds, labels = set(), [], [], Counter()
    source = {'annotation_file': str(export_path.relative_to(repo)),
              'annotation_sha256': sha(export_path), 'pack_id': pack['pack_id'],
              'pack_sha256': sha(pack_path), 'annotator_id': export['annotator_id'],
              'role': 'single_human_annotation_not_consensus_or_model_output'}
    asset_paths = {}
    for ann in export['annotations']:
        eid = ann['episode_id']
        if eid not in eps or eid in seen or ann['annotator_id'] != export['annotator_id']:
            raise ValueError('Duplicate/unknown episode or inconsistent annotator')
        seen.add(eid)
        if ann['status'] != 'complete':
            raise ValueError('Prototype requires the completed A export; does not infer missing labels')
        ep = eps[eid]
        target = derive_target(ep, ann)
        seconds.append(ann.get('seconds', 0))
        labels.update(ann['states'].values())
        evidence_ids = set(target['evidence_ids']) | {ep['reference_id'], ep['frames'][-1]['id']}
        if target.get('last_clear_no_change_id'):
            evidence_ids.add(target['last_clear_no_change_id'])
        frames = []
        for fr in ep['frames']:
            route = f"/evidence/sn7/{eid}/{fr['id']}.png"
            real_path = (pack_path.parent / fr['path']).resolve()
            if not real_path.is_relative_to(pack_path.parent.resolve()):
                raise ValueError('Unexpected frame path')
            asset_paths[route] = {'path': str(real_path), 'sha256': sha(real_path)}
            frames.append({'id': fr['id'], 'date': fr['date'], 'state': ann['states'][fr['id']],
                           'image_url': route, 'is_reference': fr['id'] == ep['reference_id']})
        records.append({
            'id': f'sn7:{eid}', 'dataset': 'sn7', 'aoi_id': ep['aoi'], 'region_id': eid,
            'quadrant': ep['region'], 'split': 'exposed_development',
            'event_type': 'structural_change', 'status': target['answer'], 'point': None,
            'event_date': None, 'observation_start': ep['frames'][0]['date'],
            'observation_end': ep['cutoff'], 'first_change_date': target['first_change_date'],
            'current_state': target['current_state'], 'raw_current_state': target['raw_current_state'],
            'reference_state': target['reference_state'],
            'last_clear_no_change_date': target['last_clear_no_change_date'],
            'evidence': [fr for fr in frames if fr['id'] in evidence_ids], 'frames': frames,
            'provenance': source,
            'limitations': ['single_rater_only', 'geometry_unavailable', 'month_precision_only',
                            'retrospective_reading_not_physical_onset', 'impact_and_cause_not_annotated'],
        })
    if seen != set(eps):
        raise ValueError('A annotation coverage is incomplete')
    summary = {'status': 'single_rater_complete_H_undetermined', 'episodes': len(records),
               'aois': len({r['aoi_id'] for r in records}), 'frames': sum(labels.values()),
               'answer_counts': dict(Counter(r['status'] for r in records)),
               'frame_state_counts': dict(labels), 'total_seconds': sum(seconds),
               'median_seconds': statistics.median(seconds), 'provenance': source}
    return records, summary, asset_paths


def build(args):
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing artifact directory: {out}')
    sys.path.insert(0, str(repo / 'code'))
    kuro_meta = repo / 'artifacts/streaming_review_20260909/kurosiwo_s1_cache/meta.jsonl'
    catalog = build_kurosiwo_catalog(kuro_meta)
    sn7, h_summary, assets = build_sn7(repo)
    catalog['records'].extend(sn7)
    catalog['schema'] = 'eo-evidence-catalog-v0'
    catalog['built_at'] = datetime.now(timezone.utc).isoformat()
    catalog['intended_use'] = 'local_research_reference_search_not_an_automatic_impact_detector'
    out.mkdir(parents=True, exist_ok=True)
    preview_count = 0
    if args.previews:
        index_path = args.previews / 'case_index.json'
        if index_path.is_file():
            preview_index = json.loads(index_path.read_text())
            # The renderer emits a top-level list or a cases list with provenance.
            cases = preview_index if isinstance(preview_index, list) else preview_index['cases']
            if isinstance(cases, dict):
                cases = [{'id': k, **v} for k, v in cases.items()]
            for case in cases:
                case_id = case.get('case_id') or case.get('id') or case.get('sample_id')
                matching = [r for r in catalog['records'] if r['dataset'] == 'kurosiwo' and r['id'] in (case_id, f'kurosiwo:{case_id}')]
                if len(matching) != 1:
                    raise ValueError(f'Preview does not match exactly one catalog record: {case_id}')
                file_name = case.get('preview_file') or case.get('file') or case.get('image_file') or case.get('png')
                src = (args.previews / file_name).resolve()
                if not src.is_relative_to(args.previews.resolve()) or not src.is_file():
                    raise ValueError('Unexpected preview image path')
                if case.get('preview_sha256') and sha(src) != case['preview_sha256']:
                    raise ValueError('Preview hash differs from renderer manifest')
                if str(case.get('event_id')) != matching[0]['aoi_id'] or case.get('event_date') != matching[0]['event_date']:
                    raise ValueError('Preview event provenance differs from catalog')
                if abs(float(case['source_tile_pflood_percent']) - matching[0]['flood_fraction_pct']) > 1e-8:
                    raise ValueError('Preview source flood fraction differs from catalog')
                dest = out / 'previews' / f'{case_id}.png'
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(src, dest)
                matching[0]['evidence'] = [{'image_url': f'/previews/{case_id}.png',
                    'kind': 'historical_sar_pair_and_reference_mask',
                    'acquisition_dates': None, 'source': case,
                    'note': 'Selected high-flood illustrative sample; reference mask, not model output.'}]
                preview_count += 1
            shutil.copyfile(index_path, out / 'preview_provenance.json')
    kuro = [r for r in catalog['records'] if r['dataset'] == 'kurosiwo']
    events = []
    for act in sorted({r['aoi_id'] for r in kuro}, key=str):
        rs = [r for r in kuro if r['aoi_id'] == act]
        dates = sorted({r['event_date'] for r in rs})
        events.append({'id': act, 'n': len(rs), 'dates': dates,
                       'positive': sum(r['flood_fraction_pct'] > 0 for r in rs),
                       'bbox': [min(r['point'][0] for r in rs), min(r['point'][1] for r in rs),
                                max(r['point'][0] for r in rs), max(r['point'][1] for r in rs)]})
    meta = {'title': 'Earth Evidence Search', 'version': '0.1 · 2026-09-25',
            'kuro_records': len(kuro), 'kuro_events': len(events), 'preview_count': preview_count,
            'kuro_positive': sum(r['flood_fraction_pct'] > 0 for r in kuro),
            'preview_example_event': next((r['aoi_id'] for r in kuro if r['evidence']), None),
            'events': events, 'h': h_summary, 'kuro_provenance': catalog['provenance'],
            'limits': ['Historical reference catalog, not current flood monitoring.',
                       'Geographic filter tests tile centroids, not intersecting footprints.',
                       'Event dates are not satellite acquisition dates.',
                       'Agricultural/forest overlap and damage are not yet measured.']}
    write_json(out / 'catalog.json', catalog)
    write_json(out / 'meta.json', meta)
    write_json(out / 'asset_routes.private.json', assets)
    write_json(out / 'h_single_rater_summary.json', h_summary)
    shutil.copyfile(Path(__file__).with_name('eo_evidence_search_v0.html'), out / 'index.html')
    print(json.dumps({'out': str(out), 'records': len(catalog['records']),
                      'events': len(events), 'previews': preview_count, 'H': h_summary['answer_counts']}, ensure_ascii=False))


def read_research_snapshot(out):
    """Read a saved report only; never poll a server or execute an experiment."""
    unavailable = {'available': False, 'checked_at': None, 'runs': []}
    try:
        raw = (out / 'research_runs.json').read_bytes()
    except FileNotFoundError:
        return {**unavailable, 'reason': 'not_created'}, 200
    except OSError:
        return {**unavailable, 'reason': 'unreadable_snapshot',
                'error': 'Saved research snapshot could not be read.'}, 503
    try:
        if len(raw) > 1024 * 1024:
            raise ValueError('Snapshot too large')
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get('schema_version') != 'eo_research_snapshot_v1':
            raise ValueError('Unknown snapshot schema')
        checked_at = data.get('checked_at')
        if not isinstance(checked_at, str) or datetime.fromisoformat(checked_at.replace('Z', '+00:00')).tzinfo is None:
            raise ValueError('Snapshot must have a timezone-aware checked_at')
        runs = data.get('runs')
        if not isinstance(runs, list) or len(runs) > 50:
            raise ValueError('Invalid run list')
        normalized = []
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError('Invalid run')
            if any(not isinstance(run.get(k), str) for k in ('id', 'status', 'next_step')):
                raise ValueError('Missing run text')
            if any(not isinstance(run.get(k), list) or not all(isinstance(x, str) for x in run[k])
                   for k in ('findings', 'limitations')):
                raise ValueError('Invalid finding or limitation list')
            entry = {k: run[k] for k in ('id', 'status', 'findings', 'limitations', 'next_step')}
            if 'title' in run:
                if not isinstance(run['title'], str):
                    raise ValueError('Invalid title')
                entry['title'] = run['title']
            if 'metrics' in run:
                metrics = run['metrics']
                if not isinstance(metrics, dict):
                    raise ValueError('Invalid metrics')
                columns, rows = metrics.get('columns'), metrics.get('rows')
                if not isinstance(columns, list) or not 1 <= len(columns) <= 12 or not all(isinstance(x, str) for x in columns):
                    raise ValueError('Invalid metric columns')
                if not isinstance(rows, list) or len(rows) > 64:
                    raise ValueError('Invalid metric rows')
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(columns):
                        raise ValueError('Metric row width mismatch')
                    if not all(x is None or isinstance(x, str) or type(x) in (int, float) for x in row):
                        raise ValueError('Invalid metric cell')
                # Reject NaN/infinity instead of turning missing measurements into numbers.
                json.dumps(rows, allow_nan=False)
                entry['metrics'] = {'columns': columns, 'rows': rows}
                if 'caption' in metrics:
                    if not isinstance(metrics['caption'], str):
                        raise ValueError('Invalid metric caption')
                    entry['metrics']['caption'] = metrics['caption']
            normalized.append(entry)
        return {'available': True, 'schema_version': data['schema_version'],
                'checked_at': checked_at, 'runs': normalized}, 200
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        return {**unavailable, 'reason': 'invalid_snapshot',
                'error': 'Saved research snapshot is invalid.'}, 503



def read_reader_case(out, tile, catalog_ids):
    """Exact catalog-ID lookup of one saved E2 answer; never execute a model."""
    unavailable = {'available': False, 'tile': tile, 'checked_at': None, 'case': None}
    if tile not in catalog_ids:
        return {**unavailable, 'reason': 'unknown_catalog_tile'}, 200
    try:
        data = json.loads((out / 'reader_cases.json').read_text())
    except FileNotFoundError:
        return {**unavailable, 'reason': 'snapshot_not_created'}, 200
    except (OSError, ValueError, UnicodeError):
        return {**unavailable, 'reason': 'invalid_snapshot'}, 503
    try:
        if not isinstance(data, dict) or data.get('schema_version') != 'eo_reader_cases_v1' or not isinstance(data.get('cases'), dict):
            raise ValueError('Invalid reader-case schema')
        if not isinstance(data.get('checked_at'), str):
            raise ValueError('Missing snapshot time')
        if tile not in data['cases']:
            return {**unavailable, 'checked_at': data['checked_at'], 'reason': 'no_saved_case'}, 200
        case = data['cases'][tile]
        if not isinstance(case, dict) or case.get('tile') != tile or case.get('run_id') != 'E2' or case.get('source_role') != 'historical_model_output':
            raise ValueError('Reader-case identity or role mismatch')
        if not isinstance(case.get('question_id'), str) or case.get('reference_label') not in ('yes', 'no', None):
            raise ValueError('Invalid question reference')
        for key in ('dates', 'slots'):
            if not isinstance(case.get(key), list) or len(case[key]) != 2 or not all(isinstance(x, str) for x in case[key]):
                raise ValueError('Invalid observation pair')
        for key in ('reader_predictions', 'blind_predictions'):
            if not isinstance(case.get(key), dict) or any(v not in ('yes', 'no', None) for v in case[key].values()):
                raise ValueError('Invalid saved predictions')
        if not isinstance(case.get('source_sha256'), dict) or not isinstance(case.get('c1_membership'), dict):
            raise ValueError('Missing case provenance')
        if not isinstance(case.get('limitations'), list) or not all(isinstance(x, str) for x in case['limitations']):
            raise ValueError('Invalid limitations')
        if 'e3_control' in case:
            control = case['e3_control']
            if not isinstance(control, dict) or control.get('run_id') != 'E3-PD-v1' or control.get('source_role') != 'historical_model_output':
                raise ValueError('Invalid E3 control identity or role')
            if control.get('question_id') != case['question_id'] or case['reference_label'] not in ('yes', 'no') or control.get('source_gold') != case['reference_label']:
                raise ValueError('E3 source question/label mismatch')
            arms = {'real', 'earlier_only', 'later_only', 'repeat_earlier', 'repeat_later', 'no_delta', 'reverse'}
            expected = arms if case['reference_label'] == 'yes' else {'real', 'later_only', 'repeat_later'}
            predictions = control.get('predictions')
            if not isinstance(predictions, dict) or set(predictions) != expected:
                raise ValueError('E3 arm coverage mismatch')
            for values in predictions.values():
                if not isinstance(values, dict) or set(values) != {'1', '2', '3'} or any(v not in ('yes', 'no', None) for v in values.values()):
                    raise ValueError('Invalid E3 saved predictions')
            hashes = control.get('source_sha256')
            if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for k, v in hashes.items()):
                raise ValueError('Missing or invalid E3 provenance')
        if 'e4_control' in case:
            control = case['e4_control']
            if not isinstance(control, dict) or control.get('run_id') != 'E4-D-v0' or control.get('source_role') != 'historical_model_output':
                raise ValueError('Invalid E4 control identity or role')
            if control.get('question_id') != case['question_id'] or case['reference_label'] not in ('yes', 'no') or control.get('source_gold') != case['reference_label']:
                raise ValueError('E4 source question/label mismatch')
            predictions = control.get('predictions')
            if not isinstance(predictions, dict) or set(predictions) != {'real', 'delta_only', 'delta_sign_flip', 'delta_feature_permute'}:
                raise ValueError('E4 arm coverage mismatch')
            for values in predictions.values():
                if not isinstance(values, dict) or set(values) != {'1', '2', '3'} or any(v not in ('yes', 'no', None) for v in values.values()):
                    raise ValueError('Invalid E4 saved predictions')
            hashes = control.get('source_sha256')
            if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for k, v in hashes.items()):
                raise ValueError('Missing or invalid E4 provenance')
        if 'e5_control' in case:
            control = case['e5_control']
            if not isinstance(control, dict) or control.get('run_id') != 'E5-EB-v0' or control.get('source_role') != 'historical_model_output':
                raise ValueError('Invalid E5 identity or role')
            if control.get('question_id') != case['question_id'] or case['reference_label'] not in ('yes', 'no') or control.get('source_gold') != case['reference_label']:
                raise ValueError('E5 source question/label mismatch')
            if control.get('dates') != case['dates'] or control.get('slots') != case['slots']:
                raise ValueError('E5 observation pair mismatch')
            member = control.get('primary_membership')
            if type(member) is not bool or member is not case['c1_membership'].get('supported_stratum'):
                raise ValueError('E5 primary membership mismatch')
            completed = control.get('completed_at')
            if not isinstance(completed, str) or datetime.fromisoformat(completed.replace('Z', '+00:00')).utcoffset() is None:
                raise ValueError('Invalid E5 completion time')
            predictions = control.get('predictions')
            expected = {'full/native', 'pair/native', 'later/native', 'delta/native', 'full/full_no_delta'}
            if not isinstance(predictions, dict) or set(predictions) != expected:
                raise ValueError('E5 evaluation coverage mismatch')
            for values in predictions.values():
                if not isinstance(values, dict) or set(values) != {'1', '2', '3'} or any(v not in ('yes', 'no', None) for v in values.values()):
                    raise ValueError('Invalid E5 saved predictions')
            hashes = control.get('source_sha256')
            if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for k, v in hashes.items()):
                raise ValueError('Missing or invalid E5 provenance')
        return {'available': True, 'tile': tile, 'checked_at': data['checked_at'], 'case': case}, 200
    except (ValueError, TypeError, KeyError):
        return {**unavailable, 'reason': 'invalid_snapshot'}, 503


def serve(args):
    out = args.out.resolve()
    catalog = json.loads((out / 'catalog.json').read_text())
    meta = json.loads((out / 'meta.json').read_text())
    assets = json.loads((out / 'asset_routes.private.json').read_text())
    reader_case_ids = {r['id'] for r in catalog['records'] if r['dataset'] == 'kurosiwo'}

    class Handler(BaseHTTPRequestHandler):
        def emit(self, data, code=200, mime='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            try:
                if route == '/api/reader-case':
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    if set(params) != {'tile'} or len(params['tile']) != 1 or not params['tile'][0]:
                        raise ValueError('Supply one nonempty catalog tile parameter')
                    snapshot, status = read_reader_case(out, params['tile'][0], reader_case_ids)
                    return self.emit(snapshot, status)
                if route == '/api/research':
                    snapshot, status = read_research_snapshot(out)
                    return self.emit(snapshot, status)
                if route == '/api/meta':
                    return self.emit(meta)
                if route == '/api/search':
                    if len(parsed.query) > 16000:
                        raise ValueError('Query too long')
                    params = parse_qs(parsed.query)
                    if set(params) - {'q'} or len(params.get('q', [])) != 1:
                        raise ValueError('Supply one JSON query parameter q')
                    query = json.loads(params['q'][0])
                    return self.emit(query_catalog(catalog, query))
                if route.startswith('/evidence/sn7/'):
                    asset = assets.get(route)
                    if asset is None:
                        return self.emit({'error': 'Unknown evidence reference'}, 404)
                    path = Path(asset['path'])
                    if sha(path) != asset['sha256']:
                        return self.emit({'error': 'Evidence file changed since build'}, 409)
                    return self.emit(path.read_bytes(), mime='image/png')
                if route == '/':
                    return self.emit((out / 'index.html').read_bytes(), mime='text/html; charset=utf-8')
                if route.startswith('/previews/'):
                    path = (out / route.lstrip('/')).resolve()
                    if path.is_relative_to(out / 'previews') and path.is_file() and path.suffix == '.png':
                        return self.emit(path.read_bytes(), mime='image/png')
                return self.emit({'error': 'Not found'}, 404)
            except (ValueError, TypeError, KeyError) as exc:
                return self.emit({'error': str(exc)}, 400)

        def log_message(self, fmt, *values):
            print(fmt % values, flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Evidence search available at http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--repo', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--previews', type=Path)
    s = sub.add_parser('serve')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--port', type=int, default=8774)
    q = sub.add_parser('query')
    q.add_argument('--catalog', type=Path, required=True)
    q.add_argument('--query', required=True, help='Strict JSON query; not a natural-language model')
    args = ap.parse_args()
    if args.command == 'build':
        build(args)
    elif args.command == 'serve':
        serve(args)
    else:
        print(json.dumps(query_catalog(json.loads(args.catalog.read_text()), json.loads(args.query)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
