#!/usr/bin/env python3
"""C1 CPU-only audit of source flood-mask quality; never opens model outputs.

The 914 fixed C0 test pos/hard_neg items are selected before reading masks.
E1 valid_frac is the fraction of pixels that are BOTH valid==1 and mask>0.
Dates and slots are copied source metadata, not verified acquisition times.
No predictions, logits, scores, fitted models, or embedding arrays are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

import numpy as np

SHAPE = (192, 192)
FLOOD_MIN = 0.02
VALID_MIN = 0.90
QA_TOLERANCE = 5e-5
C0_DIR = 'c0_linear_view_probe_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_items(items):
    """Validate fixed C0 test population; return only post-pair pos/hard rows."""
    ids = [i['id'] for i in items]
    require(len(ids) == len(set(ids)), 'Duplicate source item IDs')
    flood = [i for i in items if i['partition'] == 'test' and i['phen'] == 'flood']
    require(Counter(i['kind'] for i in flood) == {'pos': 457, 'neg': 457, 'hard_neg': 457},
            'Expected C0 test flood 457 pos / 457 neg / 457 hard_neg')
    selected = sorted((i for i in flood if i['kind'] in ('pos', 'hard_neg')), key=lambda i: i['id'])
    require(len(selected) == 914 and len({i['tile'] for i in selected}) == 914,
            'Expected 914 distinct post-pair source tiles/items')
    for item in selected:
        require(re.fullmatch(r'ks_\d{5}', item['tile']) is not None, 'Invalid tile identifier')
        suffix = 'pos' if item['kind'] == 'pos' else 'hard'
        require(item['id'] == item['tile'] + '_q1_' + suffix, 'Source ID/kind mismatch')
        require(item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Source answer/kind mismatch')
        require(item['slots'] == ['pre_2', 'post'], 'Source post-pair slots mismatch')
        require(item['fold'] == 'test' and item['type'] == 'Q1', 'Unexpected source fold/type')
        require(str(item['event']) == str(item['cluster']), 'Source event/cluster mismatch')
        require(isinstance(item['dates'], list) and len(item['dates']) == 2
                and all(isinstance(d, str) for d in item['dates']), 'Invalid source dates')
        require(type(item['flood_frac']) in (float, int) and math.isfinite(item['flood_frac'])
                and 0 <= item['flood_frac'] <= 1, 'Invalid source flood fraction')
    return selected


def mask_stats(mask, valid):
    """Exact E1 numerator/denominator plus explicit raw-mask position audit."""
    for name, value, allowed in (('mask', mask, [0, 1, 2, 3]), ('valid', valid, [0, 1])):
        require(isinstance(value, np.ndarray) and value.shape == SHAPE, name + ' must have shape 192x192')
        require(np.issubdtype(value.dtype, np.integer), name + ' must have integer dtype')
        require(np.isin(value, allowed).all(), name + ' contains an unsupported value')
    labelled = (valid == 1) & (mask > 0)
    flood = (mask == 3) & labelled
    n, flood_px = int(labelled.sum()), int(flood.sum())
    # Keep E1's max(n, 1) exactly; kind validation separately rejects invalid gold.
    stats = {'valid_frac': n / mask.size, 'flood_frac': flood_px / max(n, 1), 'flood_px': flood_px,
             'labelled_px': n, 'total_px': int(mask.size), 'source_valid_frac': float((valid == 1).mean()),
             'valid_but_unlabelled_px': int(((valid == 1) & (mask == 0)).sum()),
             'mask_dtype': str(mask.dtype), 'valid_dtype': str(valid.dtype),
             'class_valid_counts': {f'mask_{m}|valid_{v}': int(((mask == m) & (valid == v)).sum())
                                    for m in range(4) for v in range(2)},
             'position': {'axis_order': 'row_y_then_column_x; original cache pixel grid',
                          'labelled_rows_px': labelled.sum(axis=1).astype(int).tolist(),
                          'labelled_columns_px': labelled.sum(axis=0).astype(int).tolist(),
                          'flood_rows_px': flood.sum(axis=1).astype(int).tolist(),
                          'flood_columns_px': flood.sum(axis=0).astype(int).tolist()}}
    return stats


def check_source_label(item, stats):
    require(math.isclose(stats['flood_frac'], float(item['flood_frac']), rel_tol=0, abs_tol=QA_TOLERANCE),
            'Mask flood fraction differs from rounded source QA fraction: ' + item['id'])
    if item['kind'] == 'pos':
        require(stats['flood_frac'] >= FLOOD_MIN, 'Positive source item is below flood threshold: ' + item['id'])
    elif item['kind'] == 'hard_neg':
        require(stats['flood_px'] == 0 and stats['valid_frac'] >= VALID_MIN,
                'Hard-negative source item fails dry/valid threshold: ' + item['id'])
    else:
        raise ValueError('Only post-pair pos/hard_neg labels may be audited')
    return stats['valid_frac'] >= VALID_MIN


def read_tile(root, tile):
    require(re.fullmatch(r'ks_\d{5}', tile) is not None, 'Invalid tile identifier')
    cache = Path(root) / 'kurosiwo_s1_cache'
    paths = {name: cache / (name + '_u8') / (tile + '.npy') for name in ('mask', 'valid')}
    before = {name: sha(path) for name, path in paths.items()}
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    result = mask_stats(arrays['mask'], arrays['valid'])
    for name, path in paths.items():
        require(sha(path) == before[name], 'Source changed during mask read: ' + str(path))
        result[name + '_sha256'] = before[name]
        result[name + '_path'] = str(path.relative_to(root))
    return result


def quantiles(values):
    data = np.asarray(values, dtype=np.float64)
    return {'n': len(data), 'mean': float(data.mean()), 'min': float(data.min()),
            'q25': float(np.quantile(data, .25)), 'median': float(np.median(data)),
            'q75': float(np.quantile(data, .75)), 'max': float(data.max())} if len(data) else None


def audit(args):
    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    require(out != root and out.is_relative_to(root), 'Output must be a new child directory of root')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'c1-flood-label-quality-v0', 'no_model_scores': True, 'created_at': now(),
                'root': str(root), 'code_sha256': sha(__file__), 'source_files': {}}
    phase, current_id = 'initializing', None
    try:
        shutil.copyfile(__file__, out / 'source.py')
        shutil.copyfile(args.prereg, out / 'prereg.json')
        manifest['prereg_sha256'] = sha(out / 'prereg.json')
        write(out / 'manifest.json', manifest)
        write(out / 'status.json', {'status': 'running', 'started_at': now(), 'no_model_scores': True})
        cfg = json.loads((out / 'prereg.json').read_text())
        require(cfg['schema'] == 'c1-flood-label-quality-plan-v0' and cfg['no_model_scores'] is True,
                'Invalid label-only plan')
        require(cfg['thresholds'] == {'flood_min': FLOOD_MIN, 'valid_min': VALID_MIN,
                                      'rounded_qa_abs_tolerance': QA_TOLERANCE}, 'Plan threshold mismatch')
        require(cfg['expected_selected_counts'] == {'pos': 457, 'hard_neg': 457}, 'Plan population mismatch')
        phase = 'verifying_frozen_source'
        source = root / C0_DIR
        cm_path = source / 'manifest.json'
        require(sha(cm_path) == cfg['expected_c0']['manifest_sha256'], 'C0 manifest differs from plan')
        cm = json.loads(cm_path.read_text())
        require(cm['schema'] == 'c0-prepared-v1', 'Unknown C0 manifest schema')
        source_files = {str(cm_path.relative_to(root)): sha(cm_path)}
        for filename, key in (('items.jsonl', 'items_sha256'), ('prereg.json', 'prereg_sha256'), ('source.py', 'code_sha256')):
            path = source / filename
            require(sha(path) == cm[key] == cfg['expected_c0'][key], 'Frozen C0 source mismatch: ' + filename)
            source_files[str(path.relative_to(root))] = sha(path)
        manifest.update(c0_manifest_sha256=source_files[str(cm_path.relative_to(root))],
                        items_sha256=cm['items_sha256'], c0_code_sha256=cm['code_sha256'],
                        c0_prereg_sha256=cm['prereg_sha256'], source_files=source_files)
        items = [json.loads(line) for line in (source / 'items.jsonl').read_text().splitlines() if line.strip()]
        selected = selected_items(items)
        manifest['selected_ids_sha256'] = hashlib.sha256(json.dumps([i['id'] for i in selected], separators=(',', ':')).encode()).hexdigest()
        write(out / 'manifest.json', manifest)
        phase = 'auditing_masks'
        rows = []
        with (out / 'quality.jsonl').open('x') as stream:
            for item in selected:
                current_id = item['id']
                stats = read_tile(root, item['tile'])
                eligible = check_source_label(item, stats)
                row = {'id': item['id'], 'tile': item['tile'], 'event': str(item['event']),
                       'dates': item['dates'], 'slots': item['slots'], 'kind': item['kind'],
                       'source_answer': item['answer'], 'source_qa_flood_frac': item['flood_frac'],
                       'eligible_symmetric_quality': bool(eligible), **stats}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                for name in ('mask', 'valid'):
                    source_files[row[name + '_path']] = row[name + '_sha256']
        phase, current_id = 'verifying_sources_unchanged', None
        for path, digest in source_files.items():
            require(sha(root / path) == digest, 'Source changed during audit: ' + path)
        require(sha(out / 'source.py') == sha(__file__) == manifest['code_sha256'], 'Audit code changed during run')
        require(sha(out / 'prereg.json') == manifest['prereg_sha256'] == sha(args.prereg), 'Audit plan changed during run')
        by_event = defaultdict(lambda: Counter())
        for row in rows:
            by_event[row['event']][row['kind']] += 1
            if row['eligible_symmetric_quality']:
                by_event[row['event']]['symmetric_' + row['kind']] += 1
        ids = {'schema': 'c1-flood-symmetric-quality-ids-v0', 'no_model_scores': True,
               'rule': 'E1 labelled fraction ((valid==1)&(mask>0)).mean() >= 0.90 for both kinds',
               'source_items_sha256': manifest['items_sha256'],
               'ids': sorted(r['id'] for r in rows if r['eligible_symmetric_quality']),
               'excluded_ids': sorted(r['id'] for r in rows if not r['eligible_symmetric_quality']),
               'by_kind': {kind: sorted(r['id'] for r in rows if r['kind'] == kind and r['eligible_symmetric_quality'])
                           for kind in ('pos', 'hard_neg')}}
        write(out / 'symmetric_quality_ids.json', ids)
        summary = {'schema': 'c1-flood-label-quality-summary-v0', 'n': len(rows), 'n_unique_tiles': len({r['tile'] for r in rows}),
                   'no_model_scores': True, 'counts': dict(Counter(r['kind'] for r in rows)),
                   'thresholds': cfg['thresholds'], 'valid_fraction_definition': '((valid==1)&(mask>0)).sum()/36864',
                   'symmetric_counts': {kind: len(ids['by_kind'][kind]) for kind in ('pos', 'hard_neg')},
                   'symmetric_n': len(ids['ids']), 'events': {event: dict(count) for event, count in sorted(by_event.items())},
                   'by_kind': {kind: {metric: quantiles([r[metric] for r in rows if r['kind'] == kind])
                                      for metric in ('valid_frac', 'source_valid_frac', 'flood_frac', 'flood_px')}
                               for kind in ('pos', 'hard_neg')},
                   'all_source_labels_verified': True,
                   'limitations': ['Post-mask gold/quality audit only; no prediction or discrimination result.',
                                   'The source mask is not an independently reviewed temporal event label.',
                                   'Dates retain E1 event-day/-12-day approximations, not verified SAR acquisition times.',
                                   'Symmetric quality selection is fixed without reading model outputs; it need not preserve class/event/date support.',
                                   'Mask/valid byte hashes refer to the existing 192x192 cache crops, not untouched 224x224 original TIFFs.']}
        write(out / 'summary.json', summary)
        manifest.update(quality_sha256=sha(out / 'quality.jsonl'), summary_sha256=sha(out / 'summary.json'),
                        symmetric_ids_sha256=sha(out / 'symmetric_quality_ids.json'),
                        n=914, source_files=source_files, completed_at=now(),
                        environment={'python': sys.version, 'numpy': np.__version__, 'device': 'cpu'})
        write(out / 'manifest.json', manifest)
        write(out / 'status.json', {'status': 'complete', 'valid': True, 'n': 914,
                                    'no_model_scores': True, 'completed_at': now()})
        print(json.dumps({'out': str(out), 'n': 914, 'symmetric_counts': summary['symmetric_counts'],
                          'no_model_scores': True}), flush=True)
    except Exception as exc:
        manifest['source_files'] = locals().get('source_files', {})
        manifest['failed_at'] = now()
        write(out / 'manifest.json', manifest)
        error = {'status': 'invalid', 'valid': False, 'phase': phase, 'current_id': current_id,
                 'error_type': type(exc).__name__, 'error': str(exc), 'no_model_scores': True, 'failed_at': now()}
        write(out / 'error.json', error)
        write(out / 'status.json', error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/work/data/olmoearth'))
    parser.add_argument('--out', default='c1_flood_label_quality_v0')
    parser.add_argument('--prereg', type=Path, required=True)
    audit(parser.parse_args())
