#!/usr/bin/env python3
"""T0: export/validate only the top KuroSiwo TORTILLA metadata table.

No table.read(), child metadata, TIFF, NPY, embedding, model, or network reads.
The 18-byte header and bounded top Parquet footer are archived/hashed, not the
whole ~7.5GB container. Original row strings survive every validation failure.

V1 header layout reference (reviewed, not imported dynamically):
https://github.com/tacofoundation/tacoreader/blob/main/tacoreader/v1/loader_dataframe.py
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import signal
import sys
import time

META_SHA = '23c0df0e058df6664bcfaf007fb7da48a370b247153896fb15261bafe100612b'
N = 7000
SPLITS = {'train': 4000, 'validation': 1000, 'test': 2000}
MAX_SECONDS = 300
MAX_FOOTER_BYTES = 64 * 1024 * 1024
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
SELECTED = ('tortilla:id', 'tortilla:data_split', 'stac:crs', 'stac:geotransform',
            'stac:raster_shape', 'stac:time_start', 'stac:time_end', 'stac:centroid',
            'actid', 'aoiid', 'flood_date', 'pwater', 'pflood')
RULES = {
    'version': 'kuro-t0-normalization-v0',
    'source_id': 'integer decimal digits; id=ks_{source_id:05d}; exact full expected set',
    'partition': 'exact train/validation/test strings; no alias mapping',
    'event_aoi': 'unsigned decimal integer; source leading zeros retained in source_row',
    'date': 'strict ISO date or ISO datetime; original preserved, first calendar date compared',
    'fractions': 'finite percent in [0,100]; catalog agreement absolute tolerance 1e-10',
    'centroid': 'strict 2D POINT; catalog numeric equality within 1e-12 degrees',
    'centroid_grid_check': 'EPSG3857 midpoint inverse projection vs stated centroid, atol 1e-6 degrees; not a geographic join tolerance',
    'grid': 'exact EPSG:3857, shape[224,224], GDAL[x0,10,0,y0,0,-10]; no rounding/snap/rotation repair',
    'bbox': '[xmin,ymin,xmax,ymax] computed from all four exact projected corners',
    'arrays': 'numeric ndarray/list cells use complete tolist values; display strings also retained',
    'source_times': 'finite source time_start/time_end preserved; never promoted to acquisition dates',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'inode': s.st_ino, 'device': s.st_dev}


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf8')


def write_json(path, value):
    with Path(path).open('xb') as f:
        f.write(encode(value))


def status(out, value):
    # Only this new exclusive output directory's own status is replaced.
    temp = out / 'status.tmp'
    temp.write_bytes(encode(value))
    temp.replace(out / 'status.json')


def read_jsonl_bytes(data):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return [json.loads(line, object_pairs_hook=pairs_hook,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            for line in data.decode('utf8').splitlines() if line.strip()]


def uint(value, field):
    text = str(value)
    require(re.fullmatch(r'\d+', text) is not None, 'Invalid integer: ' + field)
    return int(text)


def number(value, field):
    require(not isinstance(value, bool), 'Boolean numeric value: ' + field)
    text = str(value).strip()
    require(re.fullmatch(NUMBER, text) is not None, 'Invalid numeric value: ' + field)
    result = float(text)
    require(math.isfinite(result), 'Nonfinite numeric value: ' + field)
    return result


def vector(value, count, field):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = [number(x, field) for x in value]
    else:
        text = str(value).strip()
        require(text.startswith('[') and text.endswith(']'), 'Missing vector brackets: ' + field)
        # Full-token parsing: no regex extraction from malformed strings.
        body = text[1:-1].strip()
        pieces = re.split(r'\s*,\s*|\s+', body) if body else []
        result = [number(x, field) for x in pieces]
    require(len(result) == count, 'Wrong vector length: ' + field)
    return result


def point(value):
    match = re.fullmatch(r'POINT\s*\(\s*(' + NUMBER + r')\s+(' + NUMBER + r')\s*\)', str(value))
    require(match is not None, 'Invalid 2D centroid POINT')
    result = [number(match[1], 'longitude'), number(match[2], 'latitude')]
    require(-180 <= result[0] <= 180 and -90 <= result[1] <= 90, 'Centroid outside lon/lat range')
    return result


def iso_date(value):
    text = str(value)
    require(re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?', text) is not None,
            'Invalid ISO event date/datetime')
    return datetime.fromisoformat(text.replace('Z', '+00:00')).date().isoformat()


def validate_reference(records):
    require(len(records) == N, 'Reference item count differs')
    indexed = {}
    for item in records:
        sid = item['id']
        require(isinstance(sid, str) and re.fullmatch(r'ks_\d{5}', sid) is not None, 'Malformed reference ID')
        require(sid not in indexed, 'Duplicate reference ID')
        uint(item['actid'], 'reference event'); uint(item['aoiid'], 'reference AOI')
        point(item['centroid']); iso_date(item['flood_date'])
        for field in ('pwater', 'pflood'):
            require(0 <= number(item[field], field) <= 100, 'Reference percent outside range')
        indexed[sid] = item
    require(set(indexed) == {f'ks_{i:05d}' for i in range(N)}, 'Reference exact ID coverage differs')
    require(dict(Counter(x['split'] for x in records)) == SPLITS, 'Reference split support differs')
    return indexed


def capture_row(row, index):
    source = {str(k): str(v) for k, v in row.items()}
    require(len(source) == len(row), 'Duplicate stringified column name')
    numeric_vectors = {}
    for field, count in [('stac:geotransform', 6), ('stac:raster_shape', 2)]:
        value = row.get(field)
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            # Keep full values as strings even if nonfinite or nested/malformed.
            numeric_vectors[field] = [str(x) for x in value]
    return {'table_row_index': index, 'source_row': source, 'source_numeric_vectors': numeric_vectors}


def normalize(record, reference):
    source, vectors = record['source_row'], record['source_numeric_vectors']
    require(set(SELECTED) <= set(source), 'Missing required source columns')
    source_id = uint(source['tortilla:id'], 'source_id')
    sid = f'ks_{source_id:05d}'
    require(sid in reference, 'Unknown source ID: ' + sid)
    original = reference[sid]
    partition = source['tortilla:data_split']
    require(partition == original['split'], 'partition mismatch: ' + sid)
    event, aoi = uint(source['actid'], 'event'), uint(source['aoiid'], 'AOI')
    require((event, aoi) == (uint(original['actid'], 'event'), uint(original['aoiid'], 'AOI')),
            'event/AOI mismatch: ' + sid)
    date = iso_date(source['flood_date'])
    require(date == iso_date(original['flood_date']), 'event date mismatch: ' + sid)
    center, reference_center = point(source['stac:centroid']), point(original['centroid'])
    require(all(abs(a-b) <= 1e-12 for a, b in zip(center, reference_center)), 'centroid mismatch: ' + sid)
    percentages = {}
    for field in ('pwater', 'pflood'):
        value = number(source[field], field)
        require(0 <= value <= 100 and abs(value-number(original[field], field)) <= 1e-10,
                field + ' mismatch: ' + sid)
        percentages[field] = value
    crs = source['stac:crs'].strip().upper()
    require(crs == 'EPSG:3857', 'CRS mismatch: ' + sid)
    transform = vector(vectors.get('stac:geotransform', source['stac:geotransform']), 6, 'GDAL transform')
    shape = vector(vectors.get('stac:raster_shape', source['stac:raster_shape']), 2, 'raster shape')
    require(shape == [224, 224], 'Raster shape is not 224x224: ' + sid)
    x, dx, rx, y, ry, dy = transform
    require((dx, rx, ry, dy) == (10.0, 0.0, 0.0, -10.0), 'Grid is not unrotated north-up 10m: ' + sid)
    corners = [[x + col*dx + row*rx, y + col*ry + row*dy]
               for col, row in [(0, 0), (224, 0), (224, 224), (0, 224), (0, 0)]]
    require(all(math.isfinite(v) for pair in corners for v in pair), 'Nonfinite grid bounds')
    bbox = [min(p[0] for p in corners), min(p[1] for p in corners),
            max(p[0] for p in corners), max(p[1] for p in corners)]
    require(bbox[2]-bbox[0] == 2240 and bbox[3]-bbox[1] == 2240, 'Unexpected grid span')
    projected_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    require(all(abs(v) <= 20037508.35 for v in projected_center), 'Invalid WebMercator center')
    lon = math.degrees(projected_center[0] / 6378137.0)
    lat = math.degrees(math.atan(math.sinh(projected_center[1] / 6378137.0)))
    require(max(abs(lon-center[0]), abs(lat-center[1])) <= 1e-6,
            'Grid midpoint/centroid mismatch (not a join tolerance): ' + sid)
    source_times = {key: number(source['stac:'+key], key) for key in ('time_start', 'time_end')}
    require(source_times['time_start'] <= source_times['time_end'], 'Source time interval reversed')
    return {**record, 'id': sid, 'source_id': source_id, 'partition': partition,
            'event_id': event, 'aoi_id': aoi, 'event_date': date, 'centroid_lonlat': center,
            **percentages, 'crs': crs, 'gdal_geotransform': transform,
            'raster_shape': [int(v) for v in shape], 'bbox_epsg3857': bbox,
            'corners_epsg3857': corners, 'source_times': source_times,
            'acquisition_dates': None, 'valid': True, 'mismatches': []}


def top_metadata_bytes(source):
    size = source.stat().st_size
    with source.open('rb') as handle:
        header = handle.read(18)
        require(len(header) == 18 and header[:2] in (b'#y', b'WX'), 'Unsupported/truncated TACO v1 header')
        offset = int.from_bytes(header[2:10], 'little')
        length = int.from_bytes(header[10:18], 'little')
        require(offset >= 200 and 8 <= length <= MAX_FOOTER_BYTES and offset+length <= size,
                'Invalid/oversized top metadata range')
        handle.seek(offset)
        footer = handle.read(length)
    require(len(footer) == length and footer[:4] == footer[-4:] == b'PAR1',
            'Truncated/non-Parquet top metadata')
    return header, footer, {'offset': offset, 'length': length, 'sha256': digest(footer),
                            'header_bytes': 18, 'header_sha256': digest(header)}


@contextmanager
def time_limit(seconds):
    def expired(signum, frame):
        raise TimeoutError('T0 export exceeded fixed 300-second limit')
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), 'Refusing to replace an active process timer')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def export(root, out, loader=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output exists; no overwrite or resume')
    out.mkdir(parents=True)
    started = time.monotonic()
    phase, source = 'starting', root / 'geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla'
    before = None
    try:
        with time_limit(MAX_SECONDS):
            status(out, {'schema': 'kuro-t0-status-v0', 'status': 'running', 'valid': False, 'started_at': now()})
            script = Path(__file__).resolve().read_bytes()
            (out / 'source.py').write_bytes(script)
            write_json(out / 'normalization.json', {'rules': RULES, 'required_columns': list(SELECTED),
                       'expected_n': N, 'expected_splits': SPLITS, 'meta_sha256': META_SHA,
                       'max_seconds': MAX_SECONDS, 'max_top_metadata_bytes': MAX_FOOTER_BYTES})
            require(source.is_file() and source.resolve().is_relative_to(root), 'Existing local source required')
            before = stat(source)
            phase = 'reference_metadata'
            meta_path = root / 'kurosiwo_s1_cache/meta.jsonl'
            data = meta_path.read_bytes()
            (out / 'cache_meta.jsonl').write_bytes(data)
            require(digest(data) == META_SHA, 'Pinned cache metadata SHA mismatch')
            reference = validate_reference(read_jsonl_bytes(data))
            phase = 'top_metadata_bytes'
            header, footer, footer_info = top_metadata_bytes(source)
            (out / 'header18.bin').write_bytes(header)
            (out / 'footer.parquet').write_bytes(footer)
            phase = 'load_top_table'
            if loader is None:
                import tacoreader
                loader = tacoreader.load
                reader_version = importlib.metadata.version('tacoreader')
            else:
                reader_version = 'injected test loader'
            table = loader(str(source))
            columns = [str(c) for c in table.columns]
            require(len(columns) == len(set(columns)), 'Duplicate top-level column names')
            # No table.read(), sample.read(), rasterio, or full container hashing.
            phase = 'preserve_all_source_rows'
            records = []
            with (out / 'raw_rows.jsonl').open('xb') as handle:
                for index, (_, row) in enumerate(table.iterrows()):
                    captured = capture_row(row, index)
                    handle.write(encode(captured))
                    records.append(captured)
            write_json(out / 'table_schema.json', {'columns': columns,
                       'dtypes': {str(k): str(v) for k, v in table.dtypes.items()},
                       'n_rows': len(records), 'tacoreader_version': reader_version})
            phase = 'normalize_and_crosscheck'
            normalized, mismatches, seen = [], [], Counter()
            for record in records:
                raw_id = record['source_row'].get('tortilla:id')
                try:
                    seen[uint(raw_id, 'source_id')] += 1
                    result = normalize(record, reference)
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    error = {'table_row_index': record['table_row_index'], 'source_id_raw': raw_id,
                             'error': str(exc)}
                    mismatches.append(error)
                    result = {**record, 'valid': False, 'mismatches': [str(exc)]}
                normalized.append(result)
            if len(records) != N:
                mismatches.append({'error': f'Top table count {len(records)} != {N}'})
            missing, extra = sorted(set(range(N))-set(seen)), sorted(set(seen)-set(range(N)))
            duplicates = {str(k): v for k, v in seen.items() if v != 1}
            if missing or extra or duplicates:
                mismatches.append({'error': 'Source ID coverage/uniqueness differs', 'missing': missing,
                                   'unexpected': extra, 'duplicates': duplicates})
            observed_splits = dict(Counter(r['source_row'].get('tortilla:data_split') for r in records))
            if observed_splits != SPLITS:
                mismatches.append({'error': 'Source split support differs', 'observed': observed_splits})
            with (out / 'metadata.jsonl').open('xb') as handle:
                for row in normalized:
                    handle.write(encode(row))
            with (out / 'selected_columns.jsonl').open('xb') as handle:
                for record in records:
                    handle.write(encode({'table_row_index': record['table_row_index'],
                        'values': {key: record['source_row'].get(key) for key in SELECTED},
                        'source_numeric_vectors': record['source_numeric_vectors']}))
            write_json(out / 'mismatches.json', mismatches)
            phase = 'final_source_verification'
            after = stat(source)
            require(after == before, 'Source container stat changed during export')
            header_after, footer_after, _ = top_metadata_bytes(source)
            require(header_after == header and footer_after == footer and stat(source) == before,
                    'Source header/footer changed during export')
            require(file_sha(meta_path) == META_SHA and Path(__file__).resolve().read_bytes() == script,
                    'Reference metadata/export code changed during export')
            summary = {'schema': 'kuro-t0-top-metadata-v0', 'status': 'complete' if not mismatches else 'invalid',
                       'valid': not mismatches, 'n_rows': len(records), 'n_expected': N,
                       'split_counts': observed_splits, 'n_valid_rows': sum(r['valid'] for r in normalized),
                       'n_mismatches': len(mismatches), 'all_source_rows_preserved': True,
                       'no_child_metadata_reads': True, 'no_image_reads': True,
                       'no_model_or_prediction_reads': True, 'no_source_mutations': True,
                       'acquisition_dates_claimed': False, 'created_at': now(),
                       'elapsed_s': time.monotonic()-started}
            write_json(out / 'summary.json', summary)
            files = [p.name for p in out.iterdir() if p.is_file() and p.name != 'status.json']
            manifest = {'schema': 'kuro-t0-top-metadata-manifest-v0', 'valid': not mismatches,
                        'source_container': str(source), 'source_stat_before': before, 'source_stat_after': after,
                        'container_full_sha256': None,
                        'container_hash_note': 'Full ~7.5GB archive not hashed/read. Only top header and Parquet footer read twice by this exporter; tacoreader.load reads top metadata separately. No child/image read calls.',
                        'top_metadata': footer_info, 'cache_meta_sha256': META_SHA,
                        'code_sha256': digest(script), 'files_sha256': {name: file_sha(out/name) for name in sorted(files)},
                        'raw_rows_sha256': file_sha(out/'raw_rows.jsonl'),
                        'metadata_sha256': file_sha(out/'metadata.jsonl'),
                        'selected_columns_sha256': file_sha(out/'selected_columns.jsonl'),
                        'normalization_version': RULES['version'], 'reader_version': reader_version}
            write_json(out / 'manifest.json', manifest)
            require(not mismatches, f'T0 invalid: {len(mismatches)} metadata/identity/grid mismatch record(s) preserved')
            status(out, {'schema': 'kuro-t0-status-v0', 'status': 'complete', 'valid': True,
                         'completed_at': now(), 'manifest_sha256': file_sha(out/'manifest.json')})
            return summary
    except BaseException as exc:
        # Preserve every already-written raw row and diagnostic; do not retry/reselect.
        failure = {'schema': 'kuro-t0-failure-v0', 'status': 'invalid', 'valid': False,
                   'phase': phase, 'error_type': type(exc).__name__, 'error': str(exc),
                   'partial_export_preserved': True, 'source_stat_before': before,
                   'source_stat_after': stat(source) if source.exists() else None,
                   'elapsed_s': time.monotonic()-started, 'failed_at': now()}
        write_json(out / 'failure.json', failure)
        status(out, failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/work/data/olmoearth'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.root, args.out)
    except Exception as exc:
        print(f'T0 failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
