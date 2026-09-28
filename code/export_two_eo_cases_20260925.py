#!/usr/bin/env python3
"""Draft: CPU-only export of two pre-fixed E3/E4 development cases.

Reads existing LOCAL KuroSiwo tortilla + NPY/cache files, writes a NEW export.
No HTTP/download, model, prediction, E5 file access, selection search, or label edit.
Original embedded GeoTIFF bytes are copied exactly, not re-encoded. Optional PNG
is a scientific display of the raw arrays; it is not a generated image.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

CASE_IDS = ('ks_06770', 'ks_05265')
PINS = {
    'e3_items': 'f3ae0abc245b3e800f1a39b4ca5a250b36f4cf503fe364f3003a556c7f907502',
    'quality': '8f654dfbaabc41e3b27de9a04dfe0768c1f0a748439c70acb61ddae08a17a78f',
    'c0_manifest': '5055bd2d73f1e9f162a43899a2d0f5c46cccef53137cb3a5ae31e7a2675bcb92',
}
ASSETS = {0: 'pre_event_1', 1: 'pre_event_2', 2: 'post_event', 4: 'mask', 5: 'invalid_data'}
SELECTION = ('Two IDs fixed before E5 outcomes; illustrative already-exposed test cases from '
             'event562/AOI13. Not random, representative, new evaluation, or a C1-primary matched pair. '
             'ks_06770 fails the existing symmetric label-coverage criterion; do not replace it.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


def embedded_bytes(locator, source):
    """Allow only a bounded local slice of the explicitly supplied tortilla."""
    match = re.fullmatch(r'/vsisubfile/(\d+)_(\d+),(.+)', str(locator))
    require(match is not None, 'Unexpected embedded locator; no network fallback')
    offset, length = int(match[1]), int(match[2])
    require(Path(match[3]).resolve() == source.resolve(), 'Locator points to another file')
    require(0 < length < 16 * 1024 * 1024 and offset >= 0 and offset + length <= source.stat().st_size,
            'Invalid or unexpectedly large embedded TIFF slice')
    with source.open('rb') as f:
        f.seek(offset)
        blob = f.read(length)
    require(len(blob) == length, 'Truncated original TIFF')
    return blob, {'offset': offset, 'length': length, 'sha256': hashlib.sha256(blob).hexdigest()}


def mask_summary(raw_mask, valid):
    require(raw_mask.shape == valid.shape == (192, 192), 'Expected model crop shape')
    require(set(np.unique(raw_mask)) <= {0, 1, 2, 3}, 'Unexpected raw mask classes')
    require(set(np.unique(valid)) <= {0, 1}, 'Unexpected validity classes')
    labelled = (valid == 1) & (raw_mask != 3)
    flood = labelled & (raw_mask == 2)
    return {'total_px': int(raw_mask.size), 'raw_class_counts': {str(k): int((raw_mask == k).sum()) for k in range(4)},
            'source_valid_px': int((valid == 1).sum()), 'labelled_px': int(labelled.sum()),
            'unknown_px': int((~labelled).sum()), 'flood_px': int(flood.sum()),
            'labelled_fraction': float(labelled.mean()), 'flood_fraction_all_crop': float(flood.mean()),
            'flood_fraction_labelled': float(flood.sum() / labelled.sum()) if labelled.any() else None}


def check_case_policy(sid, row):
    require(sid in CASE_IDS, 'Case selection is fixed; no replacement or additional case')
    require(str(row['tortilla:data_split']) == 'test' and int(row['actid']) == 562
            and str(row['aoiid']) == '13', 'Case scope changed')
    require(str(row['flood_date'])[:10] == '2022-01-29', 'Event date changed')


def fixed_quality_rows(records):
    selected = [r for r in records if r['tile'] in CASE_IDS]
    require(len(selected) == 2 and {r['tile'] for r in selected} == set(CASE_IDS),
            'Missing/duplicate fixed quality rows')
    result = {r['tile']: r for r in selected}
    require(result['ks_06770']['eligible_symmetric_quality'] is False
            and result['ks_05265']['eligible_symmetric_quality'] is True,
            'Fixed cases quality membership changed; do not replace low-coverage case')
    return result


def validate_c1_cache(root, quality, mask, valid):
    summary = mask_summary(mask, valid)
    remapped = np.array([1, 2, 3, 0], dtype='uint8')[mask]
    for role, array in [('mask', remapped), ('valid', valid)]:
        path = (root / quality[role + '_path']).resolve()
        require(path.is_relative_to(root.resolve()), 'C1 cache path outside root')
        require(sha(path) == quality[role + '_sha256'], 'C1 source hash changed: ' + role)
        require(np.array_equal(np.load(path, allow_pickle=False), array), 'Crop/remap differs from C1 cache')
    require((summary['flood_px'], summary['labelled_px'], summary['total_px'])
            == (quality['flood_px'], quality['labelled_px'], quality['total_px']), 'C1 coverage disagrees')
    return summary


def render(destination, cube, mask, valid, metadata, summary):
    """Raw-array scientific figure. All legends lie outside every data panel."""
    from PIL import Image, ImageDraw, ImageFont
    require(cube.shape == (2, 3, 192, 192), 'Expected two bands and three model-crop slots')
    require(summary == mask_summary(mask, valid), 'Rendered summary differs from arrays')
    fonts = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
             Path('/System/Library/Fonts/Supplemental/Arial.ttf')]
    def font(size):
        for path in fonts:
            if path.is_file():
                return ImageFont.truetype(str(path), size)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()
    image = Image.new('RGB', (1840, 1440), 'white')
    draw = ImageDraw.Draw(image)
    ink, muted = '#192e43', '#536272'
    xs, ys, size = [42, 480, 918, 1356], [214, 780], 384
    rectangles = {}
    def panel(key, pixels, col, row):
        x, y = xs[col], ys[row]
        image.paste(Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST), (x, y))
        draw.rectangle((x-1, y-1, x+size, y+size), outline='#a6b3bf', width=1)
        rectangles[key] = [x, y, x+size, y+size]
    draw.text((42, 24), f"{metadata['id']} | event {metadata['event_id']} | event date {metadata['event_date']} | test", font=font(30), fill=ink)
    whole = f"{100*summary['flood_fraction_all_crop']:.2f}%"
    labelled = 'N/A' if summary['flood_fraction_labelled'] is None else f"{100*summary['flood_fraction_labelled']:.2f}%"
    draw.text((42, 76), f'192 x 192 model crop: reference flood {whole} of whole crop / {labelled} of labelled pixels', font=font(22), fill=ink)
    draw.text((42, 116), 'Fixed illustrative cases; no damage/causal-impact claim. Event date is not a verified acquisition date.', font=font(20), fill=muted)
    for band, name in enumerate(('VV', 'VH')):
        for slot, title in enumerate(('Slot pre_1', 'Slot pre_2', 'Slot post')):
            values = cube[band, slot]
            positive = np.isfinite(values) & (values > 0)
            display = np.zeros(values.shape, dtype=np.float64)
            display[positive] = np.clip((10*np.log10(values[positive])+25)/25, 0, 1)
            gray = np.rint(display*255).astype('uint8')
            pixels = np.repeat(gray[:, :, None], 3, axis=2)
            pixels[~positive] = [187, 67, 163]
            panel(f'{name}_{slot}', pixels, slot, band)
            draw.text((xs[slot], ys[band]-66), f'{title} | {name}', font=font(24), fill=ink)
            draw.text((xs[slot], ys[band]-33), 'Acquisition date unverified', font=font(18), fill=muted)
    palette = np.array([[217,224,229], [53,106,160], [231,120,50], [255,211,78], [48,52,59]], dtype='uint8')
    shown = mask.copy()
    shown[valid != 1] = 4
    panel('reference_mask', palette[shown], 3, 0)
    draw.text((xs[3], ys[0]-66), 'Post reference mask', font=font(24), fill=ink)
    draw.text((xs[3], ys[0]-33), 'Not a model prediction', font=font(18), fill=muted)
    legends = []
    def legend(x, y, color, text):
        draw.rectangle((x, y+3, x+16, y+19), fill=color)
        draw.text((x+25, y), text, font=font(17), fill=ink)
        legends.append([x, y, x+400, y+25])
    # Dedicated whitespace BELOW the mask. No legend is painted inside pixels.
    for j, text in enumerate(('0: no-water', '1: permanent water', '2: flood', '3: label unknown', 'valid != 1')):
        legend(xs[3], 604+j*20, tuple(palette[j]), text)
    known = (valid == 1) & (mask != 3)
    support = np.empty((*known.shape, 3), dtype='uint8')
    support[:] = [255,211,78]
    support[known] = [54,143,113]
    panel('labelled_support', support, 3, 1)
    draw.text((xs[3], ys[1]-66), f"Labelled support {100*summary['labelled_fraction']:.2f}%", font=font(22), fill=ink)
    draw.text((xs[3], ys[1]-33), f"Unknown {summary['unknown_px']:,} / {summary['total_px']:,} pixels", font=font(18), fill=muted)
    legend(xs[3], 1185, (54,143,113), 'valid=1 AND raw mask!=3')
    legend(xs[3], 1214, (255,211,78), 'unknown / invalid')
    gradient = np.repeat(np.linspace(0, 255, 1220).round().astype('uint8')[None], 18, axis=0)
    image.paste(Image.fromarray(gradient).convert('RGB'), (42, 1228))
    draw.text((42, 1254), '-25 dB', font=font(18), fill=muted)
    draw.text((1215, 1254), '0 dB', font=font(18), fill=muted)
    draw.text((42, 1292), 'VV and VH: fixed clipped display; no per-image auto-contrast. Magenta marks nonpositive/nonfinite raw values.', font=font(20), fill=muted)
    draw.text((42, 1334), 'Raw mask0 is semantic no-water, even when the TIFF nodata field is 0. Raw mask3 remains unknown.', font=font(20), fill=muted)
    draw.text((42, 1376), 'All three SAR slots are shown. QA dates were approximations. Reference mask is for the post slot only.', font=font(20), fill=muted)
    image.save(destination, format='PNG')
    return {'canvas_size': list(image.size), 'data_rectangles': rectangles,
            'legend_rectangles': legends, 'sar_db_range': [-25, 0], 'font': next((str(p) for p in fonts if p.is_file()), 'Pillow default')}


def export(args):
    # Heavy geospatial dependencies are imported only when the explicitly invoked
    # export actually runs. Merely importing this draft performs no data access.
    import rasterio
    from rasterio.warp import transform as transform_coordinates
    import tacoreader
    root, out = args.root.resolve(), args.out.resolve()
    source = root / 'geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla'
    require(source.is_file(), 'Existing local tortilla required; no download fallback')
    require(not out.exists(), 'Output already exists; no overwrite')
    inputs = {'e3_items': args.e3_items, 'quality': args.quality, 'c0_manifest': args.c0_manifest}
    for name, path in inputs.items():
        require(sha(path) == PINS[name], 'Frozen metadata differs: ' + name)
    e3 = rows(args.e3_items)
    quality = fixed_quality_rows(rows(args.quality))
    manifest = json.loads(args.c0_manifest.read_text())
    local_meta = {r['id']: r for r in rows(root / 'kurosiwo_npy/meta.jsonl') if r['id'] in CASE_IDS}
    require(set(local_meta) == set(CASE_IDS), 'Raw NPY metadata missing')
    container_stat = stat(source)
    table = tacoreader.load(str(source))
    out.mkdir(parents=True)
    write(out / 'selection.json', {'ids': list(CASE_IDS), 'selection': SELECTION,
                                  'frozen_metadata_sha256': PINS, 'script_sha256': sha(__file__)})
    cases = []
    try:
        for sid in CASE_IDS:
            matches = np.flatnonzero(table['tortilla:id'].astype(int).to_numpy() == int(sid[3:]))
            require(len(matches) == 1, 'Nonunique tortilla ID')
            row_index = int(matches[0])
            row, sample = table.iloc[row_index], table.read(row_index)
            check_case_policy(sid, row)
            destination = out / sid
            destination.mkdir()
            assets, arrays, grid = [], {}, None
            for index, expected_role in ASSETS.items():
                asset_row = sample.iloc[index]
                require(str(asset_row['tortilla:id']) == expected_role, 'Asset role/order differs')
                locator = sample.read(index)
                original, slice_record = embedded_bytes(locator, source)
                target = destination / f'asset_{index}.tif'
                with target.open('xb') as handle:
                    handle.write(original)
                with rasterio.open(target) as src:
                    array = src.read(masked=False)  # raw mask class0 is semantic no-water despite TIFF nodata=0
                    this_grid = (str(src.crs), tuple(src.transform), src.height, src.width)
                    require(src.crs is not None and src.height == src.width == 224, 'Missing/unexpected source grid')
                    require(grid is None or grid == this_grid, 'Asset grids disagree')
                    grid = this_grid
                    require(array.shape == ((2, 224, 224) if index < 3 else (1, 224, 224)), 'Unexpected asset shape')
                    arrays[index] = array
                    assets.append({'index': index, 'role': expected_role, 'file': target.name,
                        'original_embedded_bytes': slice_record, 'sha256': sha(target),
                        'sample_row': {str(k): str(v) for k, v in asset_row.items()},
                        'crs': str(src.crs), 'transform': list(src.transform), 'shape': list(array.shape),
                        'dtype': str(array.dtype), 'nodata': src.nodata, 'tags': src.tags(),
                        'band_tags': [src.tags(k) for k in src.indexes], 'descriptions': list(src.descriptions)})
            raw = np.stack([arrays[i] for i in (0, 1, 2)], axis=1).astype('float32')
            raw_mask, raw_valid = arrays[4][0], arrays[5][0]
            npy_paths = {'raw': root / f'kurosiwo_npy/raw_f32/{sid}.npy',
                         'mask': root / f'kurosiwo_npy/mask_raw_u8/{sid}.npy',
                         'valid': root / f'kurosiwo_npy/valid_u8/{sid}.npy'}
            for role, array in [('raw', raw), ('mask', raw_mask), ('valid', (raw_valid == 1).astype('uint8'))]:
                saved = np.load(npy_paths[role], allow_pickle=False)
                require(np.array_equal(array, saved, equal_nan=True), 'Original TIFF and historical NPY values differ: ' + role)
            cube = raw[:, :, 16:208, 16:208]
            mask, valid = raw_mask[16:208, 16:208], (raw_valid[16:208, 16:208] == 1).astype('uint8')
            q = quality[sid]
            stats = validate_c1_cache(root, q, mask, valid)
            require(abs(float((raw_mask == 2).mean()) * 100 - float(row['pflood'])) < 1e-8, 'Whole-tile pflood differs')
            cache_path = root / f'kurosiwo_s1_cache/single_fp16/{sid}.npy'
            require(sha(cache_path) == manifest['cache_sha256'][str(cache_path)], 'C0 embedding cache changed')
            crs, transform, _, _ = grid
            affine = rasterio.Affine(*transform[:6])
            polygons = {}
            for name, extent in [('whole_tile', (0, 0, 224, 224)), ('model_crop', (16, 16, 208, 208))]:
                x0, y0, x1, y1 = extent
                points = [affine * p for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
                lon, lat = transform_coordinates(crs, 'EPSG:4326', *zip(*points))
                polygons[name] = {'type': 'Polygon', 'coordinates': [[list(p) for p in zip(lon, lat)]]}
            model_items = [i for i in e3 if i['tile'] == sid]
            require(len(model_items) == (2 if sid == 'ks_06770' else 1), 'E3 case item membership differs')
            stac_times = {a['role']: {k: v for k, v in a['sample_row'].items() if 'time' in k.lower() or 'date' in k.lower()} for a in assets if a['index'] < 3}
            report = {'id': sid, 'event_id': int(row['actid']), 'aoi_id': str(row['aoiid']),
                      'event_date': str(row['flood_date'])[:10], 'split': str(row['tortilla:data_split']),
                      'centroid_wkt_from_source': str(row['stac:centroid']), 'geometry': polygons,
                      'source_row': {str(k): str(v) for k, v in row.items()}, 'assets': assets,
                      'npys_equal_original_tiff_values': True, 'npy_sha256': {k: sha(v) for k, v in npy_paths.items()},
                      'model_embedding_sha256': sha(cache_path), 'c1_mask_valid_hashes_verified': True,
                      'model_crop': {'rows': [16, 208], 'columns': [16, 208], 'shape': [192, 192]},
                      'reference_summary': stats, 'eligible_C1_symmetric_quality': q['eligible_symmetric_quality'],
                      'qa_items': model_items, 'source_sample_time_fields': stac_times,
                      'acquisition_dates': {k: None for k in ('pre_1', 'pre_2', 'post')},
                      'acquisition_date_status': 'unverified; source sample fields and all TIFF tags retained for audit; event/stac timestamps are not promoted to acquisition dates',
                      'selection': SELECTION, 'predictions_included': False,
                      'mask_contract': 'raw0 no-water,1 permanent-water,2 flood,3 unknown; valid==1; ignore TIFF nodata=0 for categorical interpretation',
                      'limitations': ['source_label_not_model_prediction', 'post_mask_does_not_verify_pre_event_absence',
                          'already_exposed_test_cases', 'same_event_different_geographic_tiles',
                          'no_crop_damage_or_causal_impact_measurement', 'current_byte_match_does_not_prove_historical_encoder_lineage'],
                      'sar_nonpositive_or_nonfinite_crop_counts': [[int((~np.isfinite(cube[b, t]) | (cube[b, t] <= 0)).sum()) for t in range(3)] for b in range(2)]}
            np.savez_compressed(destination / 'evidence_arrays.npz', raw_linear_vv_vh=cube,
                                reference_mask_raw=mask, source_valid=valid,
                                labelled_valid=((valid == 1) & (mask != 3)).astype('uint8'))
            report['arrays_sha256'] = sha(destination / 'evidence_arrays.npz')
            if args.render:
                report['preview_layout'] = render(destination / 'preview.png', cube, mask, valid, report, stats)
                report['preview_file'], report['preview_sha256'] = 'preview.png', sha(destination / 'preview.png')
            write(destination / 'source_manifest.json', report)
            cases.append({'id': sid, 'manifest': f'{sid}/source_manifest.json',
                          'manifest_sha256': sha(destination / 'source_manifest.json'),
                          'preview_file': f'{sid}/preview.png' if args.render else None,
                          'eligible_C1_symmetric_quality': q['eligible_symmetric_quality']})
        require(stat(source) == container_stat, 'Container changed during bounded export')
        write(out / 'case_index.json', {'schema': 'eo-two-fixed-source-cases-v0', 'status': 'exported',
              'case_ids': list(CASE_IDS), 'selection': SELECTION, 'cases': cases,
              'source_container': str(source), 'container_stat': container_stat,
              'container_full_sha256': None, 'container_hash_note': 'Only selected original TIFF byte slices hashed; avoids reading the entire 7.5GB archive.',
              'metadata_pins': PINS, 'script_sha256': sha(__file__), 'created_at': datetime.now(timezone.utc).isoformat(),
              'predictions_included': False, 'no_new_model_inference': True})
    except BaseException as exc:
        write(out / 'failure.json', {'error': str(exc), 'case_ids': list(CASE_IDS), 'partial_export_preserved': True})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/work/data/olmoearth'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--e3-items', type=Path, required=True)
    p.add_argument('--quality', type=Path, required=True)
    p.add_argument('--c0-manifest', type=Path, required=True)
    p.add_argument('--render', action='store_true')
    export(p.parse_args())
