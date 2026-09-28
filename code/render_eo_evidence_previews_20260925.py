#!/usr/bin/env python3
"""Render six existing KuroSiwo evidence arrays without model predictions.

Requires NumPy and Pillow. Uses the fixed intensity convention verified in
code/fig_kurosiwo_material.py: normalized clipped VV in [-25, 0] dB.
This NPZ contains a binary flood mask, not the original four-class mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

EXPECTED_IDS = ('ks_05229', 'ks_06412', 'ks_05989', 'ks_05983', 'ks_06327', 'ks_06507')
SOURCE_RELATIVE = 'artifacts/fig_kurosiwo_material.npz'
BUILDER_RELATIVE = 'code/fig_kurosiwo_material.py'
CACHE_BUILDER_RELATIVE = 'code/extract_kurosiwo_s1_cache.py'
FLOOD_COLOR = (0, 147, 212)
OTHER_COLOR = (223, 229, 235)
INK = (21, 35, 51)
MUTED = (77, 94, 110)
FONT_PATHS = (
    Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int):
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def checked_array(npz, key: str, binary: bool = False):
    array = np.asarray(npz[key])
    if array.shape != (192, 192):
        raise ValueError(f'{key}: expected (192,192), got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError(f'{key}: non-finite pixels; do not silently fill')
    if binary:
        if not np.isin(array, (0.0, 1.0)).all():
            raise ValueError(f'{key}: expected binary flood mask')
    elif array.min() < 0 or array.max() > 1:
        raise ValueError(f'{key}: expected normalized clipped VV in [0,1]')
    return array


def render_case(case_id, meta, pre, post, label, destination: Path):
    image = Image.new('RGB', (1280, 730), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 86), fill=(241, 246, 250))
    draw.text((32, 18), f'KuroSiwo evidence | {case_id}', font=font(28), fill=INK)
    draw.text((32, 56), f'Event date: {meta["flood_date"]}   |   Event ID: {meta["actid"]}   |   Source tile pflood: {meta["pflood"]:.2f}%', font=font(17), fill=MUTED)
    panel_x = (32, 448, 864)
    titles = ('Pre-event SAR (pre_2)', 'Post-event SAR (post)', 'Reference flood mask')
    for x, title in zip(panel_x, titles):
        draw.text((x, 105), title, font=font(22), fill=INK)
    for x, array in zip(panel_x[:2], (pre, post)):
        pixels = np.round(array * 255).astype(np.uint8)
        panel = Image.fromarray(pixels).convert('RGB').resize((384, 384), resample=Image.Resampling.NEAREST)
        image.paste(panel, (x, 144))
    label_rgb = np.empty((*label.shape, 3), dtype=np.uint8)
    label_rgb[:] = OTHER_COLOR
    label_rgb[label == 1] = FLOOD_COLOR
    panel = Image.fromarray(label_rgb).resize((384, 384), resample=Image.Resampling.NEAREST)
    image.paste(panel, (panel_x[2], 144))
    draw = ImageDraw.Draw(image)
    for x in panel_x:
        draw.rectangle((x, 144, x + 383, 527), outline=(193, 204, 214), width=1)
    # Both SAR panels use one fixed display scale: no per-case auto-contrast.
    gradient = np.repeat(np.linspace(0, 255, 300).round().astype(np.uint8)[None], 12, axis=0)
    image.paste(Image.fromarray(gradient).convert('RGB'), (32, 545))
    draw.text((32, 562), '-25 dB', font=font(14), fill=MUTED)
    draw.text((297, 562), '0 dB', font=font(14), fill=MUTED)
    draw.text((360, 544), 'VV display: fixed clipped scale for both SAR panels', font=font(16), fill=MUTED)
    draw.rectangle((864, 544, 880, 560), fill=FLOOD_COLOR)
    draw.text((889, 542), 'Reference flood (label 1)', font=font(16), fill=INK)
    draw.rectangle((864, 570, 880, 586), fill=OTHER_COLOR)
    draw.text((889, 568), 'Other / unknown (label 0)', font=font(16), fill=INK)
    notes = (
        'Event date is not an image acquisition date. Actual acquisition dates are unavailable in this export.',
        'Binary mask only: no-water, permanent water and no-data are collapsed into label 0; they cannot be distinguished here.',
        '192 x 192 center crops. High-pflood selected examples: illustrative evidence only, not representative accuracy results.',
        'No model predictions displayed. SAR brightness alone is not a flood classification.',
    )
    draw.line((32, 605, 1248, 605), fill=(219, 226, 233), width=1)
    for n, note in enumerate(notes):
        draw.text((32, 620 + n * 23), note, font=font(16), fill=MUTED)
    image.save(destination, format='PNG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source = repo / SOURCE_RELATIVE
    builder = repo / BUILDER_RELATIVE
    cache_builder = repo / CACHE_BUILDER_RELATIVE
    # Keep scientific assumptions tied to the source convention, fail on drift.
    source_text = builder.read_text()
    if '"label":(mk==3).astype(np.float32)' not in source_text or '(10*np.log10(np.clip(x,1e-6,None))+25)/25' not in source_text:
        raise ValueError('Source label or VV display convention changed; inspect before rendering')
    if 'remap={3:0,0:1,1:2,2:3}' not in cache_builder.read_text():
        raise ValueError('Cache label remapping changed; inspect before rendering')
    cases = []
    with np.load(source, allow_pickle=False) as npz:
        ids = tuple(str(x) for x in npz['ids'].tolist())
        if ids != EXPECTED_IDS:
            raise ValueError(f'Expected only the six existing case IDs, got {ids}')
        metadata = json.loads(str(npz['meta'].item()))
        for case_id in ids:
            pre = checked_array(npz, f'vv_pre__{case_id}')
            post = checked_array(npz, f'vv_post__{case_id}')
            label = checked_array(npz, f'label__{case_id}', binary=True)
            output = args.out / f'{case_id}.png'
            render_case(case_id, metadata[case_id], pre, post, label, output)
            cases.append({
                'case_id': case_id,
                'event_id': metadata[case_id]['actid'],
                'event_date': metadata[case_id]['flood_date'],
                'acquisition_dates': {'pre_2': None, 'post': None},
                'source_tile_pflood_percent': metadata[case_id]['pflood'],
                'reference_mask_flood_pixels_in_shown_crop': int(np.count_nonzero(label == 1)),
                'shown_crop_total_pixels': int(label.size),
                'preview_file': output.name,
                'preview_sha256': digest(output),
                'array_keys_used': [f'vv_pre__{case_id}', f'vv_post__{case_id}', f'label__{case_id}'],
                'geographic_coordinates_available': False,
                'case_status': 'illustrative_existing_test_example_not_an_unseen_evaluation',
            })
    selected_font = next((str(p) for p in FONT_PATHS if p.is_file()), 'Pillow default')
    index = {
        'schema': 'eo-evidence-case-previews-v0',
        'created_date': '2026-09-25',
        'source_npz': SOURCE_RELATIVE,
        'source_npz_sha256': digest(source),
        'source_builder': BUILDER_RELATIVE,
        'source_builder_sha256': digest(builder),
        'source_cache_builder': CACHE_BUILDER_RELATIVE,
        'source_cache_builder_sha256': digest(cache_builder),
        'renderer_sha256': digest(Path(__file__)),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__, 'font': selected_font},
        'case_count': len(cases),
        'case_ids': list(EXPECTED_IDS),
        'selection_bias': 'Existing builder selected test examples with pflood > 3 and finite teacher3 features, sorted descending by pflood, at zero-based ranks [0,3,8,15,25,40]. High-flood illustrative cases, not representative accuracy results.',
        'sar_display': {'band': 'Sentinel-1 VV', 'stored_units': 'normalized clipped intensity, 0..1', 'display_db_range': [-25, 0], 'source_formula': 'clip((10*log10(clip(linear_vv, 1e-6, None)) + 25)/25, 0, 1)', 'auto_contrast': False, 'upsampling': 'nearest, 2x'},
        'reference_mask': {'available_classes': {'1': 'flood', '0': 'other or unknown (no-water, permanent water and no-data collapsed)'}, 'source_formula': '(remapped_mask == 3)', 'original_remapped_classes': {'0': 'no-data', '1': 'no water', '2': 'permanent water', '3': 'flood'}, 'validity_mask_available': False, 'is_model_prediction': False},
        'date_warning': 'Only event dates are included. Actual SAR acquisition dates are unavailable; synthetic encoder timestamps are not used as acquisition dates.',
        'predictions_included': False,
        'cases': cases,
    }
    (args.out / 'case_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'case_count': len(cases), 'source_npz_sha256': index['source_npz_sha256']}, indent=2))


if __name__ == '__main__':
    main()
