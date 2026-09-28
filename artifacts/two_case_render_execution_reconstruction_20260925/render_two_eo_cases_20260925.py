#!/usr/bin/env python3
"""Render a verified two-case export into a NEW copy; keep original immutable.

Requires NumPy/Pillow plus adjacent export_two_eo_cases_20260925.py. No rasterio,
tacoreader, models, network access, or source export changes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import PIL

import export_two_eo_cases_20260925 as source_code


def read(path):
    return json.loads(Path(path).read_text())


def bounded_file(root, name):
    path = Path(name)
    source_code.require(not path.is_absolute() and '..' not in path.parts, 'Unsafe export file path')
    target = root / path
    source_code.require(not target.is_symlink() and target.resolve().is_relative_to(root.resolve())
                        and target.is_file(), 'Missing/symlink/outside export file')
    return target


def validate_export(root):
    source_code.require(root.is_dir() and not root.is_symlink(), 'Source export must be a real directory')
    source_code.require(not list(root.rglob('failure.json')), 'Source export contains a failure record')
    source_code.require(not any(p.is_symlink() for p in root.rglob('*')), 'Symlinks are not allowed in export')
    index_path = bounded_file(root, 'case_index.json')
    index = read(index_path)
    source_code.require(index.get('schema') == 'eo-two-fixed-source-cases-v0' and index.get('status') == 'exported', 'Source is not a complete raw export')
    source_code.require(index['case_ids'] == list(source_code.CASE_IDS)
                        and [r['id'] for r in index['cases']] == list(source_code.CASE_IDS), 'Fixed cases/order differ')
    source_code.require(index['predictions_included'] is False and index['no_new_model_inference'] is True, 'Unexpected prediction provenance')
    source_code.require(index['metadata_pins'] == source_code.PINS, 'Frozen source metadata pins differ')
    selection = read(bounded_file(root, 'selection.json'))
    source_code.require(selection['ids'] == list(source_code.CASE_IDS)
                        and selection['frozen_metadata_sha256'] == source_code.PINS, 'Selection identity differs')
    expected_files = {'case_index.json', 'selection.json'}
    manifests = {}
    for case in index['cases']:
        sid = case['id']
        source_code.require(case['manifest'] == f'{sid}/source_manifest.json', 'Unexpected case manifest path')
        manifest_path = bounded_file(root, case['manifest'])
        source_code.require(source_code.sha(manifest_path) == case['manifest_sha256'], 'Case manifest hash mismatch')
        manifest = read(manifest_path)
        source_code.require(manifest['id'] == sid and manifest['split'] == 'test' and manifest['event_id'] == 562
                            and manifest['aoi_id'] == '13' and manifest['event_date'] == '2022-01-29', 'Case scope differs')
        source_code.require(manifest['predictions_included'] is False, 'Predictions unexpectedly included')
        source_code.require(manifest['eligible_C1_symmetric_quality'] is (sid == 'ks_05265'), 'Quality scope differs')
        source_code.require(set(manifest['acquisition_dates']) == {'pre_1', 'pre_2', 'post'}
                            and all(v is None for v in manifest['acquisition_dates'].values()), 'Unverified acquisition dates promoted')
        source_code.require(case.get('preview_file') is None and 'preview_file' not in manifest, 'Expected a raw export without previous rendering')
        source_code.require([a['index'] for a in manifest['assets']] == list(source_code.ASSETS), 'Original asset coverage differs')
        expected_files.add(case['manifest'])
        for asset in manifest['assets']:
            source_code.require(asset['file'] == f"asset_{asset['index']}.tif"
                                and asset['role'] == source_code.ASSETS[asset['index']], 'Asset role/file differs')
            rel = f"{sid}/{asset['file']}"
            path = bounded_file(root, rel)
            digest = source_code.sha(path)
            source_code.require(digest == asset['sha256'] == asset['original_embedded_bytes']['sha256'], 'Original TIFF hash mismatch')
            expected_files.add(rel)
        arrays_rel = f'{sid}/evidence_arrays.npz'
        arrays_path = bounded_file(root, arrays_rel)
        source_code.require(source_code.sha(arrays_path) == manifest['arrays_sha256'], 'Evidence arrays hash mismatch')
        expected_files.add(arrays_rel)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            source_code.require(set(arrays.files) == {'raw_linear_vv_vh', 'reference_mask_raw', 'source_valid', 'labelled_valid'}, 'Unexpected evidence array keys')
            cube, mask, valid = arrays['raw_linear_vv_vh'], arrays['reference_mask_raw'], arrays['source_valid']
            source_code.require(cube.shape == (2, 3, 192, 192) and cube.dtype == np.float32, 'Unexpected SAR cube contract')
            summary = source_code.mask_summary(mask, valid)
            source_code.require(summary == manifest['reference_summary'], 'Reference summary differs from source arrays')
            source_code.require(np.array_equal(arrays['labelled_valid'], ((valid == 1) & (mask != 3)).astype('uint8')), 'Known-support array mismatch')
        manifests[sid] = manifest
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    source_code.require(actual_files == expected_files, 'Unexpected or missing source export files')
    source_code.require(sum((root / name).stat().st_size for name in actual_files) < 50 * 1024 * 1024, 'Unexpectedly large two-case export')
    pins = {name: source_code.sha(root / name) for name in sorted(actual_files)}
    return index, manifests, pins


def render_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_code.require(not output.exists(), 'Rendered destination already exists; no overwrite')
    source_code.require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Source and destination trees overlap')
    index, manifests, before = validate_export(source)
    shutil.copytree(source, output)
    try:
        source_code.require(all(source_code.sha(output / rel) == h for rel, h in before.items()), 'Copied export bytes differ')
        provenance = {'source_directory': str(source), 'source_case_index_sha256': before['case_index.json'],
                      'source_manifest_sha256': {sid: before[f'{sid}/source_manifest.json'] for sid in source_code.CASE_IDS},
                      'source_file_sha256': before, 'helper_sha256': source_code.sha(__file__),
                      'renderer_module_sha256': source_code.sha(source_code.__file__),
                      'created_at': datetime.now(timezone.utc).isoformat(),
                      'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__}}
        for case in index['cases']:
            sid = case['id']
            manifest = manifests[sid]
            with np.load(output / sid / 'evidence_arrays.npz', allow_pickle=False) as arrays:
                layout = source_code.render(output / sid / 'preview.png', arrays['raw_linear_vv_vh'],
                    arrays['reference_mask_raw'], arrays['source_valid'], manifest, manifest['reference_summary'])
            manifest.update(preview_file='preview.png', preview_sha256=source_code.sha(output / sid / 'preview.png'),
                            preview_layout=layout, rendering_provenance=provenance)
            source_code.write(output / case['manifest'], manifest)
            case.update(manifest_sha256=source_code.sha(output / case['manifest']),
                        preview_file=f'{sid}/preview.png', preview_sha256=manifest['preview_sha256'],
                        parent_source_manifest_sha256=before[case['manifest']])
        index['rendering_provenance'] = provenance
        index['rendered_copy'] = True
        source_code.write(output / 'case_index.json', index)
        source_code.require({rel: source_code.sha(source / rel) for rel in before} == before,
                            'Original export changed during rendering')
        source_code.require({str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()} == set(before),
                            'Original export file membership changed')
        return index
    except BaseException as exc:
        source_code.write(output / 'failure.json', {'phase': 'render_local_copy', 'error': str(exc),
                           'original_export_modified_by_helper': False, 'partial_rendered_copy_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = render_copy(args.source, args.out)
    print(json.dumps({'out': str(args.out), 'case_ids': result['case_ids'], 'source_unchanged': True}))
