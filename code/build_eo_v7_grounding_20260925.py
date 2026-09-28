#!/usr/bin/env python3
"""Build a new local v7 snapshot with two pinned source previews and E5 status only.

No model execution, live polling, raster processing, prediction editing, or overwrite.
The export's raster verification is attributed to its source manifest, not rerun here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

CASE_IDS = ('ks_06770', 'ks_05265')
PLAN_SHA = 'fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
PREPARED_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
SNAPSHOT_AT = '2026-09-25T06:14:07Z'
ASSET_ROLES = {0: 'pre_event_1', 1: 'pre_event_2', 2: 'post_event', 4: 'mask', 5: 'invalid_data'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def child(directory, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Expected relative export path')
    result = (directory / name).resolve()
    require(result.is_relative_to(directory.resolve()) and result.is_file(), 'Invalid or missing export file')
    return result


def pinned(path, expected, tracking):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Missing SHA256')
    require(sha(path) == expected, 'Source hash mismatch: ' + str(path))
    tracking[str(path)] = expected
    return path


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= 1e-8


def validate_geometry(geometry, point, centroid):
    require(set(geometry) == {'whole_tile', 'model_crop'}, 'Geometry scope differs')
    match = re.fullmatch(r'POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)', centroid)
    require(match is not None and all(close(float(match[i + 1]), point[i]) for i in (0, 1)), 'Centroid mismatch')
    bounds = {}
    for role, geom in geometry.items():
        require(geom.get('type') == 'Polygon' and len(geom['coordinates']) == 1, 'Expected simple export polygon')
        ring = geom['coordinates'][0]
        require(len(ring) == 5 and ring[0] == ring[-1], 'Unclosed export geometry')
        require(all(len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p)
                    and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in ring), 'Invalid lon/lat geometry')
        xs, ys = zip(*ring)
        bounds[role] = (min(xs), min(ys), max(xs), max(ys))
        w, s, e, n = bounds[role]
        require(w < point[0] < e and s < point[1] < n, 'Centroid outside export geometry')
    whole, crop = bounds['whole_tile'], bounds['model_crop']
    require(whole[0] < crop[0] < crop[2] < whole[2] and whole[1] < crop[1] < crop[3] < whole[3],
            'Model crop not inside source tile')


def validate_case(sid, entry, export, catalog_row, reader, quality, e3_items, c0, tracking):
    require(entry['manifest'] == f'{sid}/source_manifest.json' and entry['preview_file'] == f'{sid}/preview.png',
            'Case export names differ')
    path = pinned(child(export, entry['manifest']), entry['manifest_sha256'], tracking)
    source = read(path)
    require(source['id'] == sid and source['event_id'] == 562 and source['aoi_id'] == '13'
            and source['split'] == 'test' and source['event_date'] == '2022-01-29', 'Source case scope differs')
    require(catalog_row['dataset'] == 'kurosiwo' and catalog_row['aoi_id'] == str(source['event_id'])
            and catalog_row['region_id'] == source['aoi_id'] and catalog_row['split'] == source['split']
            and catalog_row['event_date'] == source['event_date'], 'Catalog case metadata mismatch')
    require(not catalog_row['evidence'], 'Existing case evidence must not be replaced')
    require(source['predictions_included'] is False and source['npys_equal_original_tiff_values'] is True
            and source['c1_mask_valid_hashes_verified'] is True, 'Missing source verification or prediction contamination')
    require(source['acquisition_dates'] == {'pre_1': None, 'pre_2': None, 'post': None}, 'Acquisition dates are unverified')
    require(source['model_crop'] == {'rows': [16, 208], 'columns': [16, 208], 'shape': [192, 192]}, 'Crop mismatch')
    validate_geometry(source['geometry'], catalog_row['point'], source['centroid_wkt_from_source'])
    for field, target in [('pflood', 'flood_fraction_pct'), ('pwater', 'permanent_water_fraction_pct')]:
        require(close(source['source_row'][field], catalog_row[target]), 'Catalog source fraction mismatch')
    require(source['qa_items'] == [x for x in e3_items if x['tile'] == sid], 'Frozen E3 item metadata mismatch')
    require(reader['question_id'] == quality['id'] and reader['reference_label'] == quality['source_answer']
            and reader['dates'] == quality['dates'] and reader['slots'] == quality['slots'], 'Historical reader metadata mismatch')
    eligible = quality['eligible_symmetric_quality']
    require(source['eligible_C1_symmetric_quality'] is eligible and entry['eligible_C1_symmetric_quality'] is eligible
            and reader['c1_membership']['quality_eligible'] is eligible, 'C1 quality membership mismatch')
    require(eligible is (sid == 'ks_05265') and reader['c1_membership']['supported_stratum'] is eligible,
            'Fixed case C1 support differs')
    stats = source['reference_summary']
    for k in ('total_px', 'flood_px', 'labelled_px'):
        require(type(stats[k]) is int and stats[k] == quality[k], 'Reference pixel support mismatch: ' + k)
    total, known, flood = stats['total_px'], stats['labelled_px'], stats['flood_px']
    require(total == 36864 and 0 <= flood <= known <= total and stats['unknown_px'] == total - known,
            'Unknown support mismatch')
    for key, expected in [('labelled_fraction', known / total), ('flood_fraction_all_crop', flood / total),
                          ('flood_fraction_labelled', flood / known)]:
        require(close(stats[key], expected), 'Reference crop fraction mismatch: ' + key)
    counts = quality['class_valid_counts']
    expected_raw = {str(raw): sum(counts[f'mask_{remap}|valid_{v}'] for v in (0, 1))
                    for raw, remap in enumerate((1, 2, 3, 0))}
    require(stats['raw_class_counts'] == expected_raw and stats['source_valid_px'] == sum(
            counts[f'mask_{k}|valid_1'] for k in range(4)), 'Raw class / validity counts mismatch')
    cache_key = str(Path(c0['root']) / f'kurosiwo_s1_cache/single_fp16/{sid}.npy')
    require(source['model_embedding_sha256'] == c0['cache_sha256'][cache_key], 'Embedding provenance mismatch')
    assets = source['assets']
    require(len(assets) == 5 and {x['index']: x['role'] for x in assets} == ASSET_ROLES, 'Asset coverage mismatch')
    for asset in assets:
        require(asset['file'] == f"asset_{asset['index']}.tif", 'Asset file differs')
        require(asset['sha256'] == asset['original_embedded_bytes']['sha256'], 'Original slice hash mismatch')
        pinned(child(path.parent, asset['file']), asset['sha256'], tracking)
    pinned(child(path.parent, 'evidence_arrays.npz'), source['arrays_sha256'], tracking)
    require(source['preview_file'] == 'preview.png', 'Preview filename differs')
    preview = pinned(child(export, entry['preview_file']), source['preview_sha256'], tracking)
    require(preview.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'Expected PNG preview')
    unknown = 100 * (total - known) / total
    note = (f'원본 SAR VV/VH의 pre_1·pre_2·post 슬롯과 post 참조 마스크입니다. 실제 취득일은 미검증입니다. '
            f'192×192 crop의 source valid=1 {100 * stats["source_valid_px"] / total:.2f}%, '
            f'라벨 확인 가능 {100 * known / total:.2f}%, 라벨 확인 불가 {unknown:.2f}%. '
            + ('C1 주 분석에서 제외된 사례입니다. ' if not eligible else 'C1 주 분석에 포함된 비침수 참조 사례입니다. ')
            + '모델의 위치 판정·피해 탐지 결과가 아니며, 미리 고른 두 사례는 전체 자료를 대표하지 않습니다.')
    evidence = {'image_url': f'/previews/{sid}_source_grounding_v7.png',
                'kind': 'original_sar_slots_and_post_reference_mask', 'caption': note,
                'acquisition_dates': source['acquisition_dates'], 'source_role': 'original_observations_and_reference_labels',
                'source_manifest_sha256': sha(path), 'source': source}
    return evidence, preview, path


def pending_run(manifest, audit):
    require((manifest['n_items'], manifest['n_train'], manifest['n_test']) == (5989, 4234, 1755), 'E5 population mismatch')
    require(audit['eval_set_counts']['primary_same_prompt'] == 902 and audit['n_unique_caches'] == 3756,
            'E5 preparation support mismatch')
    require(audit['pool_reproduction']['bit_exact'] is True and audit['pool_reproduction']['matched_items'] == 209,
            'E5 pooled reproduction incomplete')
    return {'id': 'E5-EB', 'title': '같은 학습 예산에서 차이 입력 비교',
            'status': '학습 진행 스냅샷 · 결과 없음',
            'findings': [f'상태 확인 시각 {SNAPSHOT_AT}: seed 1 / full, 275 / 1,590 updates. 실시간 상태가 아닙니다.',
                         '고정 primary 902문항·8사건. 4개 입력 형식 × 3 seed = 12개 모델, 예정 평가 응답 26,325개.',
                         '현재 입력 3,756개 캐시를 검증하고, E4의 209개 pooled pair와 비트 단위 일치를 확인한 준비 기록입니다.'],
            'limitations': ['이 화면에 E5 모델 답변·성능·판정 결과는 없습니다. 진행량을 성공으로 세지 않습니다.',
                            '이미 노출된 개발 자료의 후속 진단이며 실제 취득 시각·물리 변화 정답의 한계가 남습니다.',
                            '준비 감사와 고정 시각의 진행 전달 기록만 표시합니다. 현재 서버 상태를 조회하지 않습니다.'],
            'next_step': '12개 모델의 고정 학습·평가를 마친 뒤 완결성과 출처를 독립 검증합니다.'}


def build(repo, base, case_export, prepared_review, out):
    repo, base, export, prepared, out = [Path(x).resolve() for x in (repo, base, case_export, prepared_review, out)]
    require(not out.exists() and not out.is_relative_to(base) and not base.is_relative_to(out), 'New exclusive output required')
    require(not (export / 'failure.json').exists(), 'Failed or partial source export')
    tracking = {}
    bundle = repo / 'code/e5_bundle_v0'
    plan = read(pinned(bundle / 'e5_equal_budget_prereg_v0.json', PLAN_SHA, tracking))
    parents = {}
    for name, digest in plan['parents'].items():
        path = pinned(child(bundle / 'frozen_inputs', name), digest, tracking)
        parents[name] = rows(path) if name.endswith('.jsonl') else read(path)
    manifest = read(pinned(prepared / 'manifest.json', PREPARED_SHA, tracking))
    require(manifest['files_sha256']['prereg.json'] == PLAN_SHA and manifest['parent_snapshot_sha256'] == plan['parents'],
            'E5 frozen plan/parents mismatch')
    audit = read(pinned(prepared / 'input_audit.json', manifest['files_sha256']['input_audit.json'], tracking))
    run = pending_run(manifest, audit)
    index_path = export / 'case_index.json'
    pinned(index_path, sha(index_path), tracking)
    index = read(index_path)
    require(index['schema'] == 'eo-two-fixed-source-cases-v0' and index['status'] == 'exported'
            and index['case_ids'] == list(CASE_IDS), 'Export status or fixed case IDs differ')
    require(index['predictions_included'] is False and index['no_new_model_inference'] is True, 'Export contains predictions')
    expected_pins = {'e3_items': plan['parents']['e3_items.jsonl'], 'quality': plan['parents']['c1_quality.jsonl'],
                     'c0_manifest': plan['parents']['c0_manifest.json']}
    require(index['metadata_pins'] == expected_pins and len(index['cases']) == 2
            and [x['id'] for x in index['cases']] == list(CASE_IDS), 'Export metadata/case coverage mismatch')
    base_hashes = {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}
    catalog, meta, research = [read(base / name) for name in ('catalog.json', 'meta.json', 'research_runs.json')]
    reader_cases = read(base / 'reader_cases.json')
    require(catalog['schema'] == 'eo-evidence-catalog-v0.2' and research['schema_version'] == 'eo_research_snapshot_v1', 'Base schema differs')
    require(all(not x['id'].startswith('E5') for x in research['runs']), 'E5 already present')
    quality = {x['tile']: x for x in parents['c1_quality.jsonl'] if x['tile'] in CASE_IDS}
    pending = []
    for sid, entry in zip(CASE_IDS, index['cases']):
        matching = [x for x in catalog['records'] if x['id'] == sid]
        require(len(matching) == 1, 'Catalog case missing or duplicated')
        evidence, png, source = validate_case(sid, entry, export, matching[0], reader_cases['cases'][sid], quality[sid],
                                               parents['e3_items.jsonl'], parents['c0_manifest.json'], tracking)
        destination = evidence['image_url'].lstrip('/')
        require(not (base / destination).exists(), 'Preview asset name already exists')
        matching[0]['evidence'].append(evidence)
        pending.append((destination, png, source))
    old_html = (base / 'index.html').read_text()
    needle = "(e.kind==='georeferenced_flood_landcover_overlap'?"
    require(old_html.count(needle) == 1, 'Unexpected evidence caption UI')
    # The inserted conditional shares the existing ternary closing parenthesis.
    html = old_html.replace(needle, "(typeof e.caption==='string'?esc(e.caption):e.kind==='georeferenced_flood_landcover_overlap'?")
    now = datetime.now(timezone.utc).isoformat()
    meta['preview_count'] += 2
    research['checked_at'] = now
    research['runs'].append(run)
    shutil.copytree(base, out)
    for destination, png, source in pending:
        shutil.copyfile(png, out / destination)
        folder = out / 'source_grounding_v7' / source.parent.name
        folder.mkdir(parents=True)
        shutil.copyfile(source, folder / 'source_manifest.json')
    shutil.copyfile(index_path, out / 'source_grounding_v7/case_index.json')
    for name, data in [('catalog.json', catalog), ('meta.json', meta), ('research_runs.json', research)]:
        write(out / name, data)
    research_sources = read(base / 'research_sources.json')
    require('source_grounding_v7' not in research_sources, 'Grounding sources already present')
    research_sources['source_grounding_v7'] = {'checked_at': now, 'e5_status_as_of': SNAPSHOT_AT,
                                             'sources': dict(tracking), 'model_outcomes_included': False}
    write(out / 'research_sources.json', research_sources)
    (out / 'index.html').write_text(html)
    for name in ('manifest.json', 'input_audit.json'):
        folder = out / 'source_grounding_v7/e5_prepared_review';folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prepared / name, folder / name)
    changed = {'catalog.json', 'meta.json', 'research_runs.json', 'research_sources.json', 'index.html'}
    for relative, digest in base_hashes.items():
        require(sha(base / relative) == digest, 'Base changed during build')
        if relative not in changed:
            require(sha(out / relative) == digest, 'Inherited file changed: ' + relative)
    for path, digest in tracking.items():
        require(sha(path) == digest, 'Input changed during build')
    output_hashes = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    report = {'schema': 'eo-v7-source-grounding-build-v0', 'created_at': now, 'base': str(base), 'out': str(out),
              'case_ids': list(CASE_IDS), 'added_evidence_images': 2, 'e5_status_as_of': SNAPSHOT_AT,
              'e5_outcomes_included': False, 'reader_cases_byte_identical': True,
              'source_sha256': tracking, 'base_files_sha256': base_hashes, 'output_files_sha256': output_hashes,
              'builder_sha256': sha(__file__),
              'limits': ['Source TIFF interpretation/CRS transforms were checked by exporter; not recomputed by builder.',
                         'Geometry stored as source provenance; query remains centroid based.',
                         'Inherited v6 check reports are historical v6 checks, not new v7 HTTP/browser verification.',
                         'E5 progress is a parent-supplied fixed-time snapshot; preparation hashes do not prove live progress.']}
    write(out / 'v7_build_manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('repo', 'base', 'case-export', 'prepared-review', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo, args.base, args.case_export, args.prepared_review, args.out)
    print(json.dumps({'out': result['out'], 'added_evidence_images': 2,
                      'reader_cases_byte_identical': result['reader_cases_byte_identical']}))
