#!/usr/bin/env python3
"""Render a deterministic, train-only observation-quality review pack on CPU.

This is a named-target quality review, not blind crop identification, an expert
consistency pilot, training, or evaluation. Query gold is never opened. No data
selection or response from this pack is applied to any existing experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = {8: 'grapevine', 14: 'leguminous_fodder'}
PARENTS = ('t31tfj', 't32ulu')
QUERY_POSITIONS = (2, 5)
NAMESPACE = 'oe10-train-quality-review-v0:'
FORBIDDEN_INPUT_KEYS = {'semantic', 'instances', 'target_mask', 'label_valid',
                        'crop_label_valid', 'labels', 'target', 'query_label'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def safe(root, ref, category):
    root = Path(root).resolve()
    require(not Path(ref).is_absolute(), 'Only relative data references are permitted')
    path = (root / ref).resolve()
    require(path.is_relative_to(root / category) and path.is_file(),
            f'Missing reference or reference outside {category}: {ref}')
    return path



def verify_catalog_lineage(prepared, catalog, support_root):
    """Check the frozen export and original metadata before opening raw inputs."""
    export_path = catalog / 'export_manifest.json'
    receipt_path = catalog / 'preparation_receipt.json'
    exported = json.loads(export_path.read_text())
    entries = exported['files']
    require(isinstance(entries, list) and bool(entries), 'Empty catalog export manifest')
    seen = set()
    for entry in entries:
        reference = entry['path']
        require(isinstance(reference, str) and not Path(reference).is_absolute(), 'Invalid export path')
        require(reference not in seen, 'Duplicate catalog export path')
        seen.add(reference)
        path = (catalog / reference).resolve()
        require(path.is_relative_to(catalog) and path.is_file(), 'Missing or escaping export file')
        require(path.stat().st_size == entry['bytes'], 'Catalog export file size mismatch: ' + reference)
        require(sha(path) == entry['sha256'], 'Catalog export file hash mismatch: ' + reference)
    require({'episodes_train.jsonl', 'training_target_plan.jsonl', 'preparation_receipt.json'} <= seen,
            'Required frozen catalog files are not covered by export manifest')
    receipt = json.loads(receipt_path.read_text())
    pinned = receipt['pinned_sources']
    manifest_path = prepared / 'manifest.jsonl'
    objects_path = support_root / 'source_objects.jsonl'
    require(sha(manifest_path) == pinned['manifest']['sha256'], 'Prepared manifest differs from pinned catalog source')
    require(sha(objects_path) == pinned['objects']['sha256'], 'Source objects differ from pinned catalog source')
    require(sha(catalog / 'episodes_train.jsonl') == receipt['public_sha256'], 'Public catalog differs from receipt')
    return {'catalog_files_hash_and_size_verified': len(entries),
            'catalog_export_sha256': sha(export_path),
            'preparation_receipt_sha256': sha(receipt_path),
            'prepared_manifest_sha256': pinned['manifest']['sha256'],
            'source_objects_sha256': pinned['objects']['sha256']}


def unique_index(rows, key):
    result = {str(row[key]): row for row in rows}
    require(len(result) == len(rows), f'Duplicate {key}')
    return result


def check_role(row):
    require(row['training_partition'] == 'train_pool', 'Source-bank/development packet forbidden')
    require(row['parent_tile'] in PARENTS, 'Unopened parent forbidden')
    require(row['supervised_training_allowed'] is True, 'Training permission is absent')


def select_cases(episodes, plans, manifest):
    """Select by fixed hash and class/parent only; never consult query gold."""
    by_plan = unique_index(plans, 'episode_id')
    by_episode = unique_index(episodes, 'episode_id')
    require(set(by_episode) == set(by_plan), 'Episode/target-plan ID mismatch')
    buckets = {(c, p): [] for c in CLASSES for p in PARENTS}
    for episode in episodes:
        require(episode['split'] == 'train', 'Non-training episode forbidden')
        pid = str(episode['query_patch_id'])
        require(pid in manifest, 'Query is absent from prepared manifest')
        row = manifest[pid]
        check_role(row)
        require(episode['query_parent_tile'] == row['parent_tile'], 'Query parent mismatch')
        require(episode['query_label_supplied'] is False, 'Query gold was supplied to public episode')
        require(not {'query_label_npz', 'target_class', 'counter_class', 'target_pixels'} & set(episode),
                'Scoring fields leaked into public episode')
        plan = by_plan[episode['episode_id']]
        target, counter = int(plan['target_class']), int(plan['counter_class'])
        require({target, counter} == set(CLASSES), 'Unexpected target/counter classes')
        require(str(plan['query_patch_id']) == pid, 'Plan query mismatch')
        require(int(plan['k_pairs']) == int(episode['k_pairs']), 'Plan K mismatch')
        if int(episode['k_pairs']) == 1:
            rank = hashlib.sha256((NAMESPACE + episode['episode_id']).encode()).hexdigest()
            buckets[target, row['parent_tile']].append((rank, episode, plan))
    used_queries, chosen = set(), []
    for (target, parent), candidates in buckets.items():
        picked = 0
        for rank, episode, plan in sorted(candidates, key=lambda item: item[0]):
            pid = str(episode['query_patch_id'])
            if pid in used_queries:
                continue
            chosen.append((episode, plan, rank))
            used_queries.add(pid)
            picked += 1
            if picked == 5:
                break
        require(picked == 5, f'Cannot select five distinct queries for {target}/{parent}')
    require(len(chosen) == len(used_queries) == 20, 'Expected 20 globally unique training queries')
    return chosen


class PacketCache:
    def __init__(self, root, manifest):
        self.root, self.manifest, self.cache = root, manifest, {}
        self.verified = {}

    def get(self, pid, reference, expected_hash, dates):
        pid = str(pid)
        require(pid in self.manifest, 'Unknown input patch')
        row = self.manifest[pid]
        check_role(row)
        require(reference == row['npz_path'], 'Input path disagrees with manifest')
        require(expected_hash == row['npz_sha256'], 'Input hash disagrees with manifest')
        require([int(d) for d in dates] == [int(d) for d in row['selected_dates']], 'Input date mismatch')
        if pid in self.cache:
            return self.cache[pid]
        path = safe(self.root, reference, 'inputs')
        require(sha(path) == expected_hash, 'Input file hash mismatch')
        with np.load(path, allow_pickle=False) as packet:
            require(not FORBIDDEN_INPUT_KEYS & set(packet.files), 'Labels found in input NPZ')
            raw = packet['raw_selected_s2'].copy()
            valid = packet['observation_valid'].copy()
            timestamps = packet['timestamps']
        require(raw.shape == (8, 10, 128, 128) and raw.dtype == np.int16, 'Unexpected raw shape/dtype')
        require(valid.shape == (8, 128, 128) and valid.dtype == np.bool_, 'Unexpected validity shape/dtype')
        require(np.array_equal(valid, ~(raw == -10000).any(axis=1)), 'Availability mask mismatch')
        parsed = [datetime.strptime(str(int(d)), '%Y%m%d') for d in dates]
        require(len(parsed) == 8 and parsed == sorted(set(parsed)), 'Dates are not eight ordered unique values')
        expected = np.asarray([[d.day, d.month - 1, d.year] for d in parsed], dtype=np.int64)
        require(timestamps.dtype == np.int64 and np.array_equal(timestamps, expected), 'Timestamp mismatch')
        self.cache[pid] = (raw, valid)
        self.verified[pid] = {'input_npz': reference, 'sha256': expected_hash,
                              'dates_yyyymmdd': [int(d) for d in dates], 'role': 'train_pool'}
        return raw, valid


def support_mask(root, support, expected_class, source_objects, mask_cache):
    key = support['object_key']
    require(key in source_objects, 'Unknown support object')
    obj = source_objects[key]
    require(obj['episode_role'] == 'train_pool', 'Source-bank object forbidden')
    require(int(obj['class_id']) == expected_class, 'Support class mismatch')
    require(str(obj['patch_id']) == str(support['patch_id']), 'Support patch mismatch')
    require(obj['parent_tile'] == support['parent_tile'] and obj['parent_tile'] in PARENTS,
            'Support parent mismatch')
    require(obj['mask_npz'] == support['mask_npz'] and obj['mask_sha256'] == support['mask_sha256'],
            'Mask source reference mismatch')
    if key not in mask_cache:
        path = safe(root, support['mask_npz'], 'support_masks')
        require(sha(path) == support['mask_sha256'], 'Support mask hash mismatch')
        with np.load(path, allow_pickle=False) as packet:
            require(set(packet.files) == {'mask'}, 'Unexpected fields in support mask NPZ')
            mask = packet['mask'].copy()
        require(mask.shape == (128, 128) and mask.dtype == np.bool_ and bool(mask.any()), 'Invalid support mask')
        require(int(mask.sum()) == int(obj['class_pixel_count']), 'Mask pixel count disagrees with source inventory')
        mask_cache[key] = mask
    return mask_cache[key]


def rgb_panel(raw, valid, mask=None):
    rgb = np.rint(np.clip(raw[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32) / 3000, 0, 1) * 255).astype(np.uint8)
    rgb[~valid] = [180, 0, 180]
    if mask is not None:
        inside = mask.copy()
        inside[1:, :] &= mask[:-1, :]
        inside[:-1, :] &= mask[1:, :]
        inside[:, 1:] &= mask[:, :-1]
        inside[:, :-1] &= mask[:, 1:]
        inside[[0, -1], :] = False
        inside[:, [0, -1]] = False
        rgb[mask & ~inside] = [255, 220, 0]
    return Image.fromarray(rgb).resize((160, 160), Image.Resampling.NEAREST)


def render_case(path, case, query, positive, counter):
    cell_w, row_h, top = 164, 208, 48
    canvas = Image.new('RGB', (9 * cell_w, top + 3 * row_h + 26), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{case['case_id']} | target: {case['target_name']} | counter: {case['counter_name']}", fill='black')
    draw.text((6, 23), 'Observation-quality review; annual source labels are shown. This is not blind crop accuracy.', fill='black')
    rows = [('QUERY - no reference mask', query, None, case['query'], QUERY_POSITIONS),
            ('POSITIVE - yellow object boundary', positive[:2], positive[2], case['positive'], tuple(range(8))),
            ('COUNTEREXAMPLE - yellow object boundary', counter[:2], counter[2], case['counterexample'], tuple(range(8)))]
    for ri, (label, packet, mask, info, positions) in enumerate(rows):
        raw, valid = packet
        y = top + ri * row_h
        draw.text((6, y), f"{label} | patch {info['patch_id']} | {info['parent_tile']} | train_pool", fill='black')
        for col, pos in enumerate(positions):
            x = col * cell_w
            draw.text((x + 3, y + 17), str(info['dates_yyyymmdd'][pos]), fill='black')
            canvas.paste(rgb_panel(raw[pos], valid[pos], mask), (x + 2, y + 33))
        if mask is None:
            draw.text((2 * cell_w + 6, y + 58), 'Only initially acquired query dates [2,5] are displayed.', fill='black')
            draw.text((2 * cell_w + 6, y + 78), 'Other query observations and query gold are not displayed.', fill='black')
        else:
            ref = np.zeros((128, 128, 3), dtype=np.uint8)
            ref[mask] = [255, 220, 0]
            canvas.paste(Image.fromarray(ref).resize((160, 160), Image.Resampling.NEAREST), (8 * cell_w + 2, y + 33))
            draw.text((8 * cell_w + 3, y + 17), 'SOURCE MASK', fill='black')
    draw.text((6, top + 3 * row_h + 4), 'Fixed RGB display: B04/B03/B02, raw/3000 clipped. Magenta = missing input. Valid input does not mean cloud-free.', fill='black')
    canvas.save(path)


def observation_template(info, positions):
    return [{'observation_id': info['observation_ids'][i], 'date_yyyymmdd': info['dates_yyyymmdd'][i],
             'visibility': 'unreviewed', 'display_saturation': 'unreviewed',
             'usable_for_visual_comparison': 'unreviewed', 'notes': ''} for i in positions]


def run(args):
    prepared, catalog = args.prepared.resolve(), args.catalog.resolve()
    support_root = args.support_root
    if support_root is None:
        for filename in ('preparation_receipt.json', 'catalog_receipt.json', 'receipt.json'):
            path = catalog / filename
            if path.is_file():
                value = json.loads(path.read_text()).get('support_mask_reference_root')
                if value:
                    support_root = Path(value)
                    break
    require(support_root is not None, 'Provide --support-root containing original source_objects.jsonl and support_masks')
    support_root = support_root.resolve()
    lineage = verify_catalog_lineage(prepared, catalog, support_root)
    source_paths = [prepared / 'manifest.jsonl', catalog / 'episodes_train.jsonl',
                    catalog / 'training_target_plan.jsonl', support_root / 'source_objects.jsonl', Path(__file__).resolve(),
                    catalog / 'export_manifest.json', catalog / 'preparation_receipt.json']
    source_hashes = {str(path): sha(path) for path in source_paths}
    manifest = unique_index(read_rows(source_paths[0]), 'patch_id')
    episodes, plans = read_rows(source_paths[1]), read_rows(source_paths[2])
    objects = unique_index(read_rows(source_paths[3]), 'object_key')
    selected = select_cases(episodes, plans, manifest)
    require(not args.out.exists(), 'Output exists; never overwrite a prior review package')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'cases').mkdir()
    cache, masks, cases, response_cases = PacketCache(prepared, manifest), {}, [], []
    for number, (episode, plan, rank) in enumerate(selected, 1):
        pid = str(episode['query_patch_id'])
        query_dates = [int(o['date_yyyymmdd']) for o in episode['query_observations']]
        require([o['observation_id'] for o in episode['query_observations']] == [f'{pid}:obs:{i}' for i in range(8)], 'Query observation IDs mismatch')
        require(episode['initial_observation_ids'] == [f'{pid}:obs:{i}' for i in QUERY_POSITIONS], 'Initial query dates changed')
        require(len(episode['support_pairs']) == 1, 'Selected episode is not K1')
        query_info = {'patch_id': pid, 'parent_tile': episode['query_parent_tile'], 'role': 'train_pool',
                      'input_npz': episode['query_input_npz'], 'input_sha256': episode['query_input_sha256'],
                      'dates_yyyymmdd': query_dates, 'observation_ids': [f'{pid}:obs:{i}' for i in range(8)],
                      'displayed_positions': list(QUERY_POSITIONS)}
        query = cache.get(pid, query_info['input_npz'], query_info['input_sha256'], query_dates)
        supporters = []
        infos = []
        target, other = int(plan['target_class']), int(plan['counter_class'])
        for role, cls in [('positive', target), ('counterexample', other)]:
            support = episode['support_pairs'][0][role]
            spid = str(support['patch_id'])
            require(spid != pid, 'Support/query patch overlap')
            require(support['observation_ids'] == [f'{spid}:obs:{i}' for i in range(8)], 'Support observation IDs mismatch')
            require(manifest[spid]['parent_tile'] == support['parent_tile'], 'Support manifest parent mismatch')
            raw, valid = cache.get(spid, support['input_npz'], support['input_sha256'], support['dates_yyyymmdd'])
            mask = support_mask(support_root, support, cls, objects, masks)
            supporters.append((raw, valid, mask))
            infos.append(dict(support, role='train_pool', displayed_positions=list(range(8)),
                              touches_image_border=objects[support['object_key']]['touches_image_border']))
        require(infos[0]['object_key'] != infos[1]['object_key'], 'Repeated support object')
        case_id = f'quality_{number:02d}'
        case = {'case_id': case_id, 'episode_id': episode['episode_id'], 'selection_hash': rank,
                'target_class': target, 'target_name': CLASSES[target], 'counter_class': other,
                'counter_name': CLASSES[other], 'query': query_info, 'positive': infos[0],
                'counterexample': infos[1], 'image': f'cases/{case_id}.png',
                'query_target_presence_not_inspected': True, 'human_response_received': False}
        render_case(args.out / case['image'], case, query, supporters[0], supporters[1])
        cases.append(case)
        response_cases.append({'case_id': case_id, 'started_utc': None, 'elapsed_seconds': None,
                               'query': observation_template(query_info, QUERY_POSITIONS),
                               'positive': observation_template(infos[0], range(8)),
                               'counterexample': observation_template(infos[1], range(8)),
                               'positive_mask_alignment': 'unreviewed', 'counterexample_mask_alignment': 'unreviewed',
                               'possible_boundary_or_duplicate_parcel_concern': 'unreviewed', 'notes': ''})
    require(all(sha(Path(path)) == digest for path, digest in source_hashes.items()), 'Source changed while rendering')
    group_counts = Counter((c['target_class'], c['query']['parent_tile']) for c in cases)
    require(set(group_counts.values()) == {5} and len(group_counts) == 4, 'Stratum imbalance')
    queue = {'schema': 'oe10_observation_quality_queue_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'task': 'named_target_observation_quality_review_not_crop_accuracy',
             'selection': 'K1; target8/14 x training parent; five per stratum; fixed SHA order; globally unique query patches; no query gold',
             'selection_namespace': NAMESPACE, 'class_names_disclosed': True,
             'original_20_expert_consistency_pilot': False, 'model_input': False,
             'actual_human_responses': 0, 'query_gold_opened': False,
             'development_packets_opened': 0, 'source_bank_packets_opened': 0,
             'automatic_quality_filtering_applied': False, 'crop_description_cards_attached': False,
             'existing_evaluation_samples_or_dates_changed': False,
             'source_hashes': source_hashes, 'catalog_lineage': lineage,
             'prepared_input_reference_root': str(prepared),
             'support_mask_reference_root': str(support_root), 'cases': cases}
    write_json(args.out / 'queue20.json', queue)
    write_json(args.out / 'response_template.json', {
        'schema': 'oe10_observation_quality_response_v1', 'reviewer_id': None,
        'task': queue['task'], 'completed': False, 'actual_human_responses': 0,
        'allowed_visibility': ['unreviewed', 'clear', 'possible_cloud_or_haze', 'obscured', 'cannot_tell'],
        'allowed_display_saturation': ['unreviewed', 'none_apparent', 'possible', 'cannot_tell'],
        'allowed_usability_and_alignment': ['unreviewed', 'yes', 'no', 'cannot_tell'],
        'cases': response_cases})
    (args.out / 'PROTOCOL.md').write_text('''# Observation-quality review

This CPU-generated pack contains 20 distinct training queries, five for each target class (grapevine or leguminous fodder) and training parent. Selection uses a fixed hash, not target presence or model scores. A target need not exist in its query.

Review each PNG and fill a copy of response_template.json. Record review duration, visible cloud/haze or obstruction, possible display saturation, visual-comparison usability, support-mask alignment, and possible border/parcel concerns. Use cannot_tell where appropriate. Annual source crop names are disclosed; do not report this as blind crop identification, independent agronomic truth, or the original 20-example expert consistency pilot. No human responses have been collected.

Only the two initially acquired query dates (positions 2 and 5) are displayed; positive and counterexample supports show all eight dates plus source masks. Query reference labels are never opened or displayed. Raw RGB is B04/B03/B02 divided by 3000 and clipped, with missing pixels magenta and support boundaries yellow. White or dark regions can reflect display limitations as well as atmosphere; availability masks do not certify cloud-free observations.

This package is for observation quality review, not model input. It does not attach crop-description cards, change any evaluation sample/date, certify global parcel independence, or automatically discard data. Any subsequent filtering must be separately recorded before a new evaluation. Source-bank and development imagery are excluded.
''')
    write_json(args.out / 'render_receipt.json', {
        'status': 'quality_review_package_prepared_not_reviewed', 'cases': len(cases),
        'unique_query_patches': len({c['query']['patch_id'] for c in cases}),
        'verified_input_packets': len(cache.verified), 'verified_support_masks': len(masks),
        'verified_input_references': cache.verified, 'source_hashes': source_hashes,
        'catalog_lineage': lineage,
        'cpu_only': True, 'new_gpu_seconds': 0, 'actual_human_responses': 0,
        'query_gold_opened': False, 'source_bank_packets_opened': 0, 'development_packets_opened': 0,
        'pixel_read_scope': 'Raw NPZ packets contain all dates; only acquired query dates [2,5] are rendered; no features or selections use unrendered query pixels.'})
    files = [{'path': str(path.relative_to(args.out)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(args.out.rglob('*')) if path.is_file()]
    write_json(args.out / 'export_manifest.json', {'files': files})
    print(json.dumps({'cases': len(cases), 'verified_inputs': len(cache.verified),
                      'verified_masks': len(masks), 'actual_human_responses': 0, 'out': str(args.out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--support-root', type=Path, help='Original episode root with source_objects.jsonl and support_masks')
    run(parser.parse_args())
