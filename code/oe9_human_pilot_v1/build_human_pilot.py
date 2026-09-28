#!/usr/bin/env python3
"""CPU-only anonymous candidate-identification pilot; no model or paid calls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

VERSION = 'oe9-human-candidate-v1'
N = 20

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def rank(namespace, value):
    return hashlib.sha256((namespace + ':' + str(value)).encode()).hexdigest()

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def safe(root, reference):
    path = (Path(root) / reference).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing source reference: {reference}')
    return path

def load_verified(root, reference, expected_sha):
    path = safe(root, reference)
    if sha(path) != expected_sha:
        raise ValueError('Source file hash mismatch: ' + reference)
    with np.load(path, allow_pickle=False) as packet:
        return {k: packet[k] for k in packet.files}

def select_queries(rows):
    source = [r for r in rows if r['training_partition'] == 'train_pool']
    if len(source) != 48 or len({str(r['patch_id']) for r in source}) != 48:
        raise ValueError('Expected frozen train_pool48, not a resampled or partial source set')
    return sorted(source, key=lambda r: rank(VERSION + ':query', r['patch_id']))[:N]

def select_candidate(instances, pid):
    ids, counts = np.unique(instances, return_counts=True)
    eligible = [int(i) for i, c in zip(ids, counts) if i > 0 and c >= 64]
    if not eligible:
        return None
    return min(eligible, key=lambda i: rank(VERSION + ':candidate', f'{pid}:{i}'))

def boundary(mask):
    p = np.pad(mask, 1)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~interior

def render(raw, valid, mask, path, bands):
    # Identical fixed display transfer for every patch/date. No per-image stretch.
    arr = np.moveaxis(raw[bands], 0, -1).astype(np.float32)
    arr = np.power(np.clip(arr / 3000., 0, 1), 1 / 2.2)
    rgb = np.asarray(arr * 255, dtype=np.uint8)
    rgb[~valid] = (90, 0, 90)
    # A single neutral outline is geometry only, not a target annotation.
    rgb[boundary(mask)] = (255, 255, 255)
    Image.fromarray(rgb).resize((384, 384), resample=Image.Resampling.NEAREST).save(path)

def render_sequence(packet, dates, mask, public, prefix):
    raw, valid = packet['raw_selected_s2'], packet['observation_valid']
    if raw.shape != (8, 10, 128, 128) or valid.shape != (8, 128, 128) or mask.shape != (128, 128):
        raise ValueError('Unexpected source image/validity/mask dimensions')
    frames = []
    for i, date in enumerate(dates):
        filenames = {}
        for name, bands in [('rgb', [2, 1, 0]), ('nir', [6, 2, 1])]:
            ref = f'assets/{prefix}_{i + 1:02d}_{name}.png'
            render(raw[i], valid[i], mask, public / ref, bands)
            filenames[name] = ref
        frames.append({'frame_id': f'{prefix}-{i + 1:02d}', 'date': str(date), **filenames})
    return frames

def private_reference(semantic, mask, target, counter):
    values, counts = np.unique(semantic[mask], return_counts=True)
    c, n = max(zip(values.tolist(), counts.tolist()), key=lambda x: (x[1], -x[0]))
    purity = n / int(mask.sum())
    if purity < .95 or c in (0, 19):
        decision = 'reference_unavailable'
    elif c == target:
        decision = 'target'
    elif c == counter:
        decision = 'counterexample'
    else:
        decision = 'neither'
    return {'reference_category': decision, 'dominant_class': c, 'dominant_class_fraction': purity,
            'candidate_pixels': int(mask.sum()), 'not_expert_gold': True}

def build(prepared, episodes, out, synthetic_fixture=False):
    prepared, episodes, out = Path(prepared), Path(episodes), Path(out)
    rows = read_jsonl(prepared / 'manifest.jsonl')
    row_map = {str(r['patch_id']): r for r in rows}
    selected = select_queries(rows)  # Freeze before loading any query labels or class histograms.
    train_episodes = read_jsonl(episodes / 'episodes_train.jsonl')
    scoring = {r['episode_id']: r for r in read_jsonl(episodes / 'scoring/scoring_train.jsonl')}
    k1 = {}
    for episode in train_episodes:
        if episode['k_pairs'] == 1:
            k1.setdefault(episode['query_patch_id'], []).append(episode)
    public, private = out / 'reviewer_package', out / 'private'
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir()
    (public / 'assets').mkdir()
    dump(private / 'query_selection_frozen_before_labels.json', {
        'namespace': VERSION + ':query', 'selected_patch_ids': [str(r['patch_id']) for r in selected],
        'selection_inputs': 'train_pool membership and patch ID only',
        'source_manifest_sha256': sha(prepared / 'manifest.jsonl')})
    cases, mappings = [], []
    for i, row in enumerate(selected):
        pid, case_id = str(row['patch_id']), f'P{i + 1:03d}'
        choices = k1.get(pid, [])
        if not choices:
            raise ValueError('Selected query has no K1 task; do not silently replace query: ' + pid)
        episode = min(choices, key=lambda e: rank(VERSION + ':pair:' + pid, e['pair_id']))
        score = scoring[episode['episode_id']]
        label = load_verified(prepared, row['label_path'], row['label_sha256'])
        inst = select_candidate(label['instances'], pid)  # Geometry only; class-blind candidate choice.
        if inst is None:
            cases.append({'case_id': case_id, 'candidate_available': False, 'frames': {},
                          'preparation_issue': 'No source geometry candidate with at least 64 pixels; not replaced.'})
            mappings.append({'case_id': case_id, 'patch_id': pid, 'reference_category': 'candidate_unavailable'})
            continue
        mask = label['instances'] == inst
        query = load_verified(prepared, row['npz_path'], row['npz_sha256'])
        frames = {'query': render_sequence(query, row['selected_dates'], mask, public, case_id + '-q')}
        support_provenance = {}
        for kind, alias in [('positive', 'positive'), ('counterexample', 'counterexample')]:
            support = episode['support_pairs'][0][kind]
            support_row = row_map[support['patch_id']]
            if support_row['training_partition'] != 'train_pool' or support['patch_id'] == pid:
                raise ValueError('Human-pilot support must be query-disjoint train_pool only')
            packet = load_verified(prepared, support['input_npz'], support['input_sha256'])
            selected_mask = load_verified(episodes, support['mask_npz'], support['mask_sha256'])['mask']
            frames[alias] = render_sequence(packet, support['dates_yyyymmdd'], selected_mask,
                                             public, case_id + ('-p' if kind == 'positive' else '-c'))
            support_provenance[kind] = {'object_key': support['object_key'],
                'input_sha256': support['input_sha256'], 'mask_sha256': support['mask_sha256']}
        cases.append({'case_id': case_id, 'candidate_available': True, 'frames': frames})
        reference = private_reference(label['semantic'], mask, score['target_class'], score['counter_class'])
        mappings.append({'case_id': case_id, 'patch_id': pid, 'instance_id': inst,
            'episode_id': episode['episode_id'], 'target_class': score['target_class'],
            'counter_class': score['counter_class'], 'query_input_sha256': row['npz_sha256'],
            'query_label_sha256': row['label_sha256'], 'supports': support_provenance, **reference})
    payload = {'schema_version': VERSION, 'case_count': N, 'cases': cases,
        'synthetic_fixture_package': bool(synthetic_fixture),
        'orders': {r: sorted([c['case_id'] for c in cases], key=lambda c: rank(VERSION + ':order:' + r, c)) for r in ('A', 'B')},
        'task': 'existing-candidate identification and annotation-time feasibility',
        'human_results_exist': False, 'source_support': 'public-label synthetic examples; not expert corrections',
        'display': {'rgb_bands': ['B04', 'B03', 'B02'], 'nir_false_color_bands': ['B08', 'B04', 'B03'],
                    'fixed_raw_clip': [0, 3000], 'gamma': 2.2, 'nodata_color': 'purple',
                    'cloud_quality_certified': False},
        'timing': {'active_time': 'foreground running time with interaction in preceding 60 seconds',
                   'idle_limit_seconds': 60, 'screen_attention_verified': False}}
    payload['asset_sha256'] = {p.relative_to(public).as_posix(): sha(p) for p in sorted((public / 'assets').glob('*.png'))}
    payload['package_id'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    # Public files contain no semantic labels, source names, private mappings, or AI/model opinions.
    dump(public / 'manifest.json', payload)
    (public / 'pilot_data.js').write_text('window.PILOT = ' + json.dumps(payload, ensure_ascii=False).replace('</', '<\\/') + ';\n')
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / 'review_ui.html', public / 'index.html')
    shutil.copyfile(here / 'review_app.js', public / 'review_app.js')
    shutil.copyfile(here / 'response_schema.json', public / 'response_schema.json')
    shutil.copyfile(here / 'PROTOCOL.md', public / 'PROTOCOL.md')
    dump(private / 'reference_mapping.json', {'package_id': payload['package_id'], 'cases': mappings})
    distribution = dict(Counter(r['reference_category'] for r in mappings))
    dump(private / 'coverage.json', {'package_id': payload['package_id'], 'case_count': N,
        'reference_availability': distribution, 'source_labels_are_not_expert_gold': True,
        'coverage_is_not_used_to_resample': True,
        'informative_reference_cases': distribution.get('target', 0) + distribution.get('counterexample', 0),
        'not_a_human_result': True})
    dump(out / 'build_contract.json', {'package_id': payload['package_id'], 'generator_sha256': sha(Path(__file__)),
        'manifest_sha256': sha(prepared / 'manifest.jsonl'), 'episode_sha256': sha(episodes / 'episodes_train.jsonl'),
        'query_count': N, 'query_partition': 'train_pool', 'support_partition': 'train_pool',
        'candidate_selection': 'class-blind hash among existing instance geometries >=64px',
        'query_selection': 'first20 hash of train_pool48 IDs before opening labels',
        'fullmask_annotation_task': False, 'free_form_expert_correction_learning_demonstrated': False,
        'human_annotations_collected': 0, 'reviewers_contacted': 0, 'paid_calls': 0, 'gpu_used': False,
        'private_reference_distribution': distribution,
        'distribution_note': 'Do not distribute private directory or this contract to annotators.',
        'distribute_only': 'reviewer_package/', 'do_not_host_private_parent_directory': True})
    return {'package_id': payload['package_id'], 'public': str(public), 'private': str(private),
            'case_count': N, 'reference_distribution': distribution, 'human_annotations_collected': 0}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root', required=True, type=Path)
    p.add_argument('--episodes-root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--synthetic-fixture', action='store_true', help='Software-test data only; marked visibly and never human evidence')
    args = p.parse_args()
    print(json.dumps(build(args.prepared_root, args.episodes_root, args.out, args.synthetic_fixture), ensure_ascii=False))

if __name__ == '__main__':
    main()
