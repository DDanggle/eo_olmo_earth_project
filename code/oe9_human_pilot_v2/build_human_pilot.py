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

VERSION = 'oe9-human-candidate-v2'
RESPONSE_VERSION = 'oe9-human-candidate-v1'  # Response schema/UI are unchanged.
N = 20
QUOTAS = {'target': 8, 'counterexample': 8, 'neither': 4}

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
    return sorted(source, key=lambda r: rank('oe9-human-candidate-v1:query', r['patch_id']))[:N]

def canonical_inventory(prepared, rows, k1, scoring):
    """Read TRAIN labels to define reference strata; never rank visibility or model success."""
    row_map = {str(r['patch_id']): r for r in rows}
    options = {}
    exclusions = Counter()
    for row in sorted(rows, key=lambda r: str(r['patch_id'])):
        if row['training_partition'] != 'train_pool':
            continue
        pid = str(row['patch_id'])
        labels = load_verified(prepared, row['label_path'], row['label_sha256'])
        semantic, instances = labels['semantic'], labels['instances']
        if semantic.shape != (128,128) or instances.shape != semantic.shape:
            raise ValueError('Source label geometry is not the expected original 128x128 grid')
        ids, sizes = np.unique(instances, return_counts=True)
        objects = []
        for instance_id, size in zip(ids.tolist(), sizes.tolist()):
            if instance_id <= 0 or size < 64:
                continue
            mask = instances == instance_id
            classes, counts = np.unique(semantic[mask], return_counts=True)
            cls, count = min(zip(classes.tolist(), counts.tolist()), key=lambda v: (-v[1],v[0]))
            if cls not in range(1,19) or count/size < .95:
                exclusions['void_background_or_mixed_candidate'] += 1
                continue
            objects.append((int(instance_id),int(cls),int(size),count/size))
        for episode in k1.get(pid,[]):
            if episode['query_patch_id'] != pid or len(episode['support_pairs']) != 1:
                raise ValueError('Inconsistent K1 episode')
            for support in episode['support_pairs'][0].values():
                if support['patch_id']==pid or row_map[support['patch_id']]['training_partition']!='train_pool':
                    raise ValueError('Support must be query-disjoint train_pool')
            target = scoring[episode['episode_id']]
            for instance_id,cls,size,purity in objects:
                category = 'target' if cls==target['target_class'] else 'counterexample' if cls==target['counter_class'] else 'neither'
                option = {'patch_id':pid,'parent_tile':row['parent_tile'],'instance_id':instance_id,
                    'episode_id':episode['episode_id'],'reference_category':category,'dominant_class':cls,
                    'candidate_pixels':size,'dominant_class_fraction':purity}
                option['canonical_hash'] = rank(VERSION+':option',f'{pid}:{instance_id}:{episode["pair_id"]}')
                key=(pid,category)
                if key not in options or option['canonical_hash']<options[key]['canonical_hash']:
                    options[key]=option
    return list(options.values()), dict(exclusions)

def match_unique_queries(options, allowed_ids, balance_regions):
    """Deterministic bipartite maximum matching; one source query per quota slot."""
    choices={(o['patch_id'],o['reference_category']):o for o in options if o['patch_id'] in allowed_ids}
    parents=sorted({o['parent_tile'] for o in choices.values()})
    if balance_regions and (len(parents)!=2 or any(n%2 for n in QUOTAS.values())):
        return None
    slots=[]
    for category,n in QUOTAS.items():
        for parent in parents if balance_regions else [None]:
            for i in range(n//2 if balance_regions else n):
                slots.append((category,parent,i))
    eligible={slot:sorted([pid for (pid,c),o in choices.items() if c==slot[0] and (slot[1] is None or o['parent_tile']==slot[1])],
                          key=lambda pid:rank(VERSION+':matching:'+str(slot),pid)) for slot in slots}
    owner={}
    def assign(slot,seen):
        for pid in eligible[slot]:
            if pid in seen:continue
            seen.add(pid)
            if pid not in owner or assign(owner[pid],seen):
                owner[pid]=slot
                return True
        return False
    for slot in sorted(slots,key=lambda s:(len(eligible[s]),rank(VERSION+':slot',str(s)))):
        if not assign(slot,set()):return None
    selected=[choices[(pid,slot[0])] for pid,slot in owner.items()]
    if len(selected)!=20 or len({o['patch_id'] for o in selected})!=20 or Counter(o['reference_category'] for o in selected)!=QUOTAS:
        raise AssertionError('Matching failed exact unique-query quota contract')
    return sorted(selected,key=lambda o:rank(VERSION+':anonymous-order',o['patch_id']))

def select_stratified(options, original_ids, all_train_ids):
    attempts=[]
    for cohort,allowed in [('retain_v1_twenty',set(original_ids)),('all_train_pool48',set(all_train_ids))]:
        for balanced in (True,False):
            selected=match_unique_queries(options,allowed,balanced)
            attempts.append({'cohort':cohort,'balanced_per_stratum_parent':balanced,'matching_exists':selected is not None})
            if selected is not None:
                return selected,{'attempts':attempts,'chosen_cohort':cohort,'balanced_per_stratum_parent':balanced,
                    'same_v1_query_ids_retained':set(o['patch_id'] for o in selected)==set(original_ids)}
    raise ValueError('No 20-unique-query 8/8/4 matching exists in train_pool48; no duplication or padding allowed')

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
    original = select_queries(rows)  # Reconstruct the exact v1 label-blind cohort first.
    train_episodes = read_jsonl(episodes / 'episodes_train.jsonl')
    scoring = {r['episode_id']: r for r in read_jsonl(episodes / 'scoring/scoring_train.jsonl')}
    k1 = {}
    for episode in train_episodes:
        if episode['k_pairs'] == 1:
            k1.setdefault(episode['query_patch_id'], []).append(episode)
    inventory, exclusions = canonical_inventory(prepared, rows, k1, scoring)
    all_train_ids=[str(r['patch_id']) for r in rows if r['training_partition']=='train_pool']
    selected_options, sampling = select_stratified(inventory,[str(r['patch_id']) for r in original],all_train_ids)
    selected=[row_map[o['patch_id']] for o in selected_options]
    episode_map={e['episode_id']:e for e in train_episodes}
    public, private = out / 'reviewer_package', out / 'private'
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir()
    (public / 'assets').mkdir()
    dump(private / 'stratified_selection.json', {
        'sampling_protocol':VERSION,'original_v1_query_ids':[str(r['patch_id']) for r in original],
        'selected_options':selected_options,'source_label_reference_quotas':QUOTAS,
        'selection_inputs':'train_pool membership, public annual semantic/instance labels and frozen K1 supports',
        'image_visibility_or_model_scores_used':False,'human_responses_seen':False,
        'public_labels_are_not_expert_gold':True,'inventory_exclusions':exclusions,
        'source_manifest_sha256':sha(prepared/'manifest.jsonl'),**sampling})
    cases, mappings = [], []
    for i, row in enumerate(selected):
        pid, case_id = str(row['patch_id']), f'P{i + 1:03d}'
        option=selected_options[i]
        episode=episode_map[option['episode_id']]
        score = scoring[episode['episode_id']]
        label = load_verified(prepared, row['label_path'], row['label_sha256'])
        inst=option['instance_id']
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
        if reference['reference_category']!=option['reference_category'] or reference['dominant_class']!=option['dominant_class']:
            raise ValueError('Rendered candidate no longer matches frozen source reference selection')
        mappings.append({'case_id': case_id, 'patch_id': pid, 'instance_id': inst,
            'episode_id': episode['episode_id'], 'target_class': score['target_class'],
            'counter_class': score['counter_class'], 'query_input_sha256': row['npz_sha256'],
            'query_label_sha256': row['label_sha256'], 'supports': support_provenance, **reference})
    payload = {'schema_version': RESPONSE_VERSION,'sampling_protocol':'reference_stratified_candidate_pilot_v2', 'case_count': N, 'cases': cases,
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
    shutil.copyfile(here / 'PRIVATE_PROTOCOL.md', private / 'PRIVATE_PROTOCOL.md')
    dump(private / 'reference_mapping.json', {'package_id': payload['package_id'], 'cases': mappings})
    distribution = dict(Counter(r['reference_category'] for r in mappings))
    dump(private / 'coverage.json', {'package_id': payload['package_id'], 'case_count': N,
        'reference_availability': distribution, 'source_labels_are_not_expert_gold': True,
        'reference_stratification_before_human_or_model_responses': True,
        'source_label_reference_quotas': QUOTAS,
        'informative_reference_cases': distribution.get('target', 0) + distribution.get('counterexample', 0),
        'not_a_human_result': True})
    dump(out / 'build_contract.json', {'package_id': payload['package_id'], 'generator_sha256': sha(Path(__file__)),
        'manifest_sha256': sha(prepared / 'manifest.jsonl'), 'episode_sha256': sha(episodes / 'episodes_train.jsonl'),
        'query_count': N, 'query_partition': 'train_pool', 'support_partition': 'train_pool',
        'candidate_selection': 'public annual class reference strata; >=64 pixel full instance with >=.95 purity; no visual/model quality ranking',
        'query_selection':sampling,
        'parent_counts':dict(Counter(row['parent_tile'] for row in selected)),
        'parent_stratum_counts':dict(Counter(o['parent_tile']+':'+o['reference_category'] for o in selected_options)),
        'v1_design_reference_distribution':{'target':2,'counterexample':1,'neither':11,'reference_unavailable':6},
        'v1_human_annotations_collected':0,
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
