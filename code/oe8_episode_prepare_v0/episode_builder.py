#!/usr/bin/env python3
"""Build bounded PASTIS support/query catalogs from already prepared raw80.

No models, network, server access, or dataset repair. Public episode creation
does not open development labels. Scoring is a separate final stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

KS = (1, 2, 4, 8)
SOURCE_PARENTS = ('t32ulu', 't31tfj')
DEV_PARENT = 't31tfm'
BANK_NAMESPACE = 'oe8-bank-v0:'


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def safe_path(root: Path, ref: str) -> Path:
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Input reference is missing or outside prepared root: {ref}')
    return path


def assign_roles(rows: list[dict[str, Any]]) -> dict[str, str]:
    if len(rows) != 80 or len({str(r['patch_id']) for r in rows}) != 80:
        raise ValueError('Expected exactly 80 unique manifest patches')
    roles: dict[str, str] = {}
    for parent in SOURCE_PARENTS:
        group = [r for r in rows if r['parent_tile'] == parent]
        if len(group) != 32 or any(r['role'] != 'train' for r in group):
            raise ValueError(f'Expected 32 source-train rows in {parent}')
        group.sort(key=lambda r: digest(BANK_NAMESPACE + str(r['patch_id'])))
        for rank, row in enumerate(group):
            roles[str(row['patch_id'])] = 'source_bank' if rank < 8 else 'train_pool'
    dev = [r for r in rows if r['parent_tile'] == DEV_PARENT]
    if len(dev) != 16 or any(r['role'] not in ('development', 'dev') for r in dev):
        raise ValueError('Expected 16 development rows in t31tfm')
    for row in dev:
        roles[str(row['patch_id'])] = 'dev_query'
    if len(roles) != 80:
        raise ValueError('Unexpected parent in manifest')
    return roles


def inspect_input(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Read input-only observation metadata/validity; never read a label path."""
    path = safe_path(root, row['npz_path'])
    with np.load(path, allow_pickle=False) as packet:
        if {'semantic', 'instances', 'label_valid', 'crop_label_valid'} & set(packet.files):
            raise ValueError('Inference input packet must not contain source labels')
        dates = np.asarray(row['selected_dates'], dtype=np.int64)
        timestamps = packet['timestamps']
        valid = packet['observation_valid']
        if dates.shape != (8,) or valid.shape != (8, 128, 128) or valid.dtype != np.bool_:
            raise ValueError('Expected 8 source dates and boolean 8x128x128 observation validity')
        parsed = [datetime.strptime(str(int(d)), '%Y%m%d') for d in dates]
        if parsed != sorted(set(parsed)):
            raise ValueError('Candidate dates must be unique and ordered')
        expected_timestamps = np.asarray([[d.day, d.month - 1, d.year] for d in parsed], dtype=np.int64)
        if timestamps.dtype != np.int64 or not np.array_equal(timestamps, expected_timestamps):
            raise ValueError('Input timestamps disagree with selected_dates: expected day, month0, year')
        if packet['raw_selected_s2'].shape != (8, 10, 128, 128) or packet['normalized_s2'].shape != (128, 128, 8, 12):
            raise ValueError('Unexpected prepared EO input dimensions')
        if bool(valid.all()) != row['strict_no_missing_input_eligible']:
            raise ValueError('Input completeness flag disagrees with observation_valid')
        meta = {'dates': [int(d) for d in dates], 'observation_valid': valid.copy(),
                'input_sha256': file_sha(path)}
    if row.get('npz_sha256') and row['npz_sha256'] != meta['input_sha256']:
        raise ValueError('Prepared input SHA does not match its manifest')
    return meta


def load_labels(path: Path, expected_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if file_sha(path) != expected_sha256:
        raise ValueError('Prepared label SHA does not match its manifest')
    with np.load(path, allow_pickle=False) as packet:
        semantic, instances, valid = packet['semantic'], packet['instances'], packet['label_valid']
        crop_valid = packet['crop_label_valid']
    if semantic.shape != (128, 128) or instances.shape != semantic.shape or valid.shape != semantic.shape:
        raise ValueError('Unaligned label arrays')
    if semantic.dtype != np.int64 or instances.dtype != np.int64 or valid.dtype != np.bool_:
        raise ValueError('Expected int64 semantic/instances and boolean label_valid')
    if (instances < 0).any() or not np.isin(semantic, np.arange(20)).all():
        raise ValueError('Unexpected semantic or instance IDs')
    if not np.array_equal(valid, semantic != 19):
        raise ValueError('label_valid must describe source semantic voids, not image visibility')
    if crop_valid.dtype != np.bool_ or not np.array_equal(crop_valid, (semantic > 0) & (semantic < 19)):
        raise ValueError('crop_label_valid disagrees with crop class IDs')
    return semantic, instances, valid


def eligible_objects(semantic: np.ndarray, instances: np.ndarray, label_valid: np.ndarray,
                     observation_valid: np.ndarray, patch_id: str, parent: str,
                     min_pixels: int = 64, min_purity: float = .95
                     ) -> tuple[list[dict[str, Any]], Counter]:
    objects: list[dict[str, Any]] = []
    excluded: Counter = Counter()
    for instance_id in sorted(int(i) for i in np.unique(instances) if i > 0):
        body = instances == instance_id
        count = int(body.sum())
        values, counts = np.unique(semantic[body], return_counts=True)
        crop = [(int(n), int(c)) for c, n in zip(values, counts) if 1 <= c <= 18]
        if not crop:
            excluded['no_crop_class'] += 1
            continue
        largest, cls = sorted(crop, key=lambda x: (-x[0], x[1]))[0]
        if largest / count < min_purity:
            excluded['class_purity_below_threshold'] += 1
            continue
        mask = body & (semantic == cls) & label_valid
        if int(mask.sum()) < min_pixels:
            excluded['fewer_than_minimum_class_pixels'] += 1
            continue
        # Check the whole source instance, not just a conveniently visible subset.
        if not bool(observation_valid[:, body].all()):
            excluded['instance_not_valid_in_all_eight_observations'] += 1
            continue
        ys, xs = np.where(mask)
        key = f'{patch_id}:{instance_id}'
        objects.append({'object_key': key, 'patch_id': patch_id, 'parent_tile': parent,
                        'instance_id': instance_id, 'class_id': cls, 'mask': mask,
                        'class_pixel_count': int(mask.sum()), 'instance_pixel_count': count,
                        'class_purity': largest / count,
                        'touches_image_border': bool(body[0].any() or body[-1].any() or body[:, 0].any() or body[:, -1].any()),
                        'bbox_xyxy_exclusive': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
    return objects, excluded


def select_classes(train_objects: list[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    patches: dict[int, set[str]] = defaultdict(set)
    object_counts: Counter = Counter()
    for obj in train_objects:
        patches[obj['class_id']].add(obj['patch_id'])
        object_counts[obj['class_id']] += 1
    eligible = [c for c in range(1, 19) if len(patches[c]) >= 4]
    eligible.sort(key=lambda c: (-len(patches[c]), c))
    coverage = {str(c): {'eligible_objects': object_counts[c], 'eligible_patches': len(patches[c])}
                for c in range(1, 19)}
    return eligible[:4], coverage


def capped_sequence(objects: list[dict[str, Any]], cls: int, excluded_patch: str | None,
                    namespace: str) -> list[dict[str, Any]]:
    candidates = [o for o in objects if o['class_id'] == cls and o['patch_id'] != excluded_patch]
    candidates.sort(key=lambda o: digest(namespace + ':' + o['object_key']))
    selected: list[dict[str, Any]] = []
    per_patch: Counter = Counter()
    seen = set()
    for obj in candidates:
        if obj['object_key'] in seen or per_patch[obj['patch_id']] >= 2:
            continue
        selected.append(obj)
        seen.add(obj['object_key'])
        per_patch[obj['patch_id']] += 1
        if len(selected) == 8:
            break
    return selected


def public_support(obj: dict[str, Any], rows: dict[str, dict[str, Any]],
                   inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pid = obj['patch_id']
    return {'object_key': obj['object_key'], 'patch_id': pid, 'parent_tile': obj['parent_tile'],
            'input_npz': rows[pid]['npz_path'], 'input_sha256': inputs[pid]['input_sha256'],
            'mask_npz': obj['mask_npz'], 'mask_sha256': obj['mask_sha256'],
            'observation_ids': [f'{pid}:obs:{i}' for i in range(8)],
            'dates_yyyymmdd': inputs[pid]['dates']}


def build_catalog(rows: list[dict[str, Any]], roles: dict[str, str],
                  inputs: dict[str, dict[str, Any]], train_objects: list[dict[str, Any]],
                  bank_objects: list[dict[str, Any]], classes: list[int], split: str,
                  initial_positions: tuple[int, int] = (2, 5)
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pure metadata/source-support construction: no filesystem or query-gold access."""
    row_map = {str(r['patch_id']): r for r in rows}
    if split not in ('train', 'development'):
        raise ValueError('Unknown episode split')
    if len(initial_positions) != 2 or len(set(initial_positions)) != 2 or any(i not in range(8) for i in initial_positions):
        raise ValueError('Expected two distinct initial candidate positions')
    query_role = 'train_pool' if split == 'train' else 'dev_query'
    queries = sorted([r for r in rows if roles[str(r['patch_id'])] == query_role],
                     key=lambda r: str(r['patch_id']))
    pool = train_objects if split == 'train' else bank_objects
    episodes, scoring_plan = [], []
    shortage, cohort = [], []
    by_k: Counter = Counter()
    for row in queries:
        pid = str(row['patch_id'])
        # Class-only namespace makes reverse tasks swap the exact same objects.
        sequences = {c: capped_sequence(pool, c, pid if split == 'train' else None,
                                       f'oe8-support-v0:{split}:{pid if split == "train" else "shared"}:class={c}')
                     for c in classes}
        for target, counter in itertools.permutations(classes, 2):
            positive, negative = sequences[target], sequences[counter]
            capacity = min(len(positive), len(negative))
            supported = [k for k in KS if k <= capacity]
            pair_id = digest(f'oe8-directed-pair-v0:{target}:{counter}')[:20]
            base_id = digest(f'oe8-episode-base-v0:{split}:{pid}:{pair_id}')[:24]
            cohort_row = {'query_patch_id': pid, 'pair_id': pair_id, 'base_id': base_id,
                          'available_ks': supported, 'k8_auc_eligible': capacity >= 8}
            cohort.append(cohort_row)
            if capacity < 8:
                shortage.append({**cohort_row, 'target_support_capacity': len(positive),
                                 'counter_support_capacity': len(negative)})
            for k in supported:
                pairs = [{'positive': public_support(p, row_map, inputs),
                          'counterexample': public_support(n, row_map, inputs)}
                         for p, n in zip(positive[:k], negative[:k])]
                keys = [o['object_key'] for pair in pairs for o in pair.values()]
                if len(keys) != len(set(keys)) or any(o['patch_id'] == pid for pair in pairs for o in pair.values()):
                    raise ValueError('Support/query overlap or duplicated support object')
                episode_id = f'{base_id}:k{k}'
                episodes.append({'episode_id': episode_id, 'base_id': base_id, 'pair_id': pair_id,
                    'split': split, 'query_patch_id': pid, 'query_parent_tile': row['parent_tile'],
                    'query_input_npz': row['npz_path'], 'query_input_sha256': inputs[pid]['input_sha256'],
                    'query_observations': [{'observation_id': f'{pid}:obs:{i}', 'date_yyyymmdd': d}
                                           for i, d in enumerate(inputs[pid]['dates'])],
                    'initial_observation_ids': [f'{pid}:obs:{i}' for i in initial_positions],
                    'strict_no_missing_input_eligible': row['strict_no_missing_input_eligible'],
                    'supervised_training_allowed': row['supervised_training_allowed'],
                    'clean_training_eligible': row['clean_training_eligible'],
                    'missing_aware_adapter_required': not row['strict_no_missing_input_eligible'],
                    'eligibility_flags_are_not_selector_features': True,
                    'parcel_disjoint_verified': False,
                    'k_pairs': k, 'support_pairs': pairs,
                    'prompt': 'Find regions matching the positive examples and exclude the counterexamples. Return a mask reference and the observation IDs you used.',
                    'definition_mode': 'unnamed_exemplar_target',
                    'query_label_supplied': False,
                    'limitations': ['annual crop reference only', 'cloud validity unavailable',
                                    'exact footprint and cross-patch parcel identity unverified'],
                    'support_origin': 'public_mask_synthetic_correction'})
                scoring_plan.append({'episode_id': episode_id, 'base_id': base_id, 'pair_id': pair_id,
                                     'query_patch_id': pid, 'query_label_npz': row['label_path'],
                                     'expected_query_label_sha256': row['label_sha256'],
                                     'target_class': target, 'counter_class': counter,
                                     'k_pairs': k, 'k8_auc_cohort': capacity >= 8})
                by_k[k] += 1
    return episodes, scoring_plan, {'query_candidates': len(queries), 'episodes': len(episodes),
        'episodes_by_k': {str(k): by_k[k] for k in KS}, 'query_pair_cohort': cohort,
        'support_shortages': shortage, 'k8_auc_base_count': sum(c['k8_auc_eligible'] for c in cohort)}


def write_scoring(root: Path, path: Path, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Gold access is confined here, after public catalogs are written/frozen."""
    label_cache = {}
    scored = []
    counts: Counter = Counter()
    for row in plan:
        ref = row['query_label_npz']
        if ref not in label_cache:
            label_path = safe_path(root, ref)
            semantic, _, valid = load_labels(label_path, row['expected_query_label_sha256'])
            label_cache[ref] = (semantic, valid, file_sha(label_path))
        semantic, valid, label_hash = label_cache[ref]
        if label_hash != row['expected_query_label_sha256']:
            raise ValueError('Conflicting label hashes in scoring plan')
        target_count = int(((semantic == row['target_class']) & valid).sum())
        counter_count = int(((semantic == row['counter_class']) & valid).sum())
        counts['target_present_episodes' if target_count else 'target_absent_episodes'] += 1
        scored.append({**row, 'query_label_sha256': label_hash,
                       'label_valid_pixels': int(valid.sum()), 'target_pixels': target_count,
                       'counter_pixels': counter_count, 'target_present': target_count > 0,
                       'counter_present': counter_count > 0,
                       'mask_rule': 'semantic == target_class AND label_valid',
                       'observability_not_inferred_from_labels': True,
                       'all_eight_observation_valid_filter_applied_to_query_gold': False})
    write_jsonl(path, scored)
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    roles = assign_roles(rows)
    for row in rows:
        pid = str(row['patch_id'])
        if row['training_partition'] != roles[pid]:
            raise ValueError(f'Prepared partition conflicts with frozen hash rule: {pid}')
        for key in ('strict_no_missing_input_eligible', 'supervised_training_allowed', 'clean_training_eligible'):
            if not isinstance(row[key], bool):
                raise ValueError(f'{key} must be an explicit bool')
        if row['supervised_training_allowed'] != (roles[pid] == 'train_pool'):
            raise ValueError('Supervised training permission disagrees with frozen partition')
        if row['clean_training_eligible'] != (row['supervised_training_allowed'] and row['strict_no_missing_input_eligible']):
            raise ValueError('clean_training_eligible must combine training permission and input completeness')
    quality_path = safe_path(args.prepared_root, 'quality_policy.json')
    quality_policy = json.loads(quality_path.read_text())
    initial_positions = tuple(quality_policy['initial_candidate_positions'])
    if initial_positions != (2, 5) or quality_policy['fixed_additional_candidate_positions'] != [0, 7]:
        raise ValueError('Prepared observation policy differs from the frozen OE8 policy')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'support_masks').mkdir()
    (args.out / 'scoring').mkdir()
    write_jsonl(args.out / 'roles.jsonl', [{'patch_id': str(r['patch_id']), 'parent_tile': r['parent_tile'],
        'original_role': r['role'], 'episode_role': roles[str(r['patch_id'])],
        'bank_partition_hash': digest(BANK_NAMESPACE + str(r['patch_id']))} for r in rows])
    inputs = {str(r['patch_id']): inspect_input(args.prepared_root, r) for r in rows}
    pools: dict[str, list[dict[str, Any]]] = {'train_pool': [], 'source_bank': []}
    exclusions = {}
    source_inventory = []
    for row in rows:
        pid = str(row['patch_id'])
        role = roles[pid]
        if role == 'dev_query':
            continue
        label_path = safe_path(args.prepared_root, row['label_path'])
        semantic, instances, label_valid = load_labels(label_path, row['label_sha256'])
        objects, reasons = eligible_objects(semantic, instances, label_valid,
                                           inputs[pid]['observation_valid'], pid, row['parent_tile'])
        exclusions[pid] = dict(reasons)
        label_sha = file_sha(label_path)
        for obj in objects:
            mask_ref = f'support_masks/{pid}_{obj["instance_id"]}.npz'
            np.savez_compressed(args.out / mask_ref, mask=obj.pop('mask'))
            obj.update(mask_npz=mask_ref, mask_sha256=file_sha(args.out / mask_ref),
                       source_label_sha256=label_sha, episode_role=role)
            pools[role].append(obj)
            source_inventory.append(obj)
    classes, train_coverage = select_classes(pools['train_pool'])
    write_jsonl(args.out / 'source_objects.jsonl', source_inventory)
    write_json(args.out / 'pair_catalog.json', {
        'class_selection_source': 'eligible object patch coverage in train_pool48 only',
        'rank_rule': 'coverage descending; numeric class ID ascending; minimum4patch; top4classes',
        'selected_classes': classes, 'training_class_coverage': train_coverage,
        'directed_pairs': [{'pair_id': digest(f'oe8-directed-pair-v0:{a}:{b}')[:20],
                            'target_class': a, 'counter_class': b}
                           for a, b in itertools.permutations(classes, 2)],
        'source_bank_used_for_class_selection': False,
        'development_labels_used_for_class_selection': False})
    public_hashes, plans, coverage = {}, {}, {}
    for split in ('train', 'development'):
        episodes, plans[split], coverage[split] = build_catalog(rows, roles, inputs,
            pools['train_pool'], pools['source_bank'], classes, split, initial_positions)
        path = args.out / f'episodes_{split}.jsonl'
        write_jsonl(path, episodes)
        public_hashes[split] = file_sha(path)
    # Freeze public tasks before any development label file is opened.
    write_json(args.out / 'public_catalog_hashes_before_scoring.json', public_hashes)
    write_json(args.out / 'coverage.json', coverage)
    scoring = {split: write_scoring(args.prepared_root, args.out / 'scoring' / f'scoring_{split}.jsonl', plan)
               for split, plan in plans.items()}
    if any(file_sha(args.out / f'episodes_{split}.jsonl') != value for split, value in public_hashes.items()):
        raise ValueError('Public episodes changed during gold-only scoring preparation')
    contract = {'status': 'bounded_episode_catalog_prepared_not_training_completed',
        'script_sha256': file_sha(Path(__file__)), 'prepared_manifest_sha256': file_sha(args.manifest),
        'prepared_input_reference_root': str(args.prepared_root.resolve()),
        'support_mask_reference_root': str(args.out.resolve()),
        'quality_policy_sha256': file_sha(quality_path),
        'initial_candidate_positions': list(initial_positions),
        'fixed_additional_candidate_positions': [0, 7],
        'candidate_count': len(rows), 'partition_counts': dict(Counter(roles.values())),
        'eligible_source_objects': {k: len(v) for k, v in pools.items()},
        'source_object_exclusions': exclusions, 'selected_classes': classes,
        'public_catalog_sha256': public_hashes, 'scoring_counts': scoring,
        'ks': list(KS), 'support_min_pixels': 64, 'support_min_class_purity': .95,
        'support_max_objects_per_patch_per_class': 2,
        'support_whole_instance_valid_in_all_eight_observations': True,
        'support_row_clean_flag_required': False,
        'training_query_clean_flag_required': False, 'development_query_clean_flag_required': False,
        'all_prepared_queries_retained_before_support_availability': True,
        'strict_trainer_must_hold_missing_queries_until_missing_aware_adapter_exists': True,
        'query_quality_flags_must_not_select_dates_or_silently_change_auc_cohort': True,
        'query_gold_validity': 'source label_valid only; no all8 observation mask filtering',
        'bank_labels_used_for_encoder_training': False,
        'spatial_disjointness': 'patch IDs separated; cross-patch parcels and exact footprints unverified',
        'parcel_disjoint_verified': False,
        'inference_loader_allowlist': ['episodes_train.jsonl', 'episodes_development.jsonl',
            'referenced inputs under prepared_input_reference_root', 'referenced support_masks'],
        'model_visible_allowlist': ['fixed generic prompt', 'acquired query observation pixels',
            'acquired observation dates and validity', 'selected support images and object masks',
            'support observation dates and validity'],
        'retrieval_scoring_only_never_model_prompt': ['episode_id', 'base_id', 'pair_id',
            'patch IDs', 'parent tile IDs', 'object_key', 'file paths', 'hashes',
            'class mapping', 'all-candidate aggregate eligibility flags'],
        'unacquired_query_pixels_features_and_per_date_quality_forbidden': True,
        'all_eight_candidate_arrays_in_npz_require_loader_acquisition_guard': True,
        'all_eight_support_observations_must_be_counted_in_cost_ledger': True,
        'new_class_transfer_demonstrated': False,
        'analysis_or_scoring_only': ['roles.jsonl', 'source_objects.jsonl', 'pair_catalog.json',
            'scoring/', 'prepared manifest class_pixel_counts and labels/'],
        'historical_pretraining_exposure_unknown': True,
        'lower_k_episodes_available': coverage['development']['episodes'] > 0,
        'k8_auc_cohort_available': coverage['development']['k8_auc_base_count'] > 0,
        'scientific_training_ready': False, 'gpu_used': False, 'model_executed': False,
        'remaining': ['independent data/episode audit', 'same-parent geometry/parcel overlap audit',
                      'real trainer and checkpoint save/reload', 'actual comparison training'],
        'support_is_human_correction': False}
    write_json(args.out / 'episode_contract.json', contract)
    print(json.dumps({k: contract[k] for k in ('status', 'partition_counts', 'selected_classes',
          'eligible_source_objects', 'lower_k_episodes_available', 'k8_auc_cohort_available')}))


if __name__ == '__main__':
    main()
