#!/usr/bin/env python3
"""CPU-only, fixed-four-class OE11 adapter for the pinned OE8 helper module.

This does not run the OE8 main, select classes, score model predictions, or train.
Source labels and metadata have already been used in CPU availability probing.
This invocation freezes task construction before reading calibration gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

BASE_SHA256 = '64c6df1fe10b40da8cc076286a6184be05ec0db261a4462af0548aaba8eff6d6'
CLASSES = (1, 3, 8, 14)
KS = (1, 2, 4, 8)
SOURCE_PARENTS = frozenset(('t31tfj', 't32ulu', 't31tfm'))
INITIAL_POSITIONS = (2, 5)
EXTRA_POSITIONS = (0, 7)
ROLE_MAP = {'train_pool': 'train_pool', 'source_bank': 'source_bank',
            'calibration': 'dev_query'}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_base(path: Path):
    """Verify source before importing; never call its hard-coded main."""
    path = path.resolve(strict=True)
    if sha(path) != BASE_SHA256:
        raise ValueError('Base builder hash differs from the reviewed OE8 helper pin')
    spec = importlib.util.spec_from_file_location('_oe11_pinned_oe8_helpers', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load pinned base builder')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.KS) != KS:
        raise ValueError('Pinned helper K contract differs')
    return module


def adapt_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Use supplied partitions, never OE8's 80-row role assignment or class rank."""
    if not rows:
        raise ValueError('Manifest is empty')
    result, roles = [], {}
    parents = set()
    for original in rows:
        row = dict(original)
        pid = str(row['patch_id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', pid) or pid in roles:
            raise ValueError('patch_id must be unique and safe as a filename component')
        partition = row['training_partition']
        if partition not in ROLE_MAP:
            raise ValueError(f'Unknown training_partition: {partition}')
        role = ROLE_MAP[partition]
        parent = row['parent_tile']
        if parent not in SOURCE_PARENTS:
            raise ValueError(f'Parent/partition outside frozen OE11 scope: {pid}')
        parents.add(parent)
        for key in ('npz_path', 'label_path'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f'Missing {key}: {pid}')
        for key in ('npz_sha256', 'label_sha256'):
            if not isinstance(row.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise ValueError(f'An explicit SHA256 is required for {key}: {pid}')
        for key in ('strict_no_missing_input_eligible', 'supervised_training_allowed',
                    'clean_training_eligible'):
            if not isinstance(row.get(key), bool):
                raise ValueError(f'{key} must be an explicit bool: {pid}')
        if row['supervised_training_allowed'] != (role == 'train_pool'):
            raise ValueError(f'Training permission conflicts with partition: {pid}')
        if row['clean_training_eligible'] != (role == 'train_pool' and row['strict_no_missing_input_eligible']):
            raise ValueError(f'Clean eligibility conflicts with input/training flags: {pid}')
        row['patch_id'] = pid
        # Retain prepared partition; the role map is the only compatibility adapter.
        roles[pid] = role
        result.append(row)
    if parents != SOURCE_PARENTS:
        raise ValueError('Manifest must contain exactly the three fixed OE11 parents')
    if set(roles.values()) != {'train_pool', 'source_bank', 'dev_query'}:
        raise ValueError('All three supplied partitions must be present')
    parent_partitions = {(row['parent_tile'], row['training_partition']) for row in result}
    if parent_partitions != set(itertools.product(SOURCE_PARENTS, ROLE_MAP)):
        raise ValueError('Each source parent must contain all three supplied partitions')
    return sorted(result, key=lambda row: row['patch_id']), roles


def validate_references(base, root: Path, rows: list[dict[str, Any]]) -> None:
    """Existence/containment checks do not read calibration label contents."""
    seen: dict[Path, str] = {}
    for row in rows:
        for key in ('npz_path', 'label_path'):
            path = base.safe_path(root, row[key])
            if path in seen:
                raise ValueError(f'Reused input/label file reference: {row["patch_id"]} {key}')
            seen[path] = row['patch_id']


def class_coverage(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(cls): {
        'eligible_objects': sum(obj['class_id'] == cls for obj in objects),
        'eligible_patches': len({obj['patch_id'] for obj in objects if obj['class_id'] == cls}),
    } for cls in CLASSES}


def annotate_coverage(coverage: dict[str, Any], episodes: list[dict[str, Any]],
                      plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Report shortages without dropping classes, padding support, or failing build."""
    expected_pairs = list(itertools.permutations(CLASSES, 2))
    actual = Counter((row['target_class'], row['counter_class'], row['k_pairs']) for row in plan)
    query_count = coverage['query_candidates']
    by_pair = [{'target_class': a, 'counter_class': b,
                'episodes_by_k': {str(k): actual[a, b, k] for k in KS}}
               for a, b in expected_pairs]
    common_ks = [k for k in KS if query_count > 0 and
                 all(actual[a, b, k] == query_count for a, b in expected_pairs)]
    return {**coverage, 'fixed_target_classes': list(CLASSES),
            'all_directed_pairs': by_pair,
            'expected_directed_pair_count': len(expected_pairs),
            'common_ks_all_queries_and_all_pairs': common_ks,
            'complete_k1_k2_k4_k8_for_all_queries_and_pairs': common_ks == list(KS),
            'intended_subset_training_ready': common_ks == list(KS),
            'catalog_construction_complete': True,
            'pair_or_class_selection_from_prediction_scores': False}


def construct_catalog(base, rows, roles, inputs, pools, split, parent_policy):
    """Optionally restrict source supports to another training parent tile."""
    if parent_policy not in ('pooled', 'cross_parent'):
        raise ValueError('Unknown training support parent policy')
    if split != 'train' or parent_policy == 'pooled':
        return base.build_catalog(rows, roles, inputs, pools['train_pool'],
                                  pools['source_bank'], list(CLASSES), split, INITIAL_POSITIONS)
    episodes, plans, parts = [], [], []
    for parent in sorted(SOURCE_PARENTS):
        group_roles = {row['patch_id']: (
            'other_train_parent_not_query' if roles[row['patch_id']] == 'train_pool'
            and row['parent_tile'] != parent else roles[row['patch_id']]) for row in rows}
        other_parent_objects = [obj for obj in pools['train_pool'] if obj['parent_tile'] != parent]
        group_episodes, group_plans, group_coverage = base.build_catalog(
            rows, group_roles, inputs, other_parent_objects, pools['source_bank'],
            list(CLASSES), split, INITIAL_POSITIONS)
        if any(support['parent_tile'] == episode['query_parent_tile']
               for episode in group_episodes for pair in episode['support_pairs']
               for support in pair.values()):
            raise ValueError('Cross-parent support policy was violated')
        episodes.extend(group_episodes)
        plans.extend(group_plans)
        parts.append(group_coverage)
    episodes.sort(key=lambda row: row['episode_id'])
    plans.sort(key=lambda row: row['episode_id'])
    coverage = {
        'query_candidates': sum(part['query_candidates'] for part in parts),
        'episodes': len(episodes),
        'episodes_by_k': {str(k): sum(part['episodes_by_k'][str(k)] for part in parts) for k in KS},
        'query_pair_cohort': [row for part in parts for row in part['query_pair_cohort']],
        'support_shortages': [row for part in parts for row in part['support_shortages']],
        'k8_auc_base_count': sum(part['k8_auc_base_count'] for part in parts),
    }
    return episodes, plans, coverage


def build(prepared_root: Path, out: Path, base_builder: Path,
          train_support_parent_policy: str = 'pooled') -> dict[str, Any]:
    if train_support_parent_policy not in ('pooled', 'cross_parent'):
        raise ValueError('Unknown training support parent policy')
    base = load_base(base_builder)
    root = prepared_root.resolve(strict=True)
    manifest_path = base.safe_path(root, 'manifest.jsonl')
    policy_path = base.safe_path(root, 'selection_policy.json')
    policy = json.loads(policy_path.read_text())
    if not isinstance(policy, dict):
        raise ValueError('selection_policy.json must contain an object')
    # The new extraction policy need not contain the old quality-policy fields.
    # Explicit conflicting fields, if present, are never silently ignored.
    for key, expected in (('initial_candidate_positions', list(INITIAL_POSITIONS)),
                          ('fixed_additional_candidate_positions', list(EXTRA_POSITIONS))):
        if key in policy and policy[key] != expected:
            raise ValueError(f'Selection policy conflicts with frozen observation positions: {key}')
    if 'source_parents' in policy and (len(policy['source_parents']) != 3 or
                                      set(policy['source_parents']) != SOURCE_PARENTS):
        raise ValueError('Selection policy source parents disagree with the fixed three-parent scope')
    if 'target_classes' in policy and policy['target_classes'] != list(CLASSES):
        raise ValueError('Selection policy target classes disagree with the fixed four-class scope')
    raw_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    rows, roles = adapt_rows(raw_rows)
    parent_partition_counts = {parent: dict(Counter(row['training_partition'] for row in rows
                               if row['parent_tile'] == parent)) for parent in sorted(SOURCE_PARENTS)}
    if 'caps_per_parent' in policy:
        caps = policy['caps_per_parent']
        if set(caps) != set(ROLE_MAP) or any(type(value) is not int or value < 1 for value in caps.values()):
            raise ValueError('Policy partition caps must contain three positive integers')
        if any(counts[partition] > caps[partition] for counts in parent_partition_counts.values()
               for partition in ROLE_MAP):
            raise ValueError('Manifest exceeds a frozen per-parent partition cap')
    validate_references(base, root, rows)
    # Read and validate observations before creating output. No query gold here.
    inputs = {row['patch_id']: base.inspect_input(root, row) for row in rows}
    out = out.resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError('Output must not be inside the prepared input packet')
    out.mkdir(parents=True, exist_ok=False)
    for subdir in ('support_masks/train_pool', 'support_masks/source_bank', 'scoring'):
        (out / subdir).mkdir(parents=True)
    base.write_jsonl(out / 'roles.jsonl', [{
        'patch_id': row['patch_id'], 'parent_tile': row['parent_tile'],
        'prepared_training_partition': row['training_partition'],
        'episode_role': roles[row['patch_id']],
        'role_assignment': 'provided_prepared_partition_not_reassigned',
    } for row in rows])
    pools: dict[str, list[dict[str, Any]]] = {'train_pool': [], 'source_bank': []}
    source_inventory, exclusions = [], {}
    for row in rows:
        pid = row['patch_id']
        role = roles[pid]
        if role == 'dev_query':
            continue
        labels = base.safe_path(root, row['label_path'])
        semantic, instances, label_valid = base.load_labels(labels, row['label_sha256'])
        objects, reasons = base.eligible_objects(semantic, instances, label_valid,
            inputs[pid]['observation_valid'], pid, row['parent_tile'])
        reasons['outside_fixed_four_class_scope'] += sum(obj['class_id'] not in CLASSES for obj in objects)
        exclusions[pid] = dict(reasons)
        for obj in objects:
            if obj['class_id'] not in CLASSES:
                continue
            mask_ref = f'support_masks/{role}/{pid}_{obj["instance_id"]}.npz'
            np.savez_compressed(out / mask_ref, mask=obj.pop('mask'))
            obj.update(mask_npz=mask_ref, mask_sha256=sha(out / mask_ref),
                       source_label_sha256=row['label_sha256'], episode_role=role)
            pools[role].append(obj)
            source_inventory.append(obj)
    base.write_jsonl(out / 'source_objects.jsonl', source_inventory)
    base.write_json(out / 'pair_catalog.json', {
        'class_selection_source': 'fixed_OE11_scope_before_model_scoring',
        'selected_classes': list(CLASSES), 'class_roles': ['target', 'counterexample'],
        'directed_pairs': [{'pair_id': base.digest(f'oe8-directed-pair-v0:{a}:{b}')[:20],
                            'target_class': a, 'counter_class': b}
                           for a, b in itertools.permutations(CLASSES, 2)],
        'training_class_coverage': class_coverage(pools['train_pool']),
        'source_bank_class_coverage': class_coverage(pools['source_bank']),
        'development_labels_used_for_class_selection_in_this_builder': False,
        'prior_cpu_label_availability_probing_occurred': True,
        'classes_ranked_or_filtered_by_coverage': False,
        'prediction_scores_used_for_class_or_pair_selection': False,
        'pair_id_namespace': 'OE8 class-pair compatibility; scope episode IDs by catalog hash',
    })
    plans, coverage, public_hashes = {}, {}, {}
    for split in ('train', 'development'):
        episodes, plans[split], raw_coverage = construct_catalog(
            base, rows, roles, inputs, pools, split, train_support_parent_policy)
        # No changes to helper task IDs, nested K supports, or observation policy.
        coverage[split] = annotate_coverage(raw_coverage, episodes, plans[split])
        coverage[split]['support_parent_policy'] = train_support_parent_policy if split == 'train' else 'global_source_bank'
        path = out / f'episodes_{split}.jsonl'
        base.write_jsonl(path, episodes)
        public_hashes[split] = sha(path)
    base.write_json(out / 'coverage.json', coverage)
    # Freeze task definitions and source masks before this invocation opens calibration gold.
    frozen_files = ['episodes_train.jsonl', 'episodes_development.jsonl', 'pair_catalog.json',
                    'source_objects.jsonl', 'roles.jsonl', 'coverage.json']
    frozen_files += sorted(str(path.relative_to(out)) for path in (out / 'support_masks').rglob('*.npz'))
    frozen_hashes = {ref: sha(out / ref) for ref in frozen_files}
    base.write_json(out / 'public_catalog_hashes_before_scoring.json', public_hashes)
    base.write_json(out / 'frozen_artifacts_before_scoring.json', frozen_hashes)
    scoring = {split: base.write_scoring(root, out / 'scoring' / f'scoring_{split}.jsonl', plans[split])
               for split in ('train', 'development')}
    if any(sha(out / ref) != expected for ref, expected in frozen_hashes.items()):
        raise ValueError('Frozen task/source artifacts changed during scoring preparation')
    complete = all(coverage[split]['complete_k1_k2_k4_k8_for_all_queries_and_pairs']
                   for split in ('train', 'development'))
    common_ks = sorted(set(coverage['train']['common_ks_all_queries_and_all_pairs']) &
                       set(coverage['development']['common_ks_all_queries_and_all_pairs']))
    contract = {
        'status': 'valid_catalog_complete_coverage' if complete else 'valid_catalog_with_support_shortages',
        'catalog_build_completed': True,
        'catalog_scope': 'OE11 expanded source/calibration fixed_four_class CPU preparation',
        'script_sha256': sha(Path(__file__)), 'base_builder_sha256': sha(base_builder),
        'base_builder_expected_sha256': BASE_SHA256,
        'base_builder_path_at_preparation': str(base_builder.resolve()),
        'base_main_called': False, 'base_assign_roles_called': False,
        'base_select_classes_called': False,
        'prepared_manifest_sha256': sha(manifest_path),
        'selection_policy_sha256': sha(policy_path),
        'prepared_input_reference_root': str(root), 'support_mask_reference_root': str(out),
        'candidate_count': len(rows), 'partition_counts': dict(Counter(roles.values())),
        'prepared_partition_counts': dict(Counter(row['training_partition'] for row in rows)),
        'allowed_source_parents': sorted(SOURCE_PARENTS),
        'parent_partition_counts': parent_partition_counts,
        'policy_caps_per_parent': policy.get('caps_per_parent'),
        'all_policy_partition_caps_filled': (all(counts[partition] == policy['caps_per_parent'][partition]
            for counts in parent_partition_counts.values() for partition in ROLE_MAP)
            if 'caps_per_parent' in policy else None),
        'calibration_parents': sorted(SOURCE_PARENTS),
        'former_development_parent_t31tfm_promoted_to_source': True,
        'calibration_is_unseen_geographic_region': False,
        'final_holdout_t30uxv_included': False,
        'selected_classes': list(CLASSES), 'winter_wheat_class_2_included_as_task': False,
        'directed_pair_count': 12, 'ks': list(KS),
        'train_support_parent_policy': train_support_parent_policy,
        'same_episode_ids_across_parent_policies': True,
        'episode_identity_requires_catalog_sha256': True,
        'initial_candidate_positions': list(INITIAL_POSITIONS),
        'fixed_additional_candidate_positions': list(EXTRA_POSITIONS),
        'acquisition_policy_changed_from_OE8': False,
        'eligible_source_objects': {key: len(value) for key, value in pools.items()},
        'source_object_exclusions': exclusions,
        'public_catalog_sha256': public_hashes,
        'frozen_artifacts_sha256': frozen_hashes,
        'scoring_file_sha256': {split: sha(out / 'scoring' / f'scoring_{split}.jsonl')
                                for split in ('train', 'development')},
        'scoring_counts': scoring,
        'all_four_classes_in_both_roles': True,
        'complete_requested_k_coverage': complete,
        'common_ks_all_queries_all_pairs_both_splits': common_ks,
        'intended_subset_training_ready': complete,
        'training_ready': False,
        'training_ready_reason': 'Catalog preparation only; launch scope, loader and geometry audits remain. '
                                 'If K8 is short, freeze a common-K scope from availability before scoring models.',
        'missing_support_not_padded': True, 'missing_classes_not_replaced': True,
        'source_masks_by_partition': True, 'calibration_support_masks_exported': False,
        'train_support_partition': ('train_pool from other source parent only' if train_support_parent_policy == 'cross_parent'
                                    else 'train_pool excluding query patch'),
        'development_support_partition': 'source_bank only',
        'calibration_query_gold_used_for_catalog_construction_in_this_invocation': False,
        'train_query_labels_can_supply_supports_to_other_train_queries': True,
        'source_and_calibration_labels_never_previously_exposed': False,
        'prior_cpu_label_availability_probing_occurred': True,
        'source_gold_used_for_support_eligibility': True,
        'catalog_frozen_before_this_invocation_reads_calibration_gold': True,
        'query_gold_validity': 'label_valid only; no all-eight observation validity filtering',
        'support_min_pixels': 64, 'support_min_class_purity': 0.95,
        'support_max_objects_per_patch_per_class': 2,
        'support_whole_instance_valid_in_all_eight_observations': True,
        'strict_trainer_must_hold_missing_queries_until_missing_aware_adapter_exists': True,
        'nonclean_queries_retained_in_public_catalog': True,
        'all_eight_query_arrays_require_acquisition_guard': True,
        'all_eight_support_observations_must_be_counted_in_cost_ledger': True,
        'bank_and_calibration_labels_allowed_for_encoder_gradient_training': False,
        'parcel_disjoint_verified': False,
        'spatial_disjointness': 'Distinct patch IDs; all three partitions share the three source parents. '
                               'Exact footprints, cross-patch parcel overlaps and upstream spatial-block '
                               'assignment are not independently verified by this builder.',
        'annual_crop_reference_only': True, 'cloud_visibility_ground_truth_available': False,
        'historical_pretraining_exposure_unknown': True,
        'support_is_human_correction': False,
        'inference_loader_allowlist': ['episodes_train.jsonl', 'episodes_development.jsonl',
            'referenced prepared input packets', 'referenced support_masks only'],
        'analysis_or_scoring_only': ['roles.jsonl', 'source_objects.jsonl', 'pair_catalog.json',
            'coverage.json', 'scoring/', 'prepared manifest class statistics and labels/'],
        'model_visible_allowlist': ['generic prompt', 'acquired query observations and dates/validity',
            'selected support observations, dates/validity and object masks'],
        'never_model_features': ['pair/class mapping', 'target/counter IDs', 'paths/hashes',
            'patch/parent/object IDs', 'all-candidate eligibility flags', 'unacquired query features'],
        'gpu_used': False, 'model_executed': False, 'performance_improvement_measured': False,
    }
    base.write_json(out / 'episode_contract.json', contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--base-builder', required=True, type=Path)
    parser.add_argument('--train-support-parent-policy', choices=('pooled', 'cross_parent'), default='pooled')
    args = parser.parse_args()
    result = build(args.prepared_root, args.out, args.base_builder, args.train_support_parent_policy)
    print(json.dumps({key: result[key] for key in ('status', 'candidate_count', 'partition_counts',
        'selected_classes', 'train_support_parent_policy', 'complete_requested_k_coverage',
        'common_ks_all_queries_all_pairs_both_splits', 'training_ready')}, sort_keys=True))


if __name__ == '__main__':
    main()
