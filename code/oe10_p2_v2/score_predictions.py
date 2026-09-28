#!/usr/bin/env python3
"""Independent CPU scorer for the locked common96 P2 development cohort.

Prediction generation must finish before this scoring-only process reads gold.
This module never supplies gold, IDs, or scoring metadata to a model and makes
no training-adequacy, fairness, equivalence, or three-seed decision claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


class ScoringError(ValueError):
    pass


KS = (1, 2, 4, 8)
AGGREGATION = 'per_query_then_target_class_macro_then_parent_macro_on_fixed_cohort'
PREDICTION_FIELDS = {'episode_id', 'base_id', 'k', 'npz_path', 'sha256'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open() as f:
        for number, line in enumerate(f, 1):
            require(bool(line.strip()), f'Blank JSONL row {number}: {path}')
            value = json.loads(line)
            require(isinstance(value, dict) and isinstance(value.get(key), str),
                    f'Invalid {key} at JSONL row {number}')
            require(value[key] not in result, f'Duplicate {key}: {value[key]}')
            result[value[key]] = value
    require(bool(result), f'Empty JSONL: {path}')
    return result


def cohort_hash(base_ids: list[str]) -> str:
    return hashlib.sha256(('\n'.join(sorted(base_ids)) + '\n').encode()).hexdigest()


def public_expected(episodes: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    p = config['p2']
    require(tuple(p['ks']) == KS and p['auc_denominator'] == 7, 'Expected fixed K1/2/4/8 and AUC denominator7')
    require(p['aggregation'] == AGGREGATION, 'Aggregation mismatch')
    k8_bases = set()
    by_base = defaultdict(dict)
    for eid, e in episodes.items():
        require(e['episode_id'] == eid and e['split'] == 'development', 'Public development episode identity mismatch')
        require(type(e['k_pairs']) is int and e['k_pairs'] in KS, 'Invalid public K')
        require(isinstance(e['base_id'], str) and bool(e['base_id']), 'Missing public base ID')
        require(e['k_pairs'] not in by_base[e['base_id']], 'Repeated K within public base')
        by_base[e['base_id']][e['k_pairs']] = e
        if e['k_pairs'] == 8:
            k8_bases.add(e['base_id'])
    c = p['cohort']
    require(len(k8_bases) == c['base_count'], 'Public common K8 base count mismatch')
    require(cohort_hash(list(k8_bases)) == c['cohort_base_ids_sha256'], 'Public common cohort hash mismatch')
    expected = {}
    for base in sorted(k8_bases):
        group = by_base[base]
        require(set(group) == set(KS), 'A common K8 base lacks one or more lower K episodes')
        ref = group[8]
        for k in KS:
            e = group[k]
            for key in ('query_patch_id', 'query_parent_tile', 'pair_id'):
                require(e[key] == ref[key], f'Public identity changes across K: {key}')
            expected[e['episode_id']] = e
    require(len(expected) == c['base_count'] * len(KS), 'Common episode count mismatch')
    require(len({e['query_patch_id'] for e in expected.values()}) == c['query_patch_count'], 'Query patch count mismatch')
    require(sorted({e['query_parent_tile'] for e in expected.values()}) == c['development_parent_ids'], 'Development parents mismatch')
    return expected


def score_metadata(expected: dict[str, dict[str, Any]], scoring: dict[str, dict[str, Any]],
                   all_public_ids: set[str]) -> None:
    require(set(scoring) == all_public_ids, 'Scoring metadata must match the entire pinned public catalog')
    by_base = {}
    for eid, e in expected.items():
        s = scoring[eid]
        for key in ('episode_id', 'base_id', 'pair_id', 'query_patch_id', 'k_pairs'):
            require(s[key] == e[key], f'Scoring/public join mismatch: {key}')
        require(s.get('k8_auc_cohort') is True, 'Common cohort missing from scoring metadata')
        for key in ('target_class', 'counter_class'):
            require(type(s[key]) is int and 1 <= s[key] <= 18, f'Invalid scoring {key}')
        require(s['target_class'] != s['counter_class'], 'Target and counter class must differ')
        for key in ('label_valid_pixels', 'target_pixels', 'counter_pixels'):
            require(type(s[key]) is int and s[key] >= 0, f'Invalid scoring count {key}')
        require(s['label_valid_pixels'] > 0, 'All-void query cannot be scored')
        require(s['target_pixels'] + s['counter_pixels'] <= s['label_valid_pixels'], 'Impossible target/counter counts')
        require(s.get('target_present') is (s['target_pixels'] > 0), 'target_present metadata mismatch')
        require(s.get('counter_present') is (s['counter_pixels'] > 0), 'counter_present metadata mismatch')
        require(s['query_label_sha256'] == s['expected_query_label_sha256'], 'Scoring label hashes disagree')
        invariant = tuple(s[key] for key in ('query_patch_id', 'target_class', 'counter_class',
                         'query_label_npz', 'query_label_sha256', 'target_pixels', 'counter_pixels', 'label_valid_pixels'))
        if e['base_id'] in by_base:
            require(by_base[e['base_id']] == invariant, 'Gold identity changes across K')
        by_base[e['base_id']] = invariant


def prediction_path(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str) and bool(relative), 'Prediction path must be a nonempty relative string')
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Prediction path must stay inside prediction directory')
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file() and path.suffix == '.npz',
            'Prediction path missing, outside root, or not NPZ')
    return path


def read_probability(root: Path, row: dict[str, Any]) -> np.ndarray:
    require(set(row) == PREDICTION_FIELDS, 'Prediction row must have exactly episode_id/base_id/k/npz_path/sha256')
    require(isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
            'Invalid prediction sha256')
    path = prediction_path(root, row['npz_path'])
    require(file_sha(path) == row['sha256'], 'Prediction file hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        require(data.files == ['probability'], 'Prediction NPZ must contain probability only')
        probability = data['probability']
    require(probability.shape == (128, 128), 'Prediction must be original128x128')
    require(np.issubdtype(probability.dtype, np.floating), 'Prediction must be floating probabilities, not class/boolean mask')
    require(bool(np.isfinite(probability).all()), 'Nonfinite prediction probability')
    require(bool(((probability >= 0) & (probability <= 1)).all()), 'Probability outside [0,1]')
    return probability


def verified_gold(gold: dict[str, Any], s: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    for key in ('target_mask', 'counter_mask', 'label_valid'):
        value = gold[key]
        require(isinstance(value, np.ndarray) and value.shape == (128, 128) and value.dtype == np.bool_,
                f'Gold {key} must be boolean128x128')
        masks.append(value)
    target, counter, valid = masks
    require(not bool((target & counter).any()), 'Target/counter masks overlap')
    require(not bool(((target | counter) & ~valid).any()), 'Target/counter mask includes voids')
    for value, key in ((target, 'target_pixels'), (counter, 'counter_pixels'), (valid, 'label_valid_pixels')):
        require(int(value.sum()) == s[key], f'Gold pixel count mismatch: {key}')
    audit = gold.get('audit', {})
    require(audit.get('purpose') == 'evaluation' and audit.get('episode_id') == s['episode_id']
            and audit.get('query_patch_id') == s['query_patch_id']
            and audit.get('label_sha256') == s['query_label_sha256'], 'Gold loader audit identity mismatch')
    return target.copy(), counter.copy(), valid.copy()


def hierarchical_macro(rows: list[dict[str, Any]], field: str) -> tuple[float, dict[str, Any]]:
    """Counter/support variants average within query, then query/class/parent macro."""
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            grouped[(row['parent'], row['target_class'], row['query_patch_id'])].append(float(value))
    require(bool(grouped), f'No eligible cases for {field}')
    per_class = defaultdict(list)
    for (parent, cls, query), values in grouped.items():
        per_class[(parent, cls)].append(math.fsum(values) / len(values))
    per_parent = defaultdict(list)
    class_values = {}
    for (parent, cls), values in sorted(per_class.items()):
        mean = math.fsum(values) / len(values)
        per_parent[parent].append(mean)
        class_values[f'{parent}:{cls}'] = {'mean': mean, 'query_count': len(values)}
    parent_values = {p: math.fsum(v)/len(v) for p, v in sorted(per_parent.items())}
    return math.fsum(parent_values.values()) / len(parent_values), {
        'by_parent_target_class': class_values, 'by_parent': parent_values,
        'eligible_query_target_groups': len(grouped),
    }


def score_core(episodes: dict[str, dict[str, Any]], scoring: dict[str, dict[str, Any]],
               predictions: dict[str, dict[str, Any]], prediction_dir: Path,
               target_accessor: Callable[[str], dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Pure scorer orchestration; production target_accessor is evaluation_target."""
    expected = public_expected(episodes, config)
    score_metadata(expected, scoring, set(episodes))
    require(set(predictions) == set(expected),
            f'Prediction cohort mismatch: missing={len(set(expected)-set(predictions))}, extra={len(set(predictions)-set(expected))}')
    for eid, row in predictions.items():
        e = expected[eid]
        require(set(row) == PREDICTION_FIELDS and row['episode_id'] == eid
                and row['base_id'] == e['base_id'] and type(row['k']) is int and row['k'] == e['k_pairs'],
                'Prediction episode/base/K identity mismatch')
    p = config['p2']
    absent_threshold = p['absence_guardrails']['absent_case_positive_if_predicted_valid_area_fraction_gt']
    require(absent_threshold == .001, 'Locked absent-case threshold must be .001')
    rows = []
    gold_cache = {}
    for eid in sorted(expected):
        e, s = expected[eid], scoring[eid]
        if e['base_id'] not in gold_cache:
            gold_cache[e['base_id']] = verified_gold(target_accessor(eid), s)
        target, counter, valid = gold_cache[e['base_id']]
        probability = read_probability(prediction_dir, predictions[eid])
        predicted = (probability > .5) & valid
        intersection = int((predicted & target).sum())
        union = int((predicted | target).sum())
        target_count, valid_count = int(target.sum()), int(valid.sum())
        predicted_count = int(predicted.sum())
        area = predicted_count / valid_count
        present = target_count > 0
        rows.append({
            'episode_id': eid, 'base_id': e['base_id'], 'k': e['k_pairs'],
            'query_patch_id': e['query_patch_id'], 'parent': e['query_parent_tile'],
            'target_class': s['target_class'], 'counter_class': s['counter_class'],
            'target_present': present, 'valid_pixels': valid_count, 'target_pixels': target_count,
            'predicted_valid_pixels': predicted_count, 'intersection': intersection, 'union': union,
            'target_iou': intersection / union if present else None,
            'absent_fp_area': None if present else area,
            'absent_fp_case': None if present else float(area > absent_threshold),
            'counter_false_positive_pixels': int((predicted & counter).sum()),
            'prediction_npz_sha256': predictions[eid]['sha256'],
            'source_label_sha256': s['query_label_sha256'],
        })
    one_k = [r for r in rows if r['k'] == 1]
    computed_cohort = {
        'base_count': len(one_k), 'query_patch_count': len({r['query_patch_id'] for r in one_k}),
        'target_present_base_count': sum(r['target_present'] for r in one_k),
        'target_absent_base_count': sum(not r['target_present'] for r in one_k),
        'target_classes': sorted({r['target_class'] for r in one_k}),
        'development_parent_ids': sorted({r['parent'] for r in one_k}),
        'cohort_base_ids_sha256': cohort_hash([r['base_id'] for r in one_k]),
        'source_scoring_sha256': p['cohort']['source_scoring_sha256'],
    }
    for key, value in computed_cohort.items():
        require(p['cohort'][key] == value, f'Gold-derived cohort mismatch: {key}')
    curves = {'target_iou_by_k': {}, 'absent_fp_area_by_k': {}, 'absent_fp_case_rate_by_k': {}}
    detail = {}
    for k in KS:
        selected = [r for r in rows if r['k'] == k]
        detail[str(k)] = {}
        for output, field in (('target_iou_by_k', 'target_iou'), ('absent_fp_area_by_k', 'absent_fp_area'),
                              ('absent_fp_case_rate_by_k', 'absent_fp_case')):
            value, breakdown = hierarchical_macro(selected, field)
            curves[output][str(k)] = value
            detail[str(k)][field] = breakdown
    c = curves['target_iou_by_k']
    area_under_curve = math.fsum((b-a)*(c[str(a)]+c[str(b)])/2 for a, b in zip(KS[:-1], KS[1:])) / 7
    return {
        'schema_version': 'oe10_single_run_prediction_scores_v1',
        'status': 'independent_probability_scoring_completed',
        'scope': 'fixed_common96_development_only_no_training_or_p2_decision',
        'aggregation': AGGREGATION, 'probability_threshold': .5,
        'probability_comparison': 'strictly_greater_than', 'absent_case_area_threshold': absent_threshold,
        'cohort': computed_cohort, 'prediction_count': len(rows), **curves, 'target_iou_auc': area_under_curve,
        'aggregation_breakdown': detail, 'per_episode_scoring_only': rows,
        'absent_classes_with_no_cases_are_not_imputed_as_zero': True,
        'training_adequacy_assertions_generated': False, 'fairness_assertions_generated': False,
        'statistical_equivalence_or_geographic_generalization_demonstrated': False,
    }


def score_predictions(*, prepared_root: Path, episodes_root: Path, prediction_dir: Path,
                      gate_config_path: Path) -> dict[str, Any]:
    from episode_loader import EpisodeLoader
    prepared_root, episodes_root, prediction_dir = map(Path, (prepared_root, episodes_root, prediction_dir))
    gate_config_path = Path(gate_config_path)
    gate_config_sha = file_sha(gate_config_path)
    config = json.loads(gate_config_path.read_text())
    require(config['status'] == 'locked_development_rules_before_P2_results_not_confirmatory_preregistration',
            'Use the locked v6 development gate configuration')
    loader = EpisodeLoader(prepared_root, episodes_root, 'development')
    episodes = loader.episodes
    scoring_path = episodes_root / 'scoring/scoring_development.jsonl'
    scoring_sha = file_sha(scoring_path)
    require(scoring_sha == config['p2']['cohort']['source_scoring_sha256'], 'Gold scoring metadata SHA mismatch')
    scoring = read_jsonl(scoring_path, 'episode_id')
    manifest_path = prediction_dir / 'predictions.jsonl'
    manifest_sha = file_sha(manifest_path)
    predictions = read_jsonl(manifest_path, 'episode_id')
    result = score_core(episodes, scoring, predictions, prediction_dir, loader.evaluation_target, config)
    require(file_sha(manifest_path) == manifest_sha, 'Prediction manifest changed during scoring')
    require(file_sha(scoring_path) == scoring_sha, 'Gold scoring metadata changed during scoring')
    require(file_sha(gate_config_path) == gate_config_sha, 'Gate configuration changed during scoring')
    result.update(gate_config_sha256=gate_config_sha, scorer_source_sha256=file_sha(Path(__file__)),
                  predictions_manifest_sha256=manifest_sha,
                  episode_loader_source_sha256=file_sha(Path(__import__('episode_loader').__file__)),
                  source_scoring_sha256=scoring_sha)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=Path, required=True)
    ap.add_argument('--episodes-root', type=Path, required=True)
    ap.add_argument('--prediction-dir', type=Path, required=True)
    ap.add_argument('--gate-config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = score_predictions(prepared_root=args.prepared_root, episodes_root=args.episodes_root,
                               prediction_dir=args.prediction_dir, gate_config_path=args.gate_config)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'prediction_count', 'target_iou_by_k',
                    'absent_fp_area_by_k', 'absent_fp_case_rate_by_k', 'target_iou_auc')}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
