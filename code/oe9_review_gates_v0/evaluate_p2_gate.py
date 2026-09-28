#!/usr/bin/env python3
"""Evaluate a strict, scorer-produced P2 development summary. No model execution.

This gate compares training seed repeats on one fixed development parent.
It never treats those seeds as independent geographies or proves equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


class InvalidInput(ValueError):
    pass


def finite_number(value: Any, name: str, low: float = 0, high: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f'{name} must be a number, not bool/string')
    value = float(value)
    if not math.isfinite(value) or value < low or value > high:
        raise InvalidInput(f'{name} outside finite [{low},{high}]')
    return value


def curve(value: Any, ks: list[int], name: str) -> dict[int, float]:
    if not isinstance(value, dict) or set(value) != {str(k) for k in ks}:
        raise InvalidInput(f'{name} must contain exactly {ks}')
    return {k: finite_number(value[str(k)], f'{name}[{k}]') for k in ks}


def auc(points: dict[int, float], ks: list[int]) -> float:
    return math.fsum((b-a) * (points[a]+points[b]) / 2
                     for a, b in zip(ks[:-1], ks[1:])) / (ks[-1]-ks[0])


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidInput(f'{name} must be an explicit boolean')
    return value


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidInput(f'{name} must be a positive integer')
    return value


def hash_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise InvalidInput(f'{name} must be a lowercase sha256')
    return value


def evaluate(summary: dict[str, Any], config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    p = config['p2']
    ks = p['ks']
    blockers: list[str] = []
    if summary.get('schema_version') != 'oe9_p2_scorer_summary_v1':
        raise InvalidInput('Unsupported scorer summary schema')
    if summary.get('gate_config_sha256') != config_sha256:
        raise InvalidInput('Gate config hash mismatch; do not change rules after results')
    if not require_bool(summary.get('protocol_locked_before_results'), 'protocol_locked_before_results'):
        blockers.append('protocol_not_locked_before_results')
    for field in ('scorer_source_sha256', 'predictions_manifest_sha256'):
        hash_string(summary.get(field), field)
    if summary.get('aggregation') != p['aggregation']:
        raise InvalidInput('Metric aggregation does not match the fixed contract')
    if summary.get('absent_case_area_threshold') != p['absence_guardrails']['absent_case_positive_if_predicted_valid_area_fraction_gt']:
        raise InvalidInput('Absent-case false positive threshold does not match the fixed contract')
    expected_cohort = p['cohort']
    cohort = summary.get('cohort', {})
    for field in ('base_count', 'query_patch_count', 'target_present_base_count',
                  'target_absent_base_count', 'target_classes', 'development_parent_ids',
                  'cohort_base_ids_sha256', 'source_scoring_sha256'):
        if cohort.get(field) != expected_cohort[field]:
            raise InvalidInput(f'Common96 cohort mismatch: {field}')
    if summary.get('comparison_axis') not in p['comparison_axis_allowed']:
        raise InvalidInput('Declare matched_data_exposure or matched_total_compute')
    for check in p['fairness_checks_required']:
        if not require_bool(summary.get('fairness_checks', {}).get(check), f'fairness.{check}'):
            blockers.append(f'fairness:{check}')

    seeds = summary.get('paired_seed_ids')
    if (not isinstance(seeds, list) or seeds != p['paired_seed_ids'] or len(seeds) != p['paired_seed_count']
            or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise InvalidInput('Exactly three unique, prelocked integer seed IDs are required')
    if not require_bool(summary.get('seed_ids_locked_before_results'), 'seed_ids_locked_before_results'):
        blockers.append('seed_ids_not_locked_before_results')
    runs = summary.get('runs')
    if not isinstance(runs, list) or len(runs) != 2 * len(seeds):
        raise InvalidInput('Exactly one run per arm and paired seed is required')
    parsed: dict[tuple[str, int], dict[str, Any]] = {}
    adequacy = p['training_adequacy']
    for r in runs:
        arm, seed = r.get('arm'), r.get('seed')
        if (arm not in p['required_arms'] or isinstance(seed, bool) or not isinstance(seed, int)
                or seed not in seeds or (arm, seed) in parsed):
            raise InvalidInput('Unknown, duplicated, or unpaired arm/seed')
        for field in ('checkpoint_sha256', 'run_receipt_sha256'):
            hash_string(r.get(field), f'{arm}/{seed}.{field}')
        if r.get('cohort_base_ids_sha256') != expected_cohort['cohort_base_ids_sha256']:
            raise InvalidInput('Run-specific evaluation cohort differs')
        points = curve(r.get('target_iou_by_k'), ks, f'{arm}/{seed}.IoU')
        area = curve(r.get('absent_fp_area_by_k'), ks, f'{arm}/{seed}.absent_area')
        cases = curve(r.get('absent_fp_case_rate_by_k'), ks, f'{arm}/{seed}.absent_cases')
        a = r.get('training_adequacy', {})
        for check in adequacy['required_checks']:
            if not require_bool(a.get(check), f'{arm}/{seed}.{check}'):
                blockers.append(f'undertraining_or_execution:{arm}/{seed}:{check}')
        updates = positive_integer(a.get('completed_updates'), f'{arm}/{seed}.completed_updates')
        minimum = positive_integer(a.get('predeclared_minimum_updates'), f'{arm}/{seed}.minimum_updates')
        if updates < minimum:
            blockers.append(f'undertraining:{arm}/{seed}:minimum_updates_not_reached')
        interval = positive_integer(a.get('predeclared_eval_interval_updates'), f'{arm}/{seed}.eval_interval')
        if require_bool(a.get('budget_cap_reached_while_improving'), f'{arm}/{seed}.cap_while_improving'):
            blockers.append(f'undertraining:{arm}/{seed}:cap_while_improving')
        windows = a.get('last_evaluation_windows')
        if not isinstance(windows, list) or len(windows) < adequacy['minimum_last_evaluation_windows']:
            blockers.append(f'undertraining:{arm}/{seed}:insufficient_evaluation_windows')
        else:
            tail = windows[-adequacy['minimum_last_evaluation_windows']:]
            steps = [positive_integer(w.get('step'), f'{arm}/{seed}.eval_step') for w in tail]
            scores = [finite_number(w.get('auc'), f'{arm}/{seed}.eval_auc') for w in tail]
            if steps[-1] != updates or any(b-a != interval for a, b in zip(steps[:-1], steps[1:])):
                raise InvalidInput('Evaluation spacing or final update does not match the declared schedule')
            if max(scores) - min(scores) > adequacy['maximum_last3_auc_range'] + 1e-12:
                blockers.append(f'undertraining:{arm}/{seed}:last_windows_not_stable')
        cost = r.get('costs', {})
        for field in ('total_gpu_seconds_including_feature_precompute', 'training_observation_exposures',
                      'inference_gpu_seconds', 'inference_distinct_query_observations',
                      'inference_support_observations'):
            finite_number(cost.get(field), f'{arm}/{seed}.cost.{field}', 0, float('inf'))
        parsed[(arm, seed)] = {'auc': auc(points, ks), 'area': area, 'cases': cases}

    rows = []
    safety_violations = []
    guards = p['absence_guardrails']
    for seed in seeds:
        a, b = parsed[('B0', seed)], parsed[('B2', seed)]
        rows.append({'seed': seed, 'B0_auc': a['auc'], 'B2_auc': b['auc'], 'delta': b['auc'] - a['auc']})
        for k in ks:
            area_delta = b['area'][k] - a['area'][k]
            case_delta = b['cases'][k] - a['cases'][k]
            if area_delta > guards['maximum_B2_minus_B0_area_fraction_increase_each_seed_and_K'] + 1e-12:
                safety_violations.append({'seed': seed, 'k': k, 'metric': 'absent_fp_area', 'delta': area_delta})
            if case_delta > guards['maximum_B2_minus_B0_case_rate_increase_each_seed_and_K'] + 1e-12:
                safety_violations.append({'seed': seed, 'k': k, 'metric': 'absent_fp_case_rate', 'delta': case_delta})
    delta = [r['delta'] for r in rows]
    margin = p['practical_margin_absolute_auc']
    if blockers:
        decision, action = 'blocked_inadequate_or_unfair_comparison', 'resolve_recorded_blockers_before_interpreting_B0_vs_B2'
    elif safety_violations:
        decision, action = 'blocked_absent_false_positive_regression', 'inspect_absent_failures_do_not_promote_encoder_benefit_claim'
    elif all(d >= margin - 1e-12 for d in delta):
        decision, action = 'development_repeated_practical_gain', 'replicate_B0_vs_B2_on_second_audited_independent_development_parent'
    elif all(d <= -margin + 1e-12 for d in delta):
        decision, action = 'development_repeated_practical_harm', 'retain_B0_and_withdraw_encoder_learning_benefit_claim_for_this_development_scope'
    elif all(abs(d) < margin - 1e-12 for d in delta):
        decision, action = 'development_observed_practical_tie', p['tie_action']
    else:
        decision, action = 'inconclusive_development_result', 'inspect_training_variance_and_errors_without_claiming_equivalence_or_expanding_method_scope'
    return {
        'schema_version': 'oe9_p2_gate_decision_v1', 'decision': decision, 'next_action': action,
        'scope': 'fixed_common96_on_t31tfm_three_training_seed_repeats_development_only',
        'gate_config_sha256': config_sha256, 'seed_results': rows,
        'mean_paired_auc_delta': statistics.mean(delta), 'practical_auc_margin': margin,
        'blockers': blockers, 'absent_guardrail_violations': safety_violations,
        'formal_equivalence_test_performed': False, 'geographic_equivalence_proven': False,
        'independent_development_parent_count_in_this_result': 1,
        'cross_geography_claim_promotion_allowed': False, 'all_EO_encoder_learning_ruled_out': False,
        'CVPR_novelty_or_acceptance_established': False, 'kuro_execution_allowed': False,
        'D1_result_changed': False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    raw = args.config.read_bytes()
    config_sha = hashlib.sha256(raw).hexdigest()
    try:
        result = evaluate(json.loads(args.summary.read_text()), json.loads(raw), config_sha)
        code = 0
    except (InvalidInput, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = {'schema_version': 'oe9_p2_gate_decision_v1', 'decision': 'invalid_input', 'error': str(exc)}
        code = 2
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        with args.out.open('x') as f:
            f.write(output)
    print(output, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
