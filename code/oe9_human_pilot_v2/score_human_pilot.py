#!/usr/bin/env python3
"""Score two independent pilot responses; source labels are availability strata, not gold."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics

VERSION = 'oe9-human-candidate-v1'
DECISIONS = ('target', 'counterexample', 'neither', 'uncertain', 'unobservable')
ISSUES = ('seasonality', 'spectral', 'shape', 'cloud', 'resolution', 'dates', 'mixed', 'examples', 'other')

def stamp(value):
    if not isinstance(value, str):
        raise ValueError('Missing timestamp')
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return d.timestamp()

def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Timing must be finite nonnegative numeric')
    return value

def validate(response, manifest, role, allow_synthetic=False):
    if response['schema_version'] != VERSION or response['package_id'] != manifest['package_id']:
        raise ValueError('Response belongs to another schema/package')
    if response['assignment'] != role or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', response['annotator_id']):
        raise ValueError('Invalid reviewer assignment/ID')
    if type(response.get('synthetic_fixture')) is not bool:
        raise ValueError('Explicit synthetic_fixture flag is required')
    if response['synthetic_fixture'] and not allow_synthetic:
        raise ValueError('Synthetic fixtures must never count as actual human results')
    stamp(response['exported_at'])
    records = response['records']
    cases = {c['case_id']: c for c in manifest['cases']}
    if len(records) != len(cases) or {r['case_id'] for r in records} != set(cases):
        raise ValueError('Response needs exactly one record for every case, including untouched records')
    result = {}
    for r in records:
        c = cases[r['case_id']]
        if r['status'] not in ('not_started', 'in_progress', 'complete'):
            raise ValueError('Invalid response status')
        if r['decision'] is not None and r['decision'] not in DECISIONS:
            raise ValueError('Invalid candidate decision')
        if r['confidence'] not in (None, 'low', 'medium', 'high') or not isinstance(r['reason'], str):
            raise ValueError('Invalid confidence/reason')
        if len(r['issues']) != len(set(r['issues'])) or any(v not in ISSUES for v in r['issues']):
            raise ValueError('Invalid reason/issue codes')
        allowed = {f['frame_id'] for frames in c['frames'].values() for f in frames}
        evidence = r['evidence_frame_ids']
        if len(evidence) != len(set(evidence)) or not set(evidence).issubset(allowed):
            raise ValueError('Invalid evidence frame IDs')
        active = number(r['active_seconds'])
        summed = 0
        previous_end = None
        for interval in r['intervals']:
            start, end = stamp(interval['started_at']), stamp(interval['ended_at'])
            elapsed, iactive = number(interval['elapsed_seconds']), number(interval['active_seconds'])
            if end < start or abs(elapsed - (end-start)) > .1 or iactive > elapsed + .1:
                raise ValueError('Inconsistent start/stop/active interval')
            if previous_end is not None and start < previous_end:
                raise ValueError('Overlapping intervals would double-count effort')
            previous_end = end
            summed += iactive
        if abs(summed - active) > .1:
            raise ValueError('Active total does not equal recorded intervals')
        if r['status'] == 'not_started':
            if r['decision'] is not None or evidence or r['reason'] or r['intervals'] or active or r['first_started_at'] or r['completed_at']:
                raise ValueError('Untouched cases must not contain fabricated annotations/times')
        if r['status'] == 'complete':
            query_frames = {f['frame_id'] for f in c['frames'].get('query', [])}
            if not c['candidate_available'] or not r['decision'] or not r['reason'].strip() or not r['confidence'] or not set(evidence) & query_frames:
                raise ValueError('Completed record lacks decision/reason/query-date evidence')
            if not r['intervals'] or active <= 0 or stamp(r['completed_at']) < stamp(r['first_started_at']):
                raise ValueError('Completed record has no plausible timed work')
            if r['decision'] == 'unobservable' and not set(r['issues']) & {'cloud', 'resolution', 'dates', 'examples', 'other'}:
                raise ValueError('Unobservable decision requires its observation limitation')
        result[r['case_id']] = r
    return result

def rate(a, b):
    return a / b if b else None

def agreement(a, b, ids):
    paired = [i for i in ids if a.get(i, {}).get('status') == 'complete' and b.get(i, {}).get('status') == 'complete']
    exact = sum(a[i]['decision'] == b[i]['decision'] for i in paired)
    ca, cb = Counter(a[i]['decision'] for i in paired), Counter(b[i]['decision'] for i in paired)
    pe = sum(ca[v]*cb[v] for v in DECISIONS)/(len(paired)**2) if paired else None
    po = rate(exact, len(paired))
    kappa = (po-pe)/(1-pe) if pe is not None and pe < 1 else None
    return {'eligible_cases':len(ids), 'both_completed':len(paired), 'exact_matches':exact,
            'exact_agreement':po, 'cohen_kappa':kappa, 'kappa_note':'Descriptive only; unstable at n=20 and undefined for degenerate marginals.',
            'a_decision_counts':dict(ca), 'b_decision_counts':dict(cb),
            'confusion_counts':dict(Counter(a[i]['decision']+' / '+b[i]['decision'] for i in paired)),
            'both_decisive_target_or_counter':sum(a[i]['decision'] in ('target','counterexample') and b[i]['decision'] in ('target','counterexample') for i in paired)}

def score(manifest, references, response_a=None, response_b=None, allow_synthetic=False):
    if references['package_id'] != manifest['package_id']:
        raise ValueError('Private availability mapping belongs to another package')
    ids = [c['case_id'] for c in manifest['cases']]
    refs = {r['case_id']:r['reference_category'] for r in references['cases']}
    if len(refs) != len(ids) or set(refs) != set(ids):
        raise ValueError('Reference availability mapping must cover exact case cohort')
    a = validate(response_a, manifest, 'A', allow_synthetic) if response_a else {}
    b = validate(response_b, manifest, 'B', allow_synthetic) if response_b else {}
    if response_a and response_b and response_a['annotator_id'] == response_b['annotator_id']:
        raise ValueError('A and B must be independently assigned distinct reviewers')
    synthetic = bool(manifest.get('synthetic_fixture_package')) or any(r and r.get('synthetic_fixture') for r in (response_a, response_b))
    informative = [i for i in ids if refs[i] in ('target','counterexample')]
    all_agree, info_agree = agreement(a,b,ids), agreement(a,b,informative)
    timing = {}
    for role, records in [('A',a),('B',b)]:
        active = [r['active_seconds'] for r in records.values() if r['status']=='complete']
        timing[role] = {'complete_count':len(active), 'total_active_seconds':sum(active),
            'median_active_seconds':statistics.median(active) if active else None,
            'p90_active_seconds':sorted(active)[math.ceil(.9*len(active))-1] if active else None,
            'started_but_incomplete':sum(r['status']=='in_progress' for r in records.values())}
    counts = Counter(refs.values())
    coverage_ok = len(informative)>=8 and counts['target']>=3 and counts['counterexample']>=3
    enough_decisive = bool(informative) and info_agree['both_decisive_target_or_counter']>=math.ceil(.75*len(informative))
    enough_agreement = bool(informative) and info_agree['exact_matches']>=math.ceil(.75*len(informative))
    diverse_decisions = all(Counter(r[i]['decision'] for i in informative if i in r and r[i]['status']=='complete')[v]>=2
                            for r in (a,b) for v in ('target','counterexample'))
    timing_ok = all(t['complete_count']==20 and t['median_active_seconds']<=600 and t['p90_active_seconds']<=900 for t in timing.values())
    full = all_agree['both_completed']==20
    gate = full and coverage_ok and enough_decisive and enough_agreement and diverse_decisions and timing_ok
    if synthetic:
        status = 'synthetic_fixture_only_not_human_results'
    elif not coverage_ok:
        status = 'pilot_design_insufficient_informative_cases'
    elif not full:
        status = 'awaiting_two_complete_independent_reviews'
    elif gate:
        status = 'provisional_candidate_identification_feasible_plan_larger_pilot'
    else:
        status = 'revise_task_or_rendering_before_expanding_annotations'
    return {'schema_version':VERSION, 'package_id':manifest['package_id'], 'status':status,
        'actual_human_complete_records':0 if synthetic else sum(t['complete_count'] for t in timing.values()),
        'human_collection_status': 'synthetic_only' if synthetic else 'both_complete' if full else 'partial' if any(t['complete_count'] for t in timing.values()) else 'not_collected',
        'insufficient_design_is_not_a_human_failure':True,
        'two_independent_reviewers_procedurally_verified':False,
        'source_reference_is_expert_gold':False, 'reference_availability_counts':dict(counts),
        'overall_agreement':all_agree, 'informative_reference_stratum_agreement':info_agree,
        'by_reference_stratum':{v:agreement(a,b,[i for i in ids if refs[i]==v]) for v in sorted(set(refs.values()))},
        'timing':timing,
        'operational_gate':{'provisional_unvalidated_assumptions':True, 'copied_from_D1':False,
            'required_complete_per_reviewer':20, 'minimum_target_reference':3, 'minimum_counter_reference':3,
            'minimum_target_plus_counter_reference':8, 'informative_decisive_fraction_minimum':.75,
            'informative_exact_agreement_fraction_minimum':.75, 'median_active_seconds_maximum':600,
            'p90_active_seconds_maximum':900, 'complete_cohort':full, 'coverage_sufficient':coverage_ok,
            'informative_decisions_sufficient':enough_decisive, 'informative_agreement_sufficient':enough_agreement,
            'minimum_each_target_and_counter_response_per_reviewer_in_informative_stratum':2,
            'nondegenerate_decision_counts':diverse_decisions,
            'timing_within_planning_limit':timing_ok, 'passed':gate and not synthetic},
        'interpretation':'Candidate discrimination and labor-time feasibility only; not new full-mask annotation, corrected-model label efficiency, classification accuracy against expert gold, or CVPR evidence.',
        'no_silent_imputation_of_absent_reviewers_or_cases':True,
        'overall_agreement_alone_never_passes_gate':True,
        'budget_change':None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--private-reference',required=True,type=Path)
    p.add_argument('--a',type=Path);p.add_argument('--b',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--allow-synthetic',action='store_true',help='Test fixtures only; result can never pass human gate')
    args=p.parse_args()
    report=score(json.loads((args.package/'manifest.json').read_text()),json.loads(args.private_reference.read_text()),
        json.loads(args.a.read_text()) if args.a else None,json.loads(args.b.read_text()) if args.b else None,args.allow_synthetic)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'actual_human_complete_records':report['actual_human_complete_records']},ensure_ascii=False))

if __name__=='__main__':
    main()
