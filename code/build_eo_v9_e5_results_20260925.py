#!/usr/bin/env python3
"""Build a new immutable evidence snapshot from independently audited E5 JSON.

Only one E5 research card changes. No model, prediction-file reads, network,
server operation, new individual reader answers, or source-date repair.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

V8_SHA = '3b5bb190ac62d80501ba209d7a9d86b493a42803da9ff14373bdc90fc58e079d'
PLAN_SHA = 'fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
PARENT_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
AUDITOR_SHA = 'a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825'
REQUIRED_NAMES = ('manifest.json', 'prereg.json', 'status.json', 'scores.json',
                  'training_summary.json', 'inference_completed.json')
V8_EXTRA = {'v8_build_manifest.json', 'browser_checks_v8.json', 'http_readback_checks_v8.json'}
CHANGED = {'research_runs.json', 'research_sources.json'}
EVALUATIONS = ('full/native', 'pair/native', 'later/native', 'delta/native', 'full/full_no_delta')
VERDICTS = {'pair_preserves_source_discrimination_at_equal_budget',
            'explicit_difference_helps_at_this_budget', 'mixed_or_inconclusive'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def child(root, name):
    p = Path(name)
    require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts, 'Unsafe relative source path')
    target = root / p
    require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root),
            'Missing or escaping snapshot source: ' + name)
    return target


def pin(path, expected, tracking):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid source digest')
    require(sha(path) == expected, 'Source hash mismatch: ' + str(path))
    tracking[str(path)] = expected


def aware(value):
    require(isinstance(value, str) and datetime.fromisoformat(value.replace('Z', '+00:00')).tzinfo is not None,
            'Timezone-aware source timestamp required')
    return value


def number(value, low, high):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, 'Invalid numeric result')
    return value


def load_results(e5, audit_path, tracking):
    """Gate six small replicas against original absolute paths in independent audit."""
    audit_digest = sha(audit_path)
    pin(audit_path, audit_digest, tracking)
    audit = read(audit_path)
    require(audit.get('schema') == 'e5-independent-result-audit-v0' and audit.get('consistent') is True
            and audit.get('checkpoint_tensors_loaded_and_checked_on_cpu') is True,
            'Complete independent CPU tensor audit required')
    require(audit.get('audit_code_sha256') == AUDITOR_SHA, 'Unexpected independent auditor source')
    require(tuple(audit.get(k) for k in ('n_models', 'n_steps', 'n_training_exposures', 'n_answers', 'primary_n', 'primary_events'))
            == (12, 19080, 152424, 26325, 902, 8), 'Independent audit population/budget differs')
    aware(audit['checked_at'])
    original = Path(audit['artifact'])
    require(original.is_absolute() and '..' not in original.parts and original.name == 'e5_equal_budget_v0',
            'Unexpected original E5 artifact path')
    require(isinstance(audit.get('hashes_verified'), dict), 'Missing independent source pins')
    require(not (e5 / 'failure.json').exists(), 'E5 failure artifact present')
    values, original_pins = {}, {}
    for name in REQUIRED_NAMES:
        key = str(original / name)
        expected = audit['hashes_verified'].get(key)
        pin(child(e5, name), expected, tracking)
        original_pins[key] = expected
        values[name] = read(e5 / name)
    manifest, plan, status, scores, summary, inference = [values[name] for name in REQUIRED_NAMES]
    require(original_pins[str(original / 'manifest.json')] == PARENT_SHA
            and original_pins[str(original / 'prereg.json')] == PLAN_SHA, 'Fixed prepared manifest/plan differ')
    require(original == Path(plan['root']) / plan['output_directory'], 'Audit original artifact differs from pinned plan')
    require(manifest['schema'] == 'e5-equal-budget-prepared-v0'
            and manifest['files_sha256']['prereg.json'] == PLAN_SHA, 'Prepared E5 contract differs')
    require(status['status'] == 'completed' and status['n_models'] == 12 and status['n_rows'] == 26325,
            'E5 training/inference is not complete')
    aware(status['at'])
    require(scores['schema'] == 'e5-equal-budget-scores-v0' and scores['valid'] is True,
            'E5 scientific scoring invalid')
    require(scores['verdict'] in VERDICTS and scores['verdict'] == status['verdict'] == audit['verdict'],
            'Registered verdict disagreement')
    require(scores['coverage'] == {'expected': 26325, 'received': 26325, 'n_items': 5989,
                                   'n_train': 4234, 'n_test': 1755, 'n_primary': 902}, 'Scored population differs')
    require(summary['schema'] == 'e5-training-summary-v0'
            and inference['schema'] == 'e5-training-inference-completed-v0', 'Training/inference schema differs')
    require(summary['n_models'] == inference['n_models'] == 12
            and inference['n_rows'] == 26325 and inference['n_prompts'] == 5989
            and summary['updates_per_model'] == 1590 and summary['exposures_per_model'] == 12702,
            'Training/inference population or exposure differs')
    require(summary['manifest_sha256'] == scores['manifest_sha256'] == PARENT_SHA
            and summary['code_snapshot_sha256'] == manifest['code_snapshot_sha256'], 'Final source freeze differs')
    require(scores['training_summary_sha256'] == original_pins[str(original / 'training_summary.json')],
            'Scores use another training summary')
    require(summary['models'] == inference['models'] and len(summary['models']) == 12, 'Final model records differ')
    identities = []
    for model in summary['models']:
        require(type(model['seed']) is int and model['updates'] == 1590 and model['exposures'] == 12702,
                'Completed model budget differs')
        identities.append((model['seed'], model['model_arm']))
    require(len(set(identities)) == 12 and set(identities) == {(s, a) for s in (1, 2, 3)
            for a in ('full', 'pair', 'later', 'delta')}, 'Twelve completed model identities differ')
    for name, key in (('predictions.jsonl', 'predictions_sha256'), ('prompts.jsonl', 'prompts_sha256'),
                      ('runtime_environment.json', 'runtime_sha256')):
        digest = audit['hashes_verified'].get(str(original / name))
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Missing audited generated-file digest')
        require(summary[key] == inference[key] == digest, 'Audited generated-file identity differs')
        if key in scores:
            require(scores[key] == digest, 'Scores refer to different generated bytes')
        original_pins[str(original / name)] = digest
    require(scores['predictions_sha256'] == original_pins[str(original / 'predictions.jsonl')], 'Prediction lineage absent')
    require(set(scores['metrics']) == set(audit['primary_metrics']) == {'1', '2', '3'}, 'Primary seed coverage differs')
    require({seed: scores['metrics'][seed]['primary_same_prompt'] for seed in ('1', '2', '3')}
            == audit['primary_metrics'], 'Saved versus independently recalculated primary metrics differ')
    require(scores['seed_decisions'] == audit['seed_decisions'], 'Seed decisions differ from independent audit')
    for seed in ('1', '2', '3'):
        primary = scores['metrics'][seed]['primary_same_prompt']
        require(set(primary['evaluations']) == set(EVALUATIONS), 'Primary arm coverage differs')
        for value in primary['evaluations'].values():
            require(value['n_events'] == 8 and value['n_items'] == 902 and len(value['events']) == 8,
                    'Primary metric support differs')
            number(value['macro_ba'], 0, 1)
        contrast = primary['contrasts']['pair_minus_full']
        require(contrast['left'] == 'pair/native' and contrast['right'] == 'full/native'
                and len(contrast['event_deltas']) == 8, 'Primary contrast direction/support differs')
        number(contrast['delta'], -1, 1)
        bounds = contrast['ci95_delta']
        require(isinstance(bounds, list) and len(bounds) == 2, 'Primary interval missing')
        require(number(bounds[0], -1, 1) <= number(bounds[1], -1, 1), 'Reversed primary interval')
    return values, audit, original_pins


def e5_card(scores, audit, completed_at):
    primary = {seed: scores['metrics'][seed]['primary_same_prompt'] for seed in ('1', '2', '3')}
    labels = [('full/native', 'full · A, B, D 학습'), ('pair/native', 'pair · A, B 학습'),
              ('later/native', 'later · B 학습'), ('delta/native', 'delta · D 학습'),
              ('full/full_no_delta', 'full_no_delta · 학습된 full에서 D 제외')]
    rows = [[label] + [round(primary[s]['evaluations'][key]['macro_ba'], 4) for s in ('1', '2', '3')]
            for key, label in labels]
    contrasts = [primary[s]['contrasts']['pair_minus_full'] for s in ('1', '2', '3')]
    rows.append(['주대조 pair − full · BA 차이'] + [round(c['delta'], 4) for c in contrasts])
    rows.append(['주대조 · 사건 bootstrap 95% 구간'] + [f"[{c['ci95_delta'][0]:.4f}, {c['ci95_delta'][1]:.4f}]" for c in contrasts])
    return {'id': 'E5-EB', 'title': '같은 학습 예산에서 차이 입력 비교', 'status': '완료 · 독립 감사 통과',
        'findings': [f"실행 완료 기록 {completed_at} · 독립 감사 {audit['checked_at']}. 저장된 결과이며 실시간 상태가 아닙니다.",
            '4개 입력 형식 × 3 seed = 12개 모델. 모델당 1,590 updates·12,702문항 노출, 전체 평가 응답 26,325개를 독립 검산했습니다.',
            '등록 판정: ' + scores['verdict'],
            '주분석은 같은 사건·질문 날짜·슬롯의 902문항(양성 445, hard-negative 457), 8사건입니다. 표에 세 seed와 모든 사전 지정 조건을 표시합니다.'],
        'metrics': {'columns': ['입력 / 측정값', 'Seed 1', 'Seed 2', 'Seed 3'], 'rows': rows,
            'caption': '원문항 참조 라벨에 대한 BA(0–1). 같은 조건 내 클래스 균형 → 사건 내 조건 균등 → 8사건 균등 평균입니다. 주대조는 별도 학습한 pair−full이며 95% 구간은 seed별 사건 bootstrap 5,000회입니다. full_no_delta는 별도 학습 모델이 아닌 full의 평가 시 입력 제거입니다. 반올림 전 값과 출처는 감사 스냅샷에 보존합니다.'},
        'limitations': ['이미 평가에 노출된 개발 자료의 후속 진단입니다. 등록 판정은 새로운 사건 일반화나 CVPR 수준의 기여를 자동으로 입증하지 않습니다.',
            '동일한 학습 문항 노출을 맞췄습니다. FLOPs·표현 난이도·최적화 난이도까지 같다는 뜻은 아닙니다.',
            'later·delta·full_no_delta는 기술통계입니다. D=B−A는 두 시점의 관측으로 계산되므로 delta 성능을 과거 정보 불필요의 증거로 읽을 수 없습니다.',
            '기존 합성 날짜와 EO 캐시를 유지했습니다. T0의 공개 날짜 후보를 입력에 소급 적용하지 않았으며, 실제 취득시각·변화 발생시점·피해 정답은 미검증입니다.',
            '이 결과는 원 라벨 구분을 측정합니다. 기억 필요성, 물리 변화·피해·인과 효과, 개별 타일의 새로운 판정을 입증하지 않습니다.'],
        'next_step': '등록된 주대조와 사건별 편차를 검토하고, 독립 사건과 검증된 날짜·라벨에서 확인할 후속 비교를 정합니다.'}


def build(base, e5, audit_path, out):
    base, e5, audit_path, out = [Path(p).resolve() for p in (base, e5, audit_path, out)]
    require(not out.exists(), 'New output required; no overwrite')
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in (base, e5)), 'Output overlaps input tree')
    tracking = {}
    pin(base / 'v8_build_manifest.json', V8_SHA, tracking)
    v8 = read(base / 'v8_build_manifest.json')
    require(v8['schema'] == 'eo-v8-source-dates-build-v0', 'Base is not the frozen v8 snapshot')
    require(not any(p.is_symlink() for p in base.rglob('*')), 'Symlink in base snapshot')
    actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    require(actual == set(v8['output_files_sha256']) | V8_EXTRA, 'Unexpected base snapshot files')
    for name, digest in v8['output_files_sha256'].items():
        pin(child(base, name), digest, tracking)
    base_hashes = {name: sha(child(base, name)) for name in sorted(actual)}
    values, audit, original_pins = load_results(e5, audit_path, tracking)
    old_research = read(base / 'research_runs.json'); old_sources = read(base / 'research_sources.json')
    require(old_research['schema_version'] == 'eo_research_snapshot_v1', 'Research schema differs')
    require(len({r['id'] for r in old_research['runs']}) == len(old_research['runs']), 'Duplicate research ID')
    matches = [i for i, r in enumerate(old_research['runs']) if r['id'] == 'E5-EB']
    require(len(matches) == 1, 'Require exactly one prior E5 card')
    position = matches[0]; previous = old_research['runs'][position]
    require(previous['status'] == '학습 진행 스냅샷 · 결과 없음', 'Prior E5 is not the fixed pending snapshot')
    require('e5_results_v9' not in old_sources, 'E5 result provenance already present')
    now = datetime.now(timezone.utc).isoformat()
    research, sources = copy.deepcopy(old_research), copy.deepcopy(old_sources)
    research['checked_at'] = now
    research['runs'][position] = e5_card(values['scores.json'], audit, values['status.json']['at'])
    require([r for r in research['runs'] if r['id'] != 'E5-EB'] == [r for r in old_research['runs'] if r['id'] != 'E5-EB'],
            'Unrelated research runs changed')
    sources['e5_results_v9'] = {'checked_at': now, 'source_role': 'independently_audited_historical_E5_aggregate',
        'original_artifact': audit['artifact'], 'local_replica': str(e5), 'registered_plan_sha256': PLAN_SHA,
        'prepared_manifest_sha256': PARENT_SHA, 'independent_audit_sha256': tracking[str(audit_path)],
        'audited_original_files_sha256': original_pins,
        'copied_files_sha256': {name: tracking[str(e5 / name)] for name in REQUIRED_NAMES},
        'registered_verdict': values['scores.json']['verdict'], 'n_models': 12, 'n_answers': 26325,
        'primary_n': 902, 'primary_events': 8, 'individual_reader_cases_modified': False,
        'prior_pending_record': 'e5_results_v9/prior_E5_card.json',
        'limits': ['The earlier v7 progress timestamp remains historical provenance; only the E5 research card is superseded.',
                   'Heavy model/prediction bytes are not recopied or re-audited by this presentation builder.',
                   'Inherited E2/E3/E4 individual reader outputs and original source dates remain unchanged.']}
    shutil.copytree(base, out)
    try:
        write(out / 'research_runs.json', research); write(out / 'research_sources.json', sources)
        folder = out / 'e5_results_v9'; folder.mkdir()
        for name in REQUIRED_NAMES:
            shutil.copyfile(e5 / name, folder / name)
            require(sha(folder / name) == tracking[str(e5 / name)], 'Result copy changed: ' + name)
        shutil.copyfile(audit_path, folder / 'independent_audit.json')
        require(sha(folder / 'independent_audit.json') == tracking[str(audit_path)], 'Audit copy changed')
        write(folder / 'prior_E5_card.json', previous)
        for name, digest in base_hashes.items():
            require(sha(base / name) == digest, 'Original v8 changed during build')
            if name not in CHANGED:
                require(sha(out / name) == digest, 'Inherited snapshot file changed: ' + name)
        for path, digest in tracking.items():
            require(sha(path) == digest, 'Pinned input changed during build')
        output_hashes = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
        report = {'schema': 'eo-v9-e5-results-build-v0', 'created_at': now, 'base': str(base), 'out': str(out),
            'registered_verdict': values['scores.json']['verdict'], 'replaced_research_run': 'E5-EB',
            'changed_existing_files': sorted(CHANGED), 'all_other_research_runs_unchanged': True,
            'catalog_meta_dates_reader_cases_html_previews_byte_identical': True,
            'source_hashes': tracking, 'base_files_sha256': base_hashes, 'output_files_sha256': output_hashes,
            'builder_sha256': sha(__file__), 'original_e5_artifact': audit['artifact'],
            'limits': ['Aggregate results only; no individual E5 predictions are added or substituted.',
                'Independent audit validation and file hashes are gates; model/tensor checks are not rerun in this builder.',
                'Inherited browser/HTTP records apply to their original snapshots. New v9 browser/HTTP validation remains necessary.',
                'No E6 execution status or performance is added.']}
        write(out / 'v9_build_manifest.json', report)
        return report
    except BaseException as error:
        write(out / 'v9_failure.json', {'error': str(error), 'partial_copy_not_valid': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('base', 'e5', 'audit', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.base, args.e5, args.audit, args.out)
    print(json.dumps({'out': result['out'], 'registered_verdict': result['registered_verdict'],
                      'catalog_meta_dates_reader_cases_html_previews_byte_identical': True}))
