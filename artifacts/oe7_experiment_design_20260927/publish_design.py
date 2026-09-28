# -*- coding: utf-8 -*-
"""Publish only the reviewed OE7 planning artifacts, preserving prior work."""
from pathlib import Path
import hashlib
import json
import re

SRC = Path('/private/tmp/oe7_experiment_design_20260927')
DST = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
DOC = 'docs/OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md'
CFG = 'config/oe7_experiment_design_v0_20260927.json'
ART = 'artifacts/oe7_experiment_design_20260927'
checks = []

def check(name, ok):
    checks.append({'name': name, 'passed': bool(ok)})
    if not ok:
        raise ValueError(name)

d = json.loads((SRC / Path(CFG).name).read_text())
prior = json.loads((DST / d['prior_execution_status']).read_text())
budget = json.loads((DST / d['budget_reference']).read_text())
check('prior_actual_inputs', d['actual_state_at_design']['engineering_inputs'] == prior['actual_native_engineering_cases'] == 2)
check('prior_training_episodes', d['actual_state_at_design']['transfer_training_episodes'] == prior['actual_transfer_training_episodes'] == 0)
check('candidate_counts', (d['actual_state_at_design']['candidate_train'], d['actual_state_at_design']['candidate_dev']) == (prior['metadata_candidates']['train'], prior['metadata_candidates']['development']))
check('not_ready_for_training', not d['actual_state_at_design']['comparative_gpu_training_ready'] and not prior['ready_for_comparative_gpu_training'])
h = d['human_plan']
check('budget_total_unchanged', h['total_krw'] == budget['total_budget'] == 1000000)
check('budget_components', [h[x] for x in ['annotator_krw','expert_krw','api_cap_krw','contingency_krw']] == [b['amount'] for b in budget['budget_items']])
check('budget_sum', sum(h[x] for x in ['annotator_krw','expert_krw','api_cap_krw','contingency_krw']) == h['total_krw'])
check('K_schedule', d['episodes']['budgets_pairs'] == [1,2,4,8])
check('matrix_complete', {(a['representation'],a['selection']) for a in d['matrix']} == {(r,s) for r in ['R0','R1'] for s in ['S0','S1']})
check('stages_unique', len({s['id'] for s in d['stages']}) == len(d['stages']) == 6)
check('no_new_execution_reported', all(v == 0 for v in d['execution_this_update'].values()))
check('physical_gpu1', d['resources']['physical_gpu'] == 1)
check('not_confirmatory', d['metrics']['confirmatory_primary'] is None and d['metrics']['confirmatory_sample_size'] is None and d['resources']['run_commands'] is None)
check('train_dev_disjoint', not set(d['data']['train_parents']) & set(d['data']['dev_parents']))

files = {
    SRC / Path(DOC).name: DOC,
    SRC / Path(CFG).name: CFG,
}
for name in ['statistical_review.md', 'method_review.md', 'execution_review.md', 'final_statistical_review.md', 'final_method_review.md', 'review_response.md', 'publish_design.py']:
    files[SRC / name] = f'{ART}/{name}'
for source, relative in files.items():
    target = DST / relative
    check('source_present:' + source.name, source.is_file())
    if target.exists():
        check('existing_artifact_identical:' + relative, target.read_bytes() == source.read_bytes())

text = (SRC / Path(DOC).name).read_text()
for link in re.findall(r'\]\(([^)]+)\)', text):
    if '://' in link or link.startswith('#'):
        continue
    target = (DST / DOC).parent / link.split('#')[0]
    planned = {str((DST / rel).resolve()) for rel in files.values()}
    check('document_link:' + link, target.exists() or str(target.resolve()) in planned)

for source, relative in files.items():
    target = DST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(source.read_bytes())

marker = '**9/27 OE7 다음 실험 설계 v6'
entry = '> **9/27 OE7 다음 실험 설계 v6 — [실험 순서·비교군·비용·판정](docs/OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md).** 세 독립 검토와 두 후속 검토를 반영해 입력80개 준비→실제 trainer→강한 기준선→표현×관측 선택2×2→독립 평가로 구체화했다. 교정 쌍1/2/4/8, 같은 support/query, 숨은 관측 비용, 실제 사람 시간, EO/새 reader 재사용을 분리한다. 현재 두 입력·80메타후보·제안 학습0 상태는 그대로이며, 이번 갱신은 설계/문서만이다. 신규성·본 평가 사전등록은 미확정. 이전 내용은 당시 이력이다.\n\n'
for name in ['GOAL.md','RESTART_HERE.md','README.md']:
    p = DST / name
    old = p.read_text()
    if marker not in old:
        first, sep, rest = old.partition('\n')
        p.write_text(first + sep + '\n' + entry + rest.lstrip('\n'))

goal = DST / 'GOAL.md'
result_marker = '### 2026-09-27 — OE7 실험 설계 v6 결과'
if result_marker not in goal.read_text():
    with goal.open('a', encoding='utf-8') as f:
        f.write('\n\n' + result_marker + '\n\n- 사용자 요청에 따라 다음 실행부터 본 평가까지 계획을 문서와 JSON으로 정리했다. 실제 상태2입력/64train+16dev메타후보/제안episode학습0을 유지했다. 표준 pair metric learning과 신규성 후보를 구분하고 동일정보 B0~B4 및 표현×관측2×2, 동일 최종 근거/동일 dense 출력 replay를 명시했다.\n- 세 독립 검토와 통계/방법 후속 검토의 원문·대응을 artifacts/oe7_experiment_design_20260927/에 보존했다. 교정1쌍·K0정의·공통K cohort·source bank 분리·미개봉 영상 접근 제한·full-window 재인코딩·test120 외 support제작비를 보강했다.\n- 검증은 기존 상태/예산과 새 JSON·상대링크·파일 해시의 정합성이다. 새 모델 성능 테스트나 GPU 학습이 아니다. 전체100만원 예산과 과거 사전등록·실패 기록·서버코드는 유지. 새 서버전송/자료다운로드/API/외부 연락 없음. 현재 GPU 점유를 새로 조회한 것은 아니다.\n- 다음 한 작업은 manifest 기반80개 raw/label 준비기와 품질/episode 계약이다. 그 뒤 실제 trainer 저장복원과 GPU1 처리량·overfit, 충분히 학습한 일반 기준선. 새 공개 PR 후보 없음.\n')

study = DST / 'STUDY.md'
study_marker = '### 2026-09-27 — 관측을 적게 골라도 계산이 더 많을 수 있다'
if study_marker not in study.read_text():
    with study.open('a',encoding='utf-8') as f:
        f.write('\n\n' + study_marker + '\n\n실험설계를 검토하며,2→3→4장의 관측 선택이 전체 temporal window를 재인코딩하면4장의 비용이 아니라2+3+4장 규모의 처리와 attention 비용을 낸다는 점을 확인했다. 후보 전체를 한 번에 읽는 baseline이 더 저렴할 수도 있어 실제 비용–품질 곡선에 포함해야 한다. 미개봉 후보 embedding을 먼저 계산한 비용, support 인코딩과 원자료 전체 CPU 로딩도 숨기면 안 된다. 네 칸의 표현×선택 비교는 같은 두 encoder checkpoint를 네 조건에서 평가할 수 있으며 네 번의 학습과 같지 않다.\n\n확인 질문: 읽은 날짜 수가 더 적은 모델을 계산 효율이 높다고 주장하려면 어떤 비용과 대조군을 추가로 확인해야 하는가?\n')

manifest = []
for rel in list(files.values()) + ['GOAL.md','RESTART_HERE.md','README.md','STUDY.md']:
    b = (DST/rel).read_bytes()
    manifest.append({'path': rel, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
out = {'scope':'planning_artifacts_only_not_model_validation','checks':checks,'passed':all(c['passed'] for c in checks),'files':manifest}
(DST/ART/'validation_and_publication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':out['passed'],'checks':len(checks),'files':len(manifest),'plan':str(DST/DOC)},ensure_ascii=False))
