#!/usr/bin/env python3
"""Publish independently audited E3 aggregates and exact historical question cases."""
import argparse, hashlib, json, shutil
from datetime import datetime,timezone
from pathlib import Path

ARMS=['real','earlier_only','later_only','repeat_earlier','repeat_later','no_delta','reverse']
LABELS=['원 입력','이전 슬롯만','이후 슬롯만','이전 슬롯 반복','이후 슬롯 반복','차이 토큰 제거','슬롯 역순']
def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(v) for v in p.read_text().splitlines() if v.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,s):
    if not ok: raise ValueError(s)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','e3','audit','base','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=a.repo.resolve();e=a.e3.resolve();base=a.base.resolve();out=a.out.resolve()
    require(not out.exists(),'Preserve earlier snapshots; new destination required')
    audit=read(a.audit);s=read(e/'scores.json');m=read(e/'manifest.json')
    require(audit['consistent'] and s['valid'] and read(e/'status.json')['status']=='completed','Verified completed E3 required')
    require(audit['verdict']==s['verdict'] and audit['n_answers']==3777,'Audit/result mismatch')
    for path,h in audit['hashes_verified'].items():require(sha(Path(path))==h,'Verified artifact changed: '+path)
    require(sha(r/'code/audit_e3_pair_results_v0.py')==audit['audit_code_sha256'],'Independent auditor changed')
    items={v['id']:v for v in rows(e/'items.jsonl')};answers={}
    for seed in [1,2,3]:
        for arm in ARMS:
            for v in rows(e/f'answers_seed{seed}_{arm}.jsonl'):
                key=(v['id'],arm,str(seed));require(key not in answers,'Duplicate E3 answer');answers[key]=v
    require(sha(base/'reader_cases.json')==read(base/'browser_checks.json')['source_sha256']['reader_cases.json'],
            'Previously verified case snapshot changed')
    cases=read(base/'reader_cases.json');linked=[]
    for case in cases['cases'].values():
        require('e3_control' not in case,'E3 cases already present')
        key=case['question_id']
        if key not in items:continue
        it=items[key]
        require(it['phen']=='flood' and it['kind'] in ['pos','hard_neg'],'Unexpected catalog-linked source item')
        require(it['tile']==case['tile'] and it['answer']==case['reference_label'] and it['dates']==case['dates'] and it['slots']==case['slots'],'Question identity differs')
        preds={}
        for arm in it['allowed_arms']:
            preds[arm]={}
            for seed in ['1','2','3']:
                v=answers[(key,arm,seed)]
                require(v['source_gold']==case['reference_label'] and v['transformed_gold'] is None,'Source-label role changed')
                preds[arm][seed]=v['parsed']
            if arm=='real':require(preds[arm]==case['reader_predictions'],'Original E2 decisions no longer reproduced for case')
        case['e3_control']={'run_id':'E3-PD-v1','question_id':key,'source_role':'historical_model_output',
             'source_gold':case['reference_label'],'predictions':preds,
             'source_sha256':{'manifest':sha(e/'manifest.json'),'scores':sha(e/'scores.json'),
                              'independent_audit':sha(a.audit),'items':m['items_sha256'],
                              **{f'{arm}_seed{seed}':sha(e/f'answers_seed{seed}_{arm}.jsonl') for arm in it['allowed_arms'] for seed in [1,2,3]}}}
        linked.append(key)
    expected=sorted(k for k,v in items.items() if v['phen']=='flood' and v['kind'] in ['pos','hard_neg'])
    require(sorted(linked)==expected and len(linked)==108,'Expected 57 positive and 51 hard-negative cases')
    verdicts={'single_second_view_sufficient_under_intervention':'이후 슬롯만으로도 원 라벨 일치 유지',
              'history_sensitive_under_intervention':'시점 제거·반복에 민감',
              'mixed_or_inconclusive':'혼합 결과 · 결론 유보'}
    next_steps={'single_second_view_sufficient_under_intervention':'같은 학습 예산의 단일 시점 모델과 비교하고, 실제 변화가 검수된 문항을 확보한 뒤 시간·기억 연구로 진행합니다.',
                'history_sensitive_under_intervention':'E4에서 차이 토큰만 남기는 통제와 차이 부호만 바꾸는 통제로, 차이 정보 의존성과 입력 형식 변화의 영향을 구분합니다.',
                'mixed_or_inconclusive':'사건·seed별 차이와 검정력을 검토해 후속 실험 하나를 고릅니다. 이번 결과에 맞춰 판정 기준을 조정하지 않습니다.'}
    metric_rows=[[label]+[round(s['metrics'][str(seed)]['flood']['paired'][arm]['macro_ba'],4) for seed in [1,2,3]] for arm,label in zip(ARMS,LABELS)]
    run={'id':'E3-PD','title':'고정 VLM의 두 시점 의존성','status':'완료 · '+verdicts[s['verdict']],
      'findings':['209문항·3,777응답을 완료했습니다. 원 입력은 세 seed·두 현상 모두 기존 답변과 100% 일치했습니다.',
                  '별도 코드로 문항 완결성·입력 계보·사건별 점수·재표집 구간·등록 판정을 재계산했습니다.',
                  '등록 판정: '+s['verdict'],
                  '같은 seed에서 두 단일 슬롯 조건을 함께 만족한 seed: 충분성 '+str(s['sufficient_seeds'])+', 민감성 '+str(s['history_sensitive_seeds'])+'.',
                  'GPU0가 비어 운영상 GPU만 변경했습니다. v0 준비물과 모델·표본·통계 규칙은 보존했습니다.'],
      'limitations':['노출된 평가 자료의 입력 통제입니다. 새 사건 일반화·기억 효과·물리적 변화 이해의 증명이 아닙니다.',
                     '변환 입력의 새 정답은 없습니다. 아래 BA는 원 질문 라벨과의 일치이며 변환 장면의 정확도가 아닙니다.',
                     '홍수 주 분석은 원래 침수 타일의 yes/no 문항 쌍입니다. 실제 비침수 타일 51개의 오탐은 별도로 보고합니다.',
                     '입력 제거·반복의 분포 변화와 합성 시간 정보가 남습니다. 두 시점 의미를 이해한다고 단정하지 않습니다.'],
      'next_step':next_steps[s['verdict']],
      'metrics':{'columns':['입력 통제','Seed 1 BA','Seed 2 BA','Seed 3 BA'],'rows':metric_rows,
                 'caption':'홍수 10사건·57타일의 원 yes/no 문항 쌍에 대한 사건평균 BA. 단일 슬롯 조건의 판정은 두 조건의 사건별 차이 구간을 함께 검사합니다.'}}
    zero_arms={'earlier_only','later_only','repeat_earlier','repeat_later','no_delta'}
    zero_rows=[v for (key,arm,seed),v in answers.items() if arm in zero_arms]
    no_count=sum(v['parsed']=='no' for v in zero_rows)
    run['findings'].insert(1,f'차이 토큰이 0인 다섯 조건의 {len(zero_rows):,}개 답 중 {no_count:,}개가 no였습니다. 두 관측을 모두 유지한 차이 토큰 제거 조건에서도 구분이 사라졌습니다.')
    run['limitations'].insert(0,'등록된 민감성 판정은 유지하지만, 과거 관측의 필요성을 분리해서 입증하지 못했습니다. 다섯 조건이 공통으로 차이 토큰을 0으로 만든다는 대안 설명이 있습니다.')
    # Publication creates a new artifact; prior HTTP/browser reports stay with their original version.
    shutil.copytree(base,out,ignore=shutil.ignore_patterns('*checks.json'))
    shutil.copyfile(r/'code/eo_evidence_search_v0.html',out/'index.html')
    stamp=datetime.now(timezone.utc).isoformat();cases['checked_at']=stamp
    cases['e3_control_provenance']={'linked_cases':len(linked),'question_ids':sorted(linked),'builder_sha256':sha(Path(__file__)),
                                   'source_role':'historical_model_output','audit_sha256':sha(a.audit)}
    (out/'reader_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    data=read(out/'research_runs.json');data['runs']=[run if v['id']=='E3-PD' else v for v in data['runs']];data['checked_at']=stamp
    (out/'research_runs.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    source=read(out/'research_sources.json');source['checked_at']=stamp
    for path in [e/'manifest.json',e/'scores.json',e/'amendment.json',a.audit.resolve()]:source['sources'][str(path.relative_to(r))]=sha(path)
    (out/'research_sources.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps({'out':str(out),'e3_cases':len(linked),'verdict':s['verdict']}))

if __name__=='__main__':main()
