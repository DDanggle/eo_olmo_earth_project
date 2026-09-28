#!/usr/bin/env python3
"""Append a verified C1 descriptive result to an existing local research snapshot."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); r=a.repo; out=a.out
    c1=r/'artifacts/c1_same_prompt_flood_v0_20260925'
    x=read(c1/'results.json')
    assert x['valid'] and read(c1/'status.json')['status']=='complete'
    audit_path=r/'artifacts/c1_independent_audit_20260925.json'
    audit=read(audit_path)
    assert audit['consistent'] and audit['controls_valid'] and not audit['errors']
    assert audit['auditor_sha256']==sha(r/'code/c1_independent_audit_20260925.py')
    for rel,digest in x['provenance']['frozen_input_sha256'].items():
        assert sha(r/rel)==digest, rel
    assert sha(c1/'analysis_plan.json')==x['provenance']['plan_sha256']
    assert sha(c1/'source.py')==x['provenance']['code_sha256']
    assert sha(c1/'selected_ids.json')==x['provenance']['selected_ids_sha256']
    s=x['subsets']['quality_symmetric']; ci=s['seed_mean_ci95_delta']
    run={'id':'C1','title':'같은 질문·날짜 안에서의 침수 구분',
         'status':'완료 · 기존 출력의 사후 진단',
         'findings':[
             '914타일의 라벨·유효 영역을 모델 답변과 별도로 감사했습니다. 양쪽 유효 영역 90% 이상을 적용해 침수 450개·비침수 457개를 고정했습니다.',
             f"두 라벨이 모두 있는 {s['n_events']}개 사건, 침수 {s['supported_n_pos']}개·비침수 {s['supported_n_hard_neg']}개를 비교했습니다. 지원이 없는 2개 사건의 양성 5개는 별도 기록했습니다.",
             '같은 문구로 묶이는 전체 914문항에서 blind의 답변 일관성을 확인했습니다. 3개 seed 모두 위반이 없었습니다.',
             '별도 코드로 표본·가중치·전체 사건 점수와 재표집 구간을 다시 계산했고 불일치는 없었습니다.',
             '사건별 seed 평균 reader−blind BA 차이 %.3f, 사건 bootstrap 95%% 구간 [%.3f, %.3f]. 이미 평가한 자료를 재집계한 탐색 결과입니다.'%(s['seed_mean_reader_minus_blind'],*ci)],
         'limitations':[
             '역사적 문구 원본은 보존되지 않아 현재 동결 자료에서 복원한 동일 문구 조건입니다.',
             '8개 사건 중 양성이 1개인 사건도 있습니다. 배경·취득 조건과 영구 수역 차이는 완전히 통제하지 못했습니다.',
             'EO 입력과 원 라벨의 연관성 근거입니다. 두 시점 비교·과거 영상의 필요성·피해·새 사건 일반화는 아직 검증하지 않았습니다.',
             'Reader와 blind는 각각 학습한 모델입니다. 같은 모델에서 영상만 제거한 인과 비교가 아닙니다.'],
         'next_step':'E3에서 같은 모델의 이전/이후 영상을 제거·반복해 과거 관측이 답에 필요한지 확인합니다.',
         'metrics':{'columns':['Seed','Reader 사건평균 BA','Blind 사건평균 BA','차이'],
                    'rows':[[seed,round(v['reader_macro_ba'],4),round(v['blind_macro_ba'],4),round(v['reader_minus_blind'],4)] for seed,v in s['per_seed'].items()],
                    'caption':'BA는 침수 검출률과 비침수 정답률의 평균입니다. 각 사건에 같은 비중을 주며, 3개 seed를 24개 독립 사건으로 세지 않습니다.'}}
    data=read(out/'research_runs.json')
    assert not any(v['id']=='C1' for v in data['runs']), 'C1 already published in this snapshot'
    data['runs'].insert(next((i for i,v in enumerate(data['runs']) if v['id']=='E3-PD'),len(data['runs'])),run)
    data['checked_at']=datetime.now(timezone.utc).isoformat()
    temp=out/'research_runs.json.tmp'
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(out/'research_runs.json')
    sources=read(out/'research_sources.json')
    sources['checked_at']=data['checked_at']
    for path in [c1/'results.json',c1/'analysis_plan.json',c1/'source.py',audit_path,r/'artifacts/c1_flood_label_quality_v0_20260925/manifest.json']:
        sources['sources'][str(path.relative_to(r))]=sha(path)
    (out/'research_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps({'runs':len(data['runs']),'out':str(out)}))

if __name__=='__main__': main()
