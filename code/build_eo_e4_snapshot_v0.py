#!/usr/bin/env python3
"""Connect only independently verified completed E4 records to a new UI artifact."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ARMS=['real','delta_only','delta_sign_flip','delta_feature_permute']
LABELS=['원 입력 [A,B,D]','차이만 [0,0,D]','차이 부호 반전 [A,B,−D]','차이 성분 순서 변경']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def require(c,s):
 if not c:raise ValueError(s)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['repo','e4','audit','base','out']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();r=a.repo.resolve();e=a.e4.resolve();base=a.base.resolve();out=a.out.resolve()
 require(not out.exists(),'New snapshot directory required')
 audit=read(a.audit);scores=read(e/'scores.json');manifest=read(e/'manifest.json')
 require(audit['consistent'] and scores['valid'] and read(e/'status.json')['status']=='completed','Audited completed run required')
 require(audit['n_answers']==2508 and audit['verdict']==scores['verdict'],'Audit mismatch')
 for path,h in audit['hashes_verified'].items():require(sha(path)==h,'Audited input changed '+path)
 require(sha(r/'code/audit_e4_delta_results_v0.py')==audit['audit_code_sha256'],'Auditor changed')
 require(sha(base/'reader_cases.json')==read(base/'browser_checks.json')['sha256']['reader_cases.json'],'Verified base changed')
 source_rows=rows(e/'items.jsonl');items={v['id']:v for v in source_rows};require(len(items)==len(source_rows)==209,'Source ID coverage')
 answers={}
 for arm in ARMS:
  for seed in ['1','2','3']:
   chunk=rows(e/f'answers_seed{seed}_{arm}.jsonl');require(len(chunk)==209,'Answer population')
   for v in chunk:
    key=(v['id'],arm,seed);require(key not in answers,'Duplicate output');answers[key]=v
 require(len(answers)==2508,'Answer coverage')
 cases=read(base/'reader_cases.json');linked=[]
 for tile,case in cases['cases'].items():
  require('e4_control' not in case,'E4 already exists')
  key=case['question_id']
  if key not in items:continue
  it=items[key]
  require(it['phen']=='flood' and it['kind'] in ['pos','hard_neg'] and it['tile']==tile,'Catalog link identity')
  require(it['answer']==case['reference_label'] and it['dates']==case['dates'] and it['slots']==case['slots'],'Dates/labels/slots changed')
  predictions={}
  for arm in ARMS:
   predictions[arm]={}
   for seed in ['1','2','3']:
    v=answers[(key,arm,seed)]
    require(v['id']==key and v['tile']==tile and v['arm']==arm and v['seed']==int(seed),'Output identity')
    require(v['source_gold']==case['reference_label'] and v['transformed_gold'] is None,'Gold role')
    predictions[arm][seed]=v['parsed']
  require(predictions['real']==case['reader_predictions']==case['e3_control']['predictions']['real'],'Real case reproduction differs')
  case['e4_control']={'run_id':'E4-D-v0','question_id':key,'source_role':'historical_model_output','source_gold':case['reference_label'],
    'predictions':predictions,'source_sha256':{'manifest':sha(e/'manifest.json'),'scores':sha(e/'scores.json'),
      'independent_audit':sha(a.audit),'items':manifest['items_sha256'],
      **{f'{arm}_seed{seed}':sha(e/f'answers_seed{seed}_{arm}.jsonl') for arm in ARMS for seed in ['1','2','3']}}}
  linked.append(key)
 expected=sorted(k for k,it in items.items() if it['phen']=='flood' and it['kind'] in ['pos','hard_neg'])
 require(sorted(linked)==expected and len(linked)==108,'Expected108 exact linked cases')
 names={'difference_block_sufficient_under_intervention':'차이 블록만으로 원 라벨 구분 유지',
        'delta_only_degrades_source_agreement':'차이 블록만 남기면 구분 감소','mixed_or_inconclusive':'혼합 · 결론 유보'}
 repro=[v['reproduction_rate'] for seed in scores['reproduction'].values() for phen in seed.values() for v in phen.values()]
 agreement=[scores['metrics'][str(seed)]['flood']['agreement_with_real']['delta_only']['agreement_all_items'] for seed in [1,2,3]]
 run={'id':'E4-D','title':'차이 블록의 역할','status':'완료 · '+names[scores['verdict']],
  'findings':[f"209문항·2,508응답을 완료했습니다. E2/E3 원 답 재현율은 최소 {min(repro)*100:.1f}%입니다.",
   '홍수 D-only 충분성 기준을 만족한 seed: '+str(scores['sufficient_seeds'])+'. 원 라벨의 사건평균 균형 일치율에 대한 판정입니다.',
   '홍수 전체165문항의 원 답과 D-only 답 일치율: '+', '.join(f'{v*100:.2f}%' for v in agreement)+'. 점수 유지와 개별 응답 재현을 구별합니다.',
   '별도 감사기로 2,508응답·기준 점수·사건별 재표집 구간·입력/코드/프롬프트 해시를 재계산했습니다.'],
  'limitations':['D=B−A 자체가 두 관측 정보를 포함합니다. 차이만으로 충분해도 과거 관측이 불필요하다는 뜻이 아닙니다.',
   'E3 결과를 본 뒤 같은 노출 자료에서 수행한 탐색 진단입니다. 새 평가 자료의 확증이 아닙니다.',
   '성분 순열은 입력 D의 값 분포만 보존합니다. 학습된 projector 이후의 노름·활성 크기까지 보존하지 않습니다.',
   '변환 장면의 새 정답은 없습니다. 원 라벨 일치와 모델 민감성을 측정하며 시간 이해·실제 피해·기억 효과를 입증하지 않습니다.'],
  'next_step':'같은 학습 예산으로 입력 형식을 처음부터 나눠 학습하는 비교를 별도 설계해, 정보의 필요성과 낯선 입력에 대한 붕괴를 구분합니다.',
  'metrics':{'columns':['입력 통제','Seed1 BA','Seed2 BA','Seed3 BA'],
   'rows':[[label]+[round(scores['metrics'][str(seed)]['flood']['paired'][arm]['macro_ba'],4) for seed in [1,2,3]] for label,arm in zip(LABELS,ARMS)],
   'caption':'홍수10사건·57개 원 침수 타일의 yes/no 문항 쌍에 대한 사건평균 원 라벨 균형 일치율입니다. 비침수51개는 별도 분석이며 새 변화 정확도가 아닙니다.'}}
 shutil.copytree(base,out,ignore=shutil.ignore_patterns('*checks.json'))
 shutil.copyfile(r/'code/eo_evidence_search_v0.html',out/'index.html');stamp=datetime.now(timezone.utc).isoformat()
 cases['checked_at']=stamp;cases['e4_control_provenance']={'linked_cases':108,'question_ids':sorted(linked),'builder_sha256':sha(__file__),'audit_sha256':sha(a.audit),'source_role':'historical_model_output'}
 (out/'reader_cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 data=read(out/'research_runs.json');require(not any(v['id']=='E4-D' for v in data['runs']),'Existing E4 run');data['runs'].append(run);data['checked_at']=stamp
 (out/'research_runs.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 source=read(out/'research_sources.json');source['checked_at']=stamp
 for path in [e/'manifest.json',e/'scores.json',e/'prereg.json',a.audit.resolve()]:source['sources'][str(path.relative_to(r))]=sha(path)
 (out/'research_sources.json').write_text(json.dumps(source,indent=2)+'\n')
 print(json.dumps({'out':str(out),'e4_cases':108,'verdict':scores['verdict']}))
if __name__=='__main__':main()
