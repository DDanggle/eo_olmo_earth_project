#!/usr/bin/env python3
"""Build a local research status snapshot from retrieved artifacts, never remote execution."""
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

def read(p): return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.repo.resolve();out=a.out.resolve()
 base=r/'artifacts/eo_evidence_search_v1_20260925'
 if out==base:raise ValueError('Preserve the prior evidence artifact')
 if not out.exists():shutil.copytree(base,out)
 shutil.copyfile(r/'code/eo_evidence_search_v0.html',out/'index.html')
 runs=[{'id':'E2','title':'기존 VLM 내용 통제의 독립 재검산','status':'등록 판정 재현','findings':['답변 22,212행과 등록 점수를 독립 재계산했습니다. 산사태 2/3 seed, 홍수 3/3 seed가 기존 통제 기준을 충족했습니다.','홍수 10개 사건은 이전 연구에서 이미 평가한 자료입니다. E2 학습과 사건이 분리됐다는 사실과 구별합니다.'],'limitations':['산사태 192타일 중 Hiroshima 186, Indonesia 6으로 지역 편중이 큽니다.','임베딩을 따른다는 근거이며 시간 추론·새 사건 일반화·피해 판정의 검증은 아닙니다.'],'next_step':'E3에서 이전/이후 영상과 차이 토큰을 제거·반복하여 어떤 입력이 답에 필요한지 확인합니다.'}]
 v0=r/'artifacts/c0_linear_view_probe_v0_invalid_20260925/failure.json'
 if v0.exists():runs.append({'id':'C0-v0','title':'CPU 기준선 첫 준비','status':'학습 전 실행 무효','findings':['캐시가 없는 QA 문항에서 중단됐습니다. 모델 학습과 예측은 생성되지 않았습니다.'],'limitations':['E2의 가용 캐시 조건을 복원하지 못한 구현 문제입니다. 성능이 낮았다는 결과가 아닙니다.'],'next_step':'실패를 보존하고, 제외 문항과 사유를 동결한 별도 v1으로 입력 계약을 수리했습니다.'})
 c0=r/'artifacts/c0_linear_view_probe_v1_20260925';result=c0/'results.json'
 if result.exists():
  c=read(result);valid=c['valid'];run={'id':'C0-v1','title':'이전·이후·두 영상의 선형 판별 정보','status':'완료 · 개발 자료 진단' if valid else '실행 무효','findings':['학습 4,234문항, 평가 1,755문항의 현재 입력을 동결했습니다. 현상별 세 조건, 총 여섯 CPU 선형 모델을 사용했습니다.'],'limitations':['이미 평가한 개발 자료이며 VLM 성능과 직접 비교하는 실험이 아닙니다.','임베딩에 시간 메타데이터도 포함됩니다. 순수 영상 내용의 효과를 분리한 결과는 아닙니다.','원 라벨과의 연관성을 측정합니다. 사건 원인·피해·시간 추론은 검증하지 않았습니다.','산사태 Indonesia는 6타일뿐이므로 지역 차이는 기술통계로만 봅니다.'],'next_step':'E3의 고정 VLM 입력 통제와 함께 해석한 뒤 후속 실험 하나를 선택합니다.'}
  if valid:
   run['findings'].append('여섯 모델의 예측·표준화·손실·기울기와 지표를 별도 코드로 재계산했고, 수렴 기준을 모두 충족했습니다.')
   run['findings'].append('두 영상 모델은 이후 영상 모델보다 hard-negative 오탐이 221/457→4/457로 줄었지만 홍수 검출도 414/457→150/457로 감소했습니다. 단순한 성능 향상으로 해석하지 않습니다.')
   run['findings'].append('사후 순위 진단에서 두 영상은 pre/pre 음성과의 구분과 여러 사건을 합친 순위가 낮아졌습니다. 판정 기준만 옮기면 해결된다고 단정할 수 없습니다.')
   flood=c['metrics']['flood'];d=flood['later_minus_pair'];run['findings'].append('홍수 이후−두 영상 BA 차이: %.3f, 사건 bootstrap 95%% 구간 [%.3f, %.3f].'%(d['delta'],*d['ci95_delta']))
   rows=[]
   for arm,label in [('earlier','이전 영상'),('later','이후 영상'),('pair','두 영상')]:
    m=flood[arm];land=c['metrics']['landslide'][arm]['events'];rows.append([label,round(m['event_macro_ba'],4),round(sum(v['recall']*v['pos'] for v in m['events'].values())/sum(v['pos'] for v in m['events'].values()),4),round(m['hard_negative']['pooled_fpr'],4),round(land['holdout_hiroshima']['ba'],4),round(land['holdout_indonesia']['ba'],4)])
   run['metrics']={'columns':['입력','홍수 사건평균 BA','홍수 검출률 (457)','홍수 hard-neg 오탐률','Hiroshima BA','Indonesia BA'],'rows':rows,'caption':'BA는 클래스별 정확도의 평균입니다. 홍수 BA는 사건별 동등 평균, 검출률과 hard-neg 오탐률은 각각 457문항을 합산했습니다. 고정 판정 기준의 결과이며 오탐과 누락을 함께 봐야 합니다.'}
  else:run['findings']+=c.get('failures',[])
  runs.append(run)
 else:runs.append({'id':'C0-v1','title':'이전·이후·두 영상의 선형 판별 정보','status':'결과 미회수','findings':[],'limitations':['이 스냅샷에는 학습 결과가 없습니다.'],'next_step':'서버 실행 기록을 회수한 뒤 독립 재계산합니다.'})
 e3=r/'artifacts/e3_pair_dependence_v0_prepared_20260925';q=read(e3/'queue.json');s=read(e3/'status.json');m=read(e3/'manifest.json');state=q.get('status',s['status'])
 runs.append({'id':'E3-PD','title':'고정 VLM의 두 시점 의존성','status':{'waiting_gpu1':'GPU 대기','completed':'실행 종료 · 결과 검토 필요','failed':'실행 오류 · 기록 확인 필요'}.get(state,state),'findings':[f"준비 입력 {m['n_items']}문항, 예정 생성 {m['n_generations']:,}회. 사용 프레임을 검사한 1,106캐시에서 비유한 값은 없었습니다.",'서버 상태 확인 시각: '+q.get('at',s.get('at','미확인'))],'limitations':['이 스냅샷에는 E3 모델 판정이 없습니다. 기다리는 상태를 성공으로 세지 않습니다.','반복·역순 입력에는 새로운 물리적 정답을 가정하지 않습니다. 원 라벨 일치와 답 변화를 측정합니다.','S1의 합성 시간 정보가 임베딩에도 들어 있어 순수 영상 내용 의존성과 구별해야 합니다.'],'next_step':'GPU가 비면 기존 답 재현부터 검증하고, 통과했을 때만 영상 제거·반복·역순 통제를 실행합니다.'})
 data={'schema_version':'eo_research_snapshot_v1','checked_at':datetime.now(timezone.utc).isoformat(),'runs':runs}
 temp=out/'research_runs.json.tmp';temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(out/'research_runs.json')
 sources=[v0,result,e3/'manifest.json',e3/'queue.json',r/'artifacts/e2_independent_audit_20260925/audit.json']
 (out/'research_sources.json').write_text(json.dumps({'checked_at':data['checked_at'],'sources':{str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources if x.exists()}},indent=2)+'\n')
 print(json.dumps({'out':str(out),'runs':len(runs),'checked_at':data['checked_at']}))
if __name__=='__main__':main()
