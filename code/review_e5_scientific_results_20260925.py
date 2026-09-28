#!/usr/bin/env python3
"""Read-only scientific reaggregation of a SHA-pinned completed E5 replica.

No production scoring/model functions imported, no model/pixel inference.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,re
import numpy as np

BASE=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/e5_equal_budget_v0_review_20260925')
ROOT=BASE/'e5_equal_budget_v0'
OUT=Path('/private/tmp/E5_SCIENTIFIC_RESULT_REVIEW_20260925.json')
MEMO=Path('/private/tmp/E5_SCIENTIFIC_RESULT_REVIEW_20260925.md')
PLAN_SHA='fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
MANIFEST_SHA='e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
EVALUATIONS=(('full','native'),('pair','native'),('later','native'),('delta','native'),('full','full_no_delta'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def check(ok,msg):
    if not ok:raise ValueError(msg)
def close(a,b):
    check(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),f'Metric mismatch {a} vs {b}')
def ci(values):
    draws=np.random.default_rng(20260925).integers(0,8,size=(5000,8))
    samples=sorted(math.fsum(values[int(j)] for j in draw)/8 for draw in draws)
    result=[]
    for p in (.025,.975):
        idx=4999*p;lo=math.floor(idx);f=idx-lo
        result.append(samples[lo]*(1-f)+samples[math.ceil(idx)]*f)
    return result

def main():
    audit=read(BASE/'e5_independent_audit_20260925.json')
    check(audit['consistent'] is True and audit['checkpoint_tensors_loaded_and_checked_on_cpu'] is True,'Upstream CPU audit required')
    check(audit['audit_code_sha256']=='a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825','Unexpected audit source')
    original=Path(audit['artifact']);check(str(original)=='/home/work/data/olmoearth/e5_equal_budget_v0','Original root differs')
    hashes={}
    for name in ('manifest.json','prereg.json','status.json','scores.json','items.jsonl','eval_sets.json','predictions.jsonl','training_summary.json'):
        hashes[name]=sha(ROOT/name)
        check(hashes[name]==audit['hashes_verified'][str(original/name)],'Replica SHA differs '+name)
    check(hashes['prereg.json']==PLAN_SHA and hashes['manifest.json']==MANIFEST_SHA,'Frozen pins differ')
    cfg=read(ROOT/'prereg.json');scores=read(ROOT/'scores.json');status=read(ROOT/'status.json')
    check(status['status']=='completed' and scores['valid'] is True,'Completion/validity')
    items=rows(ROOT/'items.jsonl');sets=read(ROOT/'eval_sets.json');answers=rows(ROOT/'predictions.jsonl')
    lookup={x['id']:(j,x) for j,x in enumerate(items)};test={x['id'] for x in items if x['partition']=='test'};primary=set(sets['primary_same_prompt'])
    check(len(items)==len(lookup)==5989 and len(test)==1755 and len(primary)==902,'Population')
    check(Counter(lookup[k][1]['kind'] for k in primary)=={'pos':445,'hard_neg':457},'Primary class counts')
    expected={(seed,arm,mode,key) for seed in (1,2,3) for arm,mode in EVALUATIONS for key in test};indexed={}
    for row in answers:
        key=(row['seed'],row['model_arm'],row['eval_arm'],row['id'])
        check(type(row['seed']) is int and key in expected and key not in indexed,'Output ID/seed/evaluation/duplication')
        j,item=lookup[row['id']]
        check(row['pair_index']==j and row['source_gold']==item['answer'] and row['transformed_gold'] is None,'Global index/gold')
        check(all(row[k]==item[k] for k in ('tile','cluster','phen','kind')),'Source metadata mismatch')
        words=re.findall(r'\b(?:yes|no)\b',row['answer_raw'].strip().lower());parsed=words[0] if words else None
        check(row['parsed']==parsed,'Raw parse mismatch');indexed[key]=row
    check(len(answers)==26325 and set(indexed)==expected,'All-output exact support')
    strata=defaultdict(list)
    for key in primary:
        item=lookup[key][1];strata[(str(item['cluster']),tuple(item['dates']),tuple(item['slots']))].append(item)
    events=sorted({k[0] for k in strata});check(len(events)==8,'Event support')
    stats={};decisions={};response_counts={}
    for seed in (1,2,3):
        conditions={}
        for arm,mode in EVALUATIONS:
            event_values=defaultdict(list)
            for (event,dates,slots),group in sorted(strata.items()):
                pos=[x for x in group if x['kind']=='pos'];neg=[x for x in group if x['kind']=='hard_neg']
                check(pos and neg,'Unsupported stratum')
                recall=sum(indexed[(seed,arm,mode,x['id'])]['parsed']=='yes' for x in pos)/len(pos)
                specificity=sum(indexed[(seed,arm,mode,x['id'])]['parsed']=='no' for x in neg)/len(neg)
                event_values[event].append((recall+specificity)/2)
            byevent={event:math.fsum(values)/len(values) for event,values in sorted(event_values.items())}
            macro=math.fsum(byevent.values())/8
            saved=scores['metrics'][str(seed)]['primary_same_prompt']['evaluations'][arm+'/'+mode]
            close(macro,saved['macro_ba'])
            for event,value in byevent.items():close(value,saved['events'][event]['ba'])
            conditions[arm+'/'+mode]={'macro_ba':macro,'event_ba':byevent}
        contrasts={}
        for name,left,right in [('pair_minus_full','pair/native','full/native'),('full_no_delta_minus_full','full/full_no_delta','full/native'),
                                ('pair_minus_full_no_delta','pair/native','full/full_no_delta'),('later_minus_full','later/native','full/native'),('delta_minus_full','delta/native','full/native')]:
            changes=[conditions[left]['event_ba'][event]-conditions[right]['event_ba'][event] for event in events]
            delta=math.fsum(changes)/8;interval=ci(changes)
            saved=scores['metrics'][str(seed)]['primary_same_prompt']['contrasts'][name]
            close(delta,saved['delta']);close(interval[0],saved['ci95_delta'][0]);close(interval[1],saved['ci95_delta'][1])
            contrasts[name]={'delta':delta,'ci95_delta':interval,'event_deltas':dict(zip(events,changes)),'primary':name=='pair_minus_full'}
        full,pair=conditions['full/native']['macro_ba'],conditions['pair/native']['macro_ba'];contrast=contrasts['pair_minus_full']
        eligible=full>=.60-1e-12 and len(events)>=5
        decision={'eligible':eligible,'pair_preserves':eligible and pair>=.60-1e-12 and contrast['ci95_delta'][0]>=-.05-1e-12,
                  'explicit_difference_helps':eligible and contrast['delta']<=-.10+1e-12 and contrast['ci95_delta'][1]<-1e-12}
        check(decision==scores['seed_decisions'][str(seed)],'Registered per-seed rule differs')
        decisions[str(seed)]=decision
        stats[str(seed)]={'conditions':conditions,'contrasts':contrasts}
        counts={}
        for name,subset in [('all_test',test),('primary_same_prompt',primary),
                            ('flood_all',{x['id'] for x in items if x['partition']=='test' and x['phen']=='flood'}),
                            ('landslide_all',set(sets['landslide'])),
                            ('primary_positive',{k for k in primary if lookup[k][1]['kind']=='pos'}),
                            ('primary_hard_negative',{k for k in primary if lookup[k][1]['kind']=='hard_neg'})]:
            group=[indexed[(seed,'full','full_no_delta',key)] for key in sorted(subset)]
            counter=Counter(r['parsed'] for r in group);raw=Counter(r['answer_raw'] for r in group)
            counts[name]={'n':len(group),'yes':counter['yes'],'no':counter['no'],'unparsed':counter[None],
                          'raw_answers':dict(raw),'all_parsed_answers_no':counter['no']==len(group)}
        response_counts[str(seed)]=counts
    preserves=[int(s) for s,d in decisions.items() if d['pair_preserves']];helps=[int(s) for s,d in decisions.items() if d['explicit_difference_helps']]
    verdict='pair_preserves_source_discrimination_at_equal_budget' if len(preserves)>=2 else 'explicit_difference_helps_at_this_budget' if len(helps)>=2 else 'mixed_or_inconclusive'
    check(verdict==scores['verdict']==status['verdict']==audit['verdict'],'Aggregate verdict')
    for name,pin in hashes.items():check(sha(ROOT/name)==pin,'Inputs changed during review')
    result={'schema':'e5-independent-scientific-review-v0','consistent':True,'checked_at':datetime.now(timezone.utc).isoformat(),
            'local_artifact':str(ROOT),'upstream_original_artifact':str(original),'upstream_audit_path':str(BASE/'e5_independent_audit_20260925.json'),
            'upstream_audit_sha256':sha(BASE/'e5_independent_audit_20260925.json'),'source_files_sha256':hashes,'review_code_sha256':sha(__file__),
            'n_answers_checked':26325,'primary_n':902,'primary_events':events,'primary_n_strata':len(strata),'primary_metrics':stats,
            'registered_seed_decisions':decisions,'pair_preserves_seeds':preserves,'explicit_difference_helps_seeds':helps,'verdict':verdict,
            'full_no_delta_actual_response_counts':response_counts,
            'interpretation':{'strongest_defensible_claim':'At the fixed exposure budget, training from the outset with [A,B,0] restores source-label discrimination relative to setting D=0 only at evaluation of a model trained with [A,B,D]. Evaluation-time removal failure therefore does not establish that A/B inputs are unlearnable or that an explicit D block is intrinsically necessary.',
                'primary_rule_explanation':'Seeds1/2 point deficits are below .05 but lower CI bounds are below -.05; seed3 point deficit exceeds .10 but its CI upper is positive. Thus no seed satisfies either complete conjunctive rule.',
                'recovery_is_secondary':'All three recovery contrasts have positive descriptive event intervals; this does not replace the inconclusive registered primary comparison.',
                'counterevidence_to_temporal_reasoning':'Later-only native BA .73735/.79687/.80295 is also substantial; primary source-label success alone does not identify use of history or temporal reasoning.'},
            'scope':'Independent CPU raw-answer/count/primary BA/paired-event percentile reaggregation; relies on pinned upstream full tensor/training audit, does not repeat it.',
            'limitations':['Exposed development source-label task; no fresh event test.','Eight event clusters, not 902 independent event samples or 24 independent seed-events.',
                           'Only pair−full is primary; recovery/later/delta contrasts are descriptive.','Input encoding and observed timing remain approximate; no verified physical onset/damage/causal or memory benefit.',
                           'Difference D is computed from both observations but cannot in general reconstruct both absolute observations.']}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# E5 독립 과학 결과 검토','',f'고정된 원 응답 26,325개와 원격 독립 감사에 기록된 SHA를 대조했고, 902문항·8사건의 점수와 seed별 사건 bootstrap 5,000회 구간·등록 규칙을 별도로 재계산했다. 불일치 없음. 주판정은 **{verdict}**이다.','',
           '| Seed | full BA | pair BA | full_no_delta BA | pair−full (95% CI) | 보존 통과 | 차이 도움 통과 |','|---|---:|---:|---:|---|---|---|']
    for s,values in stats.items():
        c=values['conditions'];d=values['contrasts']['pair_minus_full'];r=decisions[s]
        lines.append(f"| {s} | {c['full/native']['macro_ba']:.4f} | {c['pair/native']['macro_ba']:.4f} | {c['full/full_no_delta']['macro_ba']:.4f} | {d['delta']:+.4f} [{d['ci95_delta'][0]:+.4f}, {d['ci95_delta'][1]:+.4f}] | {r['pair_preserves']} | {r['explicit_difference_helps']} |")
    lines += ['',f'보존 규칙 통과 seed는 {preserves}, 차이 도움이 있다는 규칙 통과 seed는 {helps}이다. 어느 쪽도 최소 2/3 seed를 충족하지 못한다. 보존 실패는 pair가 무용하거나 full보다 확실히 나쁘다는 판정이 아니다. 관측된 재학습 회복과, 사전에 정한 0.05 허용폭 안의 성능 보존 주장은 서로 다르다.','',
              'Seed 1·2는 pair의 점추정 손실이 0.05보다 작지만 CI 하한이 각각 −0.0971·−0.0981이라 보존 관문을 통과하지 못한다. Seed 3은 점추정 손실이 0.1273이지만 CI 상한이 +0.0103이어서 차이 입력의 도움 관문도 통과하지 못한다.', '', '## 원 응답에서 직접 확인한 full_no_delta','', '| Seed | 전체 yes / no / 미파싱 | primary yes / no / 미파싱 |','|---|---|---|']
    for s,v in response_counts.items():
        x,y=v['all_test'],v['primary_same_prompt'];lines.append(f"| {s} | {x['yes']} / {x['no']} / {x['unparsed']} | {y['yes']} / {y['no']} / {y['unparsed']} |")
    lines += ['', '이 표는 BA=0.5로부터 답을 추정하지 않고 JSONL의 raw→parsed 응답을 직접 셌다. 원문 응답별 빈도와 현상·원 라벨별 분모는 동반 JSON에 보존했다.','',
              '## 재학습 회복: 사전 지정된 보조 비교','', '| Seed | pair−full_no_delta | 기술적 95% 사건 구간 |', '|---|---:|---|']
    for s,value in stats.items():
        c=value['contrasts']['pair_minus_full_no_delta'];lines.append(f"| {s} | {c['delta']:+.4f} | [{c['ci95_delta'][0]:+.4f}, {c['ci95_delta'][1]:+.4f}] |")
    lines += ['', '세 seed 모두 회복이 관측되지만, 이 보조 비교의 양의 구간은 pair−full 주대조의 판정을 대체하지 않는다. Later-only BA도 .7373/.7969/.8029로 높다는 점은 원 라벨 분류 성공을 시간 추론으로 해석하지 말아야 할 추가 이유다.', '', '## 방어 가능한 주장','',
              '이 고정 학습 예산과 개발 사건 집합에서는, 명시적 차이 입력 D를 사용하도록 학습한 모델에 평가 시 D를 0으로 바꾸는 개입과, 처음부터 D=0인 형식으로 학습하는 것은 다른 결과를 낸다. pair 재학습은 full_no_delta보다 원 라벨 구분 성능을 회복하므로, 평가 시 D 제거 실패만으로 A/B 입력이 학습 불가능하거나 명시적 D가 본질적으로 필요하다고 결론낼 수 없다. 다만 등록된 주대조는 성능 보존도 차이 입력의 일관된 이점도 확정하지 못했다.','',
              '## 논문에서 유지할 한계','',
              '- 회복 비교는 사전에 기술적 비교로 정한 보조 결과다. full 대비 비열등성·동등성·새 사건 일반화를 대신하지 않는다.',
              '- 902문항의 유효 추론 단위는 8사건이며, 3 seed를 합쳐 24개 독립 사건처럼 취급하지 않는다. 사건 간 이질성과 넓은 CI를 그대로 보여준다.',
              '- later와 delta의 성능은 시계열 추론을 자동 입증하지 않는다. D=B−A는 두 시점으로 계산되지만 두 원 관측의 모든 절대 정보를 보존하지 않는다.',
              '- EO 캐시와 질문의 합성 날짜를 유지했다. 공개 metadata의 날짜 후보, 실제 UTC 취득시각, 원 픽셀 계보 및 물리적 변화 발생시점의 검증은 별도다.',
              '- 원 라벨 구분·표현 학습의 결과이며 피해·인과 효과·기억 필요성이나 기억 방법의 성능 이득에 관한 결과가 아니다.',
              '- 이번 검토는 파일 SHA와 답·점수 재집계다. 저장된 모델 텐서/학습 예산 전체 감사는 고정된 상위 독립 감사에 의존하며 이를 재실행했다고 주장하지 않는다.','',
              f'입력: `{ROOT}`',f'상위 독립 감사: `{BASE/"e5_independent_audit_20260925.json"}`',f'검산 코드: `{Path(__file__)}`',f'정확 수치·출처 SHA: `{OUT}`','']
    MEMO.write_text('\n'.join(lines))
    print(json.dumps({'consistent':True,'verdict':verdict,'decisions':decisions,'primary':{s:{k:v['macro_ba'] for k,v in x['conditions'].items()} for s,x in stats.items()},'contrasts':{s:x['contrasts']['pair_minus_full'] for s,x in stats.items()},'no_delta_counts':response_counts,'json':str(OUT),'memo':str(MEMO)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
