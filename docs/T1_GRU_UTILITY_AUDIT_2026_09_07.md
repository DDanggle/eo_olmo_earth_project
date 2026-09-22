# T1 GRU 감사 — 가능성은 열렸지만, 95%·1/3·신규 방법을 분리한다

## 2026-09-08 00:03 KST 추가 갱신 (현재판)

Hiroshima GRU seed3가 완료되어 **.534777 / .510409 / .531251**, 평균 AP **.525479**,
teacher 격차 회복 **95.50%**, teacher와의 절대 AP 차이 **.024145**다. 세 seed 모두 90%를
넘지만 독립 지역은 여전히 하나이며 decoder는 seed1 하나다. 전체 T1은 **9/18 완료**다.
두 지역 residual은 모두 3seed 완료, Hiroshima .303452(54.10%), Chimanimani .077318(24.49%).
다른 지역 GRU와 양쪽 EMA가 남아 양지역 utility 최종 판정은 미완료다.

새 원본은 `update_20260908_0003/streaming_t1/`에 **추가 보존**했고, 이전 8-report snapshot/
summary는 덮어쓰지 않았다. 최신 집계는 `audit_summary_20260908_0003.json`이다.
현재판의 feature cosine은 GRU .940932 vs residual .938358, AP .525479 vs .303452로
아래의 해석을 바꾸지 않는다. 이하 표는 23:52–23:58에 수행한 최초 감사 기록으로 보존한다.

재현 명령:
```bash
python3 code/audit_streaming_t1_reports.py artifacts/t1_review_20260907_1452 --additional-reports artifacts/t1_review_20260907_1452/update_20260908_0003/streaming_t1
```

---

2026-09-07 23:52–23:58 KST 서버/원시 JSON 확인. 시작 로컬 HEAD `f2f2ffb`.
`artifacts/t1_review_20260907_1452/`는 다운로드 시점의 development 스냅샷이다.
진행 중 trainer/queue는 변경하지 않았고 새 GPU 실험도 실행하지 않았다.

## 1. 결론과 현재 상태

약점부터: GRU는 Hiroshima 2seed만 완료했고 다른 지역 GRU와 양쪽 EMA가 남아 있다.
현재 결과는 8/18이다. 학습 updater seed만 바꾸고 frozen decoder는 각 fold의 seed1 하나다.
이를 3개 독립 decoder·다수 독립 지역 확증으로 부르지 않는다.

긍정적 사실: Hiroshima의 GRU AP .522593, teacher .549624, stale .013356은 원시 JSON과 일치한다.
더 중요한 검증은 teacher AP가 **기존 p4_native_control report와 두 지역 모두 소수점까지 정확히
일치**한다는 점이다. 다른 실행의 teacher 값을 섞은 표가 아니다. source 통계의 추출 코드도
기존 native decoder와 같은 계산 경로이며 현재 local/server source SHA 3/3이 일치한다.
다만 현재 해시는 실행 시작 시점의 소스를 증명하지 않으며 정규화 tensor의 독립 재생 검증은 하지 않았다.

| 지역/팔 | 완료 seed | AP 관측 평균 | teacher 격차 회복 | 교사 cosine 평균 |
|---|---|---:|---:|---:|
| Hiroshima full teacher | 고정 decoder | .549624 | — | 1 |
| Hiroshima frozen c4 | 고정 decoder | .013356 | 0% | .904222 |
| Hiroshima singles mean | 무학습 | .194080 | 33.7% | .841857 |
| Hiroshima GRU | 1,2 (미완료) | .522593 | 94.96% | .940906 |
| Hiroshima residual | 1,2,3 | .303452 | 54.10% | .938358 |
| Chimanimani full teacher | 고정 decoder | .288222 | — | 1 |
| Chimanimani frozen c4 | 고정 decoder | .008909 | 0% | .900174 |
| Chimanimani singles mean | 무학습 | .147141 | 49.49% | .858407 |
| Chimanimani residual | 1,2,3 | .077318 | 24.49% | .938560 |

Hiroshima GRU seed AP는 .534777 / .510409. .027031은 teacher와 관측 평균의 **절대 AP 차이**다.
회복률은 `(student AP - frozen AP)/(teacher AP - frozen AP)`이다. 정확도 95%, 전체 모델 성능
95%, 비열등성 통과를 뜻하지 않는다. 필요 AP 허용폭과 paired spatial CI는 별개다.
Chimanimani residual의 보고 22%는 2seed 중간값이고 3seed 완료값은 24.49%다.

원 residual의 90%-양지역 **필요조건은 이미 불통과**다. 다른 팔 미완료라고 이 사실까지 미루지
않는다. 반면 GRU utility의 양지역/3seed 판정과 전체 arm 비교는 아직 미완료다.

## 2. 즉시 정정할 주장 네 개

### “재인코딩 없이” → “과거 창 전체를 재인코딩하지 않고”

새 관측은 여전히 OLMoEarth에 넣어 single-acquisition embedding을 만든다. 인코더를 전혀 안
쓰거나 raw 입력이 전혀 필요 없는 방법이 아니다. 과거 raw의 반복 읽기/인코딩을 피하는 후보다.

### “예산 맞춘 GRU” → “규모가 다른 두 updater”

JSON 실측 파라미터: GRU **3,541,248**, residual **4,721,664**. 잔차 쪽이 약 33.3% 더 크다.
GRU가 더 작은데 더 낫다는 사실은 유지되지만 strict budget-matched는 아니다. EMA는 scalar 1개를
source에서 실제 학습한다(AdamW lr .1). 무학습 singles mean과 같은 baseline이 아니다.

### “36→12, GPU 비용 1/3” → workload별 bookkeeping과 실제 latency를 분리

| 같은 요청 workload | full 재계산 timestep-unit | streaming timestep-unit |
|---|---:|---:|
| c4/6/8/10/12 모두, 초기 포함 | 40 | 12 |
| 공통 c4 생성 이후 갱신만 | 36 | 8 |
| 처음부터 최종 c12만 필요 | 12 | 12 |
| 공통 c4가 이미 있고 최종 c12만 필요 | 12 | 8 |

초기 비용을 full에서만 뺀 36 vs 12는 대칭 회계가 아니다. 어느 행도 GPU 속도 비율은 아니다.
single encoder 호출 overhead, temporal attention, updater, decode, raw I/O, caching을 계측해야 한다.
teacher 추출/업데이터 source 학습 비용도 배포 inference와 별도 보고한다. 현재 `train_s`는 학습+
일부 평가 작업을 포함하며 inference latency가 아니다. 동시 GPU 작업 시간으로 속도 우위를 주장하지 않는다.

### “EMA가 되면 방법 필요 없음 / GRU만 되면 학습 필수” → 시험한 baseline 사이의 차이

EMA가 좋으면 저비용 baseline도 유용한 것이다. 다른 관측 간격/구름/장기 rollout/새 task에서의
한계는 별도로 측정한다. GRU가 EMA를 이겨도 채널별 EMA·선형 correction·고정 누적평균 등 모든
무학습/단순 방법의 불가능성을 증명한 것은 아니다. 실제 미측정 조건을 억지로 어렵게 만들어
신규성을 만들지도 않는다. 사용자에게 필요한 deployment 조건을 먼저 동결한다.

## 3. 가장 중요한 남은 원인 분리: 관측 기여인가, 입력 계약 보정인가

frozen decoder는 e12로 source 학습·정규화되었다. e4에 같은 decoder를 적용하면:

- 새 관측 정보 부족,
- observation count/context 변화에 따른 representation shift,
- terminal label과 prefix 관측 시점의 불일치

가 섞일 수 있다. e4 AP .013을 “시간이 지나 실제 지도가 망가진 양”으로 읽으면 안 된다.
이는 **같은 final-window 판독기에 대한 compatibility gap**이며 현재 서비스 가정에서는 유효한
baseline이다. 다만 새 영상의 정보가 회복 원인인지 구분하려면 추가 대조가 필요하다.

우선순위:

1. source-only `m4 → e12` 작은 affine/static adapter: 새 영상 없이 분포 보정만 한다.
2. 같은 GRU를 새 관측 없이 동일 step 수로 학습한 대조: 공간 prior/step-count만으로 회복되는가.
3. source에서 학습한 c4-native decoder: e4 자체의 terminal-task 정보와 e12-head 호환성을 구분.
4. 입력 시점 순서/다른 타일 관측 perturbation: 보조 진단. OOD corruption으로 망가진 것만으로
   유용한 새 정보의 기여를 증명하지 않는다. 같은 계약에서 source 재학습한 no-new-input arm이 중요하다.

이 대조가 비슷하면 “스트리밍 관측 동화”보다 “prefix-to-final 표현 보정”으로 좁힌다.
GRU+실제 관측만 더 낫다면 원래 목표인 incremental evidence update의 근거가 강해진다.

## 4. 현재 결과가 가리키는 실제 개선 방향

Hiroshima 두 updater의 teacher cosine은 .9409 vs .9384로 매우 가깝지만 AP는 .5226 vs .3035다.
이는 scalar cosine만으로 판독 utility를 판정할 수 없다는 구체적 증거다. MSE도 GRU .1179 vs
residual .1362로 차이가 있으므로 “feature가 정확히 같은데 AP만 다르다”라고는 하지 않는다.

지난 턴 제안한 compact residual을 곧바로 ours로 승격하지 않는다. 이번 결과로는 **이미 작동한
GRU를 기준으로 objective를 먼저 바꾸는 것**이 더 명확한 실험이다:

- GRU + feature MSE (현 v0)
- 같은 GRU + feature MSE + frozen source-task output KL

source 데이터/파라미터/순서/검색 예산/optimizer를 맞춘다. objective 분리에서는 v0 val-MSE
checkpoint 관례를 유지하고, val-AP 선택은 별도 recipe 감도로 보고한다. 동일 feature error에서도
task utility가 더 보존되는지, 긴 rollout에서 오경보가 증가하는지 확인한다.
KL은 알려진 task를 보존하는 기존 증류 기법이다. unseen task 보존을 자동으로 보장하지 않는다.
198,208-param compact prototype은 이후 비용-성능 대조 후보이며 학습된 결과는 아직 없다.

초안: `config/t1_evidence_and_fidelity_review_v1_draft.json`.
**미등록·미실행**, threshold null은 결정되지 않았다는 표시다. 그대로 GPU runner에 넣지 않는다.

## 5. 실행/증거 보존에서 확인한 제한

- split manifest는 양 fold 각각 train600이며 train/val/test ID 중복과 교집합 모두 0.
  보고 count와 일치한다. ID 감사가 새 공간 overlap 감사를 대신하지 않는다.
- train600은 `tr[::floor(len(tr)/600)][:600]`으로 골라 정렬목록 끝을 잘라낼 수 있다. region-balanced
  random sample이 아니다. source 지역별 수는 audit summary에 보존했다. validation region도
  fold마다 다르므로 두 지역 차이를 test geography 단일 요인의 인과효과로 읽지 않는다.
- T1 output에 `code_snapshot/`이 없고 trainer는 live source를 실행한다. current local/server
  SHA 일치는 retrospective 검사다. 과거 실행 시점 동일성 증명으로 바꾸지 않는다.
- updater checkpoint, normalization snapshot, sample IDs/hash, per-tile prediction archive를
  저장하지 않는다. `best['state']`는 프로세스 메모리에만 있다. JSON에서 공간 CI·입력 perturbation
  재생을 만들 수 없으므로 다음 별도 실행부터 저장해야 한다. 현재 run을 중간에 고치지 않는다.
- 학습 seed는 updater seed뿐이며 decoder seed1 하나다. source distillation은 target 정답을
  쓰지 않지만 이미 source-label로 학습된 decoder를 사용하므로 전 과정 무라벨은 아니다.
- prefix는 전체 15장의 cloud quality로 고른 12장 안의 prefix다. encoder가 각 prefix 밖 영상을
  직접 보지는 않으나 관측 선택이 미래 품질에 의존한다. 진짜 past-available stream과 다르다.
- label은 terminal mask 하나다. c4/c6 평가를 이벤트 발생시각별 피해 탐지 성능으로 부르지 않는다.

## 6. 원래 큰 그림과 연결 — 캐시 활용에서 캐시 수명 관리로

**북극성은 유지한다:** 저장된 OLMoEarth 표현을 지역·과업·세계·버전 변화 속에서도 유용하게 쓴다.

이번 teacher는 새로 학습한 다른 거대 모델이 아니라 **동일한 frozen OLMoEarth를 prefix 전체에
다시 적용한 출력**이다. 학습한 것은 그 출력을 근사하는 작은 updater이며 OLMoEarth 가중치를
개선한 것은 아니다. source teacher로부터의 표현 증류/배포 적응으로 설명한다. 최신 .525 AP와
예전 M65의 .272 macro IoU는 지표가 달라 직접 크기를 비교하지 않는다.

| 축 | 기존 자산 | 지금 새로 여는 질문 | 아직 없는 증거 |
|---|---|---|---|
| 지역/라벨 transfer | Sen12·Solar cache/few-shot | 새로운 지역에서도 갱신 모듈이 전달되는가 | exposed 2fold 외 갱신 transfer |
| 세계 변화 | 이번 T1 single-observation update | 과거 창 반복 인코딩 없이 현재 지도를 유지하는가 | causal 관측, 반복 시점 utility, 실제 latency |
| 여러 task 공유 | Korea 같은 cube 3-task 계획 | 갱신 1회가 서로 다른 판독기의 utility를 보존하는가 | 세 과업 실제 갱신/비용 실험 |
| 모델 release 변화 | FoldRefresh/bridge | 같은 세계를 새 좌표계로 옮기는가 | 이번 T1이 release 변화까지 해결한다는 증거 |

Korea static task가 temporal teacher supervision을 자동으로 제공하지 않는다. 현재 shared-cache
재사용 시험과 dated multi-task update 시험을 분리하고, 기존 데이터 계약/희소 클래스/selection
gate가 닫히기 전 label을 열지 않는다. Nepal은 별도 응용 repo이며 새로운 확증 증거로 재활용하지 않는다.

## 7. 최근 문헌과 CVPR 포지셔닝

- [Temporal Sensitivity Analysis of Tessera Embeddings, 2026-08-27](https://arxiv.org/abs/2608.27175):
  관측 창을 줄일 때의 utility/과업 차이를 이미 분석한다. 단순 “짧은 창도 된다”는 차별점이 약하다.
  우리 후보는 매 cutoff 재계산 자체를 학습된 상태 갱신으로 대체하는 것이며, 그 차이가 비용·성능으로
  입증되어야 한다. preprint이며 이 설정을 최초로 푼다는 문헌 완전성 주장은 하지 않는다.
- [TESSERA v2, 2026-08-06 v2](https://arxiv.org/abs/2607.03949v2): 대형 teacher를 compact student로
  증류하고 유연한 임베딩 크기를 제공한다. 검색 요약의 v1 수치(21M)를 그대로 쓰지 않는다(v2는 44M).
  “작은 모델로 임베딩 성능 유지”도 이미 있는 방향이다. 현재 T1은 full OLMo를 새 영상마다 계속 써서
  encoder 자체를 compact student로 바꾸는 것과도 다르다.
- [Deep Feature Flow, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_Deep_Feature_Flow_CVPR_2017_paper.html):
  비싼 feature를 재사용/전파해 반복 비디오 인식을 가속하는 개념이 이미 있다. 위성의 irregular gaps,
  구름/결측, event change, many-task cache 제약에서 기존 방법 대비 무엇이 다른지 증명해야 한다.

가능한 기여 문장(목표, 아직 결과 아님): **불규칙한 새 관측으로 materialized EO cache를 갱신하면서,
여러 downstream task의 utility를 유지하고 반복 재인코딩 비용을 줄이는 방법.**
GRU 연결 자체, KL 자체, 95% endpoint 숫자 하나는 방법 신규성이 아니다. 기존 방법보다 실제
cost–utility 개선과 외부 재현이 있어야 한다. EMA가 성공해도 연구가 사라지는 것은 아니지만,
해결된 문제를 다시 복잡하게 만드는 것을 노벨티라고 부르지 않는다.

## 8. 다음 작업 순서

1. 기존 T1 미완료 arm을 원계약대로 끝내고 원 residual의 불통과를 보존한다.
2. 다음 재현 실행은 updater checkpoint/정규화/score/ID/실행 snapshot 저장을 먼저 포함한다.
3. no-new-input 대조로 “관측 가치 vs 계약 보정”을 분리한다.
4. 같은 GRU의 objective 비교 + 전 cutoff utility + 단독 GPU inference 계측을 별도 등록한다.
5. past-available observation stream과 외부 event/region으로 전이한다. Korea 3-task 공유 비용은
   기존 gate 후 독립적으로 측정한다. 무근거 selector/MoE를 덧붙이지 않는다.

재집계: `python3 code/audit_streaming_t1_reports.py artifacts/t1_review_20260907_1452`.
완료 조건·중복 seed·잘못된 recovery 분모·baseline 불일치·manifest count/교집합 검사
단위 테스트 **10개 + 5개 subtest 통과**. 원본 report는 변경하지 않았다.
