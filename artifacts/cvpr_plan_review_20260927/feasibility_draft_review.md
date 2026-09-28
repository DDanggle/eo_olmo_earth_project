# 종합 draft 독립 후속 검토 — 실행·통계·일정

검토일: 2026-09-27. 대상: `/private/tmp/cvpr_plan_20260927/CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md`의 첫 종합본. 이전 `feasibility_review.md`는 보존한다. 검토 중 root가 전달한 A/B 단계·frozen 정의·예산 gate·추가 CPT holdout 보완안을 아래에 별도로 구분했다. 이 문서는 성능 실험이나 논문 채택 확률 평가가 아니다.

## 판정

**방법을 반증하는 제한된 개발 실험에는 조건부 GO. 100k 본 학습 전체와 CVPR 2027 일정에는 아직 GO를 줄 근거가 없다.** 자료·처리량·main protocol이 갖춰지고 아래 조건을 만족하면 본 학습으로 전환할 수 있다. 같은 말을 여러 계획에서 반복하는 것보다, 1k 정합/실행 결과와 소량 반증 결과를 다음 판단 자료로 삼아야 한다.

이번 종합본은 원 리뷰보다 세 가지가 개선됐다. generic episodic G와 specialist를 포함해 가장 직접적인 반론을 피하지 않는다. 21 core pipelines + 12 보조 실행과 baseline 비용을 공개했다. 100k 관측에 1M query를 생성하는 것과 1M 독립 관측을 분리했고, PASTIS 4개 광역 tile 및 MADOS sparse label 한계를 드러냈다. **이 개선들은 계획의 정직성을 높였으며 실행 가능성을 자동으로 증명하지는 않는다.**

## 결정적 보완점

### F1. 21개의 의미와 A/B 학습 범위를 먼저 고정해야 한다 — Critical

검토한 첫 draft의 69행 및 142–170행에서는 C/S의 공통 B와 I/G/P의 학습 차이, I/P-frozen이 전체 encoder freeze인지 B-only freeze인지 명확하지 않았다. 이것은 예산 문제뿐 아니라 causal contrast 문제다. C도 공통 instruction B를 받으면 최종 C는 ‘원 목적 학습만 한 모델’이 아니다. 그 문장은 A 종료 checkpoint에만 적용된다.

root가 후속 메시지로 제안한 정의는 합리적이다.

- A: encoder와 개념/연결 모듈 학습. I/G/P에서 VLM 본체는 고정하지만 EO 입력을 향한 gradient 경로는 유지한다. C의 A는 원 목적만, S는 직접 사실 감독, I는 ordinary, G는 generic episodic, P는 구체화된 제안 구조다.
- B: 모든 arm에서 동일 instruction 예산으로 LoRA·연결부·field head를 적응한다. encoder는 각 arm의 A 종료 가중치에서 고정한다.
- I/P-frozen: A와 B 전체에서 encoder를 원본 가중치에 고정한다. B-only freeze가 아니다.

이 정의를 main config에 쓰고 각 단계의 trainable parameter set, reset/carry module, source exposure, updates, checkpoint lineage를 고정해야 한다. `requires_grad=False`인 VLM에서도 EO input gradient는 필요하므로 frozen VLM 전체를 `no_grad()`로 감싸면 안 된다. 이 점은 실제 gradient test가 필요하다.

배운 concept/field module을 B에 이월하는 것은 전체 모델 효과를 평가하는 데 타당하다. 단, fresh connector라고 하면서 배운 field를 보존한 점을 누락해서는 안 된다. encoder 자체의 기여는 fresh head/connector 감사로 별도 확인한다. B에서 encoder joint update를 추가하면 본 21에 포함되지 않은 새 실험이다.

**통과 조건:** 각 arm×stage의 1행 실행 표와 export/reload receipt가 생기고, P−G에서 차이가 무엇인지 코드 수준에서 한정되어야 한다. 21은 pipelines 수이며 A와 B 각각의 wall time을 더한다.

### F2. Primary mAP의 모집단과 gallery 완전성이 아직 확정되지 않았다 — Critical

첫 draft 182–210행은 완전 relevance의 gallery mAP와 cluster-aware CI를 요구한다. 방향은 맞지만 PASTIS의 소수 광역 tile만으로 광범위한 새 지역 모집단의 신뢰구간을 안정적으로 추정할 수 없다. parcel 수나 query 수를 늘려도 광역 독립 단위가 늘어나지 않는다. concept와 geography의 crossed clustering도 수가 적으면 복잡한 bootstrap으로 해결되지 않는다.

primary를 고정하기 전에 다음을 실제 수로 제출해야 한다.

1. held-out concept 수, 각 concept의 k=20 및 k=100 support 가능 수, 각 support의 고유 site 수.
2. query와 gallery의 단위(예: parcel/patch), positive/negative relevance를 결정하는 완전한 source 지원 범위.
3. train/support/query의 광역 tile·spatial cell·parcel/event 수와 overlap 제외 규칙.
4. gallery 후보 수·positive 수, no-positive query의 처리, macro 가중치, tie 규칙, query 후보 자기포함 여부.
5. paired uncertainty가 **고정된 네 지역 내 개념/관측 집합에 조건부인 것인지**, 새로운 광역 지역으로 일반화하려는 것인지.

PASTIS만 가능하면 결론을 고정 지역의 held-out concept/관계 조합으로 제한하고 tile별 결과를 전부 공개하는 경로가 가능하다. 이 경우 지역 일반화에 대한 bootstrap 유의성을 요구하거나 충족했다고 말하지 않는다. 광역 일반화가 main claim이면 독립적인 추가 geography/source가 먼저 필요하다. MADOS sparse label은 완전 retrieval relevance의 대체물이 아니다.

**통과 조건:** label-count 및 split audit에 따라 primary protocol을 dev에서 확정하고, 실제 불일치·cluster 구조를 반영한 power simulation을 한다. `.03 + 95% 하한>0`은 현재 투자 후보일 뿐 label/source 준비가 안 된 상태의 실행 가능 조건이 아니다.

### F3. 신규 CPT corpus와 평가 영역의 겹침은 ‘기록’만으로 충분하지 않다 — High

첫 draft 104행의 overlap 기록과 176–178행의 파생물 차단은 좋은 출발이다. 그러나 공개 원 checkpoint의 과거 노출이 불명확한 것과 **우리가 지금 새로 넣는 continued-pretraining 자료**의 노출은 다르다. 후자는 통제할 수 있다.

root가 제안한 locked query/support 영역의 신규 CPT 제외를 채택하는 것이 좋다. 동일 위치의 다른 날짜, 근접/중첩 subtile, 해당 위치의 파생 지도도 함께 검사한다. 제외 buffer와 공간 단위는 데이터 footprint에 근거해 정한다. source metadata가 부족해 제외가 불가능한 경우에는 새 지역 holdout 주장 대신 transductive 또는 label-held-out 평가로 명시해야 한다.

concept-held-out의 범위도 정확해야 한다. 기반 모델이 이미 coarse landcover를 학습한 사실을 인정하더라도, 이번 S/I/G/P의 label/teacher/derived query를 통해 보류한 세부 concept을 다시 넣어서는 안 된다. 공식 replay의 coarse parent label과 보류한 세부 concept의 관계를 ontology 장부에 적는다.

**통과 조건:** source→derived record의 split closure뿐 아니라 official replay→support/query의 geography overlap 검사 receipt가 있어야 한다.

### F4. 10/11–10/25 main 창에는 throughput 기반 자원 gate가 필요하다 — High

핵심 21 + 보조 12 + baseline + 개발 재실행은 2×H200이라는 모델명만으로 7주 내 가능하다고 판단할 수 없다. GPU의 공유 점유, sequence length와 timesteps, A의 frozen-VLM 입력 gradient, B의 LoRA, 데이터 decode/IO가 실제 비용을 결정한다. 현재 GPU-hour를 수치로 예측할 근거는 없다.

root가 제안한 **10/10 예산 gate**를 명시적으로 넣는 것이 적절하다. 최소한 C/S의 A, I/G/P의 A, 공통 B, frozen variants, 새로운 reader에 대해 대표 입력의 warm-up 및 측정 step을 구분한다. longest-timestep/고해상도 tail도 측정해 median만으로 OOM/시간 상한을 낙관하지 않는다.

예산 gate는 다음을 확인해야 한다.

- `합계(world_size × 단계별 실측 시간)`에 all seeds, validation, checkpoint, IO, baseline tuning, source/data auditing의 예정 시간을 더한 실행 표.
- 실제 사용할 수 있는 GPU 시간과 연구자가 지정한 실패 재실행 여유. 두 GPU 전용 점유를 기본 가정으로 두지 않음.
- 21 전체를 할 수 없을 때 test를 보기 전에 적용할 축소 순서. 1M query 증가와 10k scaling은 먼저 후속으로 옮길 수 있다. 핵심 G/specialist 통제를 없애는 방식은 부적절하다.
- main claim이 바뀌면 fresh reader·EO 감사 중 무엇이 필수인지도 다시 명시. 같은 결과에서 여러 후속 주장을 모두 유지하면서 실행만 줄이지 않음.

1k archive가 약17.9GB라는 metadata만으로 100k의 디스크·다운로드 시간을 선형 환산할 수 없다. 실제 compressed/uncompressed bytes, unique scene mapping, decode rate, free storage를 확인해야 한다. data 준비도 이 일정의 자원 항목이다.

**통과 조건:** 10/10에 자원 예산과 데이터/primary 준비가 함께 성립해야 100k GO. 아니면 실행 규모 또는 마감을 변경한다.

### F5. 최신 버전 선택은 적절하지만 v1.2 protocol 검증은 별도 작업이다 — High

v1.2-Base와 v1 공식 recipe를 구별하고 optimizer restart를 명시한 것은 타당하다. 이전 Tiny source inventory가 v1.2 model/normalizer/missing-band handling까지 확인했다는 뜻은 아니다. PASTIS 10밴드를 native 경로에 넣는 방식, band availability mask, timestamp와 CRS 정합은 실제 source/API 계약을 확인해야 한다. 편의를 위해 없는 밴드를 보간한 경우에는 zero-filled missing input과 구분하고 baseline에도 동일하게 적용해야 한다.

**통과 조건:** pinned config/weights의 공식 loader parity, normalization 1회, missing-band 정책, RGB 원 경로의 동작, EO input gradient, checkpoint 저장/복원의 한 번의 실제 실행 receipt. 이 결과는 성능 개선이 아니라 실행 준비의 증거다.

## 비용·통계 부분에서 동의하는 사항

- human120의 역할을 작은 독립 audit로 제한한 것이 맞다. 이를 새 concept×relation×K 모든 cell의 주 검정으로 확대하면 100만 원 계획과 충돌한다.
- 100k native bundles 중 audited labeled subset 크기를 별도로 보고한다는 점에 동의한다. 다만 각 arm의 replay/labelled mixture와 누적 노출도 같이 기록해야 한다.
- k-support inference와 test-time weight adaptation을 나눈 것은 적절하다. 후자는 별도 optimizer budget·실행 장부가 필요하다.
- 3 training seeds, support draw, query cluster를 구분하고 독립 n을 부풀리지 않는 원칙에 동의한다. seed별 값을 반드시 남겨야 한다.
- 직접 지도 S, generic G, specialist에 같은 source facts·negative·geometry를 허용한 통제가 꼭 필요하다. P가 이 통제를 넘지 못하면 범위를 줄이는 결론도 타당하다.

## 일정 및 최종 판단

공식 등록 2026-11-10 AoE, 본문 11-16 AoE, supplementary 11-23 AoE는 [CVPR 공식 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)와 [CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)에서 확인한 내용과 일치한다. 문서가 이를 실험 완료 보장으로 사용하지 않는 점도 맞다.

**이번 draft의 연구 규모와 정직한 비용 장부는 수용 가능하다.** 그러나 실험 표·primary 모집단·CPT holdout·실측 처리량이 확정되기 전에는 ‘큰 연구를 할 계획’과 ‘그 연구가 실행 가능함’을 구별해야 한다. 위 보완을 문서와 실제 manifest/config에 반영한 뒤 **제한된 방법 반증 pilot GO → 10/10에 main GO/축소/이관 결정**이 가장 타당하다. CVPR 채택 확률 수치는 부여하지 않는다.
