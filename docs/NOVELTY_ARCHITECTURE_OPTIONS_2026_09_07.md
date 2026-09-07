# 캐시를 고르는 연구에서, 재사용 가능한 정보를 학습하는 연구로

작성: 2026-09-07. 상태: **연구 설계 제안 / 미실행 / 사전등록 아님**.
근거: 로컬 코드·장부·등록 파일과 아래 1차 문헌. 이번 검토에서 서버 상태나 새 성능을 측정하지
않았다. 사용자에게 전달된 patch-2 진행률·예상 종료시각은 이 문서의 검증 결과가 아니다.
진행 중 체인, 봉인 설정, Korea label은 변경하지 않는다.

## 1. 판단부터

현재 결과는 쓸모 있는 empirical base다. 그러나 **선택지가 늘어나는 것**, **공유 head가 생기는 것**,
**patch-2가 +.03을 넘는 것** 중 어느 것도 그 자체로 CVPR method novelty를 주지 않는다.
반대로 이 세 가지가 안 되어도 다른 정보·학습 단계·목적을 시험할 수 있다.

가장 권하는 질문은 다음이다.

> 이미 저장한 OLMoEarth 표현에 어떤 작은 추가 정보를 보존해야, 새 지역·새 과업 또는 새 관측을
> 처리할 때 전체 시계열 재인코딩에 가까운 품질을 더 적은 저장·전송·갱신 비용으로 얻는가?

이것은 완성된 방법이나 확인된 빈 연구 영역이 아니다. 아래 선행연구와 경쟁해야 하는 **후보 질문**이다.
세 방향을 한 논문에 전부 넣지 않는다. **시간 증거 보존 → 인과적 증분 갱신**을 주 방향으로,
공간 잔차를 기존 paired-cache 자산으로 빠르게 검토할 조건부 방향으로 권한다.

## 2. 지금 보고에서 고쳐야 할 부분

| 보고의 주장 | 검토 결과 | 실무 조치 |
|---|---|---|
| patch-2가 마지막 방법 후보 | 특정 토큰화 변경의 실패는 다른 정보 경로·학습 목적의 실패가 아니다 | 이 실험의 중단선은 이 개입에만 적용 |
| 인코더는 남의 것이므로 비틀 여지가 없다 | frozen backbone에 source-trained readout/side branch를 붙이거나 허용된 fine-tuning을 할 수 있다. A3도 이미 기존 설계에 있다 | 새 발명이라고 하지 말고 미실행 baseline/후보로 구분 |
| target 라벨 5장이라 학습할 여지가 없다 | target support는 5장이지만 source 데이터로 공통 모듈을 먼저 학습할 수 있다 | source 학습과 target adaptation을 분리·비용에 모두 포함 |
| 이득이 모든 지역에서 같으면 method가 아니다 | 선택기는 필요 없어질 수 있지만, 정확도–비용 경계를 개선하는 방법은 여전히 가능하다 | 승자 이질성을 인위적으로 만들지 않음 |
| 5장으로 포화 | 이미 MS-113-AUDIT에서 철회. K20 .317 > K5 .294이며 pool .299와 nested 비교가 아니다 | 포화를 전제로 방법 가능성을 닫지 않음 |
| 작은 raw를 이겼으므로 원인이 표현임을 증명 | 파라미터 수만의 설명을 약화한다. pretraining·optimizer·head·입력의 인과 분리를 끝낸 것은 아니다 | matched OLMo encoder PEFT를 정확도·비용 상한 baseline으로 유지 |
| Sen12는 S2 12밴드×3시점+S1 | 해당 캐시 코드는 **S2 실관측 10밴드×선택된 12시점**, B01/B09 부재 masking이다 | 한국/레이더 실험의 입력과 혼동 금지 |
| 사전등록 기준은 이제 같은-trainer .291 대비 +.03 | 등록 파일과 verifier는 여전히 봉인 3-seed P4 기준이다 | 원래 gate와 같은-trainer paired effect를 별도 보고. 소급 변경 금지 |

코드 근거:

- `code/extract_sen12_fold_cache.py`: `select_timestep_indices`, `MODEL_BANDS`, `embed_crop`.
- `code/cache_decoder_train.py`: G32→64→128, G64→128→256→128. 동일 파라미터여도 공간 연산량과
  물리적 receptive field까지 같은 것은 아니다.
- 새 trainer seed 1과 봉인 3-seed 평균의 +.019는 trainer 변경과 seed 차이가 함께 섞인 값이다.
  같은-trainer arm끼리 비교하는 것이 맞지만, +.019 전부를 trainer bias로 확정하지 않는다.
- `config/second_fm_cache_prereg_v1_draft.json`의 `addendum_v1d_…` 절 `decision`과
  `code/verify_resolution_contract.py`: sealed-reference +.03/6-of-8 유지.
- 원래 설정의 pooling 결과 해석에는 상충하는 문장이 있으므로 결과와 함께 해석 정정을 병기할 것.
  실행 중 설정 파일을 고치지 않는다.

추가 계약 주의:

- variant 추출기는 cached uint16과 합성 year/day, 원래 추출기는 NetCDF 값과 실제 시각을 쓴다.
  양 경로의 입력 동등성을 확인해야 한다. 차이가 실제 성능을 바꿨다고 단정하지 않는다.
- `evidence/timestamp_asymmetry.json`은 같은 월·년 내 날짜 이동에서 5/5 embedding 동일을 이미
  보였다. 이것을 무시하고 day 차이가 원인이라고 말하면 안 된다. 이 검사는 합성 year와 전체
  materialization 경로의 동등성을 보증하는 검사도 아니다.
- `config/cost_ledger_v0.json`의 26G/11G/41G는 `du -sh` 반올림값이다. 3,825는 파일 mtime 간격이지
  계측된 accelerator active seconds가 아니다. 기존 JSON은 이 검토에서 바꾸지 않았고,
  그 값으로 만든 6–19과업 손익분기는 시나리오일 뿐 실측 결론이 아니다.

## 3. 최근 선행연구: 이미 있는 것과 실제 차별화 과제

2026-09-07 확인. arXiv만 확인된 항목은 심사 통과 논문으로 표현하지 않는다.

| 1차 자료 | 이미 하는 것 | 우리에게 요구되는 차별화 |
|---|---|---|
| [Temporal Sensitivity Analysis of Tessera Embeddings](https://arxiv.org/abs/2608.27175), 2026-08-27 preprint | frozen 임베딩, 라벨 효율, 관측 window, downstream task 비교 | 같은 곡선을 OLMo로 반복하는 것 이상: 과거 정보만 쓰는 증분 갱신 방법과 품질–비용 검증 |
| [Visual Token Codec](https://arxiv.org/abs/2608.08832), 2026-08-09 preprint | 공간·채널 token 압축, variable rate, 여러 task의 utility. 본문에 동일 bitstream의 multi-task 사용도 명시 | 단순 low-rank/양자화/shared cache는 신규 아님. 시간 갱신 또는 frozen base에 대한 incremental utility를 추가 검증 |
| [AnyUp](https://arxiv.org/abs/2510.12764), ICLR 2026 Oral; [FeatUp](https://arxiv.org/abs/2403.10516), ICLR 2024 | 외부 이미지 guidance 등을 이용한 고해상도 feature readout | 40 m feature를 예쁘게 확대하거나 RGB와 합치는 것만으로는 부족. 같은 guidance·같은 비용 대조 필요 |
| [Historical-prior EO generative compression](https://arxiv.org/abs/2605.08633), 2026-05-09 preprint | 과거 EO 자료의 prior로 압축·재구성·downstream 활용 | “지구는 같은 곳을 반복 관측하므로 residual을 저장한다” 자체도 신규 아님. 재구성 이미지가 아니라 기존 FM task output과 갱신 utility를 검증 |
| [BAN](https://arxiv.org/abs/2312.01163), 최초 2023 | frozen foundation model에 bi-temporal adaptation 경로를 결합한 변화탐지 | 시간 adapter 추가 자체가 아니라 재사용 memory·인과적 계산 예산·미학습 과업으로 차별화 |
| [Forward Compatible Training](https://machinelearning.apple.com/research/forward-compatible-training), CVPR 2022 | side-information과 변환기로 미래 embedding upgrade 지원 | “작은 sidecar로 버전 이전”을 새 아이디어로 제시하지 않음. FoldRefresh와도 역할 분리 |
| [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147), NeurIPS 2022 | 여러 차원 예산에서 쓸 수 있는 nested representation | 가변 차원만으로 기여 주장 금지 |
| [CoDEx](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Kuriyal_CoDEx_Combining_Domain_Expertise_for_Spatial_Generalization_in_Satellite_Image_CVPRW_2025_paper.html), EarthVision 2025 | 지역 전문가 결합으로 공간 일반화 | MoE라는 이름을 붙이는 것이 새로운 해결책은 아님 |

Tessera 논문의 window는 label date 중심의 양방향 window다. 이는 회고적 mapping에는 타당하다.
우리 online 문제에서는 cutoff 뒤 관측을 쓰지 않는 trailing window가 필요하다. **평가 설정의 차이**이지
그 논문이 잘못됐다는 뜻은 아니다. 인과적 benchmark만 만드는 것으로 method novelty가 완성되지도 않는다.

검색으로 미발견된 영역을 “세계 최초”라고 부르지 않는다. 위 문헌들은 최소 비교집합이며,
후보 한 개가 살아남으면 그 후보를 대상으로 streaming ViT·predictive feature coding·continual EO를
추가로 집중 조사해야 한다.

## 4. 후보 1 — 시간 증거를 보존하고, 새 관측으로 조금씩 갱신하는 캐시

### 왜 지금 이 코드에서 나오는 아이디어인가

현재 `embed_crop`은 contextualized token `h[p,t,g]`를 시간 t와 band-group g에 대해 평균내서
`z[p]` 하나로 저장한다. 이 값으로 지도 분할이 잘된다는 것은 기존 결과다.
그러나 **시간별 readout을 조금 추가하면 변화 과업에 더 도움이 되는지**는 다른 질문이다.
Transformer가 이미 시간 상호작용을 계산하므로 “평균=시간 정보 전부 소실”은 주장하지 않는다.

시점별 토큰이나 quality는 평균된 기존 `.npy`만으로 복구할 수 없다. 원래 추출 시점에서 새로
보존하거나 raw 재방문이 필요하고, 그 비용을 센다. 이것은 같은 평균 캐시에 적용한 MS-114/A2의 재포장이 아니다.

### 단계 T0: 학습 방법을 만들기 전, 추가 정보의 유용성만 측정

같은 scene·관측 cutoff·OLMo checkpoint·밴드·마스크·crop을 고정한다.

1. 현재 mean cache.
2. mean + 시간순 앞/뒤 구간 차이의 source-PCA 32/64차원 요약.
3. mean + source에서 학습한 작은 temporal sketch(시간 basis/projection; target에서 동결).
4. 시간별 token 전체를 사용하는 작은 temporal head: 압축 전 비교 기준이지 수학적 상한은 아님.

2와 3은 저장 byte와 decoder parameter를 맞춘다. first/last뿐 아니라 mean/variance, concat→PCA를
강한 단순 대조군으로 본 실험에 포함한다. 비슷한 방법을 무한 탐색하지 않고 후보·예산을 먼저 고정한다.
하나의 terminal mask밖에 없는 Sen12 결과는 최종 지도 utility만 말한다. **변화 시점·조기 탐지 성능은 못 말한다.**

### 단계 T1: 살아남으면 source-trained causal update 모듈로 확장

구조 초안:

```text
새 관측 x_t ── frozen OLMo(single-acquisition path) ── u_t
                                                    │
이전 memory m_(t-1) + 실제 Δt + 관측 quality ── 예측 ─┤
                                                    ↓
                                예상과 다른 잔차 r_t의 작은 코드
                                                    ↓
                           memory 갱신 + 시간 증거 저장
                                                    ↓
                             토지피복 / 변화 / 사건별 판독기
```

예: `r_t = Q(P(u_t) - F(m_(t-1), Δt, quality))`,
`m_t = U(m_(t-1), decode(r_t), Δt, quality)`.
F/U/P는 source에서 학습, Q는 quantization 또는 고정 차원 코드다. MoE나 RL은 첫 버전에 넣지 않는다.

학습 목표 후보는 같은 cutoff의 full-window OLMo teacher task output 유지 + 실제 source task loss +
추가 byte/계산 penalty다. teacher가 틀릴 수 있으므로 feature/logit distillation만을 정답으로 삼지 않는다.
실제 downstream label로 학생·teacher를 함께 평가한다. source module 학습은 post-training이며,
target K=5/20 head adaptation과 별도 단계다.

핵심 위험: full-window OLMo의 attention은 과거 token도 새 영상에 따라 달라진다. 따라서 새 영상 한 장의
embedding과 memory만으로 갱신하는 것은 **근사 방법**이다. 기존 transformer KV cache의 정확한 재사용이라고
말하면 안 된다. 이 근사 오차와 비용 감소의 관계가 연구 대상이다.

### 신규성 후보가 성립하는 조건

- 같은 cutoff·관측 집합에서 full-window 재인코딩과 비교한다. 한쪽만 좋은 구름/미래 관측을 받으면 안 된다.
- EMA/GRU/단순 temporal PCA와 generic feature codec보다 더 나은 quality–update-cost 결과를 보여야 한다.
- 새 과업 head를 추가할 때 과거 raw 재방문을 줄이면서 그 과업에도 유용해야 한다.
- 정상 변동과 사건 변화의 구분, 허용 오경보에서의 탐지 지연, 관측 공백별 오류를 보고한다.
- “재난 전에 지진 예측”이 아니라 **관측이 확보된 뒤 변화 파악을 갱신**하는 일이다.

이 네 조건 없이 “temporal adapter가 +.02”만 나오면 응용 개선이지 강한 새 cache 방법 주장은 아니다.

### 데이터 및 누수 경계

- `geobench_to_tiles.py`는 현재 `months_0_11=[6]*T`와 chip index를 저장한다. DEN 16,000캐시 완료는
  날짜별 같은 장소의 causal sequence가 준비됐다는 뜻이 아니다. 실제 날짜·site·label date를 먼저 복원한다.
- source-trained module도 geographic source-only split을 지킨다. target의 unlabeled 데이터를 쓰면 transductive로 별도 표시.
- full sequence에서 attention을 계산한 뒤 과거 token만 잘라 쓰면 future 정보가 남는다.
  teacher·baseline 모두 cutoff별 prefix로 계산한다.
- 12개 best-clear 장면을 전체 기간에서 고르는 기존 규칙은 online용이 아니다. 각 cutoff 이전 관측에서만 고른다.
- 실시간 게시 지연을 주장하려면 acquisition time뿐 아니라 available-at도 필요하다.
  과거 available-at을 모르면 관측시각 기준 retrospective streaming 평가로 제한한다.

## 5. 후보 2 — 40 m 캐시에 조밀한 공간 정보의 작은 잔차만 보존

**기존 paired p4/p2를 이용할 수 있어 실행 준비가 가장 싸다. 다만 p2의 추가 utility가 먼저 필요하다.**
20 m 전량 캐시가 유리한 경우에도 “비용 4배를 반드시 낸다”는 결론은 성급하다.
같은 장면의 coarse/fine feature는 상당 부분 중복일 수 있다. 중복을 얼마나 제거할 수 있는지는 미측정이다.

구조 초안:

`z40`는 기존 그대로 유지한다.
`r = Q(Cθ(z20 - Aθ(z40)))`, `z20_hat = Aθ(z40) + Dθ(r)`.

A는 공간 확대와 채널 정렬, C/D는 작은 source-trained 잔차 압축/복원기다.
patch 변경 후 feature 좌표계가 같다고 가정하지 않는다. 기존 head는 명시적으로 원래 `z40`를 읽고,
새 head만 `(z40,r)` 또는 `z20_hat`를 읽는다. 구 head 무회귀는 **구 경로 유지**의 성질이며,
새 feature를 옛 head에 넣어도 호환된다는 뜻이 아니다.

순수 FP16 배열의 설계 산술(헤더·index·weight·metadata 제외, 실측 결과 아님):

| 저장물 | tile당 payload |
|---|---:|
| 768×32×32 base | 1.50 MiB |
| 768×64×64 dense | 6.00 MiB |
| 32×64×64 residual | 0.25 MiB |
| base + 위 residual | 1.75 MiB |
| base + 64채널 residual | 2.00 MiB |

즉 질문은 “4배 캐시가 좋은가”보다 “약 1.17–1.33배 payload로 dense utility를 얼마나 회수하나”가 된다.
**이 숫자는 가능 용량이지 품질 보장이 아니다.** dense feature를 먼저 계산해서 r을 만들면 초기 인코딩
계산은 줄지 않는다. 첫 논거는 저장·전송·후속 raw revisit 절감이고, encoder 속도는 별도 측정이다.

필수 대조:

- 같은 byte의 dense PCA/quantization, 단순 concat→projection, residual PCA.
- VTC 등 일반 feature codec; 구현 불가하면 논문에서 경쟁방법 우월 주장 금지.
- bilinear, AnyUp/FeatUp guidance baseline. guidance RGB를 제공한 방법은 raw-free arm과 분리한다.
- 같은 source training 예산·decoder·input cutoff, seed별 paired test.

신규성은 residual/low-rank 자체가 아니라 **기존 캐시를 보존한 채 제한된 추가 저장량으로,
학습에 없던 task의 dense utility를 유지하는 방법과 그 근거**에서 찾아야 한다.
FCT·VTC가 가까운 경쟁이므로 이 조건을 만족해도 novelty 자동 확정은 아니다.

p2가 안 좋으면 이 exact spatial residual branch는 닫는다. temporal branch까지 닫지 않는다.
원시 고주파·DEM guidance는 별도 정보 추가 실험으로 가능하나 AnyUp/FeatUp·멀티모달 fusion과
중복이 커서 첫 method 후보보다 강한 comparator/응용 개선 후보로 둔다.

## 6. 후보 3 — source-stage OLMo post-training의 이득을 작은 branch로 옮기기

source label로 encoder의 허용된 block/LoRA를 적응하고, 그 성능을 frozen OLMo의 intermediate
feature를 보는 작은 side branch에 distill하는 방향이다. target 라벨 5장만으로 이 branch를 학습하지 않는다.

이 방향은 정확도 개선 baseline으로 중요하지만 **새로 발견한 아이디어는 아니다**.
기존 A3 설계·PEFT·feature distillation과 겹친다. A2처럼 고정 최종 cache에 residual만 붙인 실패와는
입력 정보·학습 단계가 다르지만, 그 차이만으로 CVPR novelty는 아니다.

권장 사용: 첫 번째 후보의 압축 전 quality reference와 비교상대로 둔다. LoRA teacher가 충분한
추가 utility를 못 만들면 distillation branch도 크게 돌리지 않는다. learned memory가 이 teacher에
대해서도 정확도·계산 이점을 보일 때 한 방법의 구성요소로 승격한다.

릴리스 호환성 목적까지 동시에 추가하지 않는다. 그것은 FoldRefresh/FCT와 구분해서 별도 연구해야 한다.

## 7. 다음 실행은 작은 반증 실험부터 — 아직 실행 명령 아님

### 0단계: 현재 4-arm을 자기 질문으로 닫기

원래 등록 gate, 같은-trainer p2-p4, upsample 및 pooling 대비 효과, seed 차이를 분리해 보고한다.
20 m 효과가 확인되면 source-only PCA residual pilot을 CPU에서 먼저 할 수 있다.
same-trainer 기준 변경을 “원래 사전등록”으로 서술하지 않는다.

### 1단계: 시간 정보 추가의 소규모 readout screen

- 개발 지역: 기존 노출 Sen12 china/chimanimani. Korea·새 DEN test는 보지 않는다.
- T0의 4 readout × 2 dev 지역 × seed 1/2/3 = **24 decoder run**.
- 첫 screen은 source-supervised head와 고정 source validation으로 readout utility를 측정한다.
  K5 adaptation, streaming update, 다중해상도 mixture를 동시에 넣지 않는다.
- 추가 extraction 양은 선택한 source train/val 및 두 dev region의 manifest로 먼저 고정한다.
  시간별 unpooled 전량 dump 대신 streaming accumulation/PCA 학습 표본을 사용해 disk 증폭을 피한다.
- 한 seed의 최선 지역을 보고 확증 fold를 더 고르지 않는다. 이들은 모두 development다.

실행 전 확정할 screen 기준의 **제안**: 작은 sketch가 mean 대비 AP/IoU에서 두 dev 지역 같은 방향,
한쪽 적어도 절대 .02 개선, 다른 쪽 .01 이상 악화 없음; 같은-byte 단순 sketch와도 비교한다.
이 수치는 현재 미등록이며 기존 +.03 gate의 변경이 아니다. paired 공간 CI·seed 분산·오경보를
효과 크기와 함께 본 뒤, 확증 기준은 외부 결과를 보기 전에 따로 동결한다.
단순 sketch와 learned가 같으면 시간 정보의 utility만 인정하고 learned method 승리라고 하지 않는다.

### 2단계: T0가 살아남은 경우에만 인과적 갱신 pilot

시간·site가 검증된 source/development sequence에서 full-window OLMo, one-date/EMA,
budget-matched GRU, learned update를 같은 cutoff로 비교한다. 총 encode 호출 수, peak memory,
wall time, 추가 bytes, 현 상태 분할 AP/mIoU를 계측한다. dated change label이 있으면 지연–FP도 추가한다.
초기 encoder 비용과 매 update 비용을 분리한다. 근거 없는 “12배 빠름” 같은 추정은 하지 않는다.

### 3단계: 외부 검증과 논문 주장

학습에 안 쓴 공개 과업/지역, 다른 backbone 1종에서 같은 memory 원리가 작동하는지 검증한다.
다른 FM의 계약을 일부러 불리하게 만들어 승자를 바꾸지 않는다. 학습한 task에 대한 utility와
새 head만 학습한 unseen task utility를 별도 보고한다.

Korea는 준비 gate 후 **한 번 개봉**한다. land-cover·deforestation에는 기존 source head가 없으므로
새 supervised head 과업이고, landslide만 현재 source-transfer head 외부 시험으로 볼 수 있다.
세 label이 같은 cube를 공유하는 것은 N=1/2/3 actual cost와 negative transfer를 재는 장점이지
세 독립 데이터셋이라는 뜻이 아니다. class 이름으로 넓음/중간/세밀함을 단정하지 말고
source/development label의 object area·boundary length·rare prevalence를 측정해 가설을 동결한다.
date-specific paired label이 없다면 Korea도 “조기 변화 탐지” 시험으로 부르지 않는다.

## 8. 논문과 포트폴리오에서 무엇을 남기나

- 기존 Sen12/Solar: cached representation의 유용성을 보여주는 baseline evidence.
- family/해상도/label curve: 어디서 utility가 깨지는지 설명하는 development diagnostics.
- 새 method: **보존/갱신할 증거를 학습하는 하나의 모듈**. selector를 꼭 붙일 필요 없음.
- Korea shared-cache: 새 task 추가의 실제 비용·회귀·전이 시험.
- Nepal: 별도 repo의 retrospective application. 이번 설계의 training/tuning 자료로 자동 편입하지 않음.
- release bridge/FoldRefresh: 이미 있는 호환성 자산. 새 method에 억지로 합치지 않음.

CVPR를 겨냥할 핵심은 실험 수가 아니라 다음 세 증거다.

1. 강한 단순 방법·최근 방법이 남긴 병목을 새 메커니즘이 줄인다.
2. 같은 정보·label·compute/storage 예산에서 정확도–비용 관계를 개선한다.
3. 그 이유와 효과가 unseen 지역/과업에서도 유지된다.

현재 empirical 결과는 이 실험을 시작할 자산이지 acceptance 보증은 아니다.
patch-2 한 번의 실패를 CVPR method 전체의 사형선고로 삼을 근거도 없다.
