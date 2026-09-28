# OE4: OlmoEarth–VLM의 신규성 후보를 넓게 검토한 지도

검토일 2026-09-27. 담당 `/root/cvpr_prior_art_redteam`. 기존 P-v0를 옹호하기 위한 검토가 아니다. 원 연구계획과 원 실험을 변경하지 않았고, 이 파일 자체에 GPU 실행 결과는 없다. 학습 담당의 실제 실행 기록과 분리한다. 아래는 primary paper/공식 학회 자료로 확인한 문헌과 그에 대한 연구 제안이며, '아무도 하지 않았다'는 완전한 선행 부재 증명이 아니다.

## 결론: 두 축을 남기고, 기능 수를 늘리는 축은 내려놓는다

**응용 가치가 큰 1순위:** 사람이 고친 공간 개념이 다른 지역과 작업에서도 재사용되는 학습. 단순 support prototype이나 listwise contrast에서 끝나면 독립된 신규 방법이라고 할 수 없다. 교정 범위와 다른 개념의 보존, 센서 차이를 함께 다루는 구체적 학습 설계가 필요하다.

**OlmoEarth 학습을 직접 바꾸며 먼저 반증하기 좋은 2순위:** 관측을 바꾸었을 때, 질문에 필요한 정보까지 무조건 같게 만들지 않는 표현 학습. 유지할 속성과 불확실해져야 하는 속성을 구별하고, 추가 관측으로 실제 오차가 줄어드는지를 학습·평가한다. missing-band robustness나 abstention head만 붙이는 방식으로는 부족하다.

두 축을 한꺼번에 모든 모듈에 넣지 않는다. 먼저 둘의 단순 대조군을 만들고 한 축에서 비자명한 효과가 나오는지 확인한 뒤 결합 가치를 본다. 데이터·학습량 확대만으로 신규성이 생기지는 않는다.

## 1. 공간 교정의 범위를 배우고 전이시키기 — 1순위 연구 후보

**문제:** '이 종류의 영역을 잘못 보고 있다'는 전문가 교정이 단일 질문 정답으로 저장되지 않고 다른 지역·관측에서 재사용될 수 있는가? 다른 피복이나 기존 EO 능력을 망가뜨리지 않는가?

**이미 있는 것:** [ReasonEdit v5, ICML 2026](https://arxiv.org/html/2602.02408v5)는 인간 rationale을 patch/fact codebook에 저장하고 검색하여 편집을 전이한다. 가중치를 바꾸지 않는 방식이다. [SegGPT](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.html)와 [SEEM](https://papers.neurips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html)은 이미 예시·언어 prompt 일반화를 다룬다. [M3Bench](https://arxiv.org/abs/2607.05310)는 다른 전문 분야에서도 편집의 조합 일반화와 locality가 어렵다는 평가를 제시한다. 의료 성과를 EO 성과로 전용할 수는 없다.

**남는 방법 후보:** 교정을 `(양성 영역, 구별해야 할 대조 영역, 수정하려는 속성, 적용 가능한 관측 조건)`으로 표현한다. 한 교정이 바꿔야 하는 query와 유지해야 하는 query를 독립 source annotation으로 구성해, EO 공간 표현의 수정 방향과 적용 범위를 함께 학습한다. 예시 retrieval로 매번 답을 붙이는 것과 달리 학습 후 OlmoEarth가 새 영역을 구분할 수 있어야 한다. 임의 직교 subspace가 의미 분리를 보장한다고 주장하지 않는다.

**결정 반증:** 같은 K labels, 같은 source facts와 query exposure에서 prototype/SFT/LoRA/ReasonEdit형 retrieval/generic episodic을 비교한다. 수정 대상의 새 scene 정확도, 관련 조합 전이, 비관련 개념 손상, 새 reader 전이를 함께 본다. mask가 좋아져 area도 좋아진 것은 별도 능력 전이로 세지 않는다. support를 빼도 원래 알아맞히는 사례, support alias만 외우는 사례를 분리한다.

**현재 실행성:** 기존 BEN scene-presence는 영역 교정 gold가 아니다. PASTIS parcel/reference 또는 dense annotation을 검증하면 작게 시작 가능하다. 20개 진단 영상의 센서값 교정을 '전문가 개념 교정'으로 부르면 안 된다. 시간표상 label audit가 우선 제약이다.

**폐기 조건:** 같은 prototype+negative+reweighting이 결과를 설명하거나, gain이 동일 query memorization에 국한되면 신규 학습법 주장 폐기. 실용적인 적응 도구라는 결과는 남길 수 있다.

## 2. 관측이 가진 정보와 잃은 정보를 구별하는 EO 표현 — 2순위/첫 학습 진단 후보

**문제:** 어떤 밴드·시간·공간 영역을 빼면 같은 답을 유지해도 되지만, 다른 질문에는 판단 근거가 사라진다. 이를 전부 동일 feature로 정렬하면 후자의 차이를 학습에서 지울 수 있다는 가설이다. 아직 OlmoEarth에서 실제로 그런 결함이 있다는 결과는 없다.

**이미 있는 것:** [OlmoEarth v1.2](https://arxiv.org/html/2605.20804v3)는 band dropout·masking 개선을 포함한다. [AOM](https://arxiv.org/html/2512.17224v1)은 missing band와 scale alignment를 다룬다. [VHM v4](https://arxiv.org/abs/2403.20213v4)는 deceptive questions를 학습한다. 최신 [I Don't Miss You, but I Do](https://arxiv.org/html/2609.07596v1)는 모달리티 제거/복원에 관한 모델의 설명과 실제 행동의 불일치를 직접 평가한다. 따라서 '관측 부족을 말한다'와 '정보를 복원해 비교한다' 자체도 새롭지 않다.

**남는 방법 후보:** sensor availability를 단순 augmentation tag가 아니라 task/attribute-conditioned 정보 제약으로 이용한다. 정상 관측에서만 얻을 수 있는 정밀 속성은 조건부 분포로 예측하고, 관측 가능한 속성만 안정적으로 정렬한다. EO latent의 의미 성분과 관측 의존 세부 성분을 나누는 설계는 후보지만, arbitrary factorization이나 loss 추가만으로 새롭지 않다. 핵심은 동일 입력/모델/학습량에서 무조건 consistency보다 더 나은 충분 관측 성능과 부족 관측 calibration을 동시에 얻는 것이다.

**결정 반증:** 원 목적+random band dropout, 무조건 feature consistency, heteroscedastic regression, post-hoc calibration, 제안 방식의 matched comparison. 밴드 복원에 따른 실제 error reduction, proper scoring rule, selective risk와 coverage, OOD missingness, 유지해야 할 EO task를 함께 평가한다. 모델의 자기 확신을 관측 가능성 정답으로 쓰지 않는다. 더 많은 관측이 각 샘플마다 항상 정답 확률을 높여야 한다는 잘못된 단조 조건도 강제하지 않는다.

**현재 즉시 가능한 학습:** raw S2로 native band 통계/정규화 비율의 고정된 작은 진단을 만들고 학습/holdout scene을 나눈다. DN 비율은 sensor-derived target이며 식생 건강 gold가 아니다. absent band의 수치가 통계적으로 예측 가능할 수 있으므로 '밴드가 없으면 추정 자체가 틀리다'고 채점하지 않는다. 실측값/추정값의 구분과 추정 오차를 평가한다. 원 픽셀 공식 계산기는 관측이 완전할 때의 정확한 계산 기준선으로 반드시 포함한다.

**본 연구로 가려면:** 실제 independently labeled task와 실제 관측 손실/센서 전이에서 같은 현상을 보이고, simple uncertainty baseline 이상의 이유가 필요하다. 합성 결손·쉬운 공식만 맞힌다면 공학 진단으로 끝낸다.

## 3. 해상도·영역 분할이 달라도 물리량과 의미가 맞는 학습 — 예비 후보

**이미 있는 것:** [AnySat, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html), [FlexiMo](https://arxiv.org/abs/2503.23844), AOM은 scale/sensor flexibility를 다룬다. [Geometric Consistency Protocol](https://arxiv.org/abs/2606.17564)은 RPC 기반 위성 multi-view matching의 물리적으로 타당한 평가를 제안한다. 이 논문의 stereo geometry는 우리 지역 집계와 같은 문제라고 주장하지 않는다.

**남는 후보:** georeferenced subregion의 합과 전체의 측정이 맞고, 리샘플링과 예측/집계의 순서를 바꿨을 때 정해진 양만 보존되는 task-conditioned 학습. 단순 이미지 scale invariance보다 구체적이다. 평균·합·비율의 변환 법칙이 다르고, 경계/작은 물체는 downsampling으로 사라질 수 있으므로 동일 mask를 강제하지 않는다.

**결정 반증:** area-weighted interpolation+segmentation+calculator와 동일 latent capacity 대조. unseen GSD/tile partition, boundary fragmentation, regional count/area, fine-grained EO task retention. conservation loss만 낮아지고 틀린 값을 일관되게 내면 실패다.

**실행성:** raw DN의 footprint-preserving synthetic test는 바로 가능하지만 참조 면적은 현재 BEN 자료에 없다. 10m/60m geotransform 문제는 선행 수정이며 연구 기여로 재포장하지 않는다. 올바른 affine/mask annotation 확보 뒤 실제 실험한다.

## 4. 질문에 따른 토큰·해상도·근거 선택 — 주 신규성에서 제외

[SA-GEM v1, 2026-08-15](https://arxiv.org/html/2608.15075v1)는 query router가 해상도를 고르고 공간 구조·중복·관련성을 결합해 tokens를 선택한다. [Rift v1, 2026-09-24](https://arxiv.org/html/2609.29029v1)는 위성영상의 query-conditioned tile pruning과 token budget을 함께 다룬다. TerraScope도 선택한 mask의 시각 tokens를 reasoning에 넣는다.

native EO의 spectral/temporal evidence를 선택한다는 적용 차이만으로 이 계열을 넘어섰다고 하기 어렵다. 희귀 영역의 recall과 정량 편향을 제한하면서 error/latency Pareto를 개선하는 명확한 메커니즘이 있다면 별도 연구가 될 수 있으나 지금 작은 budget으로 세 번째 주축까지 늘릴 이유가 없다. 현재 모델의 효율 부품과 강한 baseline으로 활용한다. mean/learned-query/equal-token/uncertainty allocation 대조 없이 단순 token 증가를 학습 기여로 부르지 않는다.

## 5. 개념×연산 조합 일반화 — 핵심 평가로 유지, P-v0를 새 원리로 주장하지 않음

SegGPT/SEEM과 [RSCoVLM](https://arxiv.org/html/2511.21272v2)이 강한 출발점이고, [Composition-Aware Pretraining](https://arxiv.org/html/2608.30817v1)은 장면 texture mixture 보존을 이미 다룬다. texture mixture와 semantic operation composition은 다른 개념이다.

P-v0의 independent tuple loss에 listwise contrast를 추가하는 방식은 우선 ablation이다. general episodic/meta-learning 및 supervised contrastive에 대한 차이와 이론/현상 증거가 없는 한 CVPR method의 중심으로 고정하지 않는다. 새로운 개념을 spatial relation·검색에 쓰는 held-out protocol은 1순위 교정 전이의 핵심 시험으로 가치 있다. 전체 train set의 concept×operation holdout과 episode 내부 operation withholding을 구별한다.

## 6. 여러 모달리티를 생성해 설명 능력 확장 — 현재 주축에서 제외

[TerraMind v5](https://arxiv.org/abs/2504.11171v5)는 any-to-any 생성과 Thinking-in-Modalities를 다룬다. [Galileo v3](https://arxiv.org/abs/2502.09356v3)는 global/local targets와 많은 modalities를 다루며 OlmoEarth 계열 설계에 가까운 관련 연구다.

OlmoEarth에 SAR·분광·지도 생성 head를 붙이고 그 결과를 LLM에 주는 것 자체는 차별화가 약하다. 생성된 missing modality를 실측 근거처럼 제시해서도 안 된다. 이를 이용하려면 '언제 합성 보완이 도움이 되고 언제 추정 편향만 강화하는가'라는 독립 기여가 필요하며 2순위의 관측 의존 평가와 연결될 수 있다. 지금은 수요/자료/계산이 크고 새로운 주축으로 시작하지 않는다.

## 7. 전문가·AI의 검증 가능한 감독을 대규모 학습에 연결 — 필요한 자료 기여

[MS-CLIP v3](https://arxiv.org/html/2503.15969v3)는 자동 caption과 expert review, [SPEX v2](https://arxiv.org/html/2508.05202v2)는 분광 지수 기반 언어 감독, [TerraScope](https://arxiv.org/html/2603.19039v1)는 mask-derived reasoning data를 이미 이용한다. 따라서 여러 AI의 투표, source masks에서 area 계산, 구조화된 답 생성 자체는 새롭지 않다.

가능한 연구 차이는 source ontology/부분 라벨/관측 제약을 지킨 교정이 일반 대량 synthetic supervision보다 같은 human-hours에서 더 높은 downstream transfer를 만드는지다. AI 합의율을 정답률로 쓰지 않고 인간 audit와 독립 source references로 평가한다. annotation efficiency curve, new-domain coverage, contradiction rate가 필요하다. 먼저 1순위 연구의 data engine으로 구현하고 독립적인 data-method 성과가 나온 경우에만 별도 기여를 주장한다.

## 8. 근거 획득의 순서·다음 관측 선택을 학습 — 장기 확장 후보

[NTEP](https://arxiv.org/abs/2609.03493)는 agentic VLM의 필요한 evidence/tool 호출을 감독한다. TerraScope는 mask retrieval 순서를 다루고, 최신 modality-missingness 연구도 추가 입력의 효과를 평가한다. 단순히 '정보가 부족하면 더 본다'는 새롭지 않다.

남는 EO-specific 문제는 실제 센서 비용·공간해상도·관측 시차를 고려해 '어떤 추가 관측이 해당 의사결정 오차를 가장 줄이는가'를 배우는 것이다. 전체 관측을 가진 archive에서 가린 subset부터 순차적으로 공개해 counterfactual value를 평가할 수 있다. 새 관측 요청을 실행할 필요 없이 retrospective test가 가능하지만, 현재 paired time/sensor archive completeness가 확인되지 않아 즉시 실행용 주축으로 부적합하다. 공개 archive의 비용 가정과 실제 관측 비용을 구별해야 한다.

## 9. 변경·계절·센서 차이를 분해하는 시계열 학습 — 원 자료 확보 후 확장

OlmoEarth/Galileo/AnySat는 이미 시간·모달리티 구조를 학습하고 TerraScope는 temporal questions를 지원한다. PASTIS crop type 같은 연간 라벨만으로 특정 날짜의 피해·회복·고사 정답을 만들 수 없다.

가능한 차이는 동일한 계절 state를 보존하면서 검증된 disturbance/회복에 민감한 task-conditioned representation이다. 계절 augmentation과 true change를 무조건 negative로 만들지 않는 설계가 필요하다. 참조 사건/phenology/time-alignment가 있는 data를 확보하고 반복 계절 baseline, temporal encoder, timestamp-blind VLM과 비교한다. 지금 이 축을 넣으면 정답 구축이 병목이므로 1/2순위 이후로 둔다.

## 실행 담당에게 전달할 첫 GPU 학습 판정

1. **지금 시작할 수 있는 것:** 최신 v1.2 loader/gradient/save-resume 검증 + 공식 목적의 작은 batch. 이는 프로젝트 기반을 만들지만 신규성 결과는 아니다.
2. **현재 BEN raw로 가능한 비교:** 원본 frozen/standard fine-tuning/normal consistency/task-conditioned preservation의 같은 data·step 진단. train/dev geography를 미리 고정하고 source label leakage를 막는다. native DN statistics와 결손 restoration은 기능 진단으로 한정한다. 직접 band formula·단순 회귀·heteroscedastic baseline이 강해야 한다.
3. **현재 하면 안 되는 해석:** probe 개선을 vegetation health/새 개념 교정/실제 VLM 개선으로 확대하기. 20개 scene를 반복 샘플링해 대규모 독립 표본이라고 세기. 오류나 비효과가 나오면 같은 test에서 방법을 계속 바꾸기.
4. **다음 성과의 최소 형태:** 실행 코드/입력 provenance/기존대조/학습 전후 checkpoint/held-out 결과와 실패 사례. 결과가 단순 baseline과 동등해도 그대로 기록한다.

실험에서는 각 축의 기본선 하나와 차이 하나를 선택한다. 새로운 조합을 무한히 만들어 검증력을 소모하지 않는다. 큰 논문 목표는 **OlmoEarth가 더 적은 교정으로 다양한 실제 과제에 쓰이게 만드는 모델과 학습법**으로 유지하고, 어떤 메커니즘이 그 목표에 기여하는지를 위 반증으로 선택한다.
