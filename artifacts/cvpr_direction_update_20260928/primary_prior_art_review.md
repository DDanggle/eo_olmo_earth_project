# Independent primary-source review — 2026-09-28

판정: **연구 문제에는 투자할 가치가 있지만, 현재의 positive/confuser + 지역 episodic adaptation 조합은 아직 CVPR 방법 기여가 아니다.** 아래는 신규성을 확보했다는 결론이 아니라, 확보할 수 있는지 가르는 반증 설계다. 서버·GPU·데이터 파일은 열거나 실행하지 않았다.

## 이미 주장할 수 없는 것

- EO encoder의 언어 정렬 자체: [MS-CLIP v3, 2025-08-04](https://arxiv.org/html/2503.15969v3), [DOFA-CLIP v2, 2025-07-22](https://arxiv.org/html/2503.06312v2)가 선행한다. 후자는 다중 센서와 teacher feature 보존까지 포함한다.
- 멀티스펙트럴 encoder를 실제 학습하면서 VLM 설명·분할을 결합: [SPEX v2, 2026-03-09](https://arxiv.org/html/2508.05202v2)가 가장 강한 직접 반례다. 단순한 연결부 학습 연구로 축소해서 소개하면 안 된다.
- 영역·시간·측정을 언어로 다루는 통합: [TerraScope v1, 2026-03-19](https://arxiv.org/html/2603.19039v1)의 영역 토큰, 마스크, 시간 프레임 선택을 넘어서는 것이 필요하다.
- 시간·밴드 정보를 언어 공간으로 옮긴다는 주장: [TimeSenCLIP, ISPRS JPRS 236, June 2026, 99–119](https://doi.org/10.1016/j.isprsjprs.2026.03.043)는 12개월·10밴드 Sentinel-2와 frozen CLIP 지상 사진 표현을 정렬한다. [공식 코드](https://github.com/pallavijain-pj/TimeSenCLIP)에서 입력과 학습 경로를 확인했다. 텍스트 caption을 직접 학습하지 않아도 언어 공간 활용이 가능하다.
- 지역 지식·검토 가능한 VLM 지시를 활용한다는 주장: [MetaSegNet v3, 2024-10-30](https://arxiv.org/abs/2312.12735v3)는 climate metadata를 ChatGPT 지리 프롬프트로 바꿔 cross-modal segmentation에 활용한다. [Restrict, Don't Retrain v1, 2026-09-01](https://arxiv.org/html/2609.00628v1)는 frozen segmentor에 VLM class restriction과 small-object box guidance를 공급한다. 후자는 새로운 EO encoder 교정 학습의 반례는 아니며, 성능 주장을 그대로 신뢰할 이유도 없다. 구조화된 근거 전달 자체가 새롭다는 주장만 막는다.

## 방법을 가장 직접적으로 위협하는 두 반례

**1. 지역별 support/query 학습은 이미 오래된 EO 방법이다.** [Meta-Learning for Few-Shot Land Cover Classification v1, 2020-04-28; CVPR EarthVision Workshop](https://arxiv.org/html/2004.13390v1)은 Sen12MS/DeepGlobe에 MAML을 적용한다. 소량 support로 가중치를 바꾸고 query로 meta-update한다. 다만 본문 §4.1, §6.3의 support/query는 같은 지역, 일부 실험은 같은 계절이다. 따라서 이것이 ‘A 지역의 교정만으로 B 지역을 무추가라벨로 개선’까지 입증했다고 쓰지는 말아야 한다. 그 차이는 유효한 평가 조건이지만 그 자체로 새 알고리즘은 아니다.

**2. spectral-spatial 특징과 관측 nuisance를 meta-task에 넣는 것도 선행한다.** [QMTN, Pattern Recognition 172 Part A, April 2026, 112331](https://www.sciencedirect.com/science/article/pii/S0031320325009926), DOI [10.1016/j.patcog.2025.112331](https://doi.org/10.1016/j.patcog.2025.112331)은 support/query의 업데이트 흐름을 나누는 twin network와 spectral-spatial attention을 사용한다. 출판사 본문은 atmospheric conditions/sensor calibration을 모사하는 radiometric variation과 noise를 meta-task 구성에 넣는다고 명시한다. 단일 HSI 데이터셋 안의 few-shot 분류이며, 시계열 VLM이나 지역 간 교정 전파의 직접 해결책은 아니다. 그래도 ‘EO에 맞춘 관측 변화 + episodic optimization’이라는 넓은 기여는 무너뜨린다. 출판사 검색 색인의 본문/초록/방법 요약을 읽었고, 직접 페이지 open은 403이므로 유료 전체 PDF까지 검증한 것으로 기록하지 않는다.

## 남길 대안 하나: 교정을 적용할 관측 조건까지 학습

제안 후보는 ‘지역 A에서 배운 구별을 B에 전달하되, B 관측이 그 구별을 지지할 때만 교정이 적용되도록 OlmoEarth 업데이트를 학습’하는 것이다. 예를 들어 대상/혼동대상이 RGB에서는 비슷하고 특정 시기·스펙트럼에서 구별되는 사례다. 일반 band dropout으로 모든 누락에 같은 답을 강요하지 않고, **구별 근거가 유지되는 관측에는 교정 효과를 보존하며, 근거가 없어 전문가도 판독할 수 없는 관측에는 교정된 단정을 억제**한다.

구체적 출발점은 target/confuser support로 만든 업데이트를 관측의 가용 정보에 조건화하고, 같은 query의 근거 보존/제거 쌍으로 그 업데이트의 효과를 학습하는 것이다. 관측 조건만 넣은 gating, selective prediction, invariance training의 재조합일 위험이 남는다. 이 문장만으로 신규성을 선언해서는 안 된다. ‘필수 밴드’라는 지정을 임의로 만들 수도 없다. 다른 밴드가 같은 정보를 줄 수 있으므로, 관측 부족 판정은 독립 검수와 측정 근거로 먼저 확인해야 한다.

## 방법 주장을 죽이거나 살리는 가장 작은 실험

지역 A support만 교정에 사용하고, 지리·모장면이 분리된 지역 B에서 target/confuser를 평가한다. 같은 B 샘플에 (a) 원관측, (b) 구별 근거를 보존하는 변화, (c) 독립 검수로 판독 부족이 확인된 근거 제거, (d) 제거 근거 복원을 적용한다. 모든 조건에서 교정 전후를 비교한다.

- 주요 지표: B의 target/confuser 구별 개선, 비관련 EO 과제 손상, 관측 부족에서 오답 확신, 근거 복원 후 회복. 단순히 제거 후 성능이 떨어지는 현상은 모든 모델에서 발생하므로 기여가 아니다.
- 최소 대조군: frozen OlmoEarth prototype/logistic regression; 동일 positive/confuser를 쓰는 LoRA/SFT; 동일 episodic sampling의 MAML/Reptile 계열; **동일 관측 쌍·밴드/시간 metadata·hard negative를 받는 일반 meta-learning + dropout/augmentation + calibration/gating**. 마지막 대조군을 생략하면 추가 정보 효과를 방법 효과로 오인한다.
- 고정할 것: support 수, label·근거 annotation, 관측·band·date 접근, 총 학습량, trainable parameter budget, VLM reader와 학습 데이터. wrong-support / confuser 교체와 지역명 제거도 필요하다.
- 결정 반증: 마지막 대조군이 같은 전이·보존·거절 곡선을 내면 새 교정 방법 주장은 철회한다. VLM 답만 개선되고 EO의 동일 구별 평가가 개선되지 않으면 OlmoEarth 표현 개선 주장을 철회한다. 다른 reader 재사용은 부가 검증이며 새로운 방법의 증거를 대신하지 않는다.

현재 공개 H5와 CPU gradient smoke는 이 가설의 실행 기반일 뿐이다. target/confuser와 관측 부족이 확인된 검수 쌍이 없어 이 실험을 이미 실행할 수 있다고 쓰면 안 된다. 다음 투자는 긴 새 연구계획보다 **실제 혼동 사례의 작은 검수 묶음과 위의 강한 일반 방법 대조군**에 두는 편이 타당하다. CVPR 채택 확률, 최초성, 인과적 설명 능력, 실제 성능 개선은 현재 근거로 주장할 수 없다.
