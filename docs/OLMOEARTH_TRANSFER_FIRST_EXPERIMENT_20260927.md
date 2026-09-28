# 지역을 넘어 전문가의 판독 기준을 재사용하는 OlmoEarth VLM

> **후속 설계 v4: [자료·아키텍처와 결정 비교](OLMOEARTH_VLM_ARCHITECTURE_DATA_PLAN_20260927.md).** GPU1 연결 검사 이후의 현재 압축 진단, PASTIS/Kuro 자료 계약, 같은 정보의 2×2 비교와 독립 검토를 반영했다. 이 문서의 당시 계획·결과는 이력으로 보존한다.

> **9/27 15:43 KST GPU1 완료 — [실제 VLM 실행 결과와 다음 실험](OE4_GPU1_VLM_PROGRESS_20260927.md).** native8/32step 학습·저장복원과 실제 Qwen3-VL8B FP32 연결/역전파 검사 통과. 언어손실로 OlmoEarth211개·연결부4개tensor에gradient, 추적가중치변경과 원RGB경로보존 확인. BF16캐시검사실패는보존하며 FP32통과와구분한다. VLM checkpoint저장복원·새지역전이·성능개선은미검증. 이번GPU작업종료·결과/로그회수완료. 다음은PASTIS공개라벨입력2사례계약확인이다. 아래이전대기/실패표시는당시이력이다.

2026-09-27. 통합 연구계획 v3 / OE5 개발 계획. **신규성·성능 결과나 확증 사전등록이 아니다.** [v2 통합 설계](OLMOEARTH_UNIFIED_CONTRIBUTION_20260927.md)의 큰 목표를 유지하면서, 최근 선행과 첫 실험을 구체화한다. 동반 설정은 `config/oe5_correction_transfer_development_v0.json`이다.

## 1. 크게 풀 문제

**전문가가 한 지역에서 알려준 판독 기준을, 환경과 관측 조건이 다른 곳에서도 다시 활용할 수 있게 한다.**

사용자는 대상과 혼동 대상을 몇 번 지정하고, 새 지역에서 찾기·영역 확인·조건부 비교·근거 질문을 이어간다. 모델은 그 교정이 다른 기후·계절·센서에서도 유효한지 판단하고, 실제 EO 관측에 맞춰 적용한다. 새 지역마다 같은 지식을 다시 라벨링하고 모델을 따로 만드는 비용을 줄이는 것이 응용 목표다.

실험 가능한 중심 문제는 **교정의 전이**다. 같은 개념을 가르친 예시라도 어느 환경에서는 도움이 되고 다른 환경에서는 혼동을 키울 수 있다. 그 차이를 학습할 수 있는지는 아직 모른다. 지역 정보를 넣으면 유용하다는 가정부터 고정하지 않는다.

첫 분야는 관측과 정답이 있는 작물·식생 구분, 독립 두 번째 분야는 수면·침수로 둔다. 구조물은 해상도와 라벨이 맞는 자료에서 후속 확장한다. Sentinel/Landsat만으로 개별 고사목·작은 해양 쓰레기·피해 원인을 항상 식별한다고 약속하지 않는다. 기후 배경으로 사건의 원인을 확정하지 않는다.

## 2. 2026년 선행이 이미 차지한 범위

| 1차 문헌 | 확인된 범위 | 이번 설계에 주는 제약 |
|---|---|---|
| [TerraScope](https://arxiv.org/html/2603.19039v1) | RGB/SAR·두 시점·픽셀 근거와 언어 추론을 결합 | 마스크와 설명을 함께 출력하는 것 자체는 기여가 아니다 |
| [Earth-OneVision, 2026-06](https://arxiv.org/html/2606.10819v1) | 저자 분류상 6 입력 유형·9 작업. 다중분광 밴드를 pseudo-RGB 묶음으로 나눠 독립 인코딩 | 기존 VLM이 분광 입력을 못 받는다고 쓰지 않는다. EO-native 공동 표현과 공정하게 비교한다 |
| [Few-Shot OVRS Textual Inversion, 2026-07](https://arxiv.org/html/2607.25563v1) | 소수 마스크로 고정 모델의 개념 토큰을 학습. 다른 클래스 support의 억제도 사용. §5는 고정 토큰의 외형 변화 취약성과 image-conditioned 확장을 논의 | few-shot·양성/대조·조건부 토큰만으로 신규성을 주장하기 어렵다 |
| [TMPA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Test-Time_Multi-Prompt_Adaptation_for_Open-Vocabulary_Remote_Sensing_Image_Segmentation_CVPR_2026_paper.pdf) | 설명 기반 prompt와 시각 특징을 이용한 테스트 시 prompt 적응 | 맥락 기반 prompt 보정 자체는 선행이다 |
| [Think2Seg-RS](https://arxiv.org/html/2512.19302v2) | LVLM이 frozen SAM에 위치 prompt를 전달하는 reasoning segmentation | 전문 분할 모델+동일 VLM을 강한 응용 비교군으로 둔다 |
| [DATO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Dual-Agent_Optimization_framework_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2025_paper.html) | cross-domain few-shot 특징 적응과 support–query 대응 | 단순 matching 개선이나 domain-conditioned adapter를 새 원리로 부르지 않는다 |

지역 문헌·기후와 EO 표현의 결합 선행은 [지역 맥락 검토](REGIONAL_CONTEXT_LEARNING_20260927.md)에 남긴다. 이 목록은 존재를 확인한 가까운 선행이며, 관련 연구가 없다는 증거가 아니다. 서로 다른 원 논문의 점수를 직접 순위로 비교하지 않는다.

## 3. 기존 작업을 연결하는 모델 경로

1. **OlmoEarth**가 원 밴드·날짜·공간 격자를 가진 관측을 읽는다. native 학습 목적과 독립 EO 평가를 유지한다.
2. **교정 예시**는 양성 영역, 확인된 혼동 영역, 관측 날짜와 품질, 교정의 출처를 담는다.
3. **지역·계절 맥락**은 어떤 예시와 구별 근거를 적용할지 판단하는 입력이다. 기후평년·지형·생태 정보부터 사용하고 출처·공간 범위·이용 가능한 시점을 기록한다.
4. **실제 VLM**은 같은 개념을 서로 다른 요청에 연결한다. 예: 지정한 대상 찾기 → 특정 영역/시점 조건으로 제한 → 비교와 근거 제시. 질의와 근거의 연결이 OlmoEarth까지 학습되게 한다.

원 영상과 EO token을 입력받는 Qwen3-VL 경로를 사용하되, 첫 학습 단계에서는 reader를 고정하고 encoder·연결부를 학습할 수 있다. native 학습용 mask/patch loss를 유지하는 것만으로 EO 능력 보존이 보장되지는 않으므로 직접 평가한다.

현재 사용 가능한 자산은 v1.2-Base·공식 H5·native loader/CPU 학습 경로·Qwen3-VL 연결 구현·이전 입력 통제/근거 표시 코드다. 이전 66.80/68.36/72.27 QA 점수나 E3/E5 진단은 이번 교정 전이의 성공 결과가 아니다. D1의 미완료 사람 합의도 새 실험의 gold로 전용하지 않는다.

## 4. 첫 실험: 같은 교정을 다른 환경에 적용했을 때의 효과

새 방법을 먼저 늘리기 전에 **교정 전이에서 실제로 깨지는 사례와 범위**를 만든다. 결과가 좋지 않은 사례만 사후 선별하지 않고, development에 정한 동일 query 묶음을 모든 조건에서 평가한다.

### 자료를 먼저 통과시킬 조건

- 우선 후보는 **PASTIS의 native Sentinel-2 시계열과 공간 라벨**이다. 프랑스 내부 지역 전이 개발용이며 국가 밖 전이 자료로 쓰지 않는다. [공식 코드/자료 안내](https://github.com/VSainteuf/pastis-benchmark)
- 서버 inventory에는 GEO-Bench PASTIS shard 두 개의 존재 기록이 있으나 전체 파일 완전성·실제 읽기·날짜·부모 지역 분할은 아직 통과하지 않았다. 기존 fetch 기본값은 RGB 세 밴드다. 기존 verifier의 shape 성공만으로 native contract 통과라 하지 않는다.
- PASTIS 10밴드와 v1.2 native 12밴드의 순서·누락을 검사한다. 현재 real-H5 provider를 그대로 쓰지 않는다. B01/B09 부재를 관측된 0 반사율로 표시하지 않으며, 공식 missing-band/dropout 처리와 parity를 확인한 adapter가 선행한다. 공식 PASTIS loader의 통계 정규화와 OlmoEarth 정규화를 중복 적용하지 않는다. 처리 경로가 맞지 않으면 다른 자료의 밴드 수만 보고 대체하지 말고 방사보정 수준까지 재검토한다.
- 국가 밖 dense crop 평가는 **Sen4AgriNet PAD의 Catalonia**를 후보로 감사한다. [공개 버전](https://huggingface.co/datasets/orion-ai-lab/S4A)은 FR2019/Catalonia2019·2020이며 미확보다. [원 논문 §V-C](https://arxiv.org/html/2204.00951)는 L1C 입력을 명시한다. OlmoEarth의 L2A와 자동 호환되지 않는다. PASTIS→S4A는 국가+자료셋+전처리의 복합 변화이므로 순수 환경 효과로 부르지 않는다. 동일 처리의 S4A FR→ES 평가 또는 정합된 L2A 재확보를 검토하며 label ontology·해상도·연도 효과를 감사한다. CropGlobe의 공개 feature/label NPZ를 원본 dense 영상으로 취급하지 않는다.
- 공식 1k H5는 3,996파일/999저장좌표의 구현 자료다. 원 사전학습 노출이 있고 독립 사건 정답이 없으므로 외부 전이 검증을 대신할 수 없다.

### 개발 표본과 비교

첫 frozen-readout 현상 관찰의 초안 목표는 봉인 평가를 제외한 3개 부모 지역, 지역당 12개 query 관측 묶음, 2개 이상 정합 가능한 개념이다. **총 36개라는 값은 준비 목표이며 아직 선택·확보된 episode 수가 아니다.** PASTIS의 원 부모 tile는 4개뿐이므로 두 단계를 분리한다. frozen 관찰은 3개 비봉인 tile에서 수행하고, 이후 학습 비교는 그 3개 중 2개 train/1개 dev로 나눈다. 네 번째 tile는 봉인 평가 후보로 유지한다. 학습한 VLM을 독립 dev 3개 지역에서 평가했다고 쓰지 않는다. 최종 광역 holdout 1개만으로 국가·환경 일반화를 확증할 수 없으므로 외부 지역이 추가로 필요하다.

공간상 이웃 patch나 같은 필지를 독립 지역처럼 세지 않는다. 환경 descriptor의 범위가 실제로 다른지는 결과를 보기 전에 감사한다. 각 query에서 비슷한 환경과 다른 환경의 독립 support가 모두 존재하는지 확인하고, 성립하지 않는 비교는 `not_estimable`로 기록한다. 빈 칸을 같은 필지나 가까운 patch로 채우지 않는다. 범위가 좁으면 환경 전이 주장도 제한한다.

- 같은 query와 같은 K의 support를 유지하며 `같은 지역 / 다른 지역의 비슷한 환경 / 다른 환경`의 support를 비교한다. 서로 다른 환경의 예시도 정확히 같은 개념을 정의해야 한다.
- K=5로 첫 개발을 시작하고 3번의 사전 고정 support 추출을 사용한다. 이는 **support draw이며 학습 seed 3개가 아니다.** 효과가 보이면 K=1/5/20 곡선으로 확대한다.
- 양성만/양성+대조를 볼 때 전체 검수 예산과 주석 단위를 기록한다. 미표시 영역을 자동 음성으로 만들지 않는다.
- 선택된 K와 **이미 라벨된 후보 library 전체 크기**를 구분한다. 100개를 라벨링한 뒤 5개를 골랐다면 '5개 라벨만 필요'라고 하지 않는다. 원 학습 라벨·기존 교정 library·새 지역 추가 라벨, 양성/음성 polygon 수·면적·검수 시간을 별도 비용으로 기록하고 모든 비교군에 같은 후보 pool을 준다. 기존 library를 재사용하는 운영 이득과 총 라벨 비용 절감은 별도 주장이다.
- 지역 정보는 `올바른 맥락 / 없음 / 다른 환경의 맥락`을 비교한다. 잘못된 맥락은 오류 민감성 시험이며 현실적 반사실 세계나 자동 unknown 정답이 아니다.
- support 선별은 사전 정의한 환경 거리 또는 입력 특징으로 수행한다. query gold와 관측된 support 성능으로 최고 예시를 골라 본 성능처럼 보고하지 않는다. gold를 쓴 oracle은 별도 사후 진단이다.

먼저 frozen native EO의 prototype/간단한 support readout으로 자료와 현상을 점검하고, **충분히 학습된 일반 support-conditioned VLM**에서 같은 문제가 남는지 확인한다. 학습하지 않은 새 연결부의 실패를 기반모델 한계라고 부르지 않는다. 간단한 readout의 결과만으로 실제 VLM 개선을 주장하지 않는다.

### 읽을 결과

주 개발 지표는 고정 K에서 지역별 foreground IoU의 교정 효과다. **prototype readout에서는 같은 K의 사전 고정 random support를 기준**으로 비교한다. **실제로 zero-shot 개념 판독 경로가 있는 VLM의 named-concept track에서만** `K>0−K=0`도 계산한다. 대상 이름을 줬다는 이유만으로 언어 정렬이 없는 EO prototype의 K=0이 정의되지는 않는다. 익명 새 개념은 K>0의 공통 support 기준과 비교한다. 서로 다른 기준의 차이를 같은 효과로 합치지 않는다. 환경별 평균과 support에 따른 변동, 교정 후 악화된 지역도 함께 남긴다.

환경 descriptor가 **이미지 유사도·클래스 빈도만 사용하는 선택보다 교정의 유용성을 더 잘 예측하는가**를 held-out development 지역에서 본다. 평가 지역이 너무 적으면 신뢰구간이나 통계적 유의성을 과장하지 않고 기술적 개발 결과로 남긴다.

## 5. 현상이 확인됐을 때 시험할 학습 후보

후보는 **어떤 구별 근거가 다른 환경에서도 유효한지를 학습하는 support 선택·적응 방식**이다. 같은 개념의 양성/혼동 사례에서 공간·분광·계절 차이를 표현하고 query 환경에서 사용할 근거를 선택한다. 선택한 근거로 판독하는 loss가 OlmoEarth encoder에도 전달되게 한다.

train 지역에서만, 교정을 넣기 전후의 오류 차이로 그 교정의 학습용 효용을 만들 수 있다. 효용은 특정 모델과 checkpoint에 의존하는 값이며 보편적 전문가 정답이 아니다. 학습 지역 내부를 나눠 out-of-fold로 만들고, 최종 query gold로 계산하거나 선택하지 않는다. 맥락·관측을 이용한 일반 meta-reweighting도 같은 감독을 받아야 한다. [일반 reweighting 선행](https://proceedings.mlr.press/v80/ren18a.html)

이는 **검증할 구체적 후보**이지 새 원리로 확정한 방법이 아니다. 환경이 교정 효용에 추가 정보를 주지 않으면 환경 선택을 핵심 구조로 고정하지 않고, 실제 분광/시계열 근거 보존 쪽으로 가설을 수정한다.

## 6. 방법 비교와 큰 기여의 판정

v2의 G0/G1/P0/P1을 유지한다. **주 비교 P1−G1**은 같은 context·support·원천 query labels·질문·negative pool·학습 예산을 받는다. P1에서 효용 감독을 만들었다면 G1+동일 효용 감독과 일반 learned router를 포함한다. scoring/추가 backward 비용도 합산한다.

필수 강한 비교는 frozen EO+충분히 학습한 연결부, 일반 context-conditioned episodic learning, 단순 환경/특징 거리 선택, 일반 meta-reweighting, 전문 segmentation+연산+동일 VLM이다. Textual Inversion/TMPA와의 native 입력 차이를 기록하고 RGB 공통입력 비교와 native EO 비교를 구분한다. 지원하지 않는 센서를 기존 방법에 임의로 넣고 원 논문 구현이라고 부르지 않는다.

**P1이 G1보다 좋다는 결과만으로 전체 조합의 모든 부품이 필요하다고 결론 내릴 수 없다.** 이를 나눠 검증한다.

| 주장 | 필요한 별도 증거 |
|---|---|
| 제안 학습의 효과 | 동일 정보·감독의 P1−G1 및 learned router 비교 |
| 지역 맥락의 역할 | 2×2 비교, 맥락-only/추론-only, 위치·클래스 prior 통제 |
| native EO 정보의 역할 | RGB/밴드 묶음 대비, 실제 센서 의존 과업·내용 통제 |
| VLM 응용성 | 같은 support 개념을 처음 보는 질문 조건 조합에 적용. 전문 분할+연산+동일 VLM 대비 대상·조건·근거 정확도 |
| OlmoEarth 재사용성 | 새 연결부·다른 reader를 같은 예산으로 학습한 전이, context/support 없는 독립 EO 평가 |

mask에서 계산한 면적과 그 수치 설명은 독립 성능 세 개가 아니다. 관측 가능한 시간/공간 관계를 가진 질문 조합은 원 라벨과 정합된 근거에서만 생성한다. VLM 자체의 이점이 없으면 시스템 결과와 언어모델 기여를 분리한다.

목표 결과는 (1) 같은 품질에 필요한 **독립 교정 수** 감소, (2) 지역 및 두 번째 분야의 전이, (3) 새 reader에서도 남는 효과다. 후보 library 비용까지 기록하고, 새 지역에서 추가로 필요한 교정과 전체 라벨 절감을 구분한다. 독립 EO 점수 유지에 그쳤다면 '원 EO 능력 유지'이며 '원 EO 능력 향상'이라고 쓰지 않는다. 새 대상은 우선 'downstream 추가 감독에서 보류한 support-defined target'으로 표현한다. '우리 추가학습의 미노출 개념' 주장은 native replay의 WorldCover/CDL 등 ontology·caption·negative labels와 query footprint/날짜 노출까지 감사한 뒤에만 허용한다. 원 모델 사전학습에서 처음 보는 개념이라는 뜻은 아니다.

개발 단계에서 차이가 없거나 한 지역/support draw에만 있으면 확대하지 않는다. 효과를 본 뒤 개발에서 실용적 차이·평가 규모를 추정하고 **아직 열지 않은 새 평가 지역**에 대한 주지표·허용폭·seed·분할·예산을 고정한다. 이 문서의 빈 표본 ID/숫자를 사전등록 완료라고 부르지 않는다.

## 7. 실행 순서와 현재 상태

1. **실행 선행조건 복구:** 실제 native GPU 및 실제 VLM gradient·복원 검증을 완료한다. 기존 runtime 수정 없이 새 output/controller 상태로 재시도하려면 유휴 GPU와 당시 소스/입력 일치를 먼저 확인한다.
2. **PASTIS 자료 감사와 episode manifest:** 원 밴드·실제 날짜·공간 mask·부모 지역·맥락 출처·support/query 분리를 확인한다. 자료가 실패하면 원 native 계약에 맞는 공개 자료 후보로 전환한다. 별도로 원본 OlmoEarth의 독립 EO 평가 기준선도 확보한다. 그 평가 자료를 교정 학습에 쓰지 않으며, 원래 사전학습 노출 여부와 이번 학습 노출 여부를 구분한다.
3. **교정 전이 효과의 개발 비교:** 위 36-query 초안과 간단한 readout→충분히 학습한 일반 VLM 비교. annotation에서 만든 예시는 '공개 라벨 기반 support'로 기록한다.
4. **G1/P1 결정 실험:** 실측 실패에 맞춰 후보 하나를 구현하고 동일 정보·감독의 강한 대조와 비교한다.
5. **확장:** K 곡선·새 국가/환경·독립 두 번째 분야·새 reader·EO 능력. 실제 전문가 검수는 기존 100만 원 계획의 20개 pilot 시간 측정 후 배정한다. 아직 채용·발주하지 않았다.

**2026-09-27 11:48 KST 무렵 서버 읽기 확인:** native v2는 `No idle GPU within 90 minutes; no jobs interrupted`로 04:09:25 KST 종료했고 results는 빈 배열이다. VLM v1도 이 선행 실패로 04:09:34 KST 종료했다. 따라서 두 작업은 현재 대기 중이 아니며, 이 시도에서 실제 GPU worker 성능 결과가 없다. CPU native 1step 통과와 GPU 실행 성공을 구분한다. 이번 조회는 상태 읽기였으며 서버 학습 재시작·source 변경은 없다. 원 상태 파일은 동반 artifacts에 보존한다.

데이터/계산량 1/10 절감과 FoldRefresh 학습 확장은 별도 연구 가설로 유지한다. 메인 방법의 차이를 먼저 설명할 수 있어야 하며, 여러 연구를 한꺼번에 추가해 중심 비교를 바꾸지 않는다.

## 8. 이번에 한 일과 남은 일

- 최근 primary 문헌, 데이터·기존 코드, 학습 메커니즘을 독립 검토자 세 명이 검토했다. 단순 few-shot+negative·context prompt·support weighting의 선행 충돌을 반영했다.
- 큰 문제, 첫 자료 계약, 개발 실험, G/P 비교와 부품 기여의 차이를 문서와 JSON 계획으로 작성했다. JSON의 `ready_to_train=false`는 실제 자료·VLM 선행 관문이 남았다는 뜻이다.
- 새 의미 성능, 전문가 교정 절감량, 새 사전학습 효과, CVPR 채택 확률은 이번에 산출하지 않았다. 다음 실제 작업은 실행 선행조건과 native episode 자료 준비다.
