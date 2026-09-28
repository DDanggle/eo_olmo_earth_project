# OlmoEarth capability program: independent prior-art red team

검토일: 2026-09-27. 검토자: `/root/cvpr_prior_art_redteam`.
검토 대상: `OLMOEARTH_CAPABILITY_PROGRAM_20260927.md` 및 사용자의 큰 응용 모델 요구. 웹의 논문 원문·공식 기관/학회 자료만 근거로 사용했다. 이 검토는 문헌과 설계 평가이며 새 모델 실험이나 실제 CVPR 심사 결과가 아니다.

## 판정

**조건부 연구 진행 권고. 현재 계획만으로 CVPR 제출 경쟁력을 입증했다고 볼 수 없고, 채택 확률 50% 이상이라는 숫자는 산출할 근거가 없다.** 검색·영역·측정·설명을 묶는 응용 목표는 충분히 크다. 그러나 현재 설명은 기존 모델들의 능력을 합친 시스템으로 읽힐 위험이 크다. 큰 목표를 줄일 필요는 없지만, 핵심 메커니즘과 다른 방법으로 설명되지 않는 성과를 명확하게 해야 한다.

가장 유력한 중심은 **한 개념에 대한 적은 교정이, 그 개념에 대해 감독하지 않은 작업 조합에서도 새로운 시각적 판단으로 이어지게 하는 OlmoEarth 학습법**이다. 이는 아직 신규성이나 실현 가능성이 증명된 결론이 아니라, 선행연구와 구분할 가치가 있는 검증 대상이다.

## 이미 상당 부분 수행된 것

| 근거 | 이미 다룬 능력/방법 | 현재 계획에 주는 제약 |
|---|---|---|
| [TerraScope v1, 2026-03-19](https://arxiv.org/html/2603.19039v1), §3.2, §4, §5 | InternVL 기반 mask/text 혼합 생성, mask에서 뽑은 시각 특징의 재입력, optical/SAR 선택, 다중 시점, 면적·거리·경계·건물 변화 평가. Vision encoder는 고정. | 근거 mask와 설명 일치, 면적 QA, 시간 입력, 모달리티 선택만으로 차별화 불가. |
| [SPEX v2, 2026-03-09](https://arxiv.org/html/2508.05202v2), §III-B | 분광 입력, multiscale 특징, spectral-index 유래 언어 감독, 실제 encoder 업데이트, LLM 설명 특징에서 mask 생성. 최종 단계는 SPIE subset별 학습/평가. | EO+LLM 연결이나 encoder unfreeze, 분광 설명, 공간 정보 보존만으로 최초 주장 불가. 통합 개념 전이에서는 다른 질문을 만들 여지가 있다. |
| [MS-CLIP v3, 2025-08-04](https://arxiv.org/html/2503.15969v3), §4–5 | Sentinel-2 확장 입력과 language-image continual pretraining, 검색·분류. | 다중분광 언어 사전학습과 검색 자체는 신규성이 아니다. |
| [Spectral-LLaVA v1, 2025-01-17](https://arxiv.org/html/2501.10144v1), §V | 고정 SpectralGPT와 projector/LLM 적응, 장면 설명·분류 및 language-grounded feature 평가. | 고정 EO backbone에 언어를 붙이는 초기 기준선으로 적합하다. 이를 최신 joint-training 방법의 대리로 삼으면 약하다. |
| [DOFA-CLIP v2, 2025-07-22](https://arxiv.org/abs/2503.06312v2), abstract | 여러 EO 모달리티의 image-text 학습, vision-model distillation과 modality-aware 처리. | native EO·다중센서·공간적 언어 특징·새 모달리티 전이라는 넓은 주장도 선행이 있다. |
| [RSCoVLM v2, 2026-01-09](https://arxiv.org/html/2511.21272v2), §III-A/B | 통합 언어 인터페이스에서 분류·VQA·caption·grounding·detection 공동학습, 데이터 엔진, 동적 해상도. | 여러 응용을 한 모델로 묶고 데이터 체계를 만든 사실만으로는 부족하다. |
| [Composition-Aware Pretraining v1, 2026-08-31](https://arxiv.org/html/2608.30817v1), §3 | DINOv3 patch vocabulary와 분포 target, Sinkhorn EMD 및 MIL로 EO backbone 학습. RGB 입력이며 여기서 composition은 장면 내 시각 texture 혼합을 뜻한다. | 영역 구성·분율을 보존하는 EO 사전학습도 선행이 있다. 우리의 concept×operation 일반화와 명확히 다른 정의를 써야 한다. |
| [SegGPT, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.html), [SEEM, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html) | 예시 기반 분할과 시각·언어 prompt의 조합, 새 target/prompt 일반화. | 예시 몇 장으로 새 대상을 분할하거나 text+example을 결합하는 것 자체도 기여가 아니다. |

추가로 [CVPR 2026 EarthVision workshop의 Training-Free Text-Based RS Segmentation](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Sosa_Enabling_Training-Free_Text-Based_Remote_Sensing_Segmentation_CVPRW_2026_paper.html)은 VLM/CLIP과 SAM을 조합한 강한 간단 기준선을 제시한다. 해당 방법을 검토해 모델 복잡성의 실제 이득을 확인해야 한다. 워크숍 논문이며 CVPR 본회의 논문이라고 표기하지 않는다.

## 원본 OlmoEarth에 대한 중요한 최신성 수정

현재 [공식 arXiv 2605.20804v3, 2026-08-14](https://arxiv.org/html/2605.20804v3)의 제목은 **OlmoEarth v1.2**다. 초기 v1은 v1.1 변경을 기술했고 후속 버전은 v1.2까지 통합한다. 본문 §2.2–2.5는 단일 bandset, band dropout, 비선형 입력 projection, masking/loss 변경, RoPE를 설명한다. 기존 OlmoEarth도 관측과 파생 지도 target을 함께 이용한다.

따라서 v1-Tiny의 pooling 또는 공간 artifact 진단만으로 현재 OlmoEarth의 근본 한계를 주장하면 안 된다. 현재 공개 checkpoint·코드의 실제 호환성은 별도 확인 후 버전을 pin해야 한다. v1-Tiny는 과거 진단으로 보존하고, 신규 주평가는 현행 v1.2 또는 현행 가중치를 구하지 못한 사유가 명시된 버전으로 수행하는 것이 적절하다. 현행 모델의 알려진 효율·masking 변경을 새 메커니즘으로 재발명하지 않도록 한다.

## 가장 큰 방법적 허점: 가짜 작업 전이

새 클래스의 mask를 잘 만들게 된 뒤 면적이 정확해지고, 면적 비교와 설명도 맞게 되는 것은 제품 가치가 있다. 하지만 `mask → 픽셀 합산 → 수치 비교 → 문장 생성`으로 얻을 수 있다. 이 네 결과를 네 가지 독립된 능력 전이라고 계산하면 reviewer는 같은 개선을 중복 계산했다고 지적할 것이다.

필수 대조는 **같은 OlmoEarth 분할/예시 매칭 + 결정적 기하·집계 연산 + 동일 언어모델**이다. 이 대조를 이기지 못하면 측정·설명은 유용한 인터페이스라고 보고하되, 학습된 일반화 기여와 분리한다.

전이의 주평가는 새로운 시각적 구별이 필요한 조합이어야 한다. 예컨대 새 개념을 일반 위치 찾기에서 배운 뒤, 다른 지역에서 관계 조건을 만족하는 부분만 찾아야 하는 작업, 다른 모달리티/관측 조건에서 같은 개념을 다시 식별하는 작업이 후보이다. 이들도 알고리즘 조합으로 충분히 풀리는지 반드시 강한 조합 기준선과 비교한다. 기준선을 의도적으로 약하게 만들려고 불필요한 추론을 생성해서는 안 된다.

## 권장 메커니즘을 더 구체화하는 방법

우선 한 가지 핵심을 선택할 것을 권한다: **작업과 구별되는 개념 표현을 학습하여, 새로운 개념을 넣어도 학습된 연산이 그것을 재사용하게 한다.** 다음은 검증 가능한 구현 가설이지 확정된 신규 방법이 아니다.

1. EO의 dense observation token과 예시 mask/설명을 이용해 개념을 표현한다. 질문의 작업 이름을 바꾸어도 개념 자체가 달라지지 않게 한다.
2. episode마다 개념에 대한 일부 작업의 감독만 제공하고, 별도 source scene에서 나머지 작업을 수행하도록 학습한다. 실제 평가에서는 사전에 고정한 concept×operation 조합을 전체 개발 과정에서 제외한다.
3. 공식 EO 목적의 replay를 모든 관련 대조군에 같은 양으로 준다. 제안군에만 replay를 줘서 안정성 개선을 신규 개념 학습 효과로 혼동하지 않는다.
4. 단순 shared latent·multihead·합산 loss와 동일 입력·감독·업데이트 수에서 비교한다. parameter/compute가 늘면 capacity-matched 대조도 둔다.

명시적 concept slot은 구현 선택일 뿐 신규성을 자동으로 보장하지 않는다. prototype learning, prompt-based segmentation, episodic/meta-learning과의 차이를 한 문장과 ablation으로 설명할 수 있어야 한다. 그렇지 않으면 이 구성을 더 복잡하게 만들지 말고 단순한 기준선 자체의 성능부터 본다.

## CVPR 수준 주장을 위한 결정 실험

- **기여의 중심:** 같은 양의 새 개념 감독에서, 학습에서 보지 않은 개념×작업 조합을 더 잘 수행하는가? 양성/음성 예시와 abstention도 포함해 새로운 이름만 보고 찍는 것을 막는다.
- **OlmoEarth 귀속:** 같은 시스템에서 pretrained EO를 고정/업데이트/무작위 초기화 또는 적합한 일반 시각 특징으로 대체해 비교한다. Encoder-only 전이는 보강 근거이며 전체 시스템의 유일한 자격 조건은 아니다.
- **대조군:** 원본+head, 같은 원 목적 추가 학습, 일반 공동학습, 같은 사실을 준 map/measurement 직접감독, 제안 메커니즘, segmentation+deterministic 연산 파이프라인. 동결/재사용 가능한 실행은 공유해 비용을 줄인다.
- **비교 가능 범위:** TerraScope와는 동일 rendered optical/SAR 입력의 공통 능력을 비교하고, native multispectral 이득은 같은 backbone/학습량에서 RGB와 native spectral을 통제한다. 다른 데이터의 발표 점수를 직접 나란히 놓아 승리라 하지 않는다.
- **분할:** sensor 원천 scene·지역·시간·label ontology를 기록하고, 하나의 mask에서 파생된 모든 질문은 같은 split에 둔다. 지리와 class×task holdout은 다른 축으로 보고한다.
- **평가:** atomic segmentation/retrieval, 파생 measurement, 관계 조건 query, 설명의 factual correctness를 각각 보고한다. 같은 영역을 여러 질문으로 만든 수를 독립 표본 수로 세지 않는다. 지역/사건 cluster 단위 uncertainty와 여러 seed를 사용한다.
- **규모:** 20장 probe와 256개 yes/no는 pipeline diagnosis로 남긴다. 10k/100k 학습 example보다 고유 지역·개념·독립 참조 수가 의미 있으며, 다양한 문제에서 반복 이득과 scaling 추세가 필요하다.
- **현실성:** 10m 위성영상에서 개별 고사목·작은 해양 쓰레기 등을 직접 식별한다고 약속하지 않는다. 관측 가능한 피복·군집·오염 패턴과 독립 현장 참조가 있는 범위로 응용을 정의한다.

## 조건부 진행 결정

첫 준비 작업 두 개(원본의 독립 EO 평가, 공식 목적 추가 학습 경로)는 필요하다. **이 두 개가 논문 기여 자체는 아니다.** 그 위에 최소 하나의 명확한 전이 메커니즘과 누출 없는 새 개념 평가를 함께 준비하면, 큰 응용 모델을 위한 본격 연구를 시작할 근거가 된다.

현재 권고: `GO for bounded method feasibility`, `NO for claiming CVPR-ready or >50% acceptance probability`.

첫 파일럿이 일반 공동학습·prototype+연산 대조를 넘지 못하면 실패를 공개 기록하고 메커니즘을 다시 설계한다. 성과가 나오면 native spectral, 새 지역, 다른 VLM, 데이터 규모로 넓힌다. 계획 승인률이나 agent 투표를 학회 채택률로 환산하지 않는다.
