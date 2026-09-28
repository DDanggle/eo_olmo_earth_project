# E5의 CVPR 위치: 차이 표현의 유용성과 시간 근거 사용을 구별하기

2026-09-25. 읽기 전용 검토. **E5 결과를 읽거나 새 실험을 실행하지 않았다.** 아래 평가는 고정된 E5 명세와 공개 1차 문헌에 근거한다. 문헌 검색이 완전하다는 주장이나, 검색되지 않은 항목이 신규성이라는 주장은 하지 않는다.

**판단:** E5는 현재 reader에서 명시적 차이 입력의 학습상 이점과 학습 후 입력 변경의 실패를 구별하는 유용한 개발 실험이다. 하지만 `EO features → projector → LLM`, 전후 영상 질의, 차이 모듈의 효과, 모듈 제거 후 `no change` 편향은 이미 가까운 선행이 있다. E5 단독으로 CVPR 방법 기여, 검증된 시간 이해, 기억 병목을 주장하기는 어렵다. 논문으로 확장할 때의 핵심은 **실제로 이전 관측을 읽어야 답이 달라지는 독립적인 시간 라벨**과 **VLM이 단순 판별기·사건 기록보다 무엇을 더 하는가**다.

## 검토한 로컬 근거

- [E5 고정 사전등록](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/config/e5_equal_budget_prereg_v0.json): SHA256 `fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696`.
- [E5 제안서](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/E5_EQUAL_BUDGET_PROPOSAL_20260925.md). 제안서는 원래 primary 후보가 달랐으므로, 현재 primary는 위 고정 사전등록의 **C1 902항목·8사건**이다.
- [9/24 CVPR·VLM 검토](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/CVPR_VLM_RESEARCH_UPDATE_2026_09_24.md), [9/18 지역 EO 문헌 감사](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/REGIONAL_CONTINUOUS_EO_LITERATURE_AUDIT_2026_09_18.md).

이미 로컬 문서는 TEOChat·EarthDial·ChangeChat을 직접 선행으로 다룬다. 이번 검토에서는 **DeltaVLM의 차이 모듈 ablation과 CCExpert의 difference-injection 비교**를 E5에 더 직접적인 선행으로 추가했다. 아래 상세 사실은 링크에 명시된 원문 버전에서 확인했다. 저자 보고를 자체 재현 결과로 표현하지 않았다.

## 가까운 다섯 방법

| 방법과 확인한 1차 출처 | 원문에서 확인한 사실 | E5와의 관계에 대한 판단 |
|---|---|---|
| **DeltaVLM**, [arXiv v1, §IV·§V-C, Tables VII–VIII](https://arxiv.org/html/2507.22346v1), [저자 코드](https://github.com/hanlinwu/DeltaVLM) | 두 시점 features와 `F2−F1`, CSRM 차이 처리, instruction-guided Q-former를 frozen LLM에 연결한다. EVA-ViT 마지막 두 블록은 학습한다. `w/o CSRM`의 이진 분류에서 recall **0.31%**, F1 **0.62%** 및 no-change 편향을 보고한다. | 가장 가까운 선행이다. ‘차이 처리 경로를 빼면 no로 치우침’ 자체는 새 현상이 아니다. 다만 CSRM 제거는 E5의 raw-D zeroing과 같은 개입이 아니고, OLMoEarth/S1·고정192토큰·동일초기화/순서/예산 실험을 대신하지 않는다. |
| **CCExpert**, [arXiv v1, §III-E·§IV-C3, Table VI](https://arxiv.org/html/2411.11360v1), [저자 코드](https://github.com/Meize0729/CCExpert) | 차이 정보를 원 이미지 문맥에 주입하는 모듈을 비교한다. 3단계 학습 후 enhancer 유무의 LEVIR-CC `S*m`은 7B에서 **81.31→81.80**이다. 첫 단계만 image encoder/LLM을 고정하고 이후는 모두 학습한다. | ‘명시적 차이가 필요한가’를 다루는 직접 선행이다. E5의 frozen-LLM·작은 고정예산 설정과 구별해야 한다. Caption 점수의 개선을 홍수 source-label BA와 직접 비교하거나 동일 효과크기로 해석할 수 없다. |
| **TEOChat**, [원문 v1, §3.3·§4·§5.2](https://arxiv.org/html/2410.06234v1), [저자 코드](https://github.com/ermongroup/TEOChat) | 시점별 shared image encoder와 MLP를 사용하고 image identifier를 visual sequence 사이에 넣는다. 영역·변화·시간 참조 질의를 평가하며 LLM에 LoRA를 적용한다. Projector 적응과 학습 길이도 비교한다. | 시점별 features를 LLM에 순서대로 주는 강한 관련 기준이다. 단순 pair 연결만으로 새로운 시간 VLM이라는 주장은 어렵다. E5보다 풍부한 질의 범위가 있어 향후 ‘VLM 필요성’의 과업 기준으로 유용하다. 현재의 완전히 frozen LLM과는 학습 설정이 다르다. |
| **EarthDial**, [원문 v1, §3](https://arxiv.org/html/2412.15190v1), [CVPR 2025 논문 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.pdf) | InternViT·MLP·Phi-3-mini로 EO 대화를 만들며 다중 시간·해상도·센서를 다룬다. SAR/다중분광 학습 단계에서도 MLP와 LLM을 학습한다. | S1/S2→언어라는 넓은 주장에 이미 가까운 선행이다. 다중 센서 지원이 있다고 우리 원시 SAR 배열/캐시를 수정 없이 받을 수 있다고 가정하지 않는다. Native 입력과 전처리·학습량을 확인한 외부 시스템 비교가 필요하다. |
| **ChangeChat**, [원문 v1, §2](https://arxiv.org/html/2409.08582v1), [저자 코드](https://github.com/hanlinwu/ChangeChat) | CLIP visual tokens를 MLP로 Vicuna 입력에 맞추고 두 시점 embedding을 통합한다. LLM은 LoRA로 학습하며 이진 변화, 수량, 위치, 설명을 instruction 데이터로 구성한다. | 간단한 bi-temporal adapter와 질문을 붙이는 구성 자체는 선행이다. DeltaVLM과 같은 연구 계열이므로 독립적 다섯 계열의 증거로 세지 않는다. E5 pair arm은 이 계열의 아이디어를 현재 frozen-LLM 설정에서 통제하는 내부 기준선에 가깝다. |

위 논문에서 저자가 쓰는 ‘first’, ‘reasoning’, ‘state of the art’는 이 문서의 확인된 결론으로 가져오지 않았다. 원문 버전별 구성 차이가 있을 수 있으므로 실제 재현에 들어갈 때 논문·코드 commit·가중치 버전을 함께 고정해야 한다.

## E5가 추가로 답할 수 있는 정확한 질문

현재 명세는 full `[A,B,B−A]`, pair `[A,B,0]`, later `[0,B,0]`, delta `[0,0,B−A]`를 각각 처음부터 학습하고, 새 full 모델의 학습 후 no-D 개입까지 비교한다. 12개 모델은 seed별 초기화와 배치 순서, 3 epoch·1,590 updates·12,702 exposures, 192 EO 토큰 수를 맞춘다. 주비교는 **pair/native − full/native**이다.

여기서 `B−A`는 A와 B로 계산되는 값이다. **완전한 A/B가 이미 있는 경우 D를 추가해 새 관측 정보를 제공하는 것은 아니다.** 차이는 현재 모델이 주어진 학습량에서 이용하기 쉬운 표현을 주는지, 입력 형식과 최적화가 결과에 어떤 영향을 주는지다. 이 때문에 full 우세를 ‘원본 A/B에는 시간 정보가 없다’로 해석하면 안 된다. 반대로 pair 보존은 모든 reader·예산·현상에서 D가 불필요하다는 뜻이 아니다.

학습 후 no-D는 학습 입력 분포와 다를 수 있다. pair를 처음부터 학습했을 때 회복한다면, E3의 붕괴를 단순히 시간 정보 부재로 설명할 수 없다는 근거가 된다. 그러나 다른 가중치를 학습한 비교이므로 **붕괴의 유일한 원인을 분포 변화로 식별한 것**까지는 아니다.

DeltaVLM·CCExpert가 같은 넓은 문제를 다루지만, 이번 확인으로 E5의 정확한 paired-init/fixed-budget/retraining-versus-intervention 계약을 이미 완전히 검증했다고 결론낼 수는 없다. 반대로 그 정확한 표가 검색되지 않았다는 이유로 이 통제 설계 자체를 충분한 신규 기여로 삼을 수도 없다.

## 가장 먼저 채워야 할 기준선과 과업 위험

아래는 **E5 사전등록을 바꾸라는 제안이 아니라**, 결과 확인 후 별도 명세로 판단할 후속 후보에 대한 검토다.

| 우선순위 | 빠진 비교 또는 검증 | 중요한 이유와 최소 조건 |
|---|---|---|
| 1 | **검증된 시간 변화 gold** | 현재 홍수 positive/hard는 post reference mask 기반이다. 같은 event·prompt·dates·slots를 맞춰도 배경, 영구수역, 취득 조건과 단일 post 상태로 풀 가능성이 남는다. 실제 pre/post 관측과 valid mask를 독립 검수하고, 이전 상태에 따라 변화 유무가 달라지는 사례를 포함해야 한다. 합성 날짜를 실제 취득일로 승격하지 않는다. |
| 2 | **LLM 없는 동일 입력 기준선** | C0 global mean 선형 분류기는 유용하지만, E5의 64개 공간 토큰 및 nonlinear projector와 입력·용량이 다르다. 고정된 동일 A/B/D pooled tokens를 받는 작은 판별 head를 한 개 사전 지정하고, 같은 train/test·902 membership에서 BA/FPR·시간·라벨비용을 비교하는 것이 타당하다. VLM만 성공해도 언어 일반화는 별도 문제다. |
| 3 | **질문의 의미를 바꾸는 평가** | 현재 두 yes/no 템플릿은 sensor/phenomenon과 결합돼 있어 언어를 해석하지 않는 분류기로도 풀 수 있다. 같은 근거에 대해 영역·시점·변화 종류를 바꾼 질의에서 정답이 실제로 달라져야 한다. 단순 문장 paraphrase만으로 조합적 grounding을 주장하지 않는다. |
| 4 | **한 외부 temporal VLM** | RGB의 검증된 변화 과제라면 TEOChat 또는 DeltaVLM/ChangeChat 계열 하나가 우선이다. S1 재난이라면 native SAR 지원이 확인된 EarthDial 계열을 검토한다. 외부 모델을 원래 설정대로 실행하는 시스템 비교와, 현재 backbone에 모듈만 이식하는 통제 비교를 섞지 않는다. 지금 모든 모델을 다운로드·학습할 필요는 없다. |
| 5 | **사건 detector + 구조화 기록 + query executor** | 사용자의 ‘어느 지역에 어떤 여파가 있는가’는 탐지·공간교차·시간필터·근거 ID로 먼저 풀 수 있다. 그 결과를 템플릿이나 LLM으로 설명하는 기준선은 memory 방법의 중요한 비교 대상이다. 정답 `pflood`로 정렬한 현재 브라우저를 모델 검색 정확도로 채점하지 않는다. |

추가 해석 경계는 분명하다. 902항목은 902개의 독립 사건이 아니라 **8개 사건 안의 타일들**이다. 세 seed 역시 독립 사건 수를 늘리지 않는다. 이미 E2–E4에서 본 사건이므로 E5의 사전 고정은 결과 선택을 제한하지만 새로운 confirmatory test를 만들지는 않는다. 0으로 둔 raw block도 projector bias와 type embedding을 거치므로 ‘빈 토큰’이나 ‘토큰 수 감소’와 같지 않다.

시간 제어를 더 할 때 반복·역순·교체 영상에 임의의 새 gold를 붙이면 이전 진단의 문제가 재발한다. 원 영상의 근거·방향·질문을 확인하고 의미가 정해지는 경우만 새 물리 장면 정답으로 평가한다. 나머지는 원 source label에 대한 반응 변화로 남긴다.

## 결과별 다음 의사결정의 범위

- **pair 보존 또는 pair 학습 후 회복:** 간단한 입력 경로를 선택할 근거다. 다음 자원은 독립 시간 라벨과 위치 근거에 쓰는 편이 낫다. ‘기억 불필요’로 연결하지 않는다.
- **full이 고정 예산에서 우세:** 명시적 D의 실용적 학습 이점을 기록한다. DeltaVLM·CCExpert와 연결해 기술하고, 같은 입력을 받는 판별 head 대비 이점과 새로운 사건에서의 재현을 먼저 확인한다. 새 adapter를 무조건 추가할 근거는 아니다.
- **later가 강함:** post 상태 판별로 풀 수 있는 source task일 가능성을 우선 검토한다. 이전 관측이 필수인 gold부터 확보한다.
- **모두 약하거나 혼합:** 모델이 일반적으로 변화를 못 읽는다는 결론보다 라벨·관측 가능성·입력 적응·학습량의 범위를 좁혀 보고한다. 등록 문턱을 바꾸거나 유리한 seed만 고르지 않는다.

이 네 경우 모두, E5만으로 농작물 피해액·홍수의 인과 효과·정확한 발생일은 알 수 없다. 농지와 관측 홍수의 교차는 **관측상 교차**이며 경제 피해나 원인 증명이 아니다.

## CVPR 본문에 쓸 수 있는 잠정 표현

현재 가능한 정직한 연구 질문은 다음과 같다.

> 고정된 EO 임베딩과 frozen language decoder에서, 차이 표현을 명시적으로 제공하는 것이 동일 학습 예산의 source-label 판별에 얼마나 기여하며, 학습 후 차이 입력을 지우는 진단과 처음부터 그 입력 없이 학습하는 진단은 같은 결론을 주는가?

이는 유용한 **reader 검증 단계**다. Grounded Change Memory라는 원래 주장으로 돌아가려면 실제 시간·위치 근거를 읽는 reader를 먼저 고정하고, full history / latest / uniform / detector-ledger / proposed memory를 같은 저장·검색·질의 예산에서 비교해야 한다. 메모리 우위는 아직 이 문헌 검토나 E5 설계에서 도출되지 않는다.

후속 논문 기여 후보는 ‘차이 토큰을 넣었다’보다 **질문이 요구하는 과거 근거를 검증하고, 잃었을 때 보류하며, 예산 안에서 보존했을 때 답과 근거가 함께 개선되는 것**에 있다. 이 역시 앞으로 검증할 가설이며 현재 확보된 성과가 아니다.

## 확인 범위

로컬 문서를 읽고 arXiv 원문, CVF 논문 링크와 공식 저자 저장소를 확인했다. 전문 재현·체크포인트 다운로드·실제 SAR 입력 호환성 시험은 수행하지 않았다. 웹에서 확인한 수치들은 저자 보고이며, 현재 실험 수치와 동일 척도/분포의 성능 비교표로 사용할 수 없다. E5 사전등록·소스·서버 상태·UI·D1 계약은 변경하지 않았다.
