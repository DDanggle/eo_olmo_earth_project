# Grounded Change Memory — CVPR·VLM 관점 재검토와 다음 증거

2026-09-24 · 검토 기준: 연구 저장소 `main`의 `8845014` (`git pull --ff-only`: 최신).
대상 학회는 **CVPR 2027**이다. 이 문서는 연구 방향·코드 감사·집필안이며 사전등록이나 새 모델 실험 결과가 아니다.

**현재 CVPR 본회의 방법 논문을 지지하는 증거는 부족하다.** 기억 방법의 효용, 유효한 reader, 독립 시험셋이 아직 확보되지 않았다. 다만 연구 질문은 유효하다. 다음 목표는 “기억 모듈을 하나 더 학습”하는 것이 아니라 **사람의 관측 가능성 → 실제 증거 보존 → VLM의 증거 사용**을 분리해 검증하는 것이다. D1이 그 출발점이지만, 이번 코드 감사에서 **실행 관문과 채점 의미의 결함**을 추가로 확인했다. H 판독은 준비할 수 있고, L·R의 과학적 판정은 아래 결함을 먼저 닫아야 한다.

## 1. 현재까지 무엇을 했으며 무엇이 남았나

| 단계 | 확인된 진행 | CVPR 본문에서의 위치 |
|---|---|---|
| 정적 cache·전이 연구 | Sen12/Solar의 cache 재사용 및 head 적응 자산, 과거의 양성 결과 | 기반 기술·배경. 현재 VLM 기억 가설의 직접 증거로 합치지 않는다. |
| 9/17~19 시간 진단·보간·관측 보정 | frozen 표현 민감성, 관측 수, 보간·head 일반화·오경보 진단 | 연구 질문을 좁힌 개발 기록. 원장 요약의 상한·지연·인과 해석은 9/20 감사 정정을 적용한다. |
| 9/19~20 로그·슬롯 기억·신뢰 신호 | 학습 슬롯 D와 신호 reader E의 등록 utility 관문 불통과. 계절 baseline v0.1은 통과 | 현재 구현과 계약의 음성 결과. 기억 방법 일반의 실패라는 결론은 아니다. |
| 9/22 SN7 진단 | 해상도, 질문 창, gold 위치, 교체 영상의 정답 문제를 순차 발견. MS-155 content gate 2/2 실패 | 유효한 기억 병목 증거를 아직 확보하지 못한 과정. MS-154는 방법 우위 표에서 제외한다. |
| 9/22~23 v0.5·D1 | 사람 판독 계약, 6 AOI·12 episode·172 프레임 패키지의 준비 감사, H/L/R 사전등록 | 의사결정을 위한 개발 pilot. 사람 gold·D1 모델 결과는 아직 없음. |
| 9/24 이번 감사 | MS-155 저장 답변 산술 재현, D1 합성 반례 확인, CVPR 선행·제출 요건 재점검 | 모델 결과와 구별되는 소프트웨어·해석 감사. 새 MS 성능 번호를 만들지 않는다. |

근거: [9/23 인수인계](HANDOFF_2026_09_23.md), [MS-122~149 감사](EXPERIMENT_LEDGER_AUDIT_AND_CVPR_REDESIGN_2026_09_20.md), [MS-155 해석 감사](MS155_READER_AND_EXPERT_TRANSFER_REVIEW_2026_09_22.md), [상세 측정 장부](../MEASURED_FINDINGS.md).

### 기존 요약에서 고쳐 쓸 문장

1. **“학습 후보 8개 전부 실패”는 프로젝트 요약이지 논문용 통계가 아니다.** 9/20 감사는 후보 분류의 불명확성, MS-131 누락, 실행 무효와 utility 실패의 혼합을 이미 지적했다. 안전한 문장은 “검토한 학습 방법들은 등록된 end-to-end utility 관문을 통과하지 못했다”다. 신뢰 head의 높은 구름 판별 AUC와 downstream payoff 실패는 동시에 참일 수 있다.
2. **MS-148의 통과 범위는 v0.1의 지역 내 회고적 보정이다.** Hokkaido 정상 알람 비율 `.1447→.0480`, 사건 쌍 내 localization AUC `.6687→.7170`, 통제 3지역 관문 통과가 저장 보고에 있다. “계절이 원인임을 증명”, “과거만으로 온라인 보정”, “VLM 기억 개선”은 아니다. [저장 결과](../artifacts/seasonal_baseline_v0_1/summary.json)
3. **“여섯 판 동안 규칙이 한 번도 안 바뀜”은 문자 그대로는 부정확하다.** v0.4 prereg는 기존 readings에 `content_check ≥.30 + CI`를 **추가**한다고 명시한다. 유지된 핵심 문턱과 추가된 validity control을 구별해야 한다. 더 정확한 강점은 “기존 실패를 소급 성공으로 바꾸지 않고, 설계 변경과 새 통제를 버전별 문서로 남겼다”다. 각 등록이 모든 결과 관찰보다 먼저였는지는 저장 JSON의 선언·Git 이력과 서버 실행 시각을 함께 봉인해야 완전히 입증된다. [v0](../config/bottleneck_diag_lite_prereg_v0.json), [v0.4](../config/bottleneck_diag_lite_prereg_v0_4.json)
4. **“VLM이 영상을 읽지 못했다”보다 “이 진단으로 내용 사용을 확증하지 못했다”가 맞다.** 대체 영상에도 원 gold를 쓴 결함이 있었고, v0.3→v0.4에서 대상 문항도 달라졌다. `.909→.273` 전체를 위치 효과로 분해할 수 없다.

이번 MS-155 재계산은 Qwen privileged `12/22`, wrong-content `13/22`; Molmo `6/22`, `8/22`를 확인했다. Q2는 22항목이지만 AOI 12개, 서로 다른 `(AOI, region, gold)`는 13개이고 v0.3과 같은 item ID는 2개다. 원 AOI bootstrap은 재실행하지 않았다. [재검산 결과](../artifacts/cvpr_vlm_review_20260924/ms155_reaudit.json)

## 2. D1: 등록된 의도와 실제 실행을 구별해야 한다

[D1 원본](../config/decision_experiment_d1_prereg_v0.json)은 `50dd781`에 기록되어 있다. 원본의 SHA-256은 `c8fa70a7a420176a7825e056d6a8ca8819a66c4bf881aba7364d9a2a8f85bc93`이며 이번에 수정하지 않았다.

| 단계 | 기존 사전등록 그대로의 의미 | 해석의 한계 |
|---|---|---|
| H | 12개 전체를 분모로 완전 합의율 ≥.60, 세 answer 범주 모두 존재, 총 ≤10인시 | 최소 8/12의 완전 합의가 필요하다. 단순 answer 일치나 Cohen's kappa .60이 아니다. 두 검수자가 필요하다. |
| L | agreed change episode에서 K=4의 최선 단순 선택기 tierA 보존율 `<.70`: loss / `≥.90`: no_loss | `.70≤p<.90`은 코드도 `inconclusive_between_registered_thresholds`로 반환한다. 원 결정표에는 이 분기가 빠져 있다. |
| R | swap의 donor 대비 source answer 정확도 차이 ≥.30, episode bootstrap CI가 0 제외, **real accuracy > metadata-only +.15** | 현재 구현은 아래 이유로 그대로 해석할 수 없다. 실패는 “VLM 일반이 못 읽음”이 아니다. |

### 실행 전 우선 수리할 사항: 합성 입력으로 재현됨

아래는 실제 위성영상·사람 라벨·VLM 추론이 없는 코드 검증이다. [재현 스크립트](../code/audit_d1_contract_20260924.py), [소스 해시와 감사 결과](../artifacts/cvpr_vlm_review_20260924/d1_contract_audit.json).

**A. H gate가 실행 코드에 연결되지 않았다.**

`sn7_visible_pack_v05.review_exports`의 `diagnostic_manifest_ready`는 세 범주와 교체 쌍 존재만 검사한다. 이 함수의 원 문서도 “schema/coverage readiness이며 과학 관문 아님”이라고 설명한다. 그런데 R runner는 이 값만 보고 H 완료로 간주한다. 합성 12 episode를 두 사람이 모두 판독했지만 3개만 완전 합의한 예에서 **합의율 .25인데 ready=true이고 모델 loader까지 도달**했다. 실제 모델 loader는 sentinel로 대체했으므로 GPU 호출은 없다. L의 `main`도 targets가 비어 있지 않은지만 검사한다.

수리안: `H_report`에 `n_total=12`, agreed 수, 전체 분모의 합의율, 범주별 수, 판독 완료·시간 기록을 저장한다. **H 과학 관문 통과와 matrix 준비를 별도로 검사**하고 L/R 모두 H 실패에서 종료한다. H 통과여도 날짜가 맞는 다른 정답의 donor 쌍이 없으면 R은 준비 미완료다. 이를 H 실패나 reader 실패로 바꾸지 않는다.

**B. R은 서로 다른 정답에 대한 정확도를 빼고 있다.**

pack builder는 metadata-only·blank의 정답을 올바르게 `insufficient_evidence`로 바꾼다. 이는 영상이 없을 때 보류하는지를 평가한다. 하지만 scorer는 이 정확도를 real 영상의 장면 정답 정확도와 비교한다. 모든 real 답과 swap 답을 맞히고, 영상이 없으면 올바르게 보류한 합성 reader는 `real=1.0`, `metadata=1.0`, `content_diff=1.0`, CI `[1,1]`인데도 **`1.0 > 1.0 + .15`가 거짓이라 `does_not_read`**가 된다.

수리안: (i) 입력이 없을 때의 **적절한 보류 정확도**와 (ii) 그 답이 **원 장면 정답에 우연히 맞는 비율**을 다른 열로 낸다. `real_image_reference_target`은 이미 matrix에 보존돼 있다. 두 번째 비교도 보류 지시문 때문에 낮아질 수 있으므로 이것만으로 시각 능력을 입증하지 않는다. 날짜·출력 형태·정답 분포를 맞춘 donor 통제가 핵심이다. 필요하면 강제 추측 조건을 별도로 설계하되, 기존 조건에 몰래 섞지 않는다. **이 수정은 채점 의미에 영향을 주므로 원 prereg를 보존하고 결과 확인 전 dated amendment로 등록해야 한다.**

**C. 시간만 다른 swap은 현재 content score에서 구별되지 않는다.**

`make_control_pair`는 `(answer, first_change_date)`가 다르면 쌍을 허용한다. 반면 R의 content score는 `answer_ok`만 쓴다. 둘 다 `change_supported`지만 최초 관측이 F001/F002인 합성 쌍에서 모델이 donor F002를 정확히 따라도 donor와 source의 class accuracy가 모두 1이어서 차이는 0이다. 또한 `first_acc`는 change subset에 한정하지 않고 non-change의 null 정답까지 평균한다.

수리안: class가 다른 쌍의 내용 사용과 class는 같고 최초 관측이 다른 쌍의 **시간 근거 사용**을 분리한다. first-ID는 change subset 및 answer와의 joint 정확도를 따로 보고한다. 원 .30 threshold를 새 복합 점수에 자동 이식하지 않는다. temporal 판정식·최소 유효 쌍 수는 새 결과 전에 명시해야 한다.

**D. 미완결 답변 파일도 점수 보고서가 만들어진다.**

11행 matrix에 답 1행만 준 합성 사례가 거부되지 않고 보고서로 저장됐다(판정은 null). 본 실행 전 exact ID coverage·중복·입력 해시·실패 행을 검사해야 한다. 현재 run context는 reader별 경로가 아니므로 동일 `--out`의 두 reader 실행에서 덮어써진다. 다른 matrix로 같은 답 파일을 재개하는 경우도 hash 검증이 필요하다. reader별 출력 경로와 manifest를 사용한다.

기존 visible-contract 테스트 **23개**, L selftest **12 checks**, R selftest **10 checks**는 이번에도 통과했다. 위 반례가 보여주듯 이 테스트들은 계약 전체의 타당성을 보증하지 않는다. 운영 runner·scorer는 이번 검토에서 수정하지 않았다.

### 결함과 별개로 D1에서 주의할 것

- **표본 단위:** 12 episode는 6 AOI의 두 사분면이다. 등록된 episode bootstrap을 AOI bootstrap으로 조용히 바꾸지 않는다. 원 지표를 그대로 보고하고 AOI·matched pair 단위 민감도 분석을 별도로 사전 명시한다. 6 AOI의 CI로 모집단 일반화를 강하게 주장하지 않는다.
- **보존과 충분성:** L tierA는 특정 전후 ID를 보존했는지를 본다. 기준영상 F000과의 비교가 원 질문의 정의인데 tierA는 기준영상 자체를 반드시 요구하지 않는다. tierA 통과는 질문의 충분 근거가 보존됐다는 증명이 아니다. 반대로 지정된 ID를 잃어도 대체 가능한 전후 근거가 있을 수 있다.
- **‘최초’의 비용:** tierB는 F000부터 최초 변화까지 모두 남길 것을 요구하므로 그 구간이 K보다 길면 구조적으로 실패한다. tierB 실패를 곧바로 새 방법의 headroom으로 쓰지 않는다. 대체 근거 집합과 압축된 무변화 구간 기록의 충분성은 별도 검증 대상이다.
- **offline 선택과 online 기억:** L은 cutoff까지 전체 PNG를 읽고 subset을 선택한다. 미래 cutoff를 보지 않는다는 의미는 있으나, bounded online writer가 같은 선택을 유지할 수 있음을 보이지는 않는다.
- **회색 분기:** L의 중간 구간, R의 미완결/쌍 부족/넓은 CI는 `inconclusive`로 유지한다. 목표 숫자가 나올 때까지 같은 표본의 K·쌍·질문을 바꾸지 않는다.
- **운영 상태:** 검수 팩 원영상은 이 기기에 없고, 로컬 preparation audit만 있다. “서버 작업 없음”은 9/23 인수인계 기록이며 이번에 서버를 다시 조회한 사실은 아니다. 사람 두 명의 완료 판독과 새 D1 결과도 이번에 확인된 로컬 파일에는 없다.

## 3. CVPR에서 경쟁하는 질문으로 좁히기

지금 제목은 **검증할 가설의 가제**로 유지할 수 있다. 논문 중심 문장은 다음이 좋다.

> 같은 지역의 영상이 누적될 때, 고정된 저장·질의 예산 안에서 변화 전후의 시각 근거와 관측 출처를 보존하면, 일반 영상 기억보다 **답과 확인 시점의 정확성, 근거 충실도, 근거 부족 시 보류**를 함께 개선할 수 있는가?

여기서 가장 강한 후보 기여는 **관측을 압축해도 무엇을 주장할 수 있는지 보존하는 기억**이다. 설명문에 날짜를 붙이는 것과 달리, 실제 retained pixels/features를 바꾸면 답과 근거도 올바르게 달라져야 한다. 아직 이 효과는 미측정이다.

고정 공간 R, cutoff t, 그때까지 이용 가능한 관측 `X≤t`에 대해 writer는 `M_t = W(M_{t-1}, x_t, metadata_t)`를 만든다. 질문은 작성 후에 주며 reader는 `A(M_t, q)`로 답한다. 예산에는 visual feature뿐 아니라 근거 ID·좌표·요약문·검색 인덱스도 포함한다. 과거 archive를 재조회한다면 그 저장량·읽기량·지연도 포함한다.

임의로 긴 역사 전체의 모든 과거 질문을 고정 메모리로 정확히 답한다고 약속하지 않는다. 시간 범위, 지역 단위, 지원 질문군을 정하고, 압축 때문에 근거를 잃은 경우도 평가한다. 정보량이 계속 늘어나는 문제를 작은 K로 잘라 놓고 새 방법이 필요하다고 주장해서는 안 된다.

### VLM이 필요한 이유를 실험으로 보여주기

현재 D1은 고정 사분면의 구조물 변화·최초 관측 질의다. 이것만이면 전용 detector와 변화 기록으로도 풀 수 있다. fixed ROI를 입력으로 준 상태에서 spatial grounding을 해결했다고 쓰면 안 된다.

후속 VLM 평가는 **하나의 질문 전 memory snapshot**으로 다음처럼 서로 다른, gold로 채점 가능한 질의를 처리해야 한다.

- 특정 영역에서 기준 월 이후 무엇이 달라졌는가?
- 변화가 처음 확인된 제공 관측과 이를 뒷받침하는 이전 관측은 무엇인가?
- 특정 시점 이전에 확인된 변화만 골랐을 때 어느 영역이 해당하는가?
- 마지막 영상이 가려졌을 때 과거에 확인된 사실과 현재 확인 불가능한 사실은 무엇인가?

낯선 표현만 바꾸는 paraphrase split보다 **영역×시간조건×변화 유형의 새로운 조합**을 시험한다. 전용 비전 모델+구조화 로그+질의 파서/템플릿을 강한 대조로 넣는다. VLM이 이 비교에서 이점을 보이지 않으면 자연어를 붙였다는 이유만으로 VLM 기여를 주장하지 않는다. SN7이 제공하지 않는 도로 개통·재난 원인·회복·진짜 오판 정정은 주평가에 끼워 넣지 않는다.

## 4. 2026-09-24에 확인한 가까운 선행

아래 논문 결과는 저자 보고이며 이번 환경에서 재현하지 않았다. “처음”이라는 표현은 아래 범위 차이를 실제로 입증하기 전에는 사용하지 않는다.

| 선행 | 이미 다루는 부분 | 우리에게 남는 검증 / 비교 역할 |
|---|---|---|
| [EarthDial, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.html) | 다중 센서·시간·해상도 EO 대화와 grounding | EO 영상에 언어모델을 연결하는 것 자체는 신규성이 아니다. EO reader 계열의 관련 연구·가능한 외부 대조. |
| [LongEarth-R1, 2026 preprint](https://arxiv.org/abs/2608.13344) | 장기 EO QA, 관련 프레임·영역을 연결한 reasoning; 평균 15.14·최대 30프레임, 약 120k QA | D1의 13~21프레임을 ‘새로운 긴 시계열’로 팔 수 없다. 차이는 질문 전 online memory, 독립 visible gold, 증거 손실·보류 검증이어야 한다. |
| [FluxMem, CVPR 2026](https://arxiv.org/abs/2603.02096), [공식 코드](https://github.com/ShareLab-SII/FluxMem) | 학습 없는 시간·공간 계층 압축 | 최근 K장보다 강한 우선 memory 기준선. 단순 압축·계층 구조를 기여로 세지 않는다. |
| [SelectStream, 2026 preprint](https://arxiv.org/abs/2606.16353) | 고정 용량 latent evidence graph, 쓰기·보존·질문별 검색 | 선택적으로 기억한다는 설명만으로 구별되지 않는다. EO 전후 대응과 관측 출처가 만드는 추가 효과를 측정한다. |
| [LatentStream, 2026 preprint](https://arxiv.org/abs/2609.04131) | 질문 전 계층 기억과 질문 후 latent memory 갱신 | query-agnostic memory만으로 차별화하지 않는다. 근거 보존과 내부 latent 표현의 차이를 실험으로 분해한다. |
| [R4, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sohn_R4_Retrieval-Augmented_Reasoning_for_Vision-Language_Models_in_4D_Spatio-Temporal_Space_CVPR_2026_paper.html) | 의미·공간·시간 구조를 가진 지속 기억과 검색 | 날짜·좌표·기억 그래프를 붙이는 것 자체는 신규성이 아니다. archive access와 bounded-memory 예산은 따로 맞춘다. |
| [StreamReady, CVPR 2026](https://sacrcv.github.io/StreamReady-website/) | 증거가 나타나는 시점에 맞춰 답하는 streaming video 이해 | ‘무엇을 언제 답하나’도 선행이 있다. SN7 월별 mosaic에서 촬영일 단위 latency를 주장하지 않는다. |
| [AbstainEQA, CVPR 2026](https://abstaineqa.github.io/) | 정보 부족 때 보류, 언어 shortcut, 인간 대비 진단 | ‘모른다고 답한다’ 자체보다 같은 질문의 관측·증거가 달라질 때 적절히 유지/보류하는지를 평가한다. |

**이식 비용도 중요하다.** FluxMem 공개 구현은 Qwen2.5-VL 경로이고 D1은 Qwen3-VL/Molmo2다. 곧바로 동일 reader 비교가 되는 것은 아니다. memory 방법을 같은 backbone에 이식한 비교와 각 논문의 native 시스템 비교를 분리한다. 이식 검증이 안 된 수치를 ‘방법 차이’로 해석하지 않는다. SelectStream·LatentStream·LongEarth의 실제 실행 코드/가중치 접근성은 본 비교 착수 전에 확인하며, 이번 확인만으로 실행 가능하다고 주장하지 않는다.

## 5. D1 뒤에 무엇을 하나 더 증명해야 하는가

**H+L+R 통과는 기억 연구에 들어갈 허가이지, 기억 병목의 확증이나 방법 논문 완성이 아니다.** 특히 R의 3-class 내용 사용은 정확한 최초 시점·영역 판독을 보장하지 않는다.

권장 후속은 새 데이터·reader 학습·새 writer를 동시에 여는 대신, 통과한 **한 reader를 고정한 작은 기억 비교**다. D1 종료 후 별도 사전등록으로 한다.

| 조건 | 묻는 질문 |
|---|---|
| full-prefix | 현재 reader가 전체 가용 근거로 답할 수 있는가? 비용이 큰 참조 조건이며 수학적 상한은 아니다. |
| 사람 확인의 충분 근거 subset | 어떤 프레임을 주면 정답 시점과 근거를 읽는가? 배포 불가능한 진단용 oracle. |
| latest / uniform / change / quality-diversity | 단순 선택으로 충분한가? D1 selector와 추가 baseline을 구별한다. |
| generic memory + 같은 EO metadata | 기존 memory에 날짜·공간·품질만 붙여도 해결되는가? |
| deterministic 전후 근거 묶음 | 학습 없이 관측 ID와 전후 대응을 보존해도 충분한가? learned writer 이전의 핵심 대조. |
| 제안 writer | 동일 backbone·budget·training data에서 선택/교체 학습의 추가 이득이 있는가? 앞 조건들이 효용을 보여줄 때만 학습한다. |
| detector/segmentation → 구조화 로그 → 질의 처리 | VLM을 사용하는 실제 효용은 무엇인가? 사람이 준 gold detector 출력을 입력하지 않는다. |

필요한 기전 검증은 세 가지다. (1) 일반 memory가 시각적 중복으로 보이는 핵심 전후 근거를 잃는다. (2) 그 근거를 보존/복구하면 같은 reader의 답·시점이 회복된다. (3) 근거를 제거하거나 다른 정답의 영상으로 바꾸면 답이 올바르게 보류/변경된다. 현재는 모두 후속 가설이다.

### 최소 방법 설계 후보 — 아직 구현·성능 없음

기억 단위를 개별 평균 토큰 대신 **같은 영역의 읽을 수 있는 전후 관측 묶음**으로 둔다. 관측 ID와 시각은 visual feature의 실제 출처에 연결하고, 섞은 feature에 최신 날짜만 덮어쓰지 않는다. 새 관측이 흐려도 과거의 확인된 변화를 삭제하지 않으며 현재 상태는 별도 표시한다. 오래된 증거로 현재 상태까지 확정하지 않는다.

첫 버전은 고정 규칙의 전후 bundle 유지/교체다. 이것으로 충분하면 learned writer를 발명할 필요가 없다. 부족한 현상이 반복될 때만, 동일 데이터에서 **유효 전후 근거 집합 보존**을 감독하는 작은 writer를 학습한다. 시점·질문·test gold를 write 경로에 넣지 않는다. token mixing과 ID 보존을 분리하는 소거 실험이 필요하다.

제안의 성공 조건은 복잡한 JSON schema가 아니라 **같은 byte/token·archive access 조건에서 joint utility가 높아지는 것**이다. 최종 질문이 요구하는 모든 최초/무변화 증명을 무한히 기억한다는 보장은 하지 않는다.

## 6. CVPR 본회의를 겨냥한 증거 설계

다음은 학회의 공식 의무가 아니라 이 논문의 강한 주장에 맞춘 권고다.

**주평가.** 변화가 확인된 질문에서는 `답 정확 + 허용된 최초 관측/구간 + 유효 시각 근거`의 joint 성공률을 primary 후보로 둔다. 세부 영역 gold가 만들어지기 전에는 지역 IoU를 primary에 넣지 않는다. 허용 근거가 여러 개면 유효 집합 중 하나를 만족해도 인정한다. 전체 평균뿐 아니라 세 answer 범주별 결과·시간 질문 조건부 결과를 낸다.

**보류.** 관측으로 답할 수 없는 질문의 unsupported assertion rate와, 답할 수 있는 질문에 답하는 coverage를 함께 본다. 항상 보류하는 모델이 좋은 모델이 되면 안 된다. 미래의 risk–coverage 곡선을 쓴다면 confidence 계산과 threshold 선택은 validation에서 고정한다. 없는 confidence를 사후에 만들지 않는다.

**역사와 현재.** 과거 변화 사실을 올바르게 유지하는 비율과, 최신 관측이 가려졌는데 현재도 그렇다고 부당하게 확정하는 비율을 분리한다. 실제 회복은 과거 사실의 부정이 아니다. 새 관측이 과거 해석을 반박하는 독립 gold가 없으면 belief revision을 기여에 넣지 않는다.

**예산.** K장만 맞추지 말고 visual token, feature·metadata bytes, peak GPU memory, 한 관측 도착당 갱신 비용, 질의당 prefill/생성 비용, archive raw read bytes를 함께 기록한다. 양쪽이 보유하는 최신 프레임·지역 crop·다중 해상도·외부 텍스트에도 같은 권한을 적용한다. full-prefix는 동일 예산 경쟁자라기보다 비용–품질 곡선의 참조점이다.

**일반화.** 노출된 6 AOI는 개발용으로 고정한다. 동일 AOI의 다른 사분면·cutoff·질문은 같은 split에 두고, 학습·선택에 쓰지 않은 AOI를 한 번 평가한다. 표본 수는 QA 문장 수보다 독립 AOI/사건 수로 계획한다. 최소 2 reader 계열과 두 번째 자료 출처는 강한 일반성 주장에 권장하지만 D1 이전에 전부 구축할 선행조건은 아니다.

**데이터 확장.** [SpaceNet 7](https://spacenet.ai/sn7-challenge/)의 월별 mosaic은 월 단위 관측 비교에 쓴다. 가림·불규칙한 실제 취득시각에 대한 주장은 SN7 G0 실패 이후 자동으로 살아 있지 않다. S2 장기열은 예산·접근성 확인 후 별도 단계로 열고 native OlmoEarth 입력과 맞춘다. DynamicEarthNet gap-fill의 미래 관측 사용 여부를 확인하기 전 causal replay라고 부르지 않는다.

**gold 비용.** D1은 두 사람 ≤10인시 그대로다. 이후 규모는 D1에서 측정한 판독 시간, 클래스/쌍 유효율, AOI별 변동을 바탕으로 정한다. 파일의 기존 50~100인시 장기 예산을 현재 즉시 집행할 승인으로 확대하지 않는다. 학습 gold와 시험 gold를 분리하고, 검수 불일치는 원본과 조정본을 모두 남긴다.

## 7. 네 갈래 결정의 CVPR 의미

| D1 결과 | 유지할 해석 | 다음 한 단계 / 논문 판단 |
|---|---|---|
| H 실패 | 이 질문·자료·검수 계약의 읽기 가능성이 부족 | SN7 기억 노선 중단. 다른 모든 EO 연구가 불가능하다는 뜻은 아니다. |
| H 통과, L=no_loss | 이 표본·K=4의 지정 근거쌍을 단순 선택이 잘 보존 | 새 memory 투자 이유가 약함. benchmark/reader 논문도 **별도의 발견과 외부 검증**이 있어야 하며 자동 구제책은 아니다. |
| H 통과, L=loss, R=reads_content | memory 연구를 시도할 최소 조건 | 고정 reader의 정확한 시간 판독과 deterministic bundle/generic memory pilot. 이 성공이 있어야 방법 논문을 논의한다. |
| H 통과, L=loss, R 기준 미충족 | 현재 reader/계약으로 기억 효용 판정 불가 | 목표 정합적인 reader alignment pilot을 선택할 수 있음. 학습을 자동 실행하지 않는다. |
| L 중간값 / H 통과하나 control pair 없음 / R 미완결 | 기존 네 갈래로 결론을 낼 수 없음 | inconclusive·실행 준비 부족을 명시. 같은 표본에서 threshold를 조정하지 않는다. |

이 표의 마지막 행은 원 prereg의 새 성공 기준이 아니다. 현재 원 결정표가 덮지 못하는 상태를 드러낸 **검토 제안**이다.

### 두 가지 원고 경로

**방법 논문 경로:** `Grounded Change Memory for Earth Observation Streams`를 유지한다. 본문 중심은 (a) 관측으로 확인 가능한 질문/근거 정의, (b) 관측 출처를 보존하는 budgeted writer, (c) 독립 지역의 동일 비용 비교와 기전 검증이다. MS-122~155의 긴 시행착오는 핵심 동기를 설명하는 짧은 단락과 supplement로 옮긴다.

**평가·분석 논문 경로:** 작업 가제로 `Can Vision–Language Models Ground Change in Earth Observation Streams?`를 고려한다. 주요 기여는 “우리 진단을 여러 번 고쳤다”가 아니라 **사람이 판독할 수 있는 temporal evidence를 기준으로, 흔히 쓰는 평가가 어떤 잘못된 결론을 내며 그것을 어떻게 재현 가능하게 구별하는지**다. 여러 reader, 미사용 지역, 검증된 정답 교체와 근거 제거에서 같은 문제가 재현돼야 한다. 버그 수정 기록과 12 episode만으로는 본회의 기여가 충분하지 않다.

CVPR는 새 architecture나 모든 benchmark의 SOTA만을 요구하지 않는다. 2026 공식 reviewer guide는 기술적 건전성과 지식의 진전을 보며 성능만으로 판단하지 말라고 안내한다. 다만 이것이 작은 실패 사례의 자동 채택을 뜻하지는 않는다. 이 프로젝트의 제출 준비도 판단은 위 증거 상태에 대한 연구적 판단이다. [CVPR 2026 reviewer guide](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

## 8. 일정: D1부터 한 단계씩

CVPR 2027 공식 CFP에서 **등록 2026-11-10 AoE, 본문 11-16 AoE, supplement 11-23 AoE**를 확인했다. 한국 시간 본문 마감은 **11월 17일 20:59:59 KST**다(AoE의 11/16 종료 기준). 9/24부터 약 7.5주다. 2027 세부 author/reviewer guideline은 이번 조회에서 열리지 않아 확인된 CFP와 2026 심사 관점을 구별했다. [공식 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

같은 CFP는 2026-09-15 이후 공개된 연구를 일반적으로 contemporaneous work로 다룬다고 한다. LongEarth-R1(8월), SelectStream(6월), LatentStream(9월 초)은 그 날짜 이전이다. “preprint라서 비교하지 않아도 된다”는 식으로 범위를 줄이지 말고 실질적인 차이를 논의한다. 코드 접근 불가와 선행 미고려는 다른 문제다. [공식 prior-work 안내](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

| 시점 | 구체 산출물 | 중단/축소 기준 |
|---|---|---|
| 즉시 | D1 사전 결과 감사 메모·amendment 초안, H gate/채점/완결성 수리 및 합성 검증 | H/L/R의 의미가 합의되지 않으면 모델 판정 실행 보류 |
| 두 검수자 확보 후 | 패키지 무결성 확인, H 독립 판독 ≤10인시 | H 실패 시 기존 규칙대로 L/R 중단 |
| H 통과 후 | L 원 보존곡선, 유효 donor 쌍에서 R 재실행·완결 보고 | no_loss·inconclusive·reader 불충족을 각각 기록 |
| D1 통과 시 다음 1주 | 한 reader 고정, 충분 근거 진단 + generic/deterministic memory 비교 | 회복할 오류가 없으면 learned writer를 열지 않음 |
| 제안 checkpoint: 10/15 | 유효한 새 발견, 미사용 AOI 시험계획, 구현 가능한 강한 기준선 | 하나라도 없으면 본회의 제출을 기본 일정으로 두지 않음 |
| 제안 checkpoint: 10/29 | 주요 비교·비용·근거 intervention·독립 평가의 표가 닫힘 | 미완결이면 방법 주장을 넣은 제출을 밀어붙이지 않음 |
| 마지막 2주 | 집필·그림·재현 자료·인용·한계 | submission 직전에 새 학습/데이터 축을 열지 않음 |

10/15·10/29는 이번 **일정 권고**이고 실험 사전등록 gate가 아니다. 현재 정보로 “11/16까지 방법 논문 완성 가능”을 약속할 수 없다. D1 뒤 짧은 pilot까지를 현실적 우선 목표로 삼는 9/23 판단을 유지한다.

## 9. 바로 사용할 수 있는 연구 설명

> 이 연구는 동일 지역의 위성 관측이 순차적으로 쌓일 때 제한된 기억으로 변화 여부, 처음 확인 가능한 관측, 현재의 정보 부족을 시각 근거와 함께 답할 수 있는지 묻는다. 현재까지 학습한 방법들은 등록된 utility 관문을 통과하지 못했고, SpaceNet 7의 초기 VLM 진단에서는 정답 위치·질문과 silver 라벨의 불일치·영상 교체 통제의 결함이 확인되어 기억 병목 및 SFT 선행 주장을 철회했다. 다음 단계 D1은 사람의 판독 일치도, 단순 선택기의 근거 보존, reader의 영상 내용 사용을 분리하는 개발 실험이다. 9월 24일 코드 감사에서 H 관문 연결과 R 채점의 결함을 확인했으므로 이를 결과 전에 바로잡아야 한다. D1 이후에도, 관측 출처와 전후 근거를 보존하는 기억이 강한 일반 영상 기억과 구조화 로그보다 같은 비용에서 답·시점·근거·보류를 개선하는지는 독립적으로 검증해야 한다. 이것이 CVPR 방법 논문으로 진입하기 위한 핵심 질문이다.

집필용 영어 문제 정의(결과를 주장하지 않는 초안):

> Earth observation streams accumulate evidence of geographic change, yet a useful answer must distinguish a past observation from a claim about the current state. We study whether a bounded visual memory can preserve the observations needed to answer questions about change, identify the first provided observation that clearly supports it, and abstain when the retained evidence is insufficient. Our evaluation separates human readability, evidence retention, and a vision–language reader's use of image content. This separation is necessary before attributing errors to memory compression or introducing a learned memory policy.

## 10. 검토 범위·재현 경로

- 실제 실험 저장소와 처음 열린 OlmoEarth 학습자료 저장소는 다르다. 이번 연구 업데이트는 `eo_olmo_earth_project`에 둔다. Studio 제품·PDE·채용·PR 활동은 이번 검토 범위 밖이다.
- 읽은 정본: RESTART/HANDOFF, D1 및 v0~v0.4 prereg, PLAN, CVPR_SINGLE_CLAIM, MS-155 감사, 9/20 원장 감사, 관련 MEASURED_FINDINGS 항목, 저장 답변/점수, v0.5 계약·pack·L·R 코드와 테스트.
- 새 실행: 기존 계약 테스트 23개, L/R selftest, 저장 MS-155 답변 산술 감사, 합성 D1 계약 감사. 사람 gold 생성·모델 추론·서버 재확인은 없음.
- 기본 `python3`에는 numpy가 없어 합성 scorer 감사에는 기존 `earth_paper/olmoearth/.venv-nano-lab/bin/python`의 numpy 2.5.3을 사용했다. 환경을 새로 설치하거나 변경하지 않았다.
- 재현: numpy가 있는 환경에서 `python -B code/audit_d1_contract_20260924.py --repo .`. 이 스크립트는 실제 모델 loader를 sentinel로 교체하고 임시 디렉터리만 사용한다. 감사 대상 버전의 결함을 확인하는 것이므로 코드가 고쳐지면 assertion 결과가 달라지는 것이 정상이다.
- 원 실험 코드·prereg·측정 장부는 보존했다. 이 문서의 제안은 자동 실행 지시나 사전등록된 새 판정이 아니다.
