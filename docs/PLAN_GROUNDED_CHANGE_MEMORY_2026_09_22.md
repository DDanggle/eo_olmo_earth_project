# 실행 계획 — Grounded Change Memory for EO Streams (CVPR 2027, 마감 2026-11-16 AoE)

작성 2026-09-22. 근거 문서: `CVPR_SINGLE_CLAIM_DECISION_2026_09_22.md`(방향), `EXPERIMENT_LEDGER_2026_09_20.md`(기존 실패), `REGION_LANGUAGE_VLM_RESEARCH_2026_09_21.md`(문헌·데이터 감사).
이 문서는 계획이며 사전등록이 아님. 각 단계는 실행 전 별도 prereg JSON으로 등록함. 기존 실험의 관문 수치는 바꾸지 않음.

## 0. 한 문장

불규칙하고 부분적으로 가려진 동일 지역 관측 스트림에서, 지리적으로 대응되는 전후 증거를 보존하는 기억이 일반 영상 요약·검색보다 같은 예산에서 변화 질의의 답과 근거를 더 잘 유지하는가.

## 1. 결정 구조 — 세 관문, 각 관문 실패 시 행동을 미리 고정

| 관문 | 시점 | 질문 | 통과 | 실패 시 |
|---|---|---|---|---|
| G0 데이터 적합성 | 1주 말 | 제안 데이터에 논문의 현상(가림·불규칙 간격·긴 누적)이 실제로 있는가 | SpaceNet 7 AOI별 가림 월 비율 집계 + S2 장기열 확보 가능 확인 | **SN7 실패(MS-150, 9/22)**: 사건 18.9% 가림이나 13/60 AOI에 집중. 가림 축은 S2 장기열로 이관, SN7은 변화 기억 개발 자료 |
| G1 병목 진단 | 3주 말 | 같은 reader에서 근거 선택이 병목인가 (결정 문서 §7의 4조건) | 일반 memory가 근거를 잃고 privileged evidence에서 회복 | reader 병목 → 기억 학습 중단, reader 정렬로 회귀. 동률 → 방법 주장 중단, CVPR 철회 검토. **진행(9/22)**: lite v0 TEOChat 퇴화(MS-151), v0.1 Qwen3-VL은 Q1 해상도 병목이나 Q2에서 privileged 우세 CI 밖(MS-152) → v0.2 원해상도 crop으로 재진단 |
| G2 방법 이득 | 6주 말 | 제안 모듈이 generic memory·deterministic top-K보다 답·근거·영역 joint score를 개선하는가 | 독립 출처 2곳, AOI cluster CI 0 제외 | 방법 절 삭제, 실패 분석+벤치마크 계약으로 축소하고 학회 급 재판단 |

G1이 진짜 결정점임. G1 실패면 11/16 제출을 포기하는 것을 지금 합의함.

## 2. 데이터 계약

### 2.1 개발 pilot — SpaceNet 7 (Planet 4 m, 월별 mosaic, 24개월, 건물 polygon 변화)
- 역할: 근거 후보 생성이 쉬운 개발 자료. 라벨은 omniscient(보조 고해상도·시계열 검토)이므로 **world-state gold**로만 쓰고, prefix에서 보이는지는 별도 검수(**prefix-visible gold**)로 분리함.
- 분할: 공개 라벨 AOI를 지역 단위로 개발/검증/시험으로 나눔. 공식 challenge test 점수와 구분함.
- G0 집계: UDM 마스크로 AOI×월 가림 비율, 연속 가림 길이 분포, 변화 사건의 전후 유효 관측 수.
- 한계 표기: 월 단위 순차 평가이며 촬영일 단위 online 지연이 아님.

### 2.2 현실성·native 자료 — 우리 4지역 S2 장기열 (신규 취득)
- 역할: 실제 구름·불규칙 재방문·수백 관측의 누적을 제공. OlmoEarth native 입력 계약이 맞는 유일한 자료. 결정 문서 §6 "native S2 효용은 센서 계약이 맞는 데이터에서 별도 시험"의 실체.
- 범위: hiroshima·thrissur·itogon·hokkaido 기존 타일 footprint, 사건 전후 ±12개월, 전 관측(구름 포함). 예상 약 100~150관측/타일. 용량·취득 경로(Planetary Computer 또는 AWS)와 소요를 G0에서 실측.
- 라벨: 기존 Sen12 마스크(사건 후 상태)만 있음. 변화 시점·근거 gold는 사람 검수로 소수 타일만 만듦. 이 자료의 역할은 **비용·가림 스트레스**이고, 정확도 주장의 본체는 SpaceNet 7.
- 기존 15관측 캐시·4사건 QA는 개발 진단에만 재사용, 외부 검증으로 세지 않음.

### 2.3 독립 확장 — DynamicEarthNet (Planet, 일별 gap-fill, 토지피복 변화)
- 역할: 두 번째 출처에서 같은 기억 규칙 검증. gap-fill이 가장 가까운 시점 관측으로 채워지므로 회고 평가만 하고 real-time 주장 안 함. 원시 관측 replay 가능 여부를 4주 안에 확인.

### 2.4 인코더
- SpaceNet 7·DynamicEarthNet: VHR/Planet용 encoder(공개 RGB+NIR 대응 모델 또는 일반 vision encoder) 사용. OlmoEarth에 억지로 넣지 않음.
- S2 장기열: frozen OlmoEarth-v1-Base 공간 토큰(32×32 유지, 8×8 풀링 폐기).
- 기억 모듈과 reader는 encoder 무관하게 같은 인터페이스(영역 토큰 + 좌표·시각·센서·품질)로 둠.

## 3. 방법 — 학습 대상 하나

- 기억 단위: 동일 지리 영역의 **전후 관측 묶음**(evidence unit) = {원관측 ID, 영역, 날짜/구간, 판독 가능 마스크, visual feature, 현재 상태 feature와 과거 사실 참조의 분리}.
- 학습 모듈(writer): 이전 기억 + 새 공간 토큰 + 실제 간격 + 지역 대응 + 품질 → 영역별 evidence unit의 유지/추가/교체 우선순위. 질문을 모름(causal). 
- reader: frozen LLM(Olmo-3-7B, 기존 코드 재사용) + LoRA. 기억의 visual token과 근거 참조를 읽어 영역·시간·관계 질의에 답하고 근거 ID를 냄. query-aware 검색은 read 단계에서만.
- 감독: 정답 문장 + **필요한 전후 근거 집합 보존**(대체 가능한 유효 집합 허용) + 증거 제거/교체 시 답·보류 변화.
- 출력 구분(결정 문서 §5): 상태 변화 / 지식 갱신 / 판독 불가 / 회복 / 진짜 오류 정정. 마지막은 gold 부족 시 별도 subset으로만.

## 4. 기준선 (전부 같은 encoder·reader·예산)

1. full-prefix: 현재까지 전체 관측 제공(LongEarth 방식 재계산 포함).
2. deterministic: latest-K, uniform-K, quality+diversity top-K, 변화량 top-K.
3. generic streaming memory: FluxMem(공개 코드), SelectStream(미공개면 reference implementation으로 표기). 둘에 EO 메타정보만 붙인 강한 대조 포함.
4. 구조화 대조: 학습 분할/추적 → 변화 ledger → 템플릿 질의 처리(우리 event_log_v1 계보).
5. privileged evidence selector(진단 전용, 배포 불가 표기).

## 5. 평가

- 지표: 답 정확도, 근거 ID 정확도, 영역 IoU, 보류 위험-커버리지, 부당 철회율(흐린 관측 뒤 과거 사실 삭제), 노후화 표시 정확도(마지막 확인 시점 유지 + 현재 불확실성), 비용(초기 인코딩·원본 읽기·갱신·질의 토큰/바이트).
- 단위: AOI/사건 cluster, paired 차이의 cluster bootstrap CI.
- 일반화: 지리 분리 출처 2곳에서 같은 weights.
- 질의: 학습에 없던 영역·시간 조건·관계 조합을 사전 공개 범위 안에서 포함. LLM-judge 단독 금지.
- 비용 공정성: 모든 방법의 archive 접근권·I/O 동일.

## 6. 일정 (8주, 2026-09-22 → 11-16)

| 주 | 할 일 | 산출물 | 관문 |
|---|---|---|---|
| 1 (9/22–28) | SpaceNet 7 취득·UDM 가림 집계, S2 장기열 취득 경로·용량 실측, DynamicEarthNet replay 가능성 확인, 진단 prereg 등록 | `sn7_occlusion_audit.json`, `s2_longseries_feasibility.json`, `bottleneck_diag_prereg_v0.json` | G0 |
| 2 | prefix-visible gold 소규모 검수(개발 AOI 10~20개), 4조건 진단 파이프라인, FluxMem 이식 | 진단 코드 스냅샷, gold v0 | |
| 3 | 진단 실행·판정, 사람 간 gold 일치도 | `MS-150` 진단 결과 | **G1** |
| 4 | writer v0 설계·학습(SpaceNet 7 개발 AOI), SelectStream reference 구현, S2 장기열 OlmoEarth 추출 시작 | writer prereg, 추출 캐시 | |
| 5 | 본 비교(기준선 5종), 소거(메타정보 제거·구름 합성 stress) | `MS-151~` | |
| 6 | DynamicEarthNet 확장, S2 장기열 비용·가림 스트레스, cluster CI | 결과 표 | **G2** |
| 7 | 사람 검수 시험 세트 확장, 실패 메커니즘 분석(언제 근거를 잃는가), 그림 | 표·그림 확정 | |
| 8 (11/9–16) | 집필·등록(11/10)·제출(11/16), 공개 패키지(코드·gold·평가 계약) | 제출 | |

버퍼가 없음. G1이 3주 말을 넘기면 제출 포기를 기본값으로 둠.

## 7. 사전등록 목록 (실행 전 각각 JSON)

1. `sn7_occlusion_audit_prereg_v0` — 가림 집계 정의, "현상이 있다"의 문턱(예: 변화 사건 중 ≥30%가 직전·직후 1개월 이상 가림).
2. `bottleneck_diag_prereg_v0` — 4조건, 판정 다섯 갈래(결정 문서 §7), AOI 수, cutoff 수, 예산 K, 비용 보고 항목.
3. `prefix_visible_gold_protocol_v0` — 검수자가 prefix만 보고 표시하는 절차, 두 검수자 일치도 문턱.
4. `change_memory_writer_prereg_v0` — 모듈 입력·출력, 감독, 학습 AOI, 기준선 5종, 통과 폭·CI.
5. `s2_longseries_contract_v0` — 취득 범위·구름 포함 원칙·OlmoEarth 입력 계약·역할 한정(스트레스·비용).

## 8. 기존 자산 재사용과 금지

- 재사용: Olmo-3-7B reader 코드(earthtalk_seq 계보), 사건 로그 v1(구조화 대조), 도착 순서 캐시(개발 진단), 계절 정상 적응(MS-148, 지역 정상 변동 입력), 문턱 비이식 결과(MS-141/146, 변화량 top-K 기준선의 약점 설명).
- 금지: 이전 관측 2개 필요·규칙 강건성 결과를 보편 법칙이나 이번 방법의 양성 증거로 인용(결정 문서 §9). 실패한 D·E를 "잘못된 버전"으로 무효화. 기억 크기만 줄인 인위적 문제. 최종 마스크를 이전 시점에 복사.

## 9. 역할·자원

- GPU: 서버 H200 2장(타 사용자와 공유 중 — 실행 전 nvidia-smi 확인). 진단·writer 학습은 소규모(수 GPU-hour), 큰 비용은 S2 장기열 OlmoEarth 추출(관측 수×타일 수).
- 사람 검수 예산(2026-09-22 사용자 확정): **사용자 50시간, 추가 1명 확보 시 최대 100시간.** 이 안에서 배분함.

| 용도 | 항목 | 시간 | 비고 |
|---|---|---|---|
| G1 진단 gold | AOI 12 × cutoff 2 × 질문 4 ≈ 100 + 겹침 20% | 약 25h (2명 각 12h) | 항목당 5분 → 도구로 후보 근거를 미리 잘라 주면 3분 |
| 본 시험 세트 | AOI 40 × cutoff 2 × 질문 5 ≈ 400, 단일 검수 + 겹침 20% | 약 50h | 남는 시간은 겹침 비율 올리기에 우선 |
| S2 장기열 소규모 | 타일 15 × 질문 8 ≈ 120 | 약 12h | 스트레스·비용 자료라 소규모로 충분 |
| 예비 | 프로토콜 시범·재검수 | 약 13h | |

- 총 100h. 1명(50h)만 가능하면 본 시험 세트를 200 항목으로 줄이고 겹침을 포기하며, 일치도(kappa)는 G1 세트에서만 보고함. 이 경우 논문에 "단일 검수자" 한계를 명시.
- 항목당 시간을 줄이는 도구를 2주차에 먼저 만듦: 라벨에서 후보 근거 프레임·영역을 미리 추출해 검수자는 확인/기각만 함. 검수자가 후보 밖 근거를 추가할 수 있어야 silver 편향이 gold에 그대로 옮겨가지 않음.
- 검수자는 prefix 이후 프레임을 볼 수 없게 도구에서 차단함(omniscient 라벨 유출 방지).
- 문헌 재확인: FluxMem 공식 코드 실행 가능성, SelectStream·LatentStream 공개 여부(미공개면 reference 표기), LongEarth-R1 코드·데이터 접근.

## 10. 열린 결정 (사용자·교수)

1. G1 실패 시 제출 포기를 지금 합의하는가. (2026-09-22 설명 완료, 답 대기)
2. S2 장기열 취득을 1주 안에 시작하는가(OlmoEarth·Ai2 접점 유지 여부와 직결). (답 대기)
3. 사람 검수 인력: 사용자 50h 확정, 추가 1명(+50h) 확보 여부 미정. 배분은 §9 표.
4. SpaceNet 7 서버 취득: **승인됨(2026-09-22)**.
