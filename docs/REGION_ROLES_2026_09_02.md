# 지역별 역할 — 각 지역이 "무엇을 증명하고, 무엇을 반증할 수 있는가" (2026-09-02)

원칙: 지역은 표본이 아니라 **역할**로 배정함. 한 지역을 두 역할(개발+확증)에 쓰지 않음. 통계 단위는 지역이며
seed·support draw 는 반복 측정임. 아래 표의 수치는 M65/MS-87/MS-93 봉인값(positive-tile macro IoU, seed 평균).

## 1. Sen12 확증 8지역 — zero-target transfer 의 증거이자, 새 method 의 사전등록 평가장(PT-3)

| 지역 | 성격(관측) | P4 / 최고 raw | 배정 역할 | 이 지역이 반증할 수 있는 것 |
|---|---|---:|---|---|
| **hokkaido** | 지진 유발, 재사용 이득 최대 | .386 / .221 (+.165) | *reuse 상한 지역* — 재사용이 가장 잘 되는 조건의 대표 | "재사용 이득은 작다" |
| **hiroshima** | 호우 유발, 시드 분산 큼 | .278 / .216 (+.062) | *분산 감시 지역* — 시드/서브셋 분산 보고 의무 | "결과가 seed 에 안정적이다" |
| **indonesia** | **유일하게 raw 가 이김** | .272 / .284 (−.012) | *반증 지역(falsifier)* — 새 method 는 여기서 A1 보다 나빠지면 안 됨 | "재사용이 항상 낫다" |
| **itogon** | 모든 arm 바닥(≈.15) | .152 / .148 (+.004) | *바닥 지역* — 표현이 아니라 라벨/장면 한계인지 판별 | "낮은 IoU 는 모델 탓" |
| **kyrgyzstan1** | 재사용 이득 큼, 시드 분산 큼 | .281 / .192 (+.089) | reuse 재현 지역 ② | hokkaido 결과의 단독성 |
| **kyrgyzstan2** | 양성 희소, raw 붕괴(.107) | .208 / .107 (+.101) | *희소 라벨 지역* — 라벨 예산 곡선의 저예산 끝 대표 | "라벨이 적어도 raw 가 버틴다" |
| **newzealand** | 중간 난도, P3(U-TAE) 가 P2 보다 나은 유일 지역 | .242 / .188 (+.054) | *아키텍처 민감 지역* — raw recipe audit 의 기준 | "raw baseline 이 약해서 졌다" |
| **thrissur** | 재사용 이득 큼, Presto native 가 가장 많이 회복(.139→.208) | .359 / .232 (+.127) | *Presto 최선 지역* — GeoFM 대조의 보수적 끝 | "Presto 는 항상 무력하다" |

규칙: PT-3(새 method 확증)은 이 8지역 전부에서 A2 vs A1 을 사전등록 지표로 판정하며, **indonesia 에서 A1 미만이면
전체 통과와 무관하게 "raw-우세 지역에서 실패"를 헤드라인에 병기**함. 8지역은 같은 벤치마크 수집 설계라 지리적 독립이 아님 — 외부 전이 판정은 한국에서만.

## 2. 개발(exposed) 2지역 — method·hyperparameter 개발 전용, 확증 주장 금지

| 지역 | 규모 | 배정 역할 | PT-0 봉인 결과 |
|---|---|---|---|
| **china** | 159타일 / 양성 81 | *소표본 스트레스* — K 실현 가능성·support 다양성의 하한 시험 | pool 92(양성 48) · query 52(양성 22) · buffer 제거 15 · K=5/20 가능 |
| **chimanimani** | 1,133타일 / 양성 423 | *학습곡선 개발* — K=5/20/50 곡선 모양·rank 민감도 개발 | pool 690(양성 187) · query 359(양성 195) · buffer 제거 84 · K=5/20 가능 |

두 지역의 결과는 어떤 논문 표에도 "성능"으로 들어가지 않음. PT-1 method gate 의 판정 재료일 뿐임.
china 가 실패하고 chimanimani 만 통과하면 "소표본에서 불안정"으로 기록하고 gate 는 불통과로 침(등록: 두 지역 같은 방향).

## 3. 한국 — 봉인된 최종 외부 시험 (한 번만 개봉)

- 역할 ①: **공간 분리 target few-shot** — Sen12 에서 동결한 protocol(K, arm, 작동점, update 수)을 그대로 적용.
- 역할 ②: **3-task cache amortization** — 같은 OlmoEarth cache 위에 land-cover / 벌목 / 산사태 adapter 3개. "task 가 몇 개부터
  cache+adapter 가 raw 모델 3개보다 싸지는가"를 raw I/O·GPU s·bytes 로 실측. 이것이 사업 축과 닿는 유일한 지역임.
- 금지: hyperparameter 개발·중간 열람. 개봉 전 `Korea input/ontology preflight`(밴드·해상도·라벨 정의 정합) 통과 필수.
- 반증 가능성: A2 가 한국에서 A1 이하이면 method 주장에서 한국을 빼고 in-benchmark 주장으로 강등(등록됨).

## 4. 네팔 — K=0 운영 사례 (method 증거 아님)

- 역할: target 라벨 0 에서 embedding Δ 가 검토 순위를 만드는 **실사건 데모**(NP-88/89: 외부 flood proxy 와 토큰 AUROC .846, 강 밖 .873).
- 한계 고정: proxy 라벨·사건 1건·AUPRC 에서는 post-event NDWI 우세·공간 block CI 가 0 포함. 논문에서는 그림 1장 + 운영 절.
- 반증 가능성: 없음(라벨 부재). 그래서 method 표에 넣지 않음.

## 5. 스위스 — 센서 계약 스트레스 (동일 전이라고 부르지 않음)

- 역할: 7밴드 제품 → **missing-band contract shift**. band availability gate 와 abstention action 을 평가.
- 질문: "cache 계약이 깨졌을 때 planner 가 A0~A4 대신 abstain 을 고르는가". 성능 비교 지역이 아님.

## 6. FoldRefresh — 지역이 아니라 **시간축 역할**

- OlmoEarth v1 → v1.2 처럼 encoder 가 바뀌면 cache 가 무효화됨(M85 에서 v1.2 는 광학 2승4패, 결합 Hokkaido +.032).
- 역할: "재임베딩 vs adapter 이식" 의 비용·성능 action 을 측정하는 cache-migration 시험. A3(encoder PEFT)와 같은 비용 항목으로 보고.

## 배정의 논리 한 줄
Sen12 8지역 = **주장**, china·chimanimani = **개발**, 한국 = **외부 판정**, 네팔 = **운영 데모**, 스위스 = **계약 파손**, FoldRefresh = **버전 이행**.
어느 지역도 두 칸에 걸치지 않음. 이 표는 PT-1 결과를 보기 전에 작성·커밋함.
