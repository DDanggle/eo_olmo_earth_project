# E7b — PANGAEA public twin 후보 감사 (계약 확인만, 학습 금지)

작성 2026-08-26. E6(action matrix v1) 완료 전에는 어떤 후보도 학습하지 않음
(audit-budget 규칙: 이 감사는 RQ2 public twin을 직접 막는 항목이라 허용됨).

## 왜 필요한가

M42에서 Sen12 내부의 seg vs retrieval task 쌍은 **순위 역전을 보이지 않았음**
(kill gate 발동, 주장 보류). task 이질성의 public 증거가 되려면 **물리 신호가 다른**
task 쌍이 필요함. AI-Hub 3-task는 내국인 전용이므로 headline에는 공개 짝이 필수임.

## 후보 (PANGAEA-bench, 1차 조사 2026-08-26)

| 후보 | 센서 | S2 밴드 | task | 시계열 | 접근 |
|---|---|---|---|---|---|
| PASTIS-R | S1+S2 | 11 | 작물 의미분할 | 다중시점 | HuggingFace |
| Sen1Floods11 | S1+S2 | 11 | 홍수 분할 | 단일 위주 | GitHub |
| CropTypeMapping (South Sudan) | S1+S2+Planet | 11 | 작물 분할 | 다중시점 | SustainBench |

**1차 선택 제안**: PASTIS-R(작물·계절 신호) + Sen1Floods11(수체·단일시점 신호).
산사태(Sen12)와 물리 신호가 가장 다른 조합임. 셋째는 E6 결과 후 판단.

## 통과해야 할 게이트 (학습 전, 각 후보별)

1. **밴드 계약**: 11밴드의 정체 확인 — B01/B09/B10 중 무엇이 빠졌는지, v1 band-set
   3구획에 어떻게 매핑되는지, 결측 band-set을 `MaskValue.MISSING`으로 표기 가능한지
   (M8: v1.2는 mask[...,0]만 읽으므로 **v1로만** 진행).
2. **라이선스**: 저장소에 명시 없음 — 원 논문·배포처에서 확인해 기록. 불명이면 제외.
3. **split 누수**: 공식 split의 공간 누수 감사 (M9에서 AI-Hub 공식 split이 110/110
   누수였던 전례). 필요 시 공간 블록 재분할 + SHA 봉인.
4. **시계열 계약**: PASTIS-R은 시점 수가 가변 — S12q류의 라벨 무관 시점 선택 규칙을
   사전 등록해야 함. v1 time table은 12개 한계(M-기록: 15개 입력 shape error).
5. **정규화 계약**: 반사도 스케일이 Sen12(int16 [0,10000])와 같은지.

## 하지 않는 것

- E6 완료 전 다운로드 이상의 작업 (디스크는 서버 `/home/work/data`만, 로컬 금지)
- 11밴드를 12밴드 슬롯에 억지로 채우는 것 (M28·M3 전례)
