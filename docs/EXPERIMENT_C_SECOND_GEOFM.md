# 실험 C — 두 번째 frozen GeoFM: "OLMo 고유 효과인가, 일반 GeoFM 효과인가"

작성 2026-08-27. M61의 최우선 미착수 축임. **thrissur 결과는 이미 봤으므로 이 설계는
thrissur에 대해 exploratory, hiroshima 이후 미열람 지역에 대해 confirmatory임.**

## 답하려는 질문 (하나만)

> 같은 데이터·같은 판독기에서 frozen OlmoEarth를 **다른 frozen GeoFM**으로 바꾸면
> 이득이 유지되는가, 사라지는가, 역전되는가?

- 유지되면 → "이득은 일반 GeoFM 효과"이고 논문 축은 transfer frontier(D·E)로 감
- OLMo만 크면 → 아키텍처/사전학습 데이터 고유 효과 — 무엇이 다른지 분해가 다음 질문
- 역전되면 → OLMo 선택 자체가 재검토 대상임

**어느 결과든 논문에 쓸 수 있음.** "OLMo가 이겨야 한다"는 예측이 아님.

## 후보 선정 — 타당성 순

| 후보 | 입력 계약 적합성 | 실행 타당성 | 판정 |
|---|---|---|---|
| **Presto** (nasaharvest) | **S2 10밴드 = 우리 REAL_BANDS와 일치, 12 timestep 기본, 결측 modality 마스킹 내장** | 순수 torch+einops, 단일 파일 vendoring 가능, 가중치 수 MB | **C1 채택** |
| **Clay v1 / v1.5** | S2 지원, 단일 시점(시간축은 우리가 pooling) | HF 가중치, torch로 로드 가능 | **C2 채택 — RQ3 release pair 겸용** |
| Prithvi-EO-2.0 | HLS 6밴드로 밴드 계약 불일치 | terratorch 필요(pip 없음) | 보류 |
| Galileo | 시계열 적합 | 의존성 확인 필요 | C1·C2 후 검토 |
| DOFA/SatMAE | 단일 영상 전용 | — | 제외 |

Presto가 이례적으로 잘 맞음: **S2 밴드 10개(B01·B09 제외)가 Sen12 실관측과 정확히 같고**,
12 timestep이 S12q와 같고, 결측 그룹 마스킹이 우리 MISSING 계약과 같은 사상임.
단 **픽셀 시계열 모델**이라 공간 문맥이 없음 — 이 차이 자체가 비교 축임(공간 문맥의 기여).

## Arm 정의 (동일 fold·동일 S12q·동일 seed 1/2/3·동일 선택 규칙)

| arm | encoder | 출력 격자 | 판독기 |
|---|---|---|---|
| B (기존) | frozen OlmoEarth v1 | 768ch @ 32×32 (40 m) | 작은 판독기 cin=768 |
| **C1** | frozen Presto | 128ch @ 128×128 (10 m, 픽셀별) | **같은 구조** cin=128 (proj 1×1 → 동일 블록) |
| **C2** | frozen Clay v1.5 | d @ 패치격자 (시간 평균) | 같은 구조 cin=d |
| C2b (RQ3) | frozen Clay v1.0 | 〃 | 〃 |
| A (기존) | — (scratch P2/P3) | — | — |

공정성 처리:
- 판독기는 **구조 동일, 입력 채널 수만 교체**함. 파라미터 수·FLOPs를 산출물에 기록함
- 출력 격자가 다르므로(32² vs 128²) 판독기 앞 해상도를 **encoder 본래 격자 그대로** 두고
  upsample 경로만 맞춤 — 격자를 억지로 맞추면 그 변환이 교란변수가 됨. 대신
  **토큰 격자 차이를 명시된 교란**으로 기록함(M32의 천장 논리로 상한 비교 가능)
- encoder·캐시 생성 FLOPs 포함 (M38 방식)
- FP-budget matched 평가 병행 (M44 방식)

## Label budget 축

fold의 train 라벨을 {1%, 5%, 10%, 100%}로 서브샘플(층화: 양성 타일 비율 유지,
seed별 동일 서브셋 — 서브셋 선정 seed는 20260827로 고정).
**질문**: 라벨이 적을수록 frozen이 유리한가, 어느 지점부터 scratch가 따라잡는가.

## 사전 등록 예측 (결과 관찰 전 커밋)

1. **C1(Presto)은 B(OLMo)보다 낮되 A(scratch)보다 높을 것** — 사전학습 효과는 일반적이나
   공간 문맥 부재로 OLMo에 못 미침 (틀리면: 공간 문맥 서사 재검토)
2. **1% 라벨에서 frozen 계열과 scratch의 격차가 최대**일 것
3. **틀릴 것으로 예측**: C1이 오경보(빈 타일 FP)에서 B와 동급일 것 — 픽셀 모델은
   문맥이 없어 오경보가 많을 것으로 예상함. 동급이면 "오경보 억제 = 공간 문맥" 가설 기각

## Kill gate

- C1·C2 모두 A 이하이면 → "frozen GeoFM 일반 효과" 주장 기각, OLMo 결과는 고유 효과로 격상하되 원인 분해 실험 필수
- B−C1 격차가 seed 폭 이내이면 → "OLMo 고유" 주장 금지, 일반 GeoFM 효과로 서술

## 실행 순서

1. **probe (CPU, 지금)**: Presto vendoring → Sen12 5샘플 인코딩 → 형태·결정성·결측 처리 검증
2. Clay probe (동일)
3. hiroshima 확증 종료 후 GPU1에서 C1 캐시 추출 → 3 seed 학습 (개발 fold=chimanimani에서만)
4. label budget은 C1 통과 후
5. 미열람 지역 적용은 **recipe v3 등록 후** (v2는 arm 3개로 동결돼 있음 — C축은 별도 트랙)
