# 실험 C — 두 번째 frozen GeoFM: "OLMo 고유 효과인가, 일반 GeoFM 효과인가"

작성 2026-08-27, 상태 갱신 2026-09-02. C1a common-grid와 C1b native-grid를 모두 완료했다.

> **지위 정정:** P2/P3/P4의 8-region 결과가 모두 개봉된 뒤 C1을 실행하게 됐다. 따라서 같은
> 8지역의 Presto 비교는 설정을 지금 봉인한 **matched retrospective control**이지 untouched
> confirmatory가 아니다. C1 결과를 보기 전에 계약·decoder·seed를 동결하는 가치는 남지만,
> 최초 untouched OLMo-vs-Presto 주장은 한국 또는 별도 미개봉 cohort가 맡는다.

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

### 2026-08-28 계약 감사로 닫힌 것

- 공식 upstream `nasaharvest/presto` commit을 `11e207a…`로 고정했고, 서버의 single-file code,
  normalization source, 3.3 MB weight가 upstream과 byte-identical임을 확인했다.
- 공식 normalization은 S2 전 밴드 `shift=0`, `divide=1e4`다. 기존 probe의 `/10000`은 맞았다.
- `month_to_tensor`는 2-D `[batch, 12]` tensor를 그대로 보존한다. S12q의 실제 선택 월 12개를
  넣을 수 있으므로 scalar 시작월을 쓸 이유가 없다.
- `center_lat/lon` 속성은 projected 값이므로 그대로 넣지 않는다. NetCDF CRS와 x/y로 각 픽셀의
  WGS84 좌표를 계산한다.

기계 판독 계약은 `config/presto_c1_contract.json`이다. 6,834타일 cache와 C1a common-grid
8지역×3seed는 MS-87에서 닫혔다. C1a region macro `.1092`로 P4 `.2722`와 P2 `.1966`보다
8/8 지역에서 낮았다. C1b도 `.1261`로 P4/P2에 8/8 패배해 pooling이 순위를 설명하지 못했다.
아직 안 닫힌 것은 raw recipe audit, label-budget, Korea external first-look다.

## Arm 정의 (동일 fold·동일 S12q·동일 seed 1/2/3·동일 선택 규칙)

| arm | encoder | 출력 격자 | 판독기 |
|---|---|---|---|
| B (기존) | frozen OlmoEarth v1 | 768ch @ 32×32 (40 m) | 작은 판독기 cin=768 |
| **C1a primary** | frozen Presto | 128ch @ 32×32 (native 128²를 고정 4×4 mean-pool) | **P4와 같은 공간 경로** cin=128 |
| **C1b sensitivity** | frozen Presto | 128ch @ 128×128 (10 m, 픽셀별) | `P4native`: P4와 같은 trainable layer, interpolation 없이 128²에서 실행 |
| **C2** | frozen Clay v1.5 | d @ 패치격자 (시간 평균) | 같은 구조 cin=d |
| C2b (RQ3) | frozen Clay v1.0 | 〃 | 〃 |
| A (기존) | — (scratch P2/P3) | — | — |

공정성 처리:
- **C1a가 representation-family primary**다. Presto의 native 128²를 채널별 non-overlap 4×4
  arithmetic mean으로 32²에 고정한 뒤 P4와 같은 spatial decoder/upscale path를 쓴다. signed
  feature라 GeM을 사후 선택하지 않는다.
- **C1b는 product-level sensitivity**다. native grid를 유지해 실제로 더 세밀한 Presto product를
  쓸 때의 성능과 비용을 보고한다. P4와 동일한 1×1 projection·두 conv-BN block·head를 쓰되
  두 interpolation을 제거하고 모든 block을 128²에서 실행한다. 따라서 trainable parameter는
  같고 spatial support·FLOPs만 달라진다. 이 세부 architecture는 C1a 개봉 뒤 고정됐으므로 C1b로
  OLMo 고유 우월성을 판정하거나 C1a primary를 교체하지 않는다.
- 두 arm 모두 입력 채널 projection 외 trainable parameter·FLOPs를 기록한다. pooling/readout을
  숨은 구현 상세로 두지 않는다.
- encoder·캐시 생성 FLOPs 포함 (M38 방식)
- FP-budget matched 평가 병행 (M44 방식)

## Label budget 축

fold의 train 라벨을 {1%, 5%, 10%, 100%}로 서브샘플한다. subset은 **nested**이며 region ×
positive/negative tile로 층화한다. subset seed는 20260827·20260828·20260829 최소 3개를 쓰고,
각 subset에서 arm별 sample ID가 같아야 한다. optimizer seed 3개만으로 label-sampling uncertainty를
대신하지 않는다. fraction뿐 아니라 labeled tile 수와 positive tile 수를 함께 보고한다.
**질문**: 라벨이 적을수록 frozen이 유리한가, 어느 지점부터 scratch가 따라잡는가.

1/5/10/100 네 점만으로 정확한 crossover 위치를 추정하지 않는다. 부호가 바뀌는 인접 구간이
나오면 그 사이 fraction을 추가 등록한 뒤 interpolation CI를 계산한다.

## 사전 등록 예측 (결과 관찰 전 커밋)

1. **C1(Presto)은 B(OLMo)보다 낮되 A(scratch)보다 높을 것** — 사전학습 효과는 일반적이나
   공간 문맥 부재로 OLMo에 못 미침 (틀리면: 공간 문맥 서사 재검토)
2. **1% 라벨에서 frozen 계열과 scratch의 격차가 최대**일 것
3. **틀릴 것으로 예측**: C1이 오경보(빈 타일 FP)에서 B와 동급일 것 — 픽셀 모델은
   문맥이 없어 오경보가 많을 것으로 예상함. 동급이면 "오경보 억제 = 공간 문맥" 가설 기각

## Kill gate

- C1·C2 모두 A 이하이면 → "frozen GeoFM 일반 효과" 주장 기각, OLMo 결과는 고유 효과로 격상하되 원인 분해 실험 필수
- B−C1 격차가 seed 폭 이내이면 → "OLMo 고유" 주장 금지, 일반 GeoFM 효과로 서술

## 실행 순서 — 2026-09-02 현재

1. ~~Presto 5샘플 feasibility probe~~ — **8/8 통과**.
2. ~~정규화·upstream commit·month API 확인~~ — **완료**, contract v1 봉인.
3. ~~16/64/256픽셀 smoke + exact-month/WGS84/determinism/finite 감사~~ — **완료**.
4. ~~6,834 sample Presto fp16 cache + content/file seal~~ — **완료** (`da18f121…`).
5. ~~C1a fixed mean-pool/common-grid, 동일 decoder·seed 1/2/3~~ — **완료 MS-87**.
6. ~~`P4native` exact shape·parameter parity·cache seal·source snapshot preflight~~ — **완료**.
7. ~~C1b 8지역×3seed~~ — **완료 MS-93**, `.1261`; C1a primary 유지.
8. ~~naive fusion/GeoContextGate~~ — **MS-90B/91/92 불통과, stop rule로 종료**. 승격 gate 없음.
9. raw current/official-like recipe를 source-only validation으로 감사한 뒤 label budget
   {1,5,10,100%}을 3 subset seed×3 optimizer seed로 연다(432 new runs).
10. 한국 external transfer recipe에는 P4/P2/P3/C1과 label-budget 결과를 함께 등록하고, test를 처음 열어
   OLMo-vs-Presto의 untouched 비교를 만든다.

**MS-87 claim boundary:** 결과는 “M65 이득이 이 matched Presto control에는 확장되지 않았다”를
지지한다. Presto의 native 12개월 use case 밖이고 common-grid pooling이 있으므로 “OLMoEarth가
다른 GeoFM보다 일반적으로 우월하다”는 문장은 금지한다.

**MS-93 추가 경계:** native readout은 pooling 반론을 약화하지만 Presto의 설계영역 불일치를
해결하지 않는다. `artifacts/c1b_presto_native_compact_v1.json`이 24개 원시 결과와 실행 snapshot
hash를 보존한다.
