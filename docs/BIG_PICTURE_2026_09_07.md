# 큰 그림 2026-09-07 — "캐시된 지구 표현을 언제 그대로 쓰고·head만 적응하고·조밀 재임베딩하고·raw로 돌아가는가"를 비용까지 닫는다

작성: 2026-09-07. 다른 컴퓨터의 CVPR 점수판·6분류를 장부(MEASURED_FINDINGS.md)와 대조해 검증했음. 상태 표기: **측정** / **미측정** / **철회**.

추가 검토(2026-09-07): patch-2는 특정 공간 readout의 development screen이지 방법 연구 전체의
마지막 후보가 아니다. source-trained 시간 증거/증분 갱신과 조건부 공간 잔차 후보를
[별도 설계안](NOVELTY_ARCHITECTURE_OPTIONS_2026_09_07.md)에 정리했다. 새 후보는 미실행·미등록이며,
현재 체인과 봉인 설정은 그대로 둔다. 원래 patch-2 gate와 같은-trainer 효과를 분리해 보고한다.

## 0. 이름표 (약어 금지: 모델 + 무엇을 하는지)
| 표기 | 실체 | 입력 | 학습되는 것 | 장부의 옛 이름 |
|---|---|---|---|---|
| **OlmoEarth 캐시(40 m 토큰)** | Ai2 OlmoEarth v1 Base(동결)로 128 px 칩을 한 번 인코딩해 768×32×32 fp16으로 저장. patch 크기 4 | 이 Sen12 경로는 S2 실관측 10밴드×선택 12시점. B01/B09 부재 masking; S1 미사용 | 위에 얹는 작은 conv 디코더만 | `P4`, `p4_native`, `OE@40m` |
| **OlmoEarth 캐시(20 m 토큰)** | 같은 모델·같은 칩을 patch 크기 2로 인코딩해 768×64×64로 저장. 토큰 4배, 용량 4배(41 GB vs 11 GB). 센서 해상도가 아니라 **토큰화 밀도**가 다름 | 동일 | 동일 디코더(격자만 64) | `p2_native`, `OE@20m` |
| **UNet3D raw** | 공식 Sen12 베이스라인. 원시 영상에서 처음부터 학습하는 3D U-Net | 원시 영상 | 전체 | `P2`(M65·MS-98의 ".197", ".333") |
| **U-TAE raw** | 공식 베이스라인 2, 시간 attention U-Net | 원시 영상 | 전체 | `P3`(".183") |
| **Clay v1.5 / Galileo / Prithvi 캐시** | 타 기관 EO 파운데이션 모델을 같은 방식으로 캐시한 것(MS-102: 전부 raw보다 낮음) | 각 모델 계약 | 디코더만 | `clay_cache*` 등 |

행동 이름(결정 규칙의 선택지):
| 행동 | 뜻 | 라벨 | 측정 비용(K=5, 8지역 합) |
|---|---|---|---|
| CACHED_HEAD(=A0) | 소스에서 학습한 디코더를 새 지역 캐시에 그대로 적용 | target 0 | 적응 학습 0. 캐시 생성·head 추론 비용까지 0인 것은 아님 |
| HEAD_ADAPT(=A1) | 그 디코더를 새 지역 라벨 K장으로 300 update 미세조정 | K | GPU 11–43 s, raw 읽기 0 |
| RAW_FINETUNE(=A4w) | 소스 raw UNet3D를 새 지역 라벨 K장으로 미세조정 | K | GPU 58–215 s, raw 읽기 2.7–11.2 GB |
| REEMBED | 새 지역을 더 조밀한 토큰(20 m)으로 다시 인코딩 후 head 적응 | K | **미측정**(오늘 실험 뒤 측정) |

오늘 해상도 실험의 4팔: `p4_native_control`(40 m 재현 대조군) / `p4_upsample2`(40 m 캐시를 디코더에서 64 격자로 확대만) / `p2_native`(20 m 캐시) / `p2_avgpool2`(20 m 캐시를 다시 32로 평균풀링). 20 m가 확대보다 좋고 풀링하면 이득이 사라져야 "토큰 밀도가 원인".

## 1. 점수판 대조 결과 (붙여받은 평가 vs 장부)
| 주장 | 장부 근거 | 판정 |
|---|---|---|
| 산사태 캐시 .272 > UNet3D .197 > U-TAE .183 | M65 확증 8지역 3시드 | **측정** 일치 |
| Solar 캐시 .593 vs raw .333, 8/8 | MS-98 | **측정** 일치 |
| K=5·20 캐시 head 적응 > raw 적응 16/16 | MS-96(Sen12 8/8), MS-99(Solar 층화 8/8) | **측정** 일치 — 층화 support 조건에서만 |
| (빠진 줄) Solar **random** support K=5 | MS-99: 5/8 불통과, K=20은 통과 | 논문 표에 병기. "16/16"은 층화 support 문장으로 한정 |
| (빠진 줄) 파라미터 수만으로 이점을 설명하기 어려움 | MS-97 Sen12에서 A1>A4h 7/8; MS-99 Solar 층화에서는 8/8 | **측정**. 표현이 원인이라는 인과 증명 완료는 아님 |
| (빠진 줄) 캐시 가치는 OlmoEarth에만 | MS-102/105/109: Clay·Galileo·Prithvi 캐시 < raw | **측정**(시드 1). 일반화 주장 아님 |
| raw 파라미터 줄여도 회복 안 됨 | MS-97 A4h | **측정** 일치 |
| head 적응 GPU 11–43 s·raw 0 / raw 58–215 s·2.7–11.2 GB | MS-111-v2 | **측정**(적응 단계만) |
| "빠르다" | 소스 학습·인코딩 비용 미포함 | **미측정** → §3 |
| 라벨 5장 포화 | MS-113-AUDIT | **철회** 유지 |
| selector G0 통과 | MS-112-정정 | **철회** 유지 |
| 문제 중요성 A / 방법 노벨티 C+ / 외적 일반화 B− | — | 정성 평가일 뿐 acceptance 추정이 아님. 방법 기여는 patch 실험·선택기 이외 경로도 검토 가능 |

## 2. 오늘 CPU로 새로 만든 것
1. `config/cost_ledger_v0.json` — 기존 값은 raw 26G, OE@40m 11G, OE@20m 41G라는 **반올림된 `du -sh` 수치**이며 정확한 byte 합계가 아니다. 3,825초는 첫/마지막 파일 mtime 간격이므로 계측된 GPU active seconds로 쓰면 안 된다. OE@40m 인코딩·소스 학습 시간도 **미측정**이다. 기존 ledger는 이 문서 검토에서 수정하지 않았으므로 논문 비용표에 그대로 가져오지 않는다.
2. `code/cache_breakeven.py` → `artifacts/cost/breakeven_v0.json` — 기존 가정에서 손익분기 N = 6·19 등이 나오는 **시나리오 계산**이다. 입력 비용의 불확실성과 소스 학습 비용 누락 때문에 실측 손익분기가 아니다. "캐시가 더 싸다"도 "20과업까지 회수 불가능하다"도 일반적으로 주장할 수 없다. §4에서 전체 비용을 다시 계측한다.
3. `code/policy_regret.py` → `artifacts/g0_dev/policy_regret_dev_v2.json` — 직사각 행동 행렬 위에서 정책(always:X, oracle, 향후 support-only rule)의 anchor 정규화 regret·harm·비용. 개발 2과업 재현: always:HEAD_ADAPT regret .017(=MS-111-v2 headroom), harm 1/2; always:RAW regret 1.43. 비직사각 행렬은 거부함.

## 3. 논문 문장(목표)과 현재 근거 상태
> Cached Earth representations can be more accurate than raw retraining under geographic and label scarcity (**측정**); whether they are also cheaper depends on how many tasks share the cache and on cache resolution (**미측정: 소스 단계 비용, OE@20m 효과**); we give a budget-aware rule for reuse / adapt / re-embed (**미검증: 외부 과업 regret**).

## 4. 오늘 이후 순서 (GPU 없이 되는 것 먼저) — 9/7 저녁 갱신: 다른 컴퓨터 검토(`NOVELTY_ARCHITECTURE_OPTIONS_2026_09_07.md`) 수용. "20 m 실패 = 방법 끝" 판단 철회. 후보 1(시간 증거 보존 캐시) 스크린을 등록·코드화함(`temporal_readout_screen_prereg_v0.json`). 외부 과업 후보는 `EXTERNAL_TASKS_TEMPORAL_2026_09_07.md`.
1. **[CPU] 소스 단계 비용 측정 계약**: raw UNet3D 소스 학습·캐시 head 소스 학습 GPU-s를 같은 GPU에서 1회씩 재측정(GPU 필요하나 짧음, patch 체인 뒤). 이전까지 비용 문장 금지.
2. **[GPU, 진행 중] OE@20m 4-arm** (p4 control → upsample → OE@20m → avgpool). 규칙: OE@20m ≥ OE@40m+.03 & 6/8, > upsample, 풀링 시 이득 소멸, control ±.03.
3. **[CPU] 외부 과업 프로토콜 동결**: DEN(캐시 완료)·benv2·fotw의 칩·시간·head·anchor·비용 정의 → `geobench_cache_action_prereg_v1` 직사각 행렬(3행동×3시드) 실행 준비.
4. **[CPU] 한국 3-task**: v2 큐브 제외율 보고 → 칩 격자·홀드아웃 확정 → 캐시 1회 추출(GPU, 짧음) → 라벨 개봉(1회). 시스템 검증 1건으로만 셈.
5. **[CPU] support-only 결정 규칙**: Z0/K5 특징(계약·캐시 통계·support 양성 수·비용)만으로 행동을 고르는 규칙을 `policy_regret.py`의 `rule:` 슬롯에 넣고 leave-one-task-out. 외부 과업 3개 이상에서 static 정책보다 regret·harm 모두 낮아야 방법으로 승격.
6. **[CPU] 라벨 곡선**: `label_efficiency_curve_prereg_v0` nested prefix K 실행은 GPU 필요(짧음). 순서는 2 뒤.

## 5. 중단선
- OE@20m 4-arm 불통과 → **이번 patch-2/readout 개입**은 승격하지 않는다. 이 실패만으로 다른 정보 경로·source-stage 학습까지 불가능하다고 결론내리지 않는다. 현행 selector에 유효하지 않은 행동을 억지로 추가하지 않는다.
- 외부 3과업에서 support-only 규칙이 always:HEAD_ADAPT를 못 이기면 **그 선택기 주장**은 뺀다. 기존 결과의 empirical publication과 별도로, 새 메커니즘의 개발 증거를 보고 method 방향을 결정한다. 논문 장르나 학회를 단일 gate로 자동 확정하지 않는다.
