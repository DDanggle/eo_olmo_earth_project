# 큰 그림 2026-09-07 — "캐시된 지구 표현을 언제 그대로 쓰고·head만 적응하고·조밀 재임베딩하고·raw로 돌아가는가"를 비용까지 닫는다

작성: 2026-09-07. 다른 컴퓨터의 CVPR 점수판·6분류를 장부(MEASURED_FINDINGS.md)와 대조해 검증했음. 상태 표기: **측정** / **미측정** / **철회**.

## 0. 이름표 (약어 금지: 모델 + 무엇을 하는지)
| 표기 | 실체 | 입력 | 학습되는 것 | 장부의 옛 이름 |
|---|---|---|---|---|
| **OlmoEarth 캐시(40 m 토큰)** | Ai2 OlmoEarth v1 Base(동결)로 128 px 칩을 한 번 인코딩해 768×32×32 fp16으로 저장. patch 크기 4 | S2 12밴드×3시점 + S1 | 위에 얹는 작은 conv 디코더만 | `P4`, `p4_native`, `OE@40m` |
| **OlmoEarth 캐시(20 m 토큰)** | 같은 모델·같은 칩을 patch 크기 2로 인코딩해 768×64×64로 저장. 토큰 4배, 용량 4배(41 GB vs 11 GB). 센서 해상도가 아니라 **토큰화 밀도**가 다름 | 동일 | 동일 디코더(격자만 64) | `p2_native`, `OE@20m` |
| **UNet3D raw** | 공식 Sen12 베이스라인. 원시 영상에서 처음부터 학습하는 3D U-Net | 원시 영상 | 전체 | `P2`(M65·MS-98의 ".197", ".333") |
| **U-TAE raw** | 공식 베이스라인 2, 시간 attention U-Net | 원시 영상 | 전체 | `P3`(".183") |
| **Clay v1.5 / Galileo / Prithvi 캐시** | 타 기관 EO 파운데이션 모델을 같은 방식으로 캐시한 것(MS-102: 전부 raw보다 낮음) | 각 모델 계약 | 디코더만 | `clay_cache*` 등 |

행동 이름(결정 규칙의 선택지):
| 행동 | 뜻 | 라벨 | 측정 비용(K=5, 8지역 합) |
|---|---|---|---|
| CACHED_HEAD(=A0) | 소스에서 학습한 디코더를 새 지역 캐시에 그대로 적용 | 0 | GPU 0 |
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
| (빠진 줄) 이점이 "파라미터 적어서"가 아님 | MS-97 A4h(raw를 head 크기로 축소) 8/8 패배 | **측정** |
| (빠진 줄) 캐시 가치는 OlmoEarth에만 | MS-102/105/109: Clay·Galileo·Prithvi 캐시 < raw | **측정**(시드 1). 일반화 주장 아님 |
| raw 파라미터 줄여도 회복 안 됨 | MS-97 A4h | **측정** 일치 |
| head 적응 GPU 11–43 s·raw 0 / raw 58–215 s·2.7–11.2 GB | MS-111-v2 | **측정**(적응 단계만) |
| "빠르다" | 소스 학습·인코딩 비용 미포함 | **미측정** → §3 |
| 라벨 5장 포화 | MS-113-AUDIT | **철회** 유지 |
| selector G0 통과 | MS-112-정정 | **철회** 유지 |
| 문제 중요성 A / 방법 노벨티 C+ / 외적 일반화 B− | — | 동의함. 방법 등급은 patch 실험·결정 규칙 전까지 못 올림 |

## 2. 오늘 CPU로 새로 만든 것
1. `config/cost_ledger_v0.json` — 측정 비용 고정: raw 26 GB, OE@40m 캐시 11 GB, OE@20m 캐시 41 GB(6,834타일), OE@20m 인코딩 3,825 GPU-s(H200). OE@40m 인코딩·소스 학습 시간은 **미측정**으로 명시.
2. `code/cache_breakeven.py` → `artifacts/cost/breakeven_v0.json` — 과업 수 N에 대한 캐시 경로 vs raw 경로 비용 곡선. **결과(적응 단계 비용만)**: 유리한 가정에서도 손익분기 N = 6(클라우드 가격)·19(GPU-초만), 불리한 가정에서는 20과업까지 회수 없음. **즉 "캐시가 더 싸다"는 지금 근거로는 주장 불가.** 회수 여부는 소스 단계(raw UNet3D 75 epoch vs 캐시 head 40 epoch ≈ 6분/폴드) 측정에 달림. 이게 §4의 1순위 측정.
3. `code/policy_regret.py` → `artifacts/g0_dev/policy_regret_dev_v2.json` — 직사각 행동 행렬 위에서 정책(always:X, oracle, 향후 support-only rule)의 anchor 정규화 regret·harm·비용. 개발 2과업 재현: always:HEAD_ADAPT regret .017(=MS-111-v2 headroom), harm 1/2; always:RAW regret 1.43. 비직사각 행렬은 거부함.

## 3. 논문 문장(목표)과 현재 근거 상태
> Cached Earth representations can be more accurate than raw retraining under geographic and label scarcity (**측정**); whether they are also cheaper depends on how many tasks share the cache and on cache resolution (**미측정: 소스 단계 비용, OE@20m 효과**); we give a budget-aware rule for reuse / adapt / re-embed (**미검증: 외부 과업 regret**).

## 4. 오늘 이후 순서 (GPU 없이 되는 것 먼저)
1. **[CPU] 소스 단계 비용 측정 계약**: raw UNet3D 소스 학습·캐시 head 소스 학습 GPU-s를 같은 GPU에서 1회씩 재측정(GPU 필요하나 짧음, patch 체인 뒤). 이전까지 비용 문장 금지.
2. **[GPU, 진행 중] OE@20m 4-arm** (p4 control → upsample → OE@20m → avgpool). 규칙: OE@20m ≥ OE@40m+.03 & 6/8, > upsample, 풀링 시 이득 소멸, control ±.03.
3. **[CPU] 외부 과업 프로토콜 동결**: DEN(캐시 완료)·benv2·fotw의 칩·시간·head·anchor·비용 정의 → `geobench_cache_action_prereg_v1` 직사각 행렬(3행동×3시드) 실행 준비.
4. **[CPU] 한국 3-task**: v2 큐브 제외율 보고 → 칩 격자·홀드아웃 확정 → 캐시 1회 추출(GPU, 짧음) → 라벨 개봉(1회). 시스템 검증 1건으로만 셈.
5. **[CPU] support-only 결정 규칙**: Z0/K5 특징(계약·캐시 통계·support 양성 수·비용)만으로 행동을 고르는 규칙을 `policy_regret.py`의 `rule:` 슬롯에 넣고 leave-one-task-out. 외부 과업 3개 이상에서 static 정책보다 regret·harm 모두 낮아야 방법으로 승격.
6. **[CPU] 라벨 곡선**: `label_efficiency_curve_prereg_v0` nested prefix K 실행은 GPU 필요(짧음). 순서는 2 뒤.

## 5. 중단선
- OE@20m 4-arm 불통과 → 해상도 축 삭제, 캐시 계약은 OE@40m 고정, 결정 규칙은 {reuse, adapt, raw}만. 본회의 목표 하향(EarthVision/TGRS).
- 외부 3과업에서 support-only 규칙이 always:HEAD_ADAPT를 못 이기면 "규칙"은 빼고 벤치마크+특성화 논문으로 감.
